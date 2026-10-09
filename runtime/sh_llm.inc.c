/* Llama-style decoder ops for the SmolVLA VLM prefix (SmolVLM2 text tower): RMSNorm, RoPE, SiLU-gated
 * MLP activation, masked softmax, grouped-query attention with a token mask, and the connector's pixel
 * shuffle. fp16 HBM tensors, fp16 SIMD on all three cores (the sh_rowops pattern), every op SPMD.
 *
 * Attention mask = big_vision / lerobot `make_att_2d_masks`: a uint16 per token, `tok[j]` = the
 * cumulative "attention block" id (cumsum of att_masks), SH_LLM_PAD (0xFFFF) for padding. Query i may
 * attend key j iff tok[j] <= tok[i] (an unsigned compare: PAD keys are never attended); a PAD query row
 * gets uniform probabilities (what torch's softmax of an all -inf row gives). Causal = tok[j] = j,
 * bidirectional = all 0, SmolVLA prefix = 0 for image + language tokens, PAD for language padding,
 * 1 for the state token (so nobody but the state attends the state).
 *
 * Row ops stage row blocks into TCDM (double-buffered) exactly like sh_rowop in sh_rowops.inc.c; the
 * driver here additionally gives the kernel the block's first row index (the mask needs the query
 * position), an optional per-row second input with its own width (the RoPE cos|sin table) and raw
 * fp16/uint16 parameter rows plus a per-core scratch area (the mask-class cache). */
#define SH_LLM_L1_BASE SH_ROWOPS_L1_BASE
#define SH_LLM_L1_BYTES SH_ROWOPS_L1_BYTES
#define SH_LLM_NCLS 4u                     /* cached mask classes (keep / -inf rows) per core */

typedef struct {
    uint16_t *y; const uint16_t *x; const uint16_t *b; const uint16_t *p0; const uint16_t *p1; uint8_t *scratch;
    uint32_t nr, cols, colsb, r0; const void *arg;
} sh_lblk;
typedef void (*sh_lrowfn_t)(const sh_lblk *k);
typedef struct { uint64_t y, x, b, p0, p1; uint32_t rows, cols, colsb, ldy, ldx, ldb, p0n, p1n, scratch; } sh_lrowop_args;

static void sh_llm_rowop(const sh_lrowop_args *a, sh_lrowfn_t fn, const void *arg, uint32_t cluster) {
    const uint32_t rows = a->rows, rowb = a->cols * 2, rowbb = a->b ? a->colsb * 2 : 0;
    const uint32_t p0b = (a->p0 ? a->p0n * 2 + 63 : 0) & ~63u, p1b = (a->p1 ? a->p1n * 2 + 63 : 0) & ~63u;
    const uint32_t scr = (a->scratch + 63) & ~63u, fixed = p0b + p1b + scr * ARCH_NUM_CORE_PER_CLUSTER;
    uint32_t rpb = (SH_LLM_L1_BYTES - fixed) / ((2 * rowb + rowbb) * 2); if (rpb == 0) rpb = 1; if (rpb > rows) rpb = rows;
    if (sh_set_multi(cluster)) {
        const uint32_t P = sh_set_P(cluster), want = (rows + P - 1) / P;
        if (rpb > want) rpb = want ? want : 1;
    }
    const uint32_t setb = rpb * (2 * rowb + rowbb);
    const int dm = flex_is_dm_core();
    const uint32_t ph0 = SH_LLM_L1_BASE, ph1 = ph0 + p0b, scr0 = ph1 + p1b + scr * flex_get_core_id(), sets = ph1 + p1b + scr * ARCH_NUM_CORE_PER_CLUSTER;
    const uint32_t nblk = (rows + rpb - 1) / rpb;
    uint32_t next = 0;
    while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_LSET_X(s) (sets + (s) * setb)
    #define SH_LSET_B(s) (SH_LSET_X(s) + rpb * rowb)
    #define SH_LSET_Y(s) (SH_LSET_X(s) + rpb * rowb + rpb * rowbb)
    #define SH_LLOAD(bi, s) do { const uint32_t r_ = (bi) * rpb, n_ = (rows - r_) < rpb ? (rows - r_) : rpb; \
        bare_dma_start_2d(local(SH_LSET_X(s)), a->x + (uint64_t)r_ * a->ldx * 2, rowb, rowb, a->ldx * 2, n_); \
        if (a->b) bare_dma_start_2d(local(SH_LSET_B(s)), a->b + (uint64_t)r_ * a->ldb * 2, rowbb, rowbb, a->ldb * 2, n_); } while (0)
    if (next < nblk) {
        if (dm) {
            if (a->p0) bare_dma_start_1d(local(ph0), a->p0, a->p0n * 2);
            if (a->p1) bare_dma_start_1d(local(ph1), a->p1, a->p1n * 2);
            SH_LLOAD(next, 0);
            bare_dma_wait_all();
        }
        if (scr) { volatile uint32_t *h = (volatile uint32_t *)local(scr0); for (uint32_t i = 0; i < 16; ++i) h[i] = 0xFFFFFFFFu; }
        flex_intra_cluster_sync();
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = blk * rpb, nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm && next < nblk) SH_LLOAD(next, set ^ 1);
        sh_lblk k = { (uint16_t *)local(SH_LSET_Y(set)), (const uint16_t *)local(SH_LSET_X(set)),
                      a->b ? (const uint16_t *)local(SH_LSET_B(set)) : 0, (const uint16_t *)local(ph0), (const uint16_t *)local(ph1),
                      scr ? (uint8_t *)local(scr0) : 0, nr, a->cols, a->colsb, r0, arg };
        fn(&k);
        sh_fp_fence();
        flex_intra_cluster_sync();
        if (dm) {
            bare_dma_wait_all();
            for (uint32_t r = 0; r < nr; ++r) bare_dma_start_1d(a->y + (uint64_t)(r0 + r) * a->ldy * 2, local(SH_LSET_Y(set) + r * rowb), rowb);
        }
        flex_intra_cluster_sync();
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_LSET_X
    #undef SH_LSET_B
    #undef SH_LSET_Y
    #undef SH_LLOAD
    sh_end_op(cluster);
}

/* integer stores -> later fld of the same region: feed the read-back of the last stored word into the pointer */
static inline const void *sh_llm_after_sw(const void *p, const volatile uint32_t *last) {
    uint32_t rb = *last, q;
    __asm__ volatile ("andi %0, %1, 0\n\tadd %0, %0, %2" : "=&r"(q) : "r"(rb), "r"((uint32_t)(uintptr_t)p));
    return (const void *)(uintptr_t)q;
}

/* ---- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * g (HF LlamaRMSNorm, fp32 statistics) -------------------
 * Sum of squares in fp16 SIMD with x prescaled by 1/16 (|x| up to 4095 cannot overflow a lane), folded
 * into fp32 every 16 elements; an fp32 scalar recount when the mean square is too small for that. */
static void sh_k_rmsnorm(const sh_lblk *k) {
    const float eps = *(const float *)k->arg; const uint32_t cols = k->cols, cv = cols >> 2, j0 = sh_core_rot(cv);
    const sh_v4h *g = SH_V4CP(k->p0), q4 = sh_v4_splat(0.0625f);
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; const sh_v4h *x = SH_V4CP(xr); sh_v4h *y = SH_V4P(k->y + r * cols);
        float s = 0.f; uint32_t j = 0;
        for (; j + 4 <= cv; j += 4) {
            const sh_v4h d0 = sh_v4_mul_r(x[j], q4), d1 = sh_v4_mul_r(x[j + 1], q4), d2 = sh_v4_mul_r(x[j + 2], q4), d3 = sh_v4_mul_r(x[j + 3], q4);
            sh_v4h acc = sh_v4_mul(d0, d0); acc = sh_v4_mac(acc, d1, d1); acc = sh_v4_mac(acc, d2, d2); acc = sh_v4_mac(acc, d3, d3);
            s += sh_v4_hsum(acc);
        }
        for (; j < cv; ++j) { const sh_v4h d = sh_v4_mul_r(x[j], q4); s += sh_v4_hsum(sh_v4_mul(d, d)); }
        float ms = s * (256.f / (float)cols);
        if (ms < 1e-3f) {                                   /* tiny row: fp32 recount */
            float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
            for (uint32_t i = 0; i < cols; i += 4) {
                float a0, a1, a2, a3; sh_h2f4(xr + i, &a0, &a1, &a2, &a3);
                t0 += a0 * a0; t1 += a1 * a1; t2 += a2 * a2; t3 += a3 * a3;
            }
            ms = ((t0 + t1) + (t2 + t3)) / (float)cols;
        }
        const sh_v4h rs4 = sh_v4_splat(sh_rsqrtf(ms + eps));
        for (j = j0; j < cv; ++j) y[j] = sh_v4_mul(sh_v4_mul_r(x[j], rs4), g[j]);
        for (j = 0; j < j0; ++j) y[j] = sh_v4_mul(sh_v4_mul_r(x[j], rs4), g[j]);
    }
}
void sh_rmsnorm(uint64_t y, uint64_t x, uint64_t gamma, uint32_t rows, uint32_t cols, uint32_t ld, float eps, uint32_t cluster) {
    if (cols & 3) { if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_rmsnorm] cols %u %% 4 != 0\n", cols); return; }
    sh_lrowop_args a = { y, x, 0, gamma, 0, rows, cols, 0, ld, ld, 0, cols, 0, 0 };
    sh_llm_rowop(&a, sh_k_rmsnorm, &eps, cluster);
}

/* ---- RoPE (HF Llama rotate-half convention): per head of `dh` columns, with x1 = first half, x2 = second
 * half, cos|sin the row's table (dh entries: cos[0..dh/2) then sin[0..dh/2), host precomputed for the row's
 * position id):  y1 = x1 cos - x2 sin,  y2 = x2 cos + x1 sin.  dh % 8 == 0. ------------------------- */
static void sh_k_rope(const sh_lblk *k) {
    const uint32_t dh = *(const uint32_t *)k->arg, half = dh >> 1, hv = half >> 2, cols = k->cols, nh = cols / dh;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const sh_v4h *c = SH_V4CP(k->b + r * k->colsb), *s = c + hv;
        for (uint32_t h = 0; h < nh; ++h) {
            const sh_v4h *x1 = SH_V4CP(k->x + r * cols + h * dh), *x2 = x1 + hv;
            sh_v4h *y1 = SH_V4P(k->y + r * cols + h * dh), *y2 = y1 + hv;
            for (uint32_t j = 0; j < hv; ++j) {
                const sh_v4h a = x1[j], b = x2[j], cj = c[j], sj = s[j];
                y1[j] = sh_v4_sub(sh_v4_mul(a, cj), sh_v4_mul(b, sj));
                y2[j] = sh_v4_mac(sh_v4_mul(b, cj), a, sj);
            }
        }
    }
}
void sh_rope(uint64_t y, uint64_t x, uint64_t cos_sin, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t ld_table, uint32_t dh, uint32_t cluster) {
    if ((dh & 7) || cols % dh) { if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_rope] need dh %% 8 == 0 and cols %% dh == 0 (dh=%u cols=%u)\n", dh, cols); return; }
    sh_lrowop_args a = { y, x, cos_sin, 0, 0, rows, cols, dh, ld, ld, ld_table, 0, 0, 0 };
    sh_llm_rowop(&a, sh_k_rope, &dh, cluster);
}

/* ---- y = silu(a) * b = a b / (1 + 2^(-a log2 e)) in fp16 SIMD (exponent clamped to the fp16 range) ---- */
static void sh_k_silu_mul(const sh_lblk *k) {
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h nl4 = sh_v4_splat(-SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), c15 = sh_v4_splat_h(SH_C15);
    uint32_t lo, hi; sh_share(k->nr * k->cols, 16, &lo, &hi);
    const sh_v4h *x = SH_V4CP(k->x), *b = SH_V4CP(k->b); sh_v4h *y = SH_V4P(k->y);
    #define SH_SILU_T(v) sh_v4_min_r(sh_v4_max_r(sh_v4_mul_r(v, nl4), cm14), c15)
    uint32_t i = lo >> 2; const uint32_t e = hi >> 2;
    for (; i + 4 <= e; i += 4) {
        sh_v4h t[4] = { SH_SILU_T(x[i]), SH_SILU_T(x[i + 1]), SH_SILU_T(x[i + 2]), SH_SILU_T(x[i + 3]) };
        sh_v4_exp2x4(t, &ec);
        y[i] = sh_v4_div(sh_v4_mul(x[i], b[i]), sh_v4_add_r(t[0], ec.one));             y[i + 1] = sh_v4_div(sh_v4_mul(x[i + 1], b[i + 1]), sh_v4_add_r(t[1], ec.one));
        y[i + 2] = sh_v4_div(sh_v4_mul(x[i + 2], b[i + 2]), sh_v4_add_r(t[2], ec.one)); y[i + 3] = sh_v4_div(sh_v4_mul(x[i + 3], b[i + 3]), sh_v4_add_r(t[3], ec.one));
    }
    for (; i < e; ++i) y[i] = sh_v4_div(sh_v4_mul(x[i], b[i]), sh_v4_add_r(sh_v4_exp2(SH_SILU_T(x[i]), &ec), ec.one));
    #undef SH_SILU_T
}
void sh_silu_mul(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    if (cols & 3) { if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_silu_mul] cols %u %% 4 != 0\n", cols); return; }
    sh_lrowop_args args = { y, a, b, 0, 0, rows, cols, cols, ld, ld, ld, 0, 0, 0 };
    sh_llm_rowop(&args, sh_k_silu_mul, 0, cluster);
}

/* ---- masked softmax rows ---------------------------------------------------------------------------
 * The mask of query class ti is two fp16 rows over the keys: keep[j] = 1/0 and nb[j] = 0/-inf
 * (tok[j] <= ti). They are built once per distinct ti and cached per core (SH_LLM_NCLS slots in the
 * core's scratch: a 64 B header of class ids + round-robin cursor, then keep|nb pairs of S entries). */
static inline uint32_t sh_llm_cls_bytes(uint32_t S) { return 64u + SH_LLM_NCLS * 2u * S * 2u; }
static inline void sh_llm_cls_rows(uint8_t *scr, const uint16_t *tok, uint32_t S, uint32_t ti, const sh_v4h **keep, const sh_v4h **nb) {
    volatile uint32_t *h = (volatile uint32_t *)scr;      /* h[0..NCLS) ids, h[NCLS] next slot (header initialised to ~0) */
    uint32_t slot = SH_LLM_NCLS;
    for (uint32_t c = 0; c < SH_LLM_NCLS; ++c) if (h[c] == ti) { slot = c; break; }
    const int build = slot == SH_LLM_NCLS;
    if (build) { slot = h[SH_LLM_NCLS] & (SH_LLM_NCLS - 1); h[SH_LLM_NCLS] = slot + 1; h[slot] = ti; }
    uint32_t *kw = (uint32_t *)(scr + 64 + slot * 2 * S * 2), *nw = kw + (S >> 1);
    if (build) {
        for (uint32_t j = 0; j < S; j += 2) {
            const uint32_t a = tok[j] <= ti, b = tok[j + 1] <= ti;
            kw[j >> 1] = (a ? 0x3C00u : 0u) | (b ? 0x3C000000u : 0u);
            nw[j >> 1] = (a ? 0u : 0xFC00u) | (b ? 0u : 0xFC000000u);
        }
        kw = (uint32_t *)sh_llm_after_sw(kw, &nw[(S >> 1) - 1]); nw = kw + (S >> 1);
    }
    *keep = (const sh_v4h *)kw; *nb = (const sh_v4h *)nw;
}
typedef struct { sh_v4_exp2_consts ec; sh_v4h s24, cm14, ones; float s2; } sh_sm_consts;
static inline sh_sm_consts sh_sm_init(float scale) {
    sh_sm_consts c; c.ec = sh_v4_exp2_init(); c.s2 = scale * SH_LOG2E; c.s24 = sh_v4_splat(c.s2); c.cm14 = sh_v4_splat_h(SH_CM14); c.ones = sh_v4_splat_h(0x3C00u);
    return c;
}
/* y = 2^(s2 (x + nb) - m2) keep (unnormalised, fp16), m2 = s2 max_j (x_j + nb_j); returns the fp32 row sum.
 * keep == 0: no mask. cv = cols / 4. In place allowed. */
static inline float sh_llm_softmax_row(const sh_v4h *x, sh_v4h *y, uint32_t cv, const sh_sm_consts *c, const sh_v4h *keep, const sh_v4h *nb) {
    sh_v4h m0 = sh_v4_splat_h(0xFC00u), m1 = m0; uint32_t j;
    if (keep) {
        for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_max(m0, sh_v4_add(x[j], nb[j])); m1 = sh_v4_max(m1, sh_v4_add(x[j + 1], nb[j + 1])); }
        if (j < cv) m0 = sh_v4_max(m0, sh_v4_add(x[j], nb[j]));
    } else {
        for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_max(m0, x[j]); m1 = sh_v4_max(m1, x[j + 1]); }
        if (j < cv) m0 = sh_v4_max(m0, x[j]);
    }
    const sh_v4h m24 = sh_v4_splat(sh_v4_hmax(sh_v4_max(m0, m1)) * c->s2);
    sh_v4h acc = sh_v4_splat_h(0); float sum = 0.f;
    #define SH_LSM_T(j) sh_v4_max_r(sh_v4_sub_r(sh_v4_mul_r(keep ? sh_v4_add(x[j], nb[j]) : x[j], c->s24), m24), c->cm14)
    for (j = 0; j + 4 <= cv; j += 4) {
        sh_v4h t[4] = { SH_LSM_T(j), SH_LSM_T(j + 1), SH_LSM_T(j + 2), SH_LSM_T(j + 3) };
        sh_v4_exp2x4(t, &c->ec);
        if (keep) { t[0] = sh_v4_mul(t[0], keep[j]); t[1] = sh_v4_mul(t[1], keep[j + 1]); t[2] = sh_v4_mul(t[2], keep[j + 2]); t[3] = sh_v4_mul(t[3], keep[j + 3]); }
        y[j] = t[0]; y[j + 1] = t[1]; y[j + 2] = t[2]; y[j + 3] = t[3];
        acc = sh_v4_add(sh_v4_add(acc, sh_v4_add(t[0], t[1])), sh_v4_add(t[2], t[3]));
        if ((j & 4) == 4) { sum += sh_v4_hsum(acc); acc = sh_v4_splat_h(0); }
    }
    for (; j < cv; ++j) { sh_v4h e = sh_v4_exp2(SH_LSM_T(j), &c->ec); if (keep) e = sh_v4_mul(e, keep[j]); y[j] = e; acc = sh_v4_add(acc, e); }
    #undef SH_LSM_T
    return sum + sh_v4_hsum(acc);
}
typedef struct { float scale; uint32_t S; } sh_sm_arg;
static void sh_k_softmax_masked(const sh_lblk *k) {       /* p0 = tok_k (cols), p1 = tok_q (rows), scratch = class cache */
    const sh_sm_arg *a = (const sh_sm_arg *)k->arg; const uint32_t cols = k->cols, cv = cols >> 2;
    const sh_sm_consts c = sh_sm_init(a->scale);
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const sh_v4h *x = SH_V4CP(k->x + r * cols); sh_v4h *y = SH_V4P(k->y + r * cols);
        const uint32_t ti = k->p1[k->r0 + r];
        if (ti == SH_LLM_PAD) { const sh_v4h u = sh_v4_splat(1.f / (float)cols); for (uint32_t j = 0; j < cv; ++j) y[j] = u; continue; }
        const sh_v4h *keep, *nb; sh_llm_cls_rows(k->scratch, k->p0, cols, ti, &keep, &nb);
        const float sum = sh_llm_softmax_row(x, y, cv, &c, keep, nb);
        const sh_v4h inv4 = sh_v4_splat(1.f / sum);
        for (uint32_t j = 0; j < cv; ++j) y[j] = sh_v4_mul_r(y[j], inv4);
    }
}
void sh_softmax_masked(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float scale, uint64_t tok_q, uint64_t tok_k, uint32_t cluster) {
    if ((cols & 3) || scale <= 0.f) { if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_softmax_masked] need cols %% 4 == 0 and scale > 0\n"); return; }
    sh_sm_arg arg = { scale, cols };
    sh_lrowop_args a = { y, x, 0, tok_k, tok_q, rows, cols, 0, ld, ld, 0, cols, rows, sh_llm_cls_bytes(cols) };
    sh_llm_rowop(&a, sh_k_softmax_masked, &arg, cluster);
}

/* The masked head keeps the whole head resident (no q blocking yet); limits and stamps are its own. */
#define SH_LLM_ATTN_MAX_S 256u
#define SH_LLM_STAMP(i) do { if (first) ((volatile uint32_t *)local(l.prof))[i] = sh_mcycle(); } while (0)
/* ---- one attention head with a token mask: o[S,dh] = softmax(scale q k^T + mask) v, all in TCDM --------
 * Same staging / RedMulE / deferred-normalisation structure as sh_attention_head (sh_attention.inc.c; the
 * layout helpers are reused), softmax rows in fp16 SIMD (sh_llm_softmax_row) instead of scalar fp32. The
 * token array (S uint16, or tok == 0 for no mask) is staged behind the head's buffers, then one class cache
 * per core. */
int sh_attention_head_masked(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t dh,
                             uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint64_t tok, uint32_t cluster) {
    if (cluster != SH_ALL && sh_set_r(cluster) == SH_SET_NONE) return 0;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id(), NC = ARCH_NUM_CORE_PER_CLUSTER;
    const sh_attn_l1 l = sh_attn_layout(S, dh, S, SH_ATTN_L1_BASE);   /* whole head resident (sq = S) */
    if (first) *(volatile uint32_t *)local(SH_ATTN_L1_BASE - 4) = l.prof;   /* sh_attention_profile() reads the stamps from here */
    const uint32_t ltok = (l.end + 63) & ~63u, lcls = ltok + ((S * 2 + 63) & ~63u), clsb = (sh_llm_cls_bytes(S) + 63) & ~63u, lend = lcls + NC * clsb;
    if (S > SH_LLM_ATTN_MAX_S || (S & 3) || (dh & 3) || scale <= 0.f || lend > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_attention_head_masked] S=%u dh=%u: need S <= %u, S %% 4 == 0, dh %% 4 == 0, scale > 0, L1 %u <= %u\n",
                             S, dh, SH_LLM_ATTN_MAX_S, lend, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -1;
    }
    SH_LLM_STAMP(0);
    if (dm) {
        sh_load_block_async(l.q, q, S, dh, ldq);
        sh_load_block_async(l.k, k, S, dh, ldk);
        sh_load_block_async(l.v, v, S, dh, ldv);
        if (tok) bare_dma_start_1d(local(ltok), tok, S * 2);
        sh_l1_zero_dm(l.s, S * S * 2);
        sh_l1_zero_dm(l.o, S * dh * 2);
        for (uint32_t c = 0; c < dh; ++c) bare_dma_start_2d(local(l.kt + c * S * 2), local(l.k + c * 2), 2, 2, dh * 2, S);
        bare_dma_wait_all();
    }
    { volatile uint32_t *h = (volatile uint32_t *)local(lcls + core * clsb); for (uint32_t i = 0; i < 16; ++i) h[i] = 0xFFFFFFFFu; }
    flex_intra_cluster_sync();
    SH_LLM_STAMP(1); SH_LLM_STAMP(2);
    if (first) { flex_redmule_config(S, dh, S); flex_redmule_trigger(l.q, l.kt, l.s, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_LLM_STAMP(3);
    {
        const sh_sm_consts c = sh_sm_init(scale);
        const uint16_t *tk = (const uint16_t *)local(ltok); uint8_t *scr = (uint8_t *)local(lcls + core * clsb);
        float *sum = (float *)local(l.sum); const uint32_t cv = S >> 2;
        for (uint32_t r = core; r < S; r += NC) {
            sh_v4h *row = SH_V4P(local(l.s + r * S * 2));
            if (!tok) { sum[r] = sh_llm_softmax_row(row, row, cv, &c, 0, 0); continue; }
            const uint32_t ti = tk[r];
            if (ti == SH_LLM_PAD) { for (uint32_t j = 0; j < cv; ++j) row[j] = c.ones; sum[r] = (float)S; continue; }
            const sh_v4h *keep, *nb; sh_llm_cls_rows(scr, tk, S, ti, &keep, &nb);
            sum[r] = sh_llm_softmax_row(row, row, cv, &c, keep, nb);
        }
    }
    sh_fp_fence();
    flex_intra_cluster_sync();
    SH_LLM_STAMP(4);
    if (first) { flex_redmule_config(S, S, dh); flex_redmule_trigger(l.s, l.v, l.o, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_LLM_STAMP(5);
    {
        const float *sum = (const float *)local(l.sum); const uint32_t dv = dh >> 2;
        for (uint32_t r = core; r < S; r += NC) {
            sh_v4h *ow = SH_V4P(local(l.o + r * dh * 2)); const sh_v4h inv4 = sh_v4_splat(1.f / sum[r]);
            for (uint32_t j = 0; j < dv; ++j) ow[j] = sh_v4_mul_r(ow[j], inv4);
        }
    }
    sh_fp_fence();
    flex_intra_cluster_sync();
    if (dm) sh_store_block_sync(o, l.o, S, dh, ldo);
    flex_intra_cluster_sync();
    SH_LLM_STAMP(6);
    return 0;
}

/* Grouped-query attention: H query heads (columns [h*dh, (h+1)*dh) of q / o, dh = D / H) over Hkv key/value
 * heads (columns [(h / (H/Hkv))*dh, ...) of k / v), optional token mask. cluster == SH_ALL deals head h to
 * cluster h % P and ends with a global barrier. */
SH_FAR int sh_attention_gqa(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t D, uint32_t H, uint32_t Hkv,
                     uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint64_t tok, uint32_t cluster) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, dh = H ? D / H : 0;
    int rc = 0;
    if (H == 0 || Hkv == 0 || dh * H != D || H % Hkv) {
        if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_attention_gqa] bad D=%u H=%u Hkv=%u\n", D, H, Hkv);
        return -1;
    }
    const uint32_t grp = H / Hkv;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t cl = sh_set_nth(cluster, h);   /* SH_ALL: h % P; a set: its (h mod size)-th member */
        const uint64_t qo = (uint64_t)h * dh * 2, kvo = (uint64_t)(h / grp) * dh * 2;
        int r = sh_attention_head_masked(q + qo, k + kvo, v + kvo, o + qo, S, dh, ldq, ldk, ldv, ldo, scale, tok, cl);
        if (r) rc = r;
    }
    sh_set_end(cluster);
    return rc;
}

/* ---- SmolVLM connector pixel shuffle: src [grid*grid, D] (raster patch tokens) -> dst [(grid/s)^2, D s^2]:
 * output token (gr, gb) = the s x s block of patches rows gr*s.., cols gb*s.., row-major, each D wide
 * (== SmolVLMConnector.pixel_shuffle). Pure data movement: per token s 1-D loads of s*D elements into TCDM
 * and one store; tokens dealt round-robin over clusters. ------------------------------------------------ */
SH_FAR void sh_pixel_shuffle(uint64_t dst, uint64_t src, uint32_t grid, uint32_t D, uint32_t s, uint32_t cluster) {
    const uint32_t g = grid / s, ntok = g * g, chunk = s * D * 2, P = sh_set_P(cluster), cid = sh_set_r(cluster);
    if (flex_is_dm_core() && cid != SH_SET_NONE) {
        for (uint32_t t = 0; t < ntok; ++t) {
            if ((t % P) != cid) continue;
            const uint32_t gr = t / g, gb = t % g;
            for (uint32_t i = 0; i < s; ++i)
                bare_dma_start_1d(local(SH_LLM_L1_BASE + i * chunk), src + ((uint64_t)((gr * s + i) * grid + gb * s) * D) * 2, chunk);
            bare_dma_wait_all();
            bare_dma_start_1d(dst + (uint64_t)t * s * chunk, local(SH_LLM_L1_BASE), s * chunk);
            bare_dma_wait_all();
        }
    }
    sh_end_op(cluster);
}
