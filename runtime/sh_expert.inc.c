/* SmolVLA action-expert ops (prefix sh_x_): the SiLU-gated activation with per-operand leading dimensions (and
 * plain SiLU), the flow-matching Euler update (axpy) and a GQA cross/self attention over a stationary prefix KV
 * with a token-class mask. RMSNorm / RoPE / the equal-ld silu_mul come from sh_llm.inc.c (agent/vlm-prefix);
 * see docs/SMOLVLA_EXPERT.md.
 *
 * The row ops reuse the double-buffered sh_rowop driver of sh_rowops.inc.c (block of rows staged by
 * the DM core, split over the three cores, fp16 SIMD, cols % 4 == 0; scalar fp32 fallback otherwise).
 *
 * sh_x_attention: for query head h (dh columns of q) with kv head h / (H / Hkv):
 *     keys   = [ kp (Lp prefix rows, stationary in HBM, dealt to the cluster by head) ; ko (So own rows) ]
 *     values = [ vp ; vo ]
 *     o_h = softmax(scale * q_h K^T + mask) V,  mask: prefix key j is padding when tok[j] == SH_LLM_PAD (the VLM
 *           prefix's uint16 token-class array, sh_llm.inc.c convention: 0 image/language, 1 state, 0xFFFF padding;
 *           the expert's queries are class 2, so "tok[j] <= tok[i]" reduces to "not padding"; tok == 0: all valid),
 *           own key j attended by query i only when j <= i (causal; lerobot's make_att_2d_masks with att_mask = 1
 *           on every action token)
 * entirely inside one cluster's TCDM: K rows staged and transposed by element-granular 2-D DMA into
 * kT[dh, Lpad] (Lpad = L rounded up to 32, padding columns zero), RedMulE E = q kT, a masked fp16 SIMD row
 * softmax on the cores (validity rows per core: a 1/0 multiplier and a 0/-65504 additive bias, so masked
 * and padding columns are exactly 0 after the exp), RedMulE o = E V, 1/rowsum applied to o. cluster ==
 * SH_ALL deals head h to cluster h % P and ends with a global barrier. */
#define SH_X_L1_BASE 0x1000u      /* keep clear of TCDM 0 (NULL) and the SIMD scratch at 0x800 */
#define SH_X_NPROF 8u

static inline uint32_t sh_x_up64(uint32_t b) { return (b + 63u) & ~63u; }

/* ---- silu_mul: y = silu(a) * b = a b / (1 + 2^(-a log2 e)); b == 0 -> y = silu(a) ------------------ */
static inline float sh_x_silu1(float a) { return a / (1.f + sh_exp2_clamped(sh_fminf(sh_fmaxf(-a * SH_LOG2E, -126.f), 126.f))); }
static void sh_xk_silu_mul_s(const sh_blk *k) {
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    const uint16_t *x = k->x, *b = k->b; uint16_t *y = k->y;
    for (uint32_t i = lo; i < hi; ++i) { float v = sh_x_silu1(sh_h2f(x[i])); if (b) v *= sh_h2f(b[i]); y[i] = (uint16_t)sh_f2h(v); }
}
static void sh_xk_silu_mul(const sh_blk *k) {
    if (k->cols & 3) { sh_xk_silu_mul_s(k); return; }
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h nl = sh_v4_splat(-SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), c15 = sh_v4_splat_h(SH_C15);
    uint32_t lo, hi; sh_share(k->nr * k->cols, 16, &lo, &hi);
    const sh_v4h *x = SH_V4CP(k->x), *b = k->b ? SH_V4CP(k->b) : 0; sh_v4h *y = SH_V4P(k->y);
    #define SH_X_SILU_T(v) sh_v4_min_r(sh_v4_max_r(sh_v4_mul_r(v, nl), cm14), c15)
    #define SH_X_NUM(i) (b ? sh_v4_mul(x[i], b[i]) : x[i])
    uint32_t i = lo >> 2; const uint32_t e = hi >> 2;
    for (; i + 4 <= e; i += 4) {
        sh_v4h t[4] = { SH_X_SILU_T(x[i]), SH_X_SILU_T(x[i + 1]), SH_X_SILU_T(x[i + 2]), SH_X_SILU_T(x[i + 3]) };
        sh_v4_exp2x4(t, &ec);
        y[i] = sh_v4_div(SH_X_NUM(i), sh_v4_add_r(t[0], ec.one));         y[i + 1] = sh_v4_div(SH_X_NUM(i + 1), sh_v4_add_r(t[1], ec.one));
        y[i + 2] = sh_v4_div(SH_X_NUM(i + 2), sh_v4_add_r(t[2], ec.one)); y[i + 3] = sh_v4_div(SH_X_NUM(i + 3), sh_v4_add_r(t[3], ec.one));
    }
    for (; i < e; ++i) y[i] = sh_v4_div(SH_X_NUM(i), sh_v4_add_r(sh_v4_exp2(SH_X_SILU_T(x[i]), &ec), ec.one));
    #undef SH_X_SILU_T
    #undef SH_X_NUM
}
void sh_x_silu_mul(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ldy, uint32_t lda, uint32_t ldb, uint32_t cluster) {
    sh_rowop(y, a, b, 0, 0, rows, cols, ldy, lda, ldb, sh_xk_silu_mul, 0, cluster);
}

/* ---- axpy: y = a + alpha * b (the Euler update x_t <- x_t + dt v_t) -------------------------------- */
static void sh_xk_axpy_s(const sh_blk *k) {
    const float al = *(const float *)k->arg;
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    for (uint32_t i = lo; i < hi; ++i) k->y[i] = (uint16_t)sh_f2h(sh_h2f(k->x[i]) + al * sh_h2f(k->b[i]));
}
static void sh_xk_axpy(const sh_blk *k) {
    if (k->cols & 3) { sh_xk_axpy_s(k); return; }
    const sh_v4h al4 = sh_v4_splat(*(const float *)k->arg);
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    const sh_v4h *x = SH_V4CP(k->x), *b = SH_V4CP(k->b); sh_v4h *y = SH_V4P(k->y);
    for (uint32_t i = lo >> 2, e = hi >> 2; i < e; ++i) y[i] = sh_v4_mac_r(x[i], b[i], al4);
}
void sh_x_axpy(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ldy, uint32_t lda, uint32_t ldb, float alpha, uint32_t cluster) {
    sh_rowop(y, a, b, 0, 0, rows, cols, ldy, lda, ldb, sh_xk_axpy, &alpha, cluster);
}

/* ---- GQA cross / self attention over a stationary prefix KV ---------------------------------------- */
typedef struct { uint32_t q, k, kt, s, v, o, sum, vrow, vld, prof, end, Lpad; } sh_x_attn_l1;

static inline sh_x_attn_l1 sh_x_attn_layout(uint32_t Sq, uint32_t dh, uint32_t L, uint32_t base) {
    sh_x_attn_l1 l; const uint32_t Lpad = (L + 31u) & ~31u, NC = ARCH_NUM_CORE_PER_CLUSTER;
    l.Lpad = Lpad;
    l.q = base;                               l.k = l.q + sh_x_up64(Sq * dh * 2);
    l.kt = l.k + sh_x_up64(L * dh * 2);       l.s = l.kt + sh_x_up64(dh * Lpad * 2);
    l.v = l.s + sh_x_up64(Sq * Lpad * 2);     l.o = l.v + sh_x_up64(Lpad * dh * 2);
    l.sum = l.o + sh_x_up64(Sq * dh * 2);     l.vrow = l.sum + sh_x_up64(Sq * 4);
    l.vld = l.vrow + sh_x_up64(Lpad * 2);     l.prof = l.vld + NC * 2 * sh_x_up64(Lpad * 2);
    l.end = l.prof + sh_x_up64(SH_X_NPROF * 4);
    return l;
}
uint32_t sh_x_attention_l1_bytes(uint32_t Sq, uint32_t L, uint32_t dh) { return sh_x_attn_layout(Sq, dh, L, SH_X_L1_BASE).end; }
uint32_t sh_x_attention_profile(uint32_t Sq, uint32_t L, uint32_t dh, uint32_t phase) {
    return ((volatile uint32_t *)local(sh_x_attn_layout(Sq, dh, L, SH_X_L1_BASE).prof))[phase & (SH_X_NPROF - 1)];
}
#define SH_X_STAMP(i) do { if (first) ((volatile uint32_t *)local(l.prof))[i] = sh_mcycle(); } while (0)

/* int -> FP ordering: the returned pointer equals `base` but depends on a read-back of the word holding the
 * half just stored, so an fld through it cannot issue before the integer store completed (SIMULATOR_NOTES #8). */
static inline const sh_v4h *sh_x_after_sh(volatile uint16_t *p, const sh_v4h *base) {
    uint32_t rb = *(volatile uint32_t *)((uintptr_t)p & ~3u), a;
    __asm__ volatile ("andi %0, %1, 0\n\tadd %0, %0, %2" : "=&r"(a) : "r"(rb), "r"((uint32_t)(uintptr_t)base));
    return (const sh_v4h *)(uintptr_t)a;
}

/* One score row (cv vectors = Lpad fp16): x <- exp(s2 (x - max) ) on the valid columns, 0 elsewhere; returns the fp32
 * row sum. vld = 1.0 / 0.0 per column, nb = 0 / -65504: xm = nb + x vld is the masked score. */
static inline float sh_x_softmax_row(sh_v4h *x, const sh_v4h *vld, const sh_v4h *nb, uint32_t cv, float s2,
                                     const sh_v4_exp2_consts *ec, sh_v4h s24, sh_v4h cm14) {
    sh_v4h m0 = sh_v4_splat_h(0xFC00u), m1 = m0; uint32_t j;
    for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_max(m0, sh_v4_mac(nb[j], x[j], vld[j])); m1 = sh_v4_max(m1, sh_v4_mac(nb[j + 1], x[j + 1], vld[j + 1])); }
    const float m2 = sh_v4_hmax(sh_v4_max(m0, m1)) * s2;
    const sh_v4h m24 = sh_v4_splat(m2);
    sh_v4h acc = sh_v4_splat_h(0); float sum = 0.f;
    #define SH_X_T(j) sh_v4_max_r(sh_v4_sub_r(sh_v4_mul_r(sh_v4_mac(nb[j], x[j], vld[j]), s24), m24), cm14)
    for (j = 0; j + 4 <= cv; j += 4) {   /* cv % 8 == 0 (Lpad % 32) */
        sh_v4h t[4] = { SH_X_T(j), SH_X_T(j + 1), SH_X_T(j + 2), SH_X_T(j + 3) };
        sh_v4_exp2x4(t, ec);
        t[0] = sh_v4_mul(t[0], vld[j]); t[1] = sh_v4_mul(t[1], vld[j + 1]); t[2] = sh_v4_mul(t[2], vld[j + 2]); t[3] = sh_v4_mul(t[3], vld[j + 3]);
        x[j] = t[0]; x[j + 1] = t[1]; x[j + 2] = t[2]; x[j + 3] = t[3];
        acc = sh_v4_add(sh_v4_add(acc, sh_v4_add(t[0], t[1])), sh_v4_add(t[2], t[3]));
        if ((j & 4) == 4) { sum += sh_v4_hsum(acc); acc = sh_v4_splat_h(0); }
    }
    #undef SH_X_T
    return sum + sh_v4_hsum(acc);
}

int sh_x_attention_head(uint64_t q, uint64_t kp, uint64_t vp, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o,
                        uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t dh, uint32_t ldq, uint32_t ldkp, uint32_t ldvp,
                        uint32_t ldko, uint32_t ldvo, uint32_t ldo, float scale, uint32_t cluster) {
    if (cluster != SH_ALL && flex_get_cluster_id() != cluster) return 0;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id(), NC = ARCH_NUM_CORE_PER_CLUSTER, L = Lp + So;
    const sh_x_attn_l1 l = sh_x_attn_layout(Sq, dh, L, SH_X_L1_BASE);
    const uint32_t Lpad = l.Lpad, cv = Lpad >> 2, vb = sh_x_up64(Lpad * 2);
    if ((dh & 3) || Sq == 0 || L == 0 || (ko == 0) != (So == 0) || l.end > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_x_attention_head] Sq=%u Lp=%u So=%u dh=%u: need dh %% 4 == 0, own K/V iff So > 0, L1 %u <= %u\n",
                             Sq, Lp, So, dh, l.end, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -1;
    }
    SH_X_STAMP(0);
    /* 1. stage q, K rows (prefix then own), V rows; zero kT / scores / o / V padding; transpose K by DMA */
    if (dm) {
        sh_load_block_async(l.q, q, Sq, dh, ldq);
        sh_load_block_async(l.k, kp, Lp, dh, ldkp);
        sh_load_block_async(l.v, vp, Lp, dh, ldvp);
        if (So) { sh_load_block_async(l.k + Lp * dh * 2, ko, So, dh, ldko); sh_load_block_async(l.v + Lp * dh * 2, vo, So, dh, ldvo); }
        if (tok) bare_dma_start_1d(local(l.vrow), tok, Lp * 2);
        sh_l1_zero_dm(l.kt, dh * Lpad * 2);               /* waits for everything issued so far */
        sh_l1_zero_dm(l.s, Sq * Lpad * 2);
        sh_l1_zero_dm(l.o, Sq * dh * 2);
        if (Lpad > L) sh_l1_zero_dm(l.v + L * dh * 2, (Lpad - L) * dh * 2);
        for (uint32_t c = 0; c < dh; ++c)                   /* kT row c = column c of K: L elements of 2 B, stride dh*2 */
            bare_dma_start_2d(local(l.kt + c * Lpad * 2), local(l.k + c * 2), 2, 2, dh * 2, L);
        bare_dma_wait_all();
    }
    flex_intra_cluster_sync();
    /* 2. per-core validity rows: prefix from `tok` (padding class 0xFFFF -> 0, else 1; all 1 without tok), own columns
     *    [0, lo] of this core's first row, padding columns 0 */
    uint32_t lo, hi; sh_share(Sq, 1, &lo, &hi);
    volatile uint16_t *vld = (volatile uint16_t *)local(l.vld + core * 2 * vb), *nb = vld + (vb >> 1);
    {
        const volatile uint16_t *vr = (const volatile uint16_t *)local(l.vrow);
        for (uint32_t j = 0; j < Lpad; ++j) {
            uint32_t v;
            if (j < Lp) v = tok ? (vr[j] != SH_LLM_PAD) : 1u;
            else if (j < L) v = (j - Lp) <= lo;
            else v = 0;
            vld[j] = v ? 0x3C00u : 0u; nb[j] = v ? 0u : 0xFBFFu;
        }
    }
    SH_X_STAMP(1);
    /* 3. E = q kT on RedMulE */
    if (first) { flex_redmule_config(Sq, dh, Lpad); flex_redmule_trigger(l.q, l.kt, l.s, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_X_STAMP(2);
    /* 4. masked row softmax numerators (rows split over the cores), fp32 row sums */
    {
        const sh_v4_exp2_consts ec = sh_v4_exp2_init();
        const float s2 = scale * SH_LOG2E;
        const sh_v4h s24 = sh_v4_splat(s2), cm14 = sh_v4_splat_h(SH_CM14);
        float *sum = (float *)local(l.sum);
        const sh_v4h *vv = (const sh_v4h *)(uintptr_t)vld, *nn = (const sh_v4h *)(uintptr_t)nb;
        vv = sh_x_after_sh(nb + Lpad - 1, vv);
        for (uint32_t r = lo; r < hi; ++r) {
            if (So && r > lo) { vld[Lp + r] = 0x3C00u; nb[Lp + r] = 0u; vv = sh_x_after_sh(nb + Lp + r, vv); }
            sum[r] = sh_x_softmax_row(SH_V4P(local(l.s + r * Lpad * 2)), vv, nn, cv, s2, &ec, s24, cm14);
        }
        sh_fp_fence();
    }
    flex_intra_cluster_sync();
    SH_X_STAMP(3);
    /* 5. o = E V on RedMulE */
    if (first) { flex_redmule_config(Sq, Lpad, dh); flex_redmule_trigger(l.s, l.v, l.o, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_X_STAMP(4);
    /* 6. o_i /= sum_i (fp16 SIMD), store */
    {
        const float *sum = (const float *)local(l.sum); const uint32_t dv = dh >> 2;
        for (uint32_t r = lo; r < hi; ++r) {
            sh_v4h *ov = SH_V4P(local(l.o + r * dh * 2)); const sh_v4h inv4 = sh_v4_splat(1.f / sum[r]);
            for (uint32_t j = 0; j < dv; ++j) ov[j] = sh_v4_mul_r(ov[j], inv4);
        }
        sh_fp_fence();
    }
    flex_intra_cluster_sync();
    if (dm) sh_store_block_sync(o, l.o, Sq, dh, ldo);
    flex_intra_cluster_sync();
    SH_X_STAMP(5);
    return 0;
}

int sh_x_attention(uint64_t q, uint64_t kp, uint64_t vp, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o,
                   uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t H, uint32_t Hkv, uint32_t dh,
                   uint32_t ldq, uint32_t ldkp, uint32_t ldvp, uint32_t ldko, uint32_t ldvo, uint32_t ldo,
                   float scale, uint32_t cluster) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    int rc = 0;
    if (H == 0 || Hkv == 0 || H % Hkv) {
        if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_x_attention] H=%u not a multiple of Hkv=%u\n", H, Hkv);
        return -1;
    }
    const uint32_t grp = H / Hkv;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t cl = (cluster == SH_ALL) ? h % P : cluster, kvh = h / grp;
        const uint64_t qo = (uint64_t)h * dh * 2, ko_ = (uint64_t)kvh * dh * 2;
        int r = sh_x_attention_head(q + qo, kp + ko_, vp + ko_, ko ? ko + ko_ : 0, vo ? vo + ko_ : 0, tok, o + qo,
                                    Sq, Lp, So, dh, ldq, ldkp, ldvp, ldko, ldvo, ldo, scale, cl);
        if (r) rc = r;
    }
    if (cluster == SH_ALL) flex_global_barrier_xy();
    return rc;
}

/* ---- test helper: print every element of an HBM fp16 matrix ("<tag> r c hex"), first core of cluster 0 -------- */
void sh_test_dump_all(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, const char *tag) {
    for (uint32_t i = 0; i < rows; ++i) {
        const volatile uint16_t *row = (const volatile uint16_t *)(uintptr_t)(a + (uint64_t)i * ld * 2);
        for (uint32_t j = 0; j < cols; ++j) sh_printf("%s %u %u %04x\n", tag, i, j, (uint32_t)row[j]);
    }
}
void sh_test_dump_all_idx(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, const char *tag, uint32_t idx) {
    char t[32]; snprintf(t, sizeof t, "%s%u", tag, idx);
    sh_test_dump_all(a, rows, cols, ld, t);
}
