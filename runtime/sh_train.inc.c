/* Training ops (prefix sh_t_): the backward primitives and the optimizer step of a LoRA adapter trained at test
 * time on the SmolVLA action expert (docs/TTT.md). Gradients never enter the frozen VLM: everything here runs on
 * the expert's [50 x 720..4096] activations, its frozen fp16 weights and the fp32 master copies of the adapters.
 *
 *  - sh_t_rowop: the row-op driver of sh_rowops / sh_llm generalised to up to four input and four output streams,
 *    each with its own width, leading dimension and element size (fp16 = 2 B or fp32 = 4 B), plus one raw
 *    parameter row. Same double-buffered staging (block i+1 in and block i-1 out while block i computes), blocks
 *    dealt round-robin over the clusters for SH_ALL, all three cores compute. This is the fp32 HBM tensor path the
 *    optimizer needs (fp32 master weights and moments).
 *  - backward kernels: rmsnorm (+ residual gradient), silu_mul (fp16 SIMD), softmax (row dot), MSE loss gradient,
 *    grouped-query reduction of per-head key/value gradients, and the fused attention backward of one head
 *    (scores recomputed in TCDM, dP = dO V^T, dS = P (dP - rowdot(dO, O)), dq = dS K, own dK = dS^T q,
 *    own dV = P^T dO, all four GEMMs on RedMulE; FlashAttention-2's recompute schedule with the whole key row
 *    resident).
 *  - sh_t_gemm_tr: Z = op(X) op(W) where op transposes physically into an HBM scratch first (RedMulE reads both
 *    operands row-major), so the cost of every transposed operand is a separate sh_transpose call.
 *  - sh_t_optim: SGD / Adam on fp32 masters with an fp16 copy for the next forward, gradients fp16 or fp32 with a
 *    loss-scale divisor.
 *  - sh_t_allreduce: the data-parallel gradient reduction over all clusters with the NoC's in-network 16-bit
 *    REDADD (fp16 directly, or exact through two integer limbs; see the function). */
#define SH_T_L1_BASE SH_ROWOPS_L1_BASE
#define SH_T_L1_BYTES SH_ROWOPS_L1_BYTES
#define SH_T_NS 4u

/* Control code (drivers, wrappers, the backward compositions) is compiled for size: the 64 KB instruction memory holds the
 * whole library + program, and none of it is on a per-element path. The per-element kernels (sh_tk_*, the attention head)
 * keep the -O3 of the build. */
#define SH_T_COLD __attribute__((optimize("Os")))
/* The SDK's iDMA helpers are C99 `inline` without `static`: -O3 always inlines them, -Os may call them, so the unity build
 * provides the external definitions (C99 6.7.4: an `extern` declaration of an inline function emits one). */
extern inline uint32_t bare_dma_start_1d(uint64_t dst, uint64_t src, size_t size);
extern inline uint32_t bare_dma_start_2d(uint64_t dst, uint64_t src, size_t size, size_t dst_stride, size_t src_stride, size_t repeat);
extern inline void bare_dma_wait_all();
extern inline void bare_dma_set_mask(uint16_t row_mask, uint16_t col_mask);
extern inline uint32_t bare_dma_start_1d_reduction(uint64_t dst, uint64_t src, size_t size, collective_compute_format_t fmt, uint16_t row_mask, uint16_t col_mask);
/* memcpy / memset for the cores: the toolchain's newlib is built with the C extension (compressed instructions) and the
 * Snitch cores of this model do not execute RVC ("Executing illegal instruction ... opcode 0x...433d" = c.li inside
 * newlib's memset). GCC emits these calls for struct copies / zeroing (-Os: compound-literal configs), so the library
 * provides plain-RV32 versions, which the linker takes before libc's. Loop-to-libcall conversion is off inside them. */
__attribute__((optimize("no-tree-loop-distribute-patterns"))) void *memcpy(void *d, const void *s, size_t n) {
    if ((((uintptr_t)d | (uintptr_t)s | n) & 3u) == 0) {
        uint32_t *dw = (uint32_t *)d; const uint32_t *sw = (const uint32_t *)s;
        for (size_t i = 0; i < n / 4; ++i) dw[i] = sw[i];
    } else {
        uint8_t *db = (uint8_t *)d; const uint8_t *sb = (const uint8_t *)s;
        for (size_t i = 0; i < n; ++i) db[i] = sb[i];
    }
    return d;
}
__attribute__((optimize("no-tree-loop-distribute-patterns"))) void *memset(void *d, int c, size_t n) {
    uint8_t *db = (uint8_t *)d;
    for (size_t i = 0; i < n; ++i) db[i] = (uint8_t)c;
    return d;
}
static inline uint32_t sh_t_up64(uint32_t b) { return (b + 63u) & ~63u; }
static inline uint32_t sh_t_root(void) { return flex_get_cluster_id() == 0 && flex_is_first_core(); }

/* ---- generic multi-stream row-op driver ---------------------------------------------------------------------- */
typedef struct { uint64_t a; uint32_t cols, ld, esz; } sh_t_str;              /* a == 0: stream unused */
typedef struct { sh_t_str in[SH_T_NS], out[SH_T_NS]; uint64_t p0; uint32_t p0bytes, rows; } sh_t_args;
typedef struct { uint8_t *in[SH_T_NS]; uint8_t *out[SH_T_NS]; const uint8_t *p0; uint32_t nr, r0; const sh_t_args *a; const void *arg; } sh_tblk;
typedef void (*sh_tfn_t)(const sh_tblk *k);

SH_T_COLD static void sh_t_rowop(const sh_t_args *a, sh_tfn_t fn, const void *arg, uint32_t cluster) {
    const uint32_t rows = a->rows;
    uint32_t rin = 0, rout = 0;
    for (uint32_t s = 0; s < SH_T_NS; ++s) {
        if (a->in[s].a) rin += a->in[s].cols * a->in[s].esz;
        if (a->out[s].a) rout += a->out[s].cols * a->out[s].esz;
    }
    const uint32_t p0b = a->p0 ? sh_t_up64(a->p0bytes) : 0;
    /* each stream's block is 64 B aligned: up to 2 x 8 x 64 B of slack per set */
    uint32_t rpb = (SH_T_L1_BYTES - p0b - 2048u) / (2u * (rin + rout)); if (rpb == 0) rpb = 1; if (rpb > rows) rpb = rows;
    if (cluster == SH_ALL) {
        const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, want = (rows + P - 1) / P;
        if (rpb > want) rpb = want ? want : 1;
    }
    uint32_t oin[SH_T_NS], oout[SH_T_NS], setb = 0;
    for (uint32_t s = 0; s < SH_T_NS; ++s) { oin[s] = setb; if (a->in[s].a) setb += sh_t_up64(rpb * a->in[s].cols * a->in[s].esz); }
    for (uint32_t s = 0; s < SH_T_NS; ++s) { oout[s] = setb; if (a->out[s].a) setb += sh_t_up64(rpb * a->out[s].cols * a->out[s].esz); }
    const int dm = flex_is_dm_core();
    const uint32_t lp0 = SH_T_L1_BASE, sets = SH_T_L1_BASE + p0b;
    const uint32_t nblk = (rows + rpb - 1) / rpb;
    uint32_t next = 0;
    while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_TSET(s_) (sets + (s_) * setb)
    #define SH_TLOAD(bi, s_) do { const uint32_t r_ = (bi) * rpb, n_ = (rows - r_) < rpb ? (rows - r_) : rpb; \
        for (uint32_t q_ = 0; q_ < SH_T_NS; ++q_) { const sh_t_str *t_ = &a->in[q_]; if (!t_->a) continue; \
            const uint32_t rb_ = t_->cols * t_->esz; \
            if (t_->ld == t_->cols) bare_dma_start_1d(local(SH_TSET(s_) + oin[q_]), t_->a + (uint64_t)r_ * rb_, rb_ * n_); \
            else bare_dma_start_2d(local(SH_TSET(s_) + oin[q_]), t_->a + (uint64_t)r_ * t_->ld * t_->esz, rb_, rb_, t_->ld * t_->esz, n_); } } while (0)
    if (next < nblk) {
        if (dm) {
            if (a->p0) bare_dma_start_1d(local(lp0), a->p0, a->p0bytes);
            SH_TLOAD(next, 0);
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = blk * rpb, nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm && next < nblk) SH_TLOAD(next, set ^ 1);
        sh_tblk k;
        for (uint32_t s = 0; s < SH_T_NS; ++s) {
            k.in[s] = a->in[s].a ? (uint8_t *)local(SH_TSET(set) + oin[s]) : 0;
            k.out[s] = a->out[s].a ? (uint8_t *)local(SH_TSET(set) + oout[s]) : 0;
        }
        k.p0 = a->p0 ? (const uint8_t *)local(lp0) : 0; k.nr = nr; k.r0 = r0; k.a = a; k.arg = arg;
        fn(&k);
        sh_fp_fence();
        flex_intra_cluster_sync();
        if (dm) {
            bare_dma_wait_all();
            for (uint32_t s = 0; s < SH_T_NS; ++s) {
                const sh_t_str *t = &a->out[s]; if (!t->a) continue;
                const uint32_t rb = t->cols * t->esz;
                if (t->ld == t->cols) bare_dma_start_1d(t->a + (uint64_t)r0 * rb, local(SH_TSET(set) + oout[s]), rb * nr);
                else for (uint32_t r = 0; r < nr; ++r) bare_dma_start_1d(t->a + (uint64_t)(r0 + r) * t->ld * t->esz, local(SH_TSET(set) + oout[s] + r * rb), rb);
            }
        }
        flex_intra_cluster_sync();
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_TSET
    #undef SH_TLOAD
    sh_end_op(cluster);
}
/* zero an argument block without a libc memset (the libc is built with compressed instructions, which the cores do not
 * execute: `= { 0 }` on a struct this size becomes a memset call) */
static inline void sh_t_clear(sh_t_args *a) { volatile uint32_t *w = (volatile uint32_t *)a; for (uint32_t i = 0; i < sizeof *a / 4; ++i) w[i] = 0; }
static inline sh_t_str sh_t_s16(uint64_t a, uint32_t cols, uint32_t ld) { sh_t_str s = { a, cols, ld, 2 }; return s; }
static inline sh_t_str sh_t_s32(uint64_t a, uint32_t cols, uint32_t ld) { sh_t_str s = { a, cols, ld, 4 }; return s; }
static inline int sh_t_bad(int cond, const char *what) { if (cond && sh_t_root()) sh_printf("[sh_t] %s\n", what); return cond; }

/* ---- RMSNorm backward: y = x r g, r = rsqrt(mean x^2 + eps) ------------------------------------------------
 * dx = r g dy - x r^3 / D sum_j (g dy x)_j  (+ dres: the residual stream's gradient, fused). fp32 statistics. */
static void sh_tk_rmsnorm_bwd(const sh_tblk *k) {
    const float eps = *(const float *)k->arg; const uint32_t C = k->a->in[0].cols; const float invC = 1.f / (float)C;
    const uint16_t *g = (const uint16_t *)k->p0;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *x = (const uint16_t *)k->in[0] + r * C, *dy = (const uint16_t *)k->in[1] + r * C;
        const uint16_t *dr = k->in[2] ? (const uint16_t *)k->in[2] + r * C : 0;
        uint16_t *dx = (uint16_t *)k->out[0] + r * C;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        for (uint32_t i = 0; i < C; i += 4) {
            float x0, x1, x2, x3, y0, y1, y2, y3, g0, g1, g2, g3;
            sh_h2f4(x + i, &x0, &x1, &x2, &x3); sh_h2f4(dy + i, &y0, &y1, &y2, &y3); sh_h2f4(g + i, &g0, &g1, &g2, &g3);
            s0 += x0 * x0; s1 += x1 * x1; s2 += x2 * x2; s3 += x3 * x3;
            d0 += g0 * y0 * x0; d1 += g1 * y1 * x1; d2 += g2 * y2 * x2; d3 += g3 * y3 * x3;
        }
        const float rs = sh_rsqrtf(((s0 + s1) + (s2 + s3)) * invC + eps);
        const float c = ((d0 + d1) + (d2 + d3)) * rs * rs * rs * invC;
#ifdef SH_T_RMSBWD_F32
        for (uint32_t i = 0; i < C; i += 4) {
            float x0, x1, x2, x3, y0, y1, y2, y3, g0, g1, g2, g3, e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
            sh_h2f4(x + i, &x0, &x1, &x2, &x3); sh_h2f4(dy + i, &y0, &y1, &y2, &y3); sh_h2f4(g + i, &g0, &g1, &g2, &g3);
            if (dr) sh_h2f4(dr + i, &e0, &e1, &e2, &e3);
            sh_f2h4(dx + i, rs * g0 * y0 - c * x0 + e0, rs * g1 * y1 - c * x1 + e1, rs * g2 * y2 - c * x2 + e2, rs * g3 * y3 - c * x3 + e3);
        }
#else   /* output pass in fp16 SIMD (4 lanes per instruction; the statistics above stay fp32): (dy rs) g - c x (+ dres) */
        const sh_v4h rs4 = sh_v4_splat(rs), c4 = sh_v4_splat(c);
        const sh_v4h *xv = SH_V4CP(x), *yv = SH_V4CP(dy), *gv = SH_V4CP(g), *rv = dr ? SH_V4CP(dr) : 0; sh_v4h *ov = SH_V4P(dx);
        if (rv) for (uint32_t j = 0; j < (C >> 2); ++j) ov[j] = sh_v4_add(sh_v4_sub(sh_v4_mul(sh_v4_mul_r(yv[j], rs4), gv[j]), sh_v4_mul_r(xv[j], c4)), rv[j]);
        else    for (uint32_t j = 0; j < (C >> 2); ++j) ov[j] = sh_v4_sub(sh_v4_mul(sh_v4_mul_r(yv[j], rs4), gv[j]), sh_v4_mul_r(xv[j], c4));
#endif
    }
}
SH_T_COLD void sh_t_rmsnorm_bwd(uint64_t dx, uint64_t x, uint64_t dy, uint64_t dres, uint64_t gamma, uint32_t rows, uint32_t cols,
                      uint32_t lddx, uint32_t ldx, uint32_t lddy, uint32_t lddres, float eps, uint32_t cluster) {
    if (sh_t_bad(cols & 3, "rmsnorm_bwd: cols % 4 != 0")) return;
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows; a.in[0] = sh_t_s16(x, cols, ldx); a.in[1] = sh_t_s16(dy, cols, lddy);
    if (dres) a.in[2] = sh_t_s16(dres, cols, lddres);
    a.out[0] = sh_t_s16(dx, cols, lddx); a.p0 = gamma; a.p0bytes = cols * 2;
    sh_t_rowop(&a, sh_tk_rmsnorm_bwd, &eps, cluster);
}

/* ---- silu_mul backward: y = silu(a) b  ->  db = dy silu(a), da = dy b s (1 + a (1 - s)), s = sigmoid(a) ------
 * fp16 SIMD (the forward's exp2 kernel), elements split over the cores. */
static void sh_tk_silu_mul_bwd(const sh_tblk *k) {
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h nl = sh_v4_splat(-SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), c15 = sh_v4_splat_h(SH_C15);
    uint32_t lo, hi; sh_share(k->nr * k->a->in[0].cols, 16, &lo, &hi);
    const sh_v4h *A = SH_V4CP(k->in[0]), *B = SH_V4CP(k->in[1]), *DY = SH_V4CP(k->in[2]);
    sh_v4h *DA = SH_V4P(k->out[0]), *DB = SH_V4P(k->out[1]);
    #define SH_T_SILU_T(v) sh_v4_min_r(sh_v4_max_r(sh_v4_mul_r(v, nl), cm14), c15)
    uint32_t i = lo >> 2; const uint32_t e = hi >> 2;
    for (; i + 4 <= e; i += 4) {
        sh_v4h t[4] = { SH_T_SILU_T(A[i]), SH_T_SILU_T(A[i + 1]), SH_T_SILU_T(A[i + 2]), SH_T_SILU_T(A[i + 3]) };
        sh_v4_exp2x4(t, &ec);
        for (int j = 0; j < 4; ++j) {
            const sh_v4h a = A[i + j], s = sh_v4_div(ec.one, sh_v4_add_r(t[j], ec.one)), sil = sh_v4_mul(a, s), dy = DY[i + j];
            DB[i + j] = sh_v4_mul(dy, sil);
            DA[i + j] = sh_v4_mul(sh_v4_mul(dy, B[i + j]), sh_v4_mul(s, sh_v4_add_r(sh_v4_sub(a, sil), ec.one)));
        }
    }
    for (; i < e; ++i) {
        const sh_v4h a = A[i], s = sh_v4_div(ec.one, sh_v4_add_r(sh_v4_exp2(SH_T_SILU_T(a), &ec), ec.one)), sil = sh_v4_mul(a, s);
        DB[i] = sh_v4_mul(DY[i], sil);
        DA[i] = sh_v4_mul(sh_v4_mul(DY[i], B[i]), sh_v4_mul(s, sh_v4_add_r(sh_v4_sub(a, sil), ec.one)));
    }
    #undef SH_T_SILU_T
}
SH_T_COLD void sh_t_silu_mul_bwd(uint64_t da, uint64_t db, uint64_t a, uint64_t b, uint64_t dy, uint32_t rows, uint32_t cols,
                       uint32_t ldda, uint32_t lddb, uint32_t lda, uint32_t ldb, uint32_t lddy, uint32_t cluster) {
    if (sh_t_bad(cols & 3, "silu_mul_bwd: cols % 4 != 0")) return;
    sh_t_args s; sh_t_clear(&s);
    s.rows = rows; s.in[0] = sh_t_s16(a, cols, lda); s.in[1] = sh_t_s16(b, cols, ldb); s.in[2] = sh_t_s16(dy, cols, lddy);
    s.out[0] = sh_t_s16(da, cols, ldda); s.out[1] = sh_t_s16(db, cols, lddb);
    sh_t_rowop(&s, sh_tk_silu_mul_bwd, 0, cluster);
}

/* ---- softmax backward: y = softmax(scale x)  ->  dx = scale y (dy - sum_j y_j dy_j)  (fp32 row dot) ---------- */
static void sh_tk_softmax_bwd(const sh_tblk *k) {
    const float sc = *(const float *)k->arg; const uint32_t C = k->a->in[0].cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *y = (const uint16_t *)k->in[0] + r * C, *dy = (const uint16_t *)k->in[1] + r * C;
        uint16_t *dx = (uint16_t *)k->out[0] + r * C;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        for (uint32_t i = 0; i < C; i += 4) {
            float y0, y1, y2, y3, g0, g1, g2, g3;
            sh_h2f4(y + i, &y0, &y1, &y2, &y3); sh_h2f4(dy + i, &g0, &g1, &g2, &g3);
            d0 += y0 * g0; d1 += y1 * g1; d2 += y2 * g2; d3 += y3 * g3;
        }
        const float dot = (d0 + d1) + (d2 + d3);
        for (uint32_t i = 0; i < C; i += 4) {
            float y0, y1, y2, y3, g0, g1, g2, g3;
            sh_h2f4(y + i, &y0, &y1, &y2, &y3); sh_h2f4(dy + i, &g0, &g1, &g2, &g3);
            sh_f2h4(dx + i, sc * y0 * (g0 - dot), sc * y1 * (g1 - dot), sc * y2 * (g2 - dot), sc * y3 * (g3 - dot));
        }
    }
}
SH_T_COLD void sh_t_softmax_bwd(uint64_t dx, uint64_t y, uint64_t dy, uint32_t rows, uint32_t cols, uint32_t ld, float scale, uint32_t cluster) {
    if (sh_t_bad(cols & 3, "softmax_bwd: cols % 4 != 0")) return;
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows; a.in[0] = sh_t_s16(y, cols, ld); a.in[1] = sh_t_s16(dy, cols, ld); a.out[0] = sh_t_s16(dx, cols, ld);
    sh_t_rowop(&a, sh_tk_softmax_bwd, &scale, cluster);
}

/* ---- MSE loss: L = mean (pred - tgt)^2 over rows x cols; dy = gscale (pred - tgt) with gscale = 2 / (rows cols) x
 * the loss scale (chosen by the caller); loss != 0: per-row sum of squares in fp32 ([rows, 1]). ------------- */
static void sh_tk_mse(const sh_tblk *k) {
    const float gs = *(const float *)k->arg; const uint32_t C = k->a->in[0].cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *p = (const uint16_t *)k->in[0] + r * C, *t = (const uint16_t *)k->in[1] + r * C;
        uint16_t *dy = (uint16_t *)k->out[0] + r * C;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (uint32_t i = 0; i < C; i += 4) {
            float p0, p1, p2, p3, t0, t1, t2, t3;
            sh_h2f4(p + i, &p0, &p1, &p2, &p3); sh_h2f4(t + i, &t0, &t1, &t2, &t3);
            p0 -= t0; p1 -= t1; p2 -= t2; p3 -= t3;
            s0 += p0 * p0; s1 += p1 * p1; s2 += p2 * p2; s3 += p3 * p3;
            sh_f2h4(dy + i, gs * p0, gs * p1, gs * p2, gs * p3);
        }
        if (k->out[1]) ((float *)k->out[1])[r] = (s0 + s1) + (s2 + s3);
    }
}
SH_T_COLD void sh_t_mse_grad(uint64_t dy, uint64_t loss_rows, uint64_t pred, uint64_t tgt, uint32_t rows, uint32_t cols, uint32_t ld,
                   float gscale, uint32_t cluster) {
    if (sh_t_bad(cols & 3, "mse_grad: cols % 4 != 0")) return;
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows; a.in[0] = sh_t_s16(pred, cols, ld); a.in[1] = sh_t_s16(tgt, cols, ld); a.out[0] = sh_t_s16(dy, cols, ld);
    if (loss_rows) a.out[1] = sh_t_s32(loss_rows, 1, 1);
    sh_t_rowop(&a, sh_tk_mse, &gscale, cluster);
}

/* ---- grouped-query reduction: y[:, g dh ..] = sum_{j < grp} x[:, (g grp + j) dh ..] (per-head key / value grads
 * of the query heads sharing kv head g). fp16 SIMD. -------------------------------------------------------- */
typedef struct { uint32_t grp, dh; } sh_t_gqa_arg;
static void sh_tk_gqa_sum(const sh_tblk *k) {
    const sh_t_gqa_arg *ga = (const sh_t_gqa_arg *)k->arg;
    const uint32_t CX = k->a->in[0].cols, CY = k->a->out[0].cols, dv = ga->dh >> 2, ng = CY / ga->dh;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const sh_v4h *x = SH_V4CP((const uint16_t *)k->in[0] + r * CX); sh_v4h *y = SH_V4P((uint16_t *)k->out[0] + r * CY);
        for (uint32_t g = 0; g < ng; ++g)
            for (uint32_t j = 0; j < dv; ++j) {
                sh_v4h acc = x[(g * ga->grp) * dv + j];
                for (uint32_t m = 1; m < ga->grp; ++m) acc = sh_v4_add(acc, x[(g * ga->grp + m) * dv + j]);
                y[g * dv + j] = acc;
            }
    }
}
SH_T_COLD void sh_t_gqa_sum(uint64_t y, uint64_t x, uint32_t rows, uint32_t Hkv, uint32_t grp, uint32_t dh, uint32_t ldy, uint32_t ldx, uint32_t cluster) {
    if (sh_t_bad(dh & 3, "gqa_sum: dh % 4 != 0")) return;
    sh_t_gqa_arg g = { grp, dh };
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows; a.in[0] = sh_t_s16(x, Hkv * grp * dh, ldx); a.out[0] = sh_t_s16(y, Hkv * dh, ldy);
    sh_t_rowop(&a, sh_tk_gqa_sum, &g, cluster);
}

/* ---- optimizer: fp32 master weights (+ Adam moments), fp16 copy for the forward ------------------------------
 * g: fp16 (gesz 2) or fp32 (gesz 4) gradients, multiplied by inv_scale (the loss scale's inverse).
 * SGD:  w <- w - lr g.   Adam: m <- b1 m + (1 - b1) g; v <- b2 v + (1 - b2) g^2; w <- w - lr (m / bc1) / (sqrt(v / bc2) + eps)
 * (bc1 = 1 - b1^t, bc2 = 1 - b2^t computed by the caller). All tensors are rows x cols with leading dim = cols
 * (a flat parameter arena). In place: the fp32 outputs alias the inputs. */
typedef struct { uint32_t adam, gesz; float lr, inv_scale, b1, b2, eps, bc1, bc2; } sh_t_opt_arg;
static void sh_tk_optim(const sh_tblk *k) {
    const sh_t_opt_arg *o = (const sh_t_opt_arg *)k->arg;
    uint32_t lo, hi; sh_share(k->nr * k->a->in[1].cols, 4, &lo, &hi);
    const float *w = (const float *)k->in[1]; float *wo = (float *)k->out[0]; uint16_t *w16 = (uint16_t *)k->out[1];
    const float lr = o->lr, is = o->inv_scale;
    if (!o->adam) {
        for (uint32_t i = lo; i < hi; i += 4) {
            float g0, g1, g2, g3;
            if (o->gesz == 2) sh_h2f4((const uint16_t *)k->in[0] + i, &g0, &g1, &g2, &g3);
            else { const float *gf = (const float *)k->in[0] + i; g0 = gf[0]; g1 = gf[1]; g2 = gf[2]; g3 = gf[3]; }
            const float n0 = w[i] - lr * (g0 * is), n1 = w[i + 1] - lr * (g1 * is), n2 = w[i + 2] - lr * (g2 * is), n3 = w[i + 3] - lr * (g3 * is);
            wo[i] = n0; wo[i + 1] = n1; wo[i + 2] = n2; wo[i + 3] = n3;
            sh_f2h4(w16 + i, n0, n1, n2, n3);
        }
        return;
    }
    const float *m = (const float *)k->in[2], *v = (const float *)k->in[3]; float *mo = (float *)k->out[2], *vo = (float *)k->out[3];
    const float b1 = o->b1, b2 = o->b2, c1 = 1.f - b1, c2 = 1.f - b2, ib1 = 1.f / o->bc1, ib2 = 1.f / o->bc2, eps = o->eps;
    for (uint32_t i = lo; i < hi; ++i) {
        float g;
        if (o->gesz == 2) g = sh_h2f(((const uint16_t *)k->in[0])[i]); else g = ((const float *)k->in[0])[i];
        g *= is;
        const float mn = b1 * m[i] + c1 * g, vn = b2 * v[i] + c2 * g * g;
        const float vh = vn * ib2, den = vh * sh_rsqrtf(vh + 1e-30f) + eps;
        const float wn = w[i] - lr * (mn * ib1) / den;
        mo[i] = mn; vo[i] = vn; wo[i] = wn;
        w16[i] = (uint16_t)sh_f2h(wn);
    }
}
SH_T_COLD void sh_t_optim(uint64_t w32, uint64_t w16, uint64_t m32, uint64_t v32, uint64_t g, uint32_t gesz, uint32_t rows, uint32_t cols,
                uint32_t adam, float lr, float inv_scale, float b1, float b2, float eps, float bc1, float bc2, uint32_t cluster) {
    if (sh_t_bad((cols & 3) || (adam && (!m32 || !v32)), "optim: cols % 4 != 0 or Adam without moments")) return;
    sh_t_opt_arg o = { adam, gesz, lr, inv_scale, b1, b2, eps, bc1, bc2 };
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows;
    a.in[0] = gesz == 2 ? sh_t_s16(g, cols, cols) : sh_t_s32(g, cols, cols);
    a.in[1] = sh_t_s32(w32, cols, cols); a.out[0] = sh_t_s32(w32, cols, cols); a.out[1] = sh_t_s16(w16, cols, cols);
    if (adam) { a.in[2] = sh_t_s32(m32, cols, cols); a.in[3] = sh_t_s32(v32, cols, cols); a.out[2] = a.in[2]; a.out[3] = a.in[3]; }
    sh_t_rowop(&a, sh_tk_optim, &o, cluster);
}

/* ---- GEMM with transposed operands ----------------------------------------------------------------------------
 * Z[M,N] (+)= op(X) op(W): with tx, X is stored as X^T [K, M] (leading dim ldx); with tw, W is stored as W^T [N, K].
 * RedMulE reads both operands row-major from TCDM, so each transposed operand is first copied transposed into the
 * HBM scratch (sh_transpose: X' [M, K] at scratch, W' [K, N] behind it) and the plain sh_gemm runs on the copies.
 * scratch needs (tx ? M K : 0) + (tw ? K N : 0) fp16 elements. */
SH_T_COLD int sh_t_gemm_tr(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K, uint32_t ldx, uint32_t ldw, uint32_t ldz,
                 uint32_t tx, uint32_t tw, uint64_t scratch, const sh_gemm_cfg *cfg, uint32_t cluster) {
    if (tx) { sh_transpose(scratch, x, K, M, ldx, K, cluster); x = scratch; ldx = K; scratch += (uint64_t)M * K * 2; }
    if (tw) { sh_transpose(scratch, w, N, K, ldw, N, cluster); w = scratch; ldw = N; }
    if (cluster != SH_ALL) flex_intra_cluster_sync();
    return sh_gemm(x, w, z, M, N, K, ldx, ldw, ldz, cfg, cluster);
}

/* ---- copy and scale (the training programs' only elementwise needs besides the backward kernels) -----------------
 * sh_t_copy: HBM -> HBM rows by 1-D DMA on the DM cores (rows dealt over the clusters), no staging. */
SH_T_COLD void sh_t_copy(uint64_t dst, uint64_t src, uint32_t rows, uint32_t cols, uint32_t ldd, uint32_t lds, uint32_t cluster) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, cid = flex_get_cluster_id();
    if (flex_is_dm_core() && (cluster == SH_ALL || cid == cluster)) {
        for (uint32_t r = cluster == SH_ALL ? cid : 0; r < rows; r += cluster == SH_ALL ? P : 1)
            bare_dma_start_1d(dst + (uint64_t)r * ldd * 2, src + (uint64_t)r * lds * 2, cols * 2);
        bare_dma_wait_all();
    }
    sh_end_op(cluster);
}
static void sh_tk_scale(const sh_tblk *k) {
    const sh_v4h s4 = sh_v4_splat(*(const float *)k->arg);
    uint32_t lo, hi; sh_share(k->nr * k->a->in[0].cols, 4, &lo, &hi);
    const sh_v4h *x = SH_V4CP(k->in[0]); sh_v4h *y = SH_V4P(k->out[0]);
    for (uint32_t i = lo >> 2; i < (hi >> 2); ++i) y[i] = sh_v4_mul_r(x[i], s4);
    /* the block is packed, so an odd row width (e.g. 50) leaves a 1..3 element tail on the last core */
    const uint16_t *xs = (const uint16_t *)k->in[0]; uint16_t *ys = (uint16_t *)k->out[0]; const float s = *(const float *)k->arg;
    for (uint32_t i = (hi & ~3u) > lo ? (hi & ~3u) : lo; i < hi; ++i) ys[i] = (uint16_t)sh_f2h(sh_h2f(xs[i]) * s);
}
SH_T_COLD static void sh_t_scale(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float s, uint32_t cluster) {
    sh_t_args a; sh_t_clear(&a);
    a.rows = rows; a.in[0] = sh_t_s16(x, cols, ld); a.out[0] = sh_t_s16(y, cols, ld);
    sh_t_rowop(&a, sh_tk_scale, &s, cluster);
}
/* y = a + b (per-operand leading dims; cols % 4 == 0). The residual add of the training forward: seeding the residual into
 * the accumulating GEMM instead would round every one of its K fp16 FMAs at |h| (v error 3x the inference program's). */
static void sh_tk_add(const sh_tblk *k) {
    uint32_t lo, hi; sh_share(k->nr * k->a->in[0].cols, 4, &lo, &hi);
    const sh_v4h *a = SH_V4CP(k->in[0]), *b = SH_V4CP(k->in[1]); sh_v4h *y = SH_V4P(k->out[0]);
    for (uint32_t i = lo >> 2; i < (hi >> 2); ++i) y[i] = sh_v4_add(a[i], b[i]);
}
SH_T_COLD void sh_t_add(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ldy, uint32_t lda, uint32_t ldb, uint32_t cluster) {
    if (sh_t_bad(cols & 3, "add: cols % 4 != 0")) return;
    sh_t_args s; sh_t_clear(&s);
    s.rows = rows; s.in[0] = sh_t_s16(a, cols, lda); s.in[1] = sh_t_s16(b, cols, ldb); s.out[0] = sh_t_s16(y, cols, ldy);
    sh_t_rowop(&s, sh_tk_add, 0, cluster);
}

/* ---- LoRA forward and the backward of a (LoRA-)linear layer -------------------------------------------------------
 * Tile helpers: the largest candidate dividing n that still gives >= 15 tiles (16 clusters), else the smallest divisor. */
SH_T_COLD static uint32_t sh_t_tile_n(uint32_t n) {
    static const uint32_t c[] = { 256, 128, 64, 48, 32, 16 };
    for (uint32_t i = 0; i < 6; ++i) if (n % c[i] == 0 && n / c[i] >= 15) return c[i];
    for (int i = 5; i >= 0; --i) if (n % c[i] == 0) return c[i];
    return n;
}
SH_T_COLD static uint32_t sh_t_tile_k(uint32_t n, uint32_t cap) {   /* the largest divisor of n <= cap that is a multiple of 16 (or n) */
    if (n <= cap) return n;
    for (uint32_t t = cap & ~15u; t >= 16; t -= 16) if (n % t == 0) return t;
    return n;
}
SH_T_COLD static int sh_t_g(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K, uint32_t ldx, uint32_t ldw, uint32_t ldz,
                  uint32_t tm, uint32_t tn, uint32_t tk, uint32_t acc, uint32_t cluster) {
    sh_gemm_cfg c = { tm, tn, tk, 1, acc, SH_FP16, 0 };
    return sh_gemm(x, w, z, M, N, K, ldx, ldw, ldz, &c, cluster);
}
static inline void sh_t_sync(uint32_t cluster) { if (cluster != SH_ALL) flex_intra_cluster_sync(); }

/* Y[M, N] += s (X[M, K] A[K, r]) B[r, N]; t = s X A [M, r] is kept for the backward. tk | K: the K tile of X A. */
SH_T_COLD int sh_t_lora_fwd(uint64_t y, uint64_t x, uint64_t a, uint64_t b, uint64_t t, uint32_t M, uint32_t K, uint32_t N, uint32_t r,
                  uint32_t ldy, uint32_t ldx, float s, uint32_t tk, uint32_t cluster) {
    int rc = sh_t_g(x, a, t, M, r, K, ldx, r, r, M, r, tk, 0, cluster);
    sh_t_sync(cluster);
    sh_t_scale(t, t, M, r, r, s, cluster);
    rc |= sh_t_g(t, b, y, M, N, r, r, N, ldy, M, sh_t_tile_n(N), r, 1, cluster);
    sh_t_sync(cluster);
    return rc;
}

/* Backward of y[M, N] = x[M, K] W[K, N] (+ s (x A) B on the first nl output columns: the LoRA) w.r.t. x and, with a LoRA,
 * its A [K, r] / B [r, nl]. Every product with the frozen W is formed transposed so RedMulE reads W in its stored
 * [in, out] layout and only the M-row activations are transposed (M = 50 << K, N):
 *     dyT = dy^T                       [N, M]     (sh_transpose)
 *     dxT = W dyT                      [K, M]     (tiles tm x M x tk)
 *   LoRA:
 *     dB  = t^T dy[:, :nl]             [r, nl]    (t = s x A from the forward; sh_t_gemm_tr with the transposed t)
 *     uT  = s B dyT[:nl]               [r, M]
 *     dA  = (uT x)^T                   [K, r]
 *     dxT += A uT
 *     dx  = dxT^T                      [M, K]
 * scratch (fp16): N M + K M + r M + r K + M r elements. a == 0: no LoRA. Returns 0 or < 0 (a GEMM constraint). */
SH_T_COLD int sh_t_linear_bwd(uint64_t dx, uint64_t dy, uint64_t w, uint32_t M, uint32_t K, uint32_t N, uint32_t lddx, uint32_t lddy, uint32_t ldw,
                    uint64_t a, uint64_t b, uint64_t t, uint64_t x, uint64_t da, uint64_t db, uint32_t r, uint32_t nl, uint32_t ldx, float s,
                    uint64_t scratch, uint32_t tm, uint32_t tk, uint32_t cluster) {
    const uint64_t dyT = scratch, dxT = dyT + (uint64_t)N * M * 2, uT = dxT + (uint64_t)K * M * 2, aT = uT + (uint64_t)r * M * 2,
                   scr = aT + (uint64_t)r * K * 2;
    int rc = 0;
    sh_transpose(dyT, dy, M, N, lddy, M, cluster); sh_t_sync(cluster);
    rc |= sh_t_g(w, dyT, dxT, K, M, N, ldw, M, M, tm, M, tk, 0, cluster); sh_t_sync(cluster);
    if (a) {
        sh_gemm_cfg c = { r, sh_t_tile_n(nl), M, 1, 0, SH_FP16, 0 };
        rc |= sh_t_gemm_tr(t, dy, db, r, nl, M, r, lddy, nl, 1, 0, scr, &c, cluster); sh_t_sync(cluster);
        rc |= sh_t_g(b, dyT, uT, r, M, nl, nl, M, M, r, M, sh_t_tile_k(nl, 1024), 0, cluster); sh_t_sync(cluster);
        sh_t_scale(uT, uT, r, M, M, s, cluster); sh_t_sync(cluster);
        rc |= sh_t_g(uT, x, aT, r, K, M, M, ldx, K, r, sh_t_tile_n(K), M, 0, cluster); sh_t_sync(cluster);
        sh_transpose(da, aT, r, K, K, r, cluster); sh_t_sync(cluster);
        rc |= sh_t_g(a, uT, dxT, K, M, r, r, M, M, tm, M, r, 1, cluster); sh_t_sync(cluster);
    }
    sh_transpose(dx, dxT, K, M, M, lddx, cluster); sh_t_sync(cluster);
    return rc;
}

/* ---- attention backward ------------------------------------------------------------------------------------------
 * The forward of sh_x_attention_head (Sq queries over L = Lp prefix keys + So own causal keys, prefix padding mask
 * from the token-class array) differentiated for one query head, everything inside one cluster's TCDM:
 *   1. stage q, K = [kp ; ko] (Lpad rows, zero padding), V likewise, o, dO; K^T and V^T by element-granular DMA
 *   2. RedMulE  S = q K^T;  cores: P = softmax(scale S + mask) (normalised, fp16), D_i = sum_j dO_ij O_ij (fp32)
 *   3. RedMulE  dP = dO V^T;  cores: dS = scale P (dP - D_i)    (in place over dP; masked columns are 0 since P is)
 *   4. RedMulE  dq = dS K
 *   5. own keys only (So > 0): DMA-transpose the So own columns of dS and P, RedMulE dk = dS_own^T q, dv = P_own^T dO
 * The prefix keys/values are the frozen VLM's: no gradient is formed for them. dk / dv are per QUERY head (ld of the
 * caller's per-head scratch); sh_t_attention_bwd sums them over each kv group.
 * dout == 0 is the FORWARD of the same head (training programs use it instead of sh_x_attention_head so the staging,
 * masking and softmax code exists once in the 64 KB instruction memory): steps 1-2 without o / dO, then RedMulE
 * o = P V and o is stored (ldo). */
typedef struct { uint32_t q, k, kt, v, vt, s, dp, o, dout, dq, dst, pt, dk, dv, sum, vrow, vld, end, Lpad; } sh_t_attn_l1;
static inline sh_t_attn_l1 sh_t_attn_layout(uint32_t Sq, uint32_t dh, uint32_t L, uint32_t So, uint32_t base) {
    sh_t_attn_l1 l; const uint32_t Lpad = (L + 31u) & ~31u, NC = ARCH_NUM_CORE_PER_CLUSTER, qb = sh_t_up64(Sq * dh * 2), kb = sh_t_up64(Lpad * dh * 2);
    l.Lpad = Lpad;
    l.q = base; l.k = l.q + qb; l.kt = l.k + kb; l.v = l.kt + kb; l.vt = l.v + kb; l.s = l.vt + kb;
    l.dp = l.s + sh_t_up64(Sq * Lpad * 2); l.o = l.dp + sh_t_up64(Sq * Lpad * 2); l.dout = l.o + qb; l.dq = l.dout + qb;
    l.dst = l.dq + qb; l.pt = l.dst + sh_t_up64(So * Sq * 2); l.dk = l.pt + sh_t_up64(So * Sq * 2); l.dv = l.dk + sh_t_up64(So * dh * 2);
    l.sum = l.dv + sh_t_up64(So * dh * 2); l.vrow = l.sum + sh_t_up64(Sq * 4); l.vld = l.vrow + sh_t_up64(Lpad * 2);
    l.end = l.vld + NC * 2 * sh_t_up64(Lpad * 2);
    return l;
}
uint32_t sh_t_attention_bwd_l1_bytes(uint32_t Sq, uint32_t L, uint32_t So, uint32_t dh) { return sh_t_attn_layout(Sq, dh, L, So, SH_X_L1_BASE).end; }

int sh_t_attention_bwd_head(uint64_t q, uint64_t kp, uint64_t vp, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o, uint64_t dout,
                            uint64_t dq, uint64_t dk, uint64_t dv, uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t dh,
                            uint32_t ldq, uint32_t ldkp, uint32_t ldvp, uint32_t ldko, uint32_t ldvo, uint32_t ldo, uint32_t lddo,
                            uint32_t lddq, uint32_t lddk, uint32_t lddv, float scale, uint32_t cluster) {
    if (cluster != SH_ALL && flex_get_cluster_id() != cluster) return 0;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id(), L = Lp + So;
    const sh_t_attn_l1 l = sh_t_attn_layout(Sq, dh, L, So, SH_X_L1_BASE);
    const uint32_t Lpad = l.Lpad, cv = Lpad >> 2, vb = sh_t_up64(Lpad * 2);
    if ((dh & 3) || Sq == 0 || L == 0 || (ko == 0) != (So == 0) || l.end > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_t_attention_bwd_head] Sq=%u Lp=%u So=%u dh=%u: need dh %% 4 == 0, own K/V iff So > 0, L1 %u <= %u\n",
                             Sq, Lp, So, dh, l.end, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -1;
    }
    /* 1. staging */
    if (dm) {
        sh_load_block_async(l.q, q, Sq, dh, ldq);
        sh_load_block_async(l.k, kp, Lp, dh, ldkp);
        sh_load_block_async(l.v, vp, Lp, dh, ldvp);
        if (So) { sh_load_block_async(l.k + Lp * dh * 2, ko, So, dh, ldko); sh_load_block_async(l.v + Lp * dh * 2, vo, So, dh, ldvo); }
        if (dout) { sh_load_block_async(l.o, o, Sq, dh, ldo); sh_load_block_async(l.dout, dout, Sq, dh, lddo); }
        if (tok) bare_dma_start_1d(local(l.vrow), tok, Lp * 2);
        sh_l1_zero_dm(l.kt, dh * Lpad * 2);                   /* waits for everything issued so far */
        sh_l1_zero_dm(l.s, Sq * Lpad * 2);
        if (dout) {
            sh_l1_zero_dm(l.vt, dh * Lpad * 2);
            sh_l1_zero_dm(l.dp, Sq * Lpad * 2);
            sh_l1_zero_dm(l.dq, Sq * dh * 2);
            if (So) { sh_l1_zero_dm(l.dk, So * dh * 2); sh_l1_zero_dm(l.dv, So * dh * 2); }
        } else sh_l1_zero_dm(l.o, Sq * dh * 2);
        if (Lpad > L) { sh_l1_zero_dm(l.k + L * dh * 2, (Lpad - L) * dh * 2); sh_l1_zero_dm(l.v + L * dh * 2, (Lpad - L) * dh * 2); }
        for (uint32_t c = 0; c < dh; ++c) {
            bare_dma_start_2d(local(l.kt + c * Lpad * 2), local(l.k + c * 2), 2, 2, dh * 2, L);
            if (dout) bare_dma_start_2d(local(l.vt + c * Lpad * 2), local(l.v + c * 2), 2, 2, dh * 2, L);
        }
        bare_dma_wait_all();
    }
    flex_intra_cluster_sync();
    /* validity rows of this core's first query row (the forward's scheme: causal own columns enabled one per row) */
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
    /* 2. S = q K^T, P, D */
    if (first) { flex_redmule_config(Sq, dh, Lpad); flex_redmule_trigger(l.q, l.kt, l.s, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    {
        const sh_v4_exp2_consts ec = sh_v4_exp2_init();
        const float s2 = scale * SH_LOG2E;
        const sh_v4h s24 = sh_v4_splat(s2), cm14 = sh_v4_splat_h(SH_CM14);
        float *D = (float *)local(l.sum);
        const sh_v4h *vv = (const sh_v4h *)(uintptr_t)vld, *nn = (const sh_v4h *)(uintptr_t)nb;
        vv = sh_x_after_sh(nb + Lpad - 1, vv);
        for (uint32_t r = lo; r < hi; ++r) {
            if (So && r > lo) { vld[Lp + r] = 0x3C00u; nb[Lp + r] = 0u; vv = sh_x_after_sh(nb + Lp + r, vv); }
            sh_v4h *row = SH_V4P(local(l.s + r * Lpad * 2));
            const sh_v4h inv4 = sh_v4_splat(1.f / sh_x_softmax_row(row, vv, nn, cv, s2, &ec, s24, cm14));
            for (uint32_t j = 0; j < cv; ++j) row[j] = sh_v4_mul_r(row[j], inv4);
            if (!dout) continue;
            const uint16_t *orow = (const uint16_t *)local(l.o + r * dh * 2), *grow = (const uint16_t *)local(l.dout + r * dh * 2);
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            for (uint32_t j = 0; j < dh; j += 4) {
                float o0, o1, o2, o3, g0, g1, g2, g3;
                sh_h2f4(orow + j, &o0, &o1, &o2, &o3); sh_h2f4(grow + j, &g0, &g1, &g2, &g3);
                d0 += o0 * g0; d1 += o1 * g1; d2 += o2 * g2; d3 += o3 * g3;
            }
            D[r] = (d0 + d1) + (d2 + d3);
        }
        sh_fp_fence();
    }
    flex_intra_cluster_sync();
    if (!dout) {   /* forward: o = P V */
        if (first) { flex_redmule_config(Sq, Lpad, dh); flex_redmule_trigger(l.s, l.v, l.o, REDMULE_FP_16); flex_redmule_wait(); }
        flex_intra_cluster_sync();
        if (dm) { for (uint32_t r = 0; r < Sq; ++r) bare_dma_start_1d(o + (uint64_t)r * ldo * 2, local(l.o + r * dh * 2), dh * 2); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
        return 0;
    }
    /* 3. dP = dO V^T; dS = scale P (dP - D) */
    if (first) { flex_redmule_config(Sq, dh, Lpad); flex_redmule_trigger(l.dout, l.vt, l.dp, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    {
        const float *D = (const float *)local(l.sum); const sh_v4h sc4 = sh_v4_splat(scale);
        for (uint32_t r = lo; r < hi; ++r) {
            const sh_v4h *p = SH_V4CP(local(l.s + r * Lpad * 2)); sh_v4h *d = SH_V4P(local(l.dp + r * Lpad * 2));
            const sh_v4h d4 = sh_v4_splat(D[r]);
            for (uint32_t j = 0; j < cv; ++j) d[j] = sh_v4_mul_r(sh_v4_mul(p[j], sh_v4_sub_r(d[j], d4)), sc4);
        }
        sh_fp_fence();
    }
    flex_intra_cluster_sync();
    /* 4. dq = dS K;  5. own dk = dS_own^T q, dv = P_own^T dO */
    if (dm && So) {
        for (uint32_t c = 0; c < So; ++c) {
            bare_dma_start_2d(local(l.dst + c * Sq * 2), local(l.dp + (Lp + c) * 2), 2, 2, Lpad * 2, Sq);
            bare_dma_start_2d(local(l.pt + c * Sq * 2), local(l.s + (Lp + c) * 2), 2, 2, Lpad * 2, Sq);
        }
    }
    if (first) { flex_redmule_config(Sq, Lpad, dh); flex_redmule_trigger(l.dp, l.k, l.dq, REDMULE_FP_16); flex_redmule_wait(); }
    if (dm && So) bare_dma_wait_all();
    flex_intra_cluster_sync();
    if (So && first) {
        flex_redmule_config(So, Sq, dh); flex_redmule_trigger(l.dst, l.q, l.dk, REDMULE_FP_16); flex_redmule_wait();
        flex_redmule_config(So, Sq, dh); flex_redmule_trigger(l.pt, l.dout, l.dv, REDMULE_FP_16); flex_redmule_wait();
    }
    flex_intra_cluster_sync();
    if (dm) {
        for (uint32_t r = 0; r < Sq; ++r) bare_dma_start_1d(dq + (uint64_t)r * lddq * 2, local(l.dq + r * dh * 2), dh * 2);
        for (uint32_t r = 0; r < So; ++r) {
            bare_dma_start_1d(dk + (uint64_t)r * lddk * 2, local(l.dk + r * dh * 2), dh * 2);
            bare_dma_start_1d(dv + (uint64_t)r * lddv * 2, local(l.dv + r * dh * 2), dh * 2);
        }
        bare_dma_wait_all();
    }
    flex_intra_cluster_sync();
    return 0;
}

/* Multi-head: query head h (columns h dh of q / o / dO / dq) with kv head h / (H / Hkv). With own keys (So > 0) the
 * per-head dk / dv go to `scratch` ([So, 2 H dh]: dk heads then dv heads) and are summed over each kv group into
 * dko / dvo (So x Hkv dh). cluster == SH_ALL deals head h to cluster h % P. */
SH_T_COLD int sh_t_attention_bwd(uint64_t q, uint64_t kp, uint64_t vp, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o, uint64_t dout,
                       uint64_t dq, uint64_t dko, uint64_t dvo, uint64_t scratch,
                       uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t H, uint32_t Hkv, uint32_t dh,
                       uint32_t ldq, uint32_t ldkp, uint32_t ldvp, uint32_t ldko, uint32_t ldvo, uint32_t ldo, uint32_t lddo,
                       uint32_t lddq, uint32_t lddko, uint32_t lddvo, float scale, uint32_t cluster) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    if (H == 0 || Hkv == 0 || H % Hkv || (So && dout && !scratch)) {
        if (sh_t_root()) sh_printf("[sh_t_attention_bwd] H=%u Hkv=%u So=%u scratch=%u: bad arguments\n", H, Hkv, So, (uint32_t)(scratch != 0));
        return -1;
    }
    const uint32_t grp = H / Hkv, lds = 2 * H * dh;
    int rc = 0;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t cl = (cluster == SH_ALL) ? h % P : cluster;
        const uint64_t qo = (uint64_t)h * dh * 2, kvo = (uint64_t)(h / grp) * dh * 2;
        const uint64_t dkh = So && dout ? scratch + qo : 0, dvh = So && dout ? scratch + (uint64_t)H * dh * 2 + qo : 0;
        int r = sh_t_attention_bwd_head(q + qo, kp + kvo, vp + kvo, So ? ko + kvo : 0, So ? vo + kvo : 0, tok, o + qo, dout ? dout + qo : 0,
                                        dq + qo, dkh, dvh, Sq, Lp, So, dh, ldq, ldkp, ldvp, ldko, ldvo, ldo, lddo, lddq, lds, lds, scale, cl);
        if (r) rc = r;
    }
    if (cluster == SH_ALL) flex_global_barrier_xy(); else flex_intra_cluster_sync();
    if (So && dout) {
        sh_t_gqa_sum(dko, scratch, So, Hkv, grp, dh, lddko, lds, cluster);
        sh_t_gqa_sum(dvo, scratch + (uint64_t)H * dh * 2, So, Hkv, grp, dh, lddvo, lds, cluster);
    }
    return rc;
}

int sh_t_attention_fwd(uint64_t q, uint64_t kp, uint64_t vp, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o,
                       uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t H, uint32_t Hkv, uint32_t dh,
                       uint32_t ldq, uint32_t ldkp, uint32_t ldvp, uint32_t ldko, uint32_t ldvo, uint32_t ldo, float scale, uint32_t cluster) {
    return sh_t_attention_bwd(q, kp, vp, ko, vo, tok, o, 0, 0, 0, 0, 0, Sq, Lp, So, H, Hkv, dh, ldq, ldkp, ldvp, ldko, ldvo, ldo, 0, 0, 0, 0, scale, cluster);
}

/* ---- data-parallel gradient reduction with the in-network REDADD -----------------------------------------------------
 * Every cluster c holds a full gradient vector of n fp16 elements at src + c * src_stride (bytes); dst (n elements)
 * receives the sum over all clusters. The vector is cut into chunks of SH_T_RED_CHUNK bytes, chunk i is owned by
 * cluster i % P (a reduce-scatter: 16 roots at once, each pulling its chunk from every cluster with one REDADD), and
 * the owner stores its summed chunk to dst. Since all clusters share HBM, no all-gather is needed: the optimizer
 * reads dst. Per round every cluster stages P chunks (one per owner) at fixed TCDM slots, global barrier, each owner
 * issues its reduction (destination-initiated), global barrier, owners store.
 *
 * mode 0: COLLECTIVE_REDADD_FP_16 on the fp16 gradients (the gvsoc NoC adds in fp32 but converts back with a
 *         truncating fp16 conversion and reads fp16 subnormals as 0, floonoc.cpp process_collective_operations).
 * mode 1: exact: every element becomes a fixed-point integer q = round(g 2^e) with one global power-of-two scale
 *         (2^e = 2^19 / max|g| over all clusters, the max found with one REDMAX_FP_16), split into a signed high
 *         limb q >> 12 (|hi| < 2^7) and an unsigned low limb q & 0xFFF; two REDADDs (INT_16, UINT_16) cannot
 *         overflow for 16 clusters, the owner recombines hi 4096 + lo in fp32 and writes fp32 gradients (dst then holds
 *         n fp32). Twice the NoC bytes plus the conversions on the cores.
 * Returns the number of rounds. */
#define SH_T_RED_CHUNK 0x4000u          /* 16 KB per chunk: 16 slots x (hi | lo | out) fit TCDM */
static inline uint16_t sh_t_mask_all(uint32_t dim) { return (uint16_t)~(dim - 1u); }
/* gvsoc collective model: a destination-initiated reduction accumulates the pulled data INTO the read-burst buffer the
 * root's iDMA back-end hands the request (and every intermediate router's copy of it), and those 4 KB buffers are a
 * static FIFO pool of ARCH_IDMA_OUTSTAND_BURST entries that keeps whatever the burst 256 reads earlier carried. So
 * REDADD returns sum + stale (pool contents) once the pool has wrapped (first reductions of a run looked right). Real
 * hardware starts from zero. Workaround: read ARCH_IDMA_OUTSTAND_BURST x 4 KB of zeros from the cluster's zero memory
 * (on the same AXI back-end) right before the reduction, so the bursts it gets (<= 256, FIFO order) hold zeros.
 * Cost: 1 MB of local zero reads per owner and reduction step (~16 k cycles). */
#define SH_T_RED_ZSCR 0xC0000u                 /* 128 KB TCDM sink for the zero reads (above the reduction's buffers) */
SH_T_COLD static void sh_t_red_flush(void) {
    for (uint32_t i = 0; i < ARCH_IDMA_OUTSTAND_BURST * 4096u / ARCH_CLUSTER_ZOMEM_SIZE; ++i)
        bare_dma_start_1d(local(SH_T_RED_ZSCR), zomem(0), ARCH_CLUSTER_ZOMEM_SIZE);
    bare_dma_wait_all();
}
static inline void sh_t_red_stage(uint32_t slots, uint64_t mine, uint32_t rd, uint32_t P, uint32_t nchunk, uint32_t n) {
    const uint32_t CH = SH_T_RED_CHUNK, ce = CH / 2;
    for (uint32_t r = 0; r < P; ++r) {
        const uint32_t c = rd * P + r; if (c >= nchunk) break;
        const uint32_t ne = (n - c * ce) < ce ? (n - c * ce) : ce;
        bare_dma_start_1d(local(slots + r * CH), mine + (uint64_t)c * CH, ne * 2);
    }
    bare_dma_wait_all();
}

SH_T_COLD int sh_t_allreduce(uint64_t dst, uint64_t src, uint32_t src_stride, uint32_t n, uint32_t mode, uint64_t scal) {
#ifdef SH_T_NO_RED_EXACT   /* programs that only use the fp16 REDADD can drop the exact path (instruction memory) */
    mode = 0;
#endif
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, cid = flex_get_cluster_id(), CH = SH_T_RED_CHUNK, ce = CH / 2;
    const int dm = flex_is_dm_core();
    const uint16_t rm = sh_t_mask_all(ARCH_NUM_CLUSTER_X), cm = sh_t_mask_all(ARCH_NUM_CLUSTER_Y);
    const uint64_t mine = src + (uint64_t)cid * src_stride;
    /* TCDM: slots [P] x CH (staged fp16 or hi limbs), lo slots [P] x CH (mode 1), reduction results hi / lo, fp32 out,
     * per-core max partials */
    const uint32_t slots = SH_T_L1_BASE, lslots = slots + P * CH, rhi = lslots + (mode ? P * CH : 0), rlo = rhi + CH, rout = rlo + CH;
    const uint32_t rmax = rout + 2 * CH;
    const uint32_t nchunk = (n + ce - 1) / ce, rounds = (nchunk + P - 1) / P;
    float iq = 1.f; int32_t emax = 15;
    if (mode == 1) {
        /* pass 1: local max |g| (fp16 SIMD on the staged rounds, all cores), as fp16 into a 64 B slot of every
         * cluster, one REDMAX_FP_16 at cluster 0, result through HBM (`scal`, 64 B) */
        sh_v4h m4 = sh_v4_splat_h(0);
        for (uint32_t rd = 0; rd < rounds; ++rd) {
            if (dm) sh_t_red_stage(slots, mine, rd, P, nchunk, n);
            flex_intra_cluster_sync();
            const uint32_t nloc = ((rd * P + P <= nchunk ? P : nchunk - rd * P) * ce) & ~15u;
            uint32_t lo, hi; sh_share(nloc, 16, &lo, &hi);
            const sh_v4h *x = SH_V4CP(local(slots)); const sh_v4h z = sh_v4_splat_h(0);
            for (uint32_t j = lo >> 2; j < (hi >> 2); ++j) m4 = sh_v4_max(m4, sh_v4_max(x[j], sh_v4_sub(z, x[j])));
            flex_intra_cluster_sync();
        }
        ((volatile float *)local(rmax))[flex_get_core_id()] = sh_v4_hmax(m4);
        sh_fp_fence();
        flex_intra_cluster_sync();
        if (flex_is_first_core()) {
            float m = 0.f;
            for (uint32_t c = 0; c < ARCH_NUM_CORE_PER_CLUSTER; ++c) m = sh_fmaxf(m, ((volatile float *)local(rmax))[c]);
            volatile uint16_t *sl = (volatile uint16_t *)local(rmax + 64);
            sl[0] = (uint16_t)sh_f2h(m);
            for (uint32_t i = 1; i < 32; ++i) sl[i] = 0;
        }
        flex_global_barrier_xy();
        if (cid == 0 && dm) {
            flex_dma_async_reduction(rmax + 128, rmax + 64, 64, COLLECTIVE_REDMAX_FP_16, rm, cm);
            flex_dma_async_wait_all();
            bare_dma_start_1d(scal, local(rmax + 128), 64); bare_dma_wait_all();
        }
        flex_global_barrier_xy();
        /* scale 2^e from the max's fp16 exponent Emax: |g| < 2^(Emax - 14), so q = g 2^e with e = 33 - Emax stays below
         * 2^19 (the NoC's truncation can only lower the max by one ulp, inside the same binade or the one below) */
        emax = (int32_t)((((const volatile uint16_t *)(uintptr_t)scal)[0] >> 10) & 31u) + 1; if (emax > 31) emax = 31;
        union { float f; uint32_t u; } s; s.u = (uint32_t)(127 - (33 - emax)) << 23; iq = s.f;
    }
    for (uint32_t rd = 0; rd < rounds; ++rd) {
        /* stage: chunk rd P + r into slot r (all clusters) */
        if (dm) sh_t_red_stage(slots, mine, rd, P, nchunk, n);
        flex_intra_cluster_sync();
        if (mode == 1) {   /* fp16 -> (hi, lo) limbs in place: slot r = hi, lslot r = lo; all cores, element-split */
            const uint32_t nloc = (rd * P + P <= nchunk ? P : nchunk - rd * P) * ce;
            uint32_t lo, hi; sh_share(nloc, 4, &lo, &hi);
            uint16_t *hp = (uint16_t *)local(slots), *lp = (uint16_t *)local(lslots);
            /* integer only, from the fp16 bits (no FP <-> int moves, each a ~15-cycle round trip on these cores):
             * |g| = M 2^(E - 25), M = 1024 + mant (E > 0) or mant (E = 0, E taken as 1); q = M 2^(E - 25 + e), and with
             * e = 33 - emax the shift E - emax + 8 is <= 8; right shifts round half up */
            for (uint32_t i = lo; i < hi; ++i) {
                const uint32_t b = hp[i], E = (b >> 10) & 31u, M = (b & 1023u) | (E ? 1024u : 0u);
                const int32_t sh = (int32_t)(E ? E : 1u) - emax + 8;
                int32_t qv = sh >= 0 ? (int32_t)(M << sh) : (sh > -12 ? (int32_t)((M + (1u << (-sh - 1))) >> -sh) : 0);
                if (b & 0x8000u) qv = -qv;
                hp[i] = (uint16_t)(qv >> 12); lp[i] = (uint16_t)(qv & 0xFFF);
            }
        }
        const uint32_t c = rd * P + cid;
        if (c < nchunk && dm) sh_t_red_flush();   /* all owners at once: only the owner's own iDMA reads until its reduction */
        flex_global_barrier_xy();
        /* all 16 owners issue at once (checked exact against constant per-cluster data, 0.71 vs 0.76 ms serialised for a
         * 3 MiB arena); SH_T_RED_SERIAL makes them take turns (one reduction on the NoC at a time). */
#ifdef SH_T_RED_SERIAL
        for (uint32_t owner = 0; owner < P; ++owner) {
            if (owner == cid && c < nchunk && dm) {
#else
        {
            if (c < nchunk && dm) {
#endif
                const uint32_t ne = (n - c * ce) < ce ? (n - c * ce) : ce;
                if (mode == 0) {
                    flex_dma_async_reduction(rhi, slots + cid * CH, ne * 2, COLLECTIVE_REDADD_FP_16, rm, cm);
                    flex_dma_async_wait_all();
                } else {
                    flex_dma_async_reduction(rhi, slots + cid * CH, ne * 2, COLLECTIVE_REDADD_INT_16, rm, cm);
                    flex_dma_async_wait_all();
                    flex_dma_async_reduction(rlo, lslots + cid * CH, ne * 2, COLLECTIVE_REDADD_UINT_16, rm, cm);
                    flex_dma_async_wait_all();
                }
            }
            flex_global_barrier_xy();      /* every pull done: the slots may be overwritten / the NoC is free */
        }
        if (c < nchunk) {
            const uint32_t ne = (n - c * ce) < ce ? (n - c * ce) : ce;
            if (mode == 0) {
                if (dm) { bare_dma_start_1d(dst + (uint64_t)c * CH, local(rhi), ne * 2); bare_dma_wait_all(); }
            } else {
                uint32_t lo, hi; sh_share(ne, 4, &lo, &hi);
                const int16_t *hs = (const int16_t *)local(rhi); const uint16_t *ls = (const uint16_t *)local(rlo); float *of = (float *)local(rout);
                for (uint32_t i = lo; i < hi; ++i) {
                    const int32_t qv = (int32_t)hs[i] * 4096 + (int32_t)ls[i];
                    float f;
                    __asm__ volatile ("fcvt.s.w %0, %1" : "=f"(f) : "r"(qv));
                    of[i] = f * iq;
                }
                sh_fp_fence();
                flex_intra_cluster_sync();
                if (dm) { bare_dma_start_1d(dst + (uint64_t)c * ce * 4, local(rout), ne * 4); bare_dma_wait_all(); }
            }
        }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
    return (int)rounds;
}

/* ---- test helper: fp32 samples ("<tag> r c XXXXXXXX", the IEEE bits), first core ------------------------------- */
SH_T_COLD void sh_t_dump_samples_f32(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, uint32_t nsamples, const char *tag) {
    uint32_t s = seed ^ 0x9e3779b9u;
    for (uint32_t n = 0; n < nsamples; ++n) {
        uint32_t i = sh_lcg(&s) % rows, j = sh_lcg(&s) % cols;
        uint32_t v = ((const volatile uint32_t *)(uintptr_t)(a + (uint64_t)i * ld * 4))[j];
        sh_printf("%s %u %u %08x\n", tag, i, j, v);
    }
}
