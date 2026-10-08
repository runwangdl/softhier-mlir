/* Row-wise / elementwise fp16 ops on HBM tensors. Row blocks are staged into TCDM by the DM core
 * (double-buffered: block i+1 streams in and block i-1 streams out while block i is computed), ALL
 * THREE cores of the cluster compute a share of every block in fp32, and the DM core streams the
 * result back. `cluster` selects the executing cluster, or SH_ALL to deal the blocks round-robin over
 * all clusters (ends with a global barrier).
 *
 * fp16 <-> fp32 goes through the Zfh register conversions (sh_h2f4 / sh_f2h4 in sh_ops.h: integer
 * lhu/sh + fmv + fcvt; scalar flh/fsh are broken in gvsoc, docs/SIMULATOR_NOTES.md #2). The Snitch
 * FPU model overlaps independent instructions but serialises dependent ones (~6 cycles per dependent
 * op, ~2 per independent one), so every kernel works on 8 elements at a time through the 4-wide
 * interleaved converters and keeps 8 partial accumulators. Broadcast parameter rows (gamma, beta,
 * bias) are converted to fp32 once per op so the inner loops read them with a single flw. */
#define SH_ROWOPS_L1_BASE 0x1000u       /* staging area: keep clear of TCDM 0 (NULL for the compiler, SDK L1 allocator at 0x10) */
#define SH_ROWOPS_L1_BYTES 0x80000u     /* params + 2 x (x [+b] + y) */
#define SH_ROWOPS_MAX_PCOLS 4096u       /* parameter rows: fp16 copy + fp32 copy each */
#define SH_ROWOPS_PARAM_BYTES (SH_ROWOPS_MAX_PCOLS * 12u)   /* 2 x fp16 (2 B) + 2 x fp32 (4 B) per column = 48 KB */

static inline int sh_my_block(uint32_t blk, uint32_t cluster) {
    if (cluster == SH_ALL) return (blk % (ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y)) == flex_get_cluster_id();
    return flex_get_cluster_id() == cluster;
}
static inline void sh_end_op(uint32_t cluster) {
    flex_intra_cluster_sync();
    if (cluster == SH_ALL) flex_global_barrier_xy();
}
/* this core's contiguous share [lo, hi) of n items (rows or elements), in units of `q`.
 * SH_ROWOPS_CORES (default: all cores of the cluster) can be lowered for scaling experiments. */
#ifndef SH_ROWOPS_CORES
#define SH_ROWOPS_CORES ARCH_NUM_CORE_PER_CLUSTER
#endif
static inline void sh_share(uint32_t n, uint32_t q, uint32_t *lo, uint32_t *hi) {
    const uint32_t c = flex_get_core_id(), nc = SH_ROWOPS_CORES, nq = (n + q - 1) / q;
    if (c >= nc) { *lo = *hi = n; return; }
    *lo = (nq * c / nc) * q; *hi = (nq * (c + 1) / nc) * q; if (*hi > n) *hi = n; if (*lo > n) *lo = n;
}
static inline float sh_fmaxf(float a, float b) { float r; __asm__ ("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
static inline float sh_fminf(float a, float b) { float r; __asm__ ("fmin.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
/* 2^t for t already clamped to [-126, 126]: 2^k * p(f), k = round(t), f = t - k in [-0.5, 0.5],
 * degree-4 p (rel err < 6e-6), 9 FPU ops. The rounding is done in pure FP (t + 1.5*2^23 - 1.5*2^23)
 * and 2^k is built from the bits of that sum: the obvious fcvt.w.s -> fcvt.s.w pair returns a stale
 * integer when several independent pairs are issued back to back (docs/SIMULATOR_NOTES.md #6). */
static inline float sh_exp2_clamped(float t) {
    const float M = 12582912.f;                    /* 1.5 * 2^23: t + M is integer-valued (RNE) */
    float m = t + M; uint32_t mb; float e2k;
    __asm__ ("" : "+f"(m));                                               /* opaque: -ffast-math must not fold (t + M) - M */
    __asm__ ("fmv.x.w %0, %1" : "=r"(mb) : "f"(m));                       /* bits(m) = 0x4B400000 + k */
    __asm__ ("fmv.w.x %0, %1" : "=f"(e2k) : "r"((mb << 23) + ((127u - 0x4B400000u) << 23)));   /* (k + 127) << 23 */
    float kf = m - M;
    __asm__ ("" : "+f"(kf));
    float f = t - kf;
    float p = 1.f + f * (0.69312805f + f * (0.24023677f + f * (0.055870272f + f * 0.00959024f)));
    return p * e2k;
}
#define SH_LOG2E 1.442695041f
static inline float sh_expf(float x) { return sh_exp2_clamped(sh_fminf(sh_fmaxf(x * SH_LOG2E, -126.f), 127.f)); }
static inline float sh_rsqrtf(float x) { union { float f; uint32_t u; } u; u.f = x; u.u = 0x5f3759df - (u.u >> 1); float y = u.f; y = y * (1.5f - 0.5f * x * y * y); y = y * (1.5f - 0.5f * x * y * y); return y; }
/* gelu (tanh form) == x * sigmoid(2u), u = 0.79788(x + 0.044715 x^3)  ->  x / (1 + 2^(x (k0 + k1 x^2))) */
#define SH_GELU_K0 (-2.3022082f)
#define SH_GELU_K1 (-0.10294324f)
static inline float sh_gelu1(float x) {
    float t = sh_fminf(sh_fmaxf(x * (SH_GELU_K0 + SH_GELU_K1 * x * x), -126.f), 126.f);
    return x / (1.f + sh_exp2_clamped(t));
}

/* 8 elements per step: load/convert, apply F to each, convert/store. */
#define SH_LOAD8(p, i) float a0, a1, a2, a3, b0, b1, b2, b3; sh_h2f4((p) + (i), &a0, &a1, &a2, &a3); sh_h2f4((p) + (i) + 4, &b0, &b1, &b2, &b3)
#define SH_LOAD8B(p, i) float c0, c1, c2, c3, d0, d1, d2, d3; sh_h2f4((p) + (i), &c0, &c1, &c2, &c3); sh_h2f4((p) + (i) + 4, &d0, &d1, &d2, &d3)
#define SH_STORE8(p, i) sh_f2h4((p) + (i), a0, a1, a2, a3); sh_f2h4((p) + (i) + 4, b0, b1, b2, b3)
#define SH_APPLY8(F) a0 = F(a0); a1 = F(a1); a2 = F(a2); a3 = F(a3); b0 = F(b0); b1 = F(b1); b2 = F(b2); b3 = F(b3)

/* One staged block as seen by a kernel. p0/p1: the two broadcast parameter rows as fp32 (cols each). */
typedef struct {
    uint16_t *y; const uint16_t *x; const uint16_t *b; const float *p0; const float *p1;
    uint32_t nr, cols; const void *arg;
} sh_blk;
typedef void (*sh_rowfn_t)(const sh_blk *k);

/* Generic driver. x (and optionally a per-row second input b) are streamed in `rpb`-row blocks;
 * p0/p1 are single parameter rows staged once. Every core calls this; kernels split the block. */
static void sh_rowop(uint64_t y, uint64_t x, uint64_t b, uint64_t p0, uint64_t p1, uint32_t rows, uint32_t cols,
                     uint32_t ldy, uint32_t ldx, uint32_t ldb, sh_rowfn_t fn, const void *arg, uint32_t cluster) {
    const uint32_t rowb = cols * 2, nbuf = b ? 3 : 2;
    const uint32_t pbytes = (p0 || p1) ? SH_ROWOPS_PARAM_BYTES : 0;
    uint32_t rpb = (SH_ROWOPS_L1_BYTES - pbytes) / (rowb * nbuf * 2); if (rpb == 0) rpb = 1; if (rpb > rows) rpb = rows;
    if (cluster == SH_ALL) {   /* enough blocks to keep every cluster busy */
        const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, want = (rows + P - 1) / P;
        if (rpb > want) rpb = want ? want : 1;
    }
    const uint32_t setb = rpb * rowb * nbuf;                 /* bytes per buffer set */
    const int dm = flex_is_dm_core();
    /* parameter rows: fp16 staging copies, then fp32 copies the kernels read */
    const uint32_t ph0 = SH_ROWOPS_L1_BASE, ph1 = ph0 + SH_ROWOPS_MAX_PCOLS * 2, pf0 = ph1 + SH_ROWOPS_MAX_PCOLS * 2, pf1 = pf0 + SH_ROWOPS_MAX_PCOLS * 4;
    const uint32_t nblk = (rows + rpb - 1) / rpb;
    uint32_t next = 0;                                       /* next block index to stage */
    while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_SET_X(s)  (SH_ROWOPS_L1_BASE + pbytes + (s) * setb)
    #define SH_SET_B(s)  (SH_SET_X(s) + rpb * rowb)
    #define SH_SET_Y(s)  (SH_SET_X(s) + (nbuf - 1) * rpb * rowb)
    #define SH_LOAD_BLK(bi, s) do { const uint32_t r_ = (bi) * rpb, n_ = (rows - r_) < rpb ? (rows - r_) : rpb; \
        bare_dma_start_2d(local(SH_SET_X(s)), x + (uint64_t)r_ * ldx * 2, rowb, rowb, ldx * 2, n_); \
        if (b) bare_dma_start_2d(local(SH_SET_B(s)), b + (uint64_t)r_ * ldb * 2, rowb, rowb, ldb * 2, n_); } while (0)
    if (next < nblk) {
        if (dm) {
            if (p0) bare_dma_start_1d(local(ph0), p0, rowb);
            if (p1) bare_dma_start_1d(local(ph1), p1, rowb);
            SH_LOAD_BLK(next, 0);
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
        if (pbytes) {                                        /* fp16 -> fp32 parameter rows, split over the cores */
            uint32_t lo, hi; sh_share(cols, 8, &lo, &hi);
            for (uint32_t i = lo; i < hi; i += 4) {          /* lo/hi are multiples of 8, cols need not be: pad is harmless */
                if (p0) sh_h2f4((const uint16_t *)local(ph0) + i, (float *)local(pf0) + i, (float *)local(pf0) + i + 1, (float *)local(pf0) + i + 2, (float *)local(pf0) + i + 3);
                if (p1) sh_h2f4((const uint16_t *)local(ph1) + i, (float *)local(pf1) + i, (float *)local(pf1) + i + 1, (float *)local(pf1) + i + 2, (float *)local(pf1) + i + 3);
            }
            flex_intra_cluster_sync();
        }
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = blk * rpb, nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm && next < nblk) SH_LOAD_BLK(next, set ^ 1);  /* stream the following block in during compute */
        sh_blk k = { (uint16_t *)local(SH_SET_Y(set)), (const uint16_t *)local(SH_SET_X(set)),
                     b ? (const uint16_t *)local(SH_SET_B(set)) : 0, (const float *)local(pf0), (const float *)local(pf1), nr, cols, arg };
        fn(&k);
        flex_intra_cluster_sync();
        if (dm) {                                            /* next block landed (and older stores drained); stream this one out */
            bare_dma_wait_all();
            for (uint32_t r = 0; r < nr; ++r) bare_dma_start_1d(y + (uint64_t)(r0 + r) * ldy * 2, local(SH_SET_Y(set) + r * rowb), rowb);
        }
        flex_intra_cluster_sync();
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_SET_X
    #undef SH_SET_B
    #undef SH_SET_Y
    #undef SH_LOAD_BLK
    sh_end_op(cluster);
}

/* ---- kernels on one L1 block (each core: its share of rows / elements) ---------------------- */
static inline float sh_row_sum(const uint16_t *xr, uint32_t cols) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f; uint32_t i = 0;
    for (; i + 8 <= cols; i += 8) { SH_LOAD8(xr, i); s0 += a0; s1 += a1; s2 += a2; s3 += a3; t0 += b0; t1 += b1; t2 += b2; t3 += b3; }
    for (; i < cols; ++i) s0 += sh_h2f(xr[i]);
    return ((s0 + s1) + (s2 + s3)) + ((t0 + t1) + (t2 + t3));
}
static inline float sh_row_sqdev(const uint16_t *xr, uint32_t cols, float mean) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f; uint32_t i = 0;
    for (; i + 8 <= cols; i += 8) {
        SH_LOAD8(xr, i);
        a0 -= mean; a1 -= mean; a2 -= mean; a3 -= mean; b0 -= mean; b1 -= mean; b2 -= mean; b3 -= mean;
        s0 += a0 * a0; s1 += a1 * a1; s2 += a2 * a2; s3 += a3 * a3; t0 += b0 * b0; t1 += b1 * b1; t2 += b2 * b2; t3 += b3 * b3;
    }
    for (; i < cols; ++i) { float d = sh_h2f(xr[i]) - mean; s0 += d * d; }
    return ((s0 + s1) + (s2 + s3)) + ((t0 + t1) + (t2 + t3));
}
static void sh_k_layernorm(const sh_blk *k) {
    const float eps = *(const float *)k->arg; const float *g = k->p0, *be = k->p1; const uint32_t cols = k->cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols;
        const float mean = sh_row_sum(xr, cols) / (float)cols;
        const float rs = sh_rsqrtf(sh_row_sqdev(xr, cols, mean) / (float)cols + eps);
        uint32_t i = 0;
        for (; i + 8 <= cols; i += 8) {
            SH_LOAD8(xr, i);
            a0 = (a0 - mean) * (rs * g[i]) + be[i];         a1 = (a1 - mean) * (rs * g[i + 1]) + be[i + 1];
            a2 = (a2 - mean) * (rs * g[i + 2]) + be[i + 2]; a3 = (a3 - mean) * (rs * g[i + 3]) + be[i + 3];
            b0 = (b0 - mean) * (rs * g[i + 4]) + be[i + 4]; b1 = (b1 - mean) * (rs * g[i + 5]) + be[i + 5];
            b2 = (b2 - mean) * (rs * g[i + 6]) + be[i + 6]; b3 = (b3 - mean) * (rs * g[i + 7]) + be[i + 7];
            SH_STORE8(yr, i);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h((sh_h2f(xr[i]) - mean) * (rs * g[i]) + be[i]);
    }
}
static void sh_k_softmax(const sh_blk *k) {
    const float s2 = *(const float *)k->arg * SH_LOG2E;   /* work in the log2 domain: exp(s x - m) = 2^(s2 x - m2) */
    const uint32_t cols = k->cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols;
        float m0 = -3.0e38f, m1 = m0, m2 = m0, m3 = m0, n0 = m0, n1 = m0, n2 = m0, n3 = m0; uint32_t i = 0;
        for (; i + 8 <= cols; i += 8) {
            SH_LOAD8(xr, i);
            m0 = sh_fmaxf(m0, a0 * s2); m1 = sh_fmaxf(m1, a1 * s2); m2 = sh_fmaxf(m2, a2 * s2); m3 = sh_fmaxf(m3, a3 * s2);
            n0 = sh_fmaxf(n0, b0 * s2); n1 = sh_fmaxf(n1, b1 * s2); n2 = sh_fmaxf(n2, b2 * s2); n3 = sh_fmaxf(n3, b3 * s2);
        }
        for (; i < cols; ++i) m0 = sh_fmaxf(m0, sh_h2f(xr[i]) * s2);
        const float m2x = sh_fmaxf(sh_fmaxf(sh_fmaxf(m0, m1), sh_fmaxf(m2, m3)), sh_fmaxf(sh_fmaxf(n0, n1), sh_fmaxf(n2, n3)));
        float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f, t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
        #define SH_E(v) sh_exp2_clamped(sh_fmaxf((v) * s2 - m2x, -126.f))     /* <= 0 by construction */
        for (i = 0; i + 8 <= cols; i += 8) {
            SH_LOAD8(xr, i);
            SH_APPLY8(SH_E);
            SH_STORE8(yr, i);
            q0 += a0; q1 += a1; q2 += a2; q3 += a3; t0 += b0; t1 += b1; t2 += b2; t3 += b3;
        }
        for (; i < cols; ++i) { float e = SH_E(sh_h2f(xr[i])); yr[i] = (uint16_t)sh_f2h(e); q0 += e; }
        #undef SH_E
        const float inv = 1.f / (((q0 + q1) + (q2 + q3)) + ((t0 + t1) + (t2 + t3)));
        for (i = 0; i + 8 <= cols; i += 8) {
            SH_LOAD8(yr, i);
            a0 *= inv; a1 *= inv; a2 *= inv; a3 *= inv; b0 *= inv; b1 *= inv; b2 *= inv; b3 *= inv;
            SH_STORE8(yr, i);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h(sh_h2f(yr[i]) * inv);
    }
}
static void sh_k_gelu(const sh_blk *k) {
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    const uint16_t *x = k->x; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 8 <= hi; i += 8) { SH_LOAD8(x, i); SH_APPLY8(sh_gelu1); SH_STORE8(y, i); }
    for (; i < hi; ++i) y[i] = (uint16_t)sh_f2h(sh_gelu1(sh_h2f(x[i])));
}
static void sh_k_add(const sh_blk *k) {          /* y = x + b, b per row */
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    const uint16_t *x = k->x, *b = k->b; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 8 <= hi; i += 8) {
        SH_LOAD8(x, i); SH_LOAD8B(b, i);
        a0 += c0; a1 += c1; a2 += c2; a3 += c3; b0 += d0; b1 += d1; b2 += d2; b3 += d3;
        SH_STORE8(y, i);
    }
    for (; i < hi; ++i) y[i] = (uint16_t)sh_f2h(sh_h2f(x[i]) + sh_h2f(b[i]));
}
static void sh_k_add_bias(const sh_blk *k) {     /* y = x + p0 (one row broadcast) */
    const uint32_t cols = k->cols; const float *bias = k->p0;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols; uint32_t i = 0;
        for (; i + 8 <= cols; i += 8) {
            SH_LOAD8(xr, i);
            a0 += bias[i]; a1 += bias[i + 1]; a2 += bias[i + 2]; a3 += bias[i + 3]; b0 += bias[i + 4]; b1 += bias[i + 5]; b2 += bias[i + 6]; b3 += bias[i + 7];
            SH_STORE8(yr, i);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h(sh_h2f(xr[i]) + bias[i]);
    }
}
static void sh_k_scale(const sh_blk *k) {
    const float s = *(const float *)k->arg;
    uint32_t lo, hi; sh_share(k->nr * k->cols, 8, &lo, &hi);
    const uint16_t *x = k->x; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 8 <= hi; i += 8) { SH_LOAD8(x, i); a0 *= s; a1 *= s; a2 *= s; a3 *= s; b0 *= s; b1 *= s; b2 *= s; b3 *= s; SH_STORE8(y, i); }
    for (; i < hi; ++i) y[i] = (uint16_t)sh_f2h(sh_h2f(x[i]) * s);
}

/* ---- public ops ---------------------------------------------------------------------------- */
void sh_layernorm(uint64_t y, uint64_t x, uint64_t gamma, uint64_t beta, uint32_t rows, uint32_t cols, uint32_t ld, float eps, uint32_t cluster) {
    sh_rowop(y, x, 0, gamma, beta, rows, cols, ld, ld, 0, sh_k_layernorm, &eps, cluster);
}
void sh_softmax_rows(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float scale, uint32_t cluster) {
    sh_rowop(y, x, 0, 0, 0, rows, cols, ld, ld, 0, sh_k_softmax, &scale, cluster);
}
void sh_gelu(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    sh_rowop(y, x, 0, 0, 0, rows, cols, ld, ld, 0, sh_k_gelu, 0, cluster);
}
void sh_add(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    sh_rowop(y, a, b, 0, 0, rows, cols, ld, ld, ld, sh_k_add, 0, cluster);
}
void sh_add_bias(uint64_t y, uint64_t x, uint64_t bias_row, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    sh_rowop(y, x, 0, bias_row, 0, rows, cols, ld, ld, 0, sh_k_add_bias, 0, cluster);
}
void sh_scale(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float s, uint32_t cluster) {
    sh_rowop(y, x, 0, 0, 0, rows, cols, ld, ld, 0, sh_k_scale, &s, cluster);
}

/* dst[cols, rows] = src[rows, cols]^T in BxB blocks through TCDM (double-buffered, blocks dealt
 * round-robin over clusters, rows of the block split over the 3 cores, 2x2 micro-tiles via 32-bit
 * loads/stores so each core moves 4 elements per 2 loads + 2 stores). The staged tiles use a row
 * pitch of B+2 elements so the three cores and the strided stores do not all hit one TCDM bank. */
void sh_transpose(uint64_t dst, uint64_t src, uint32_t rows, uint32_t cols, uint32_t ld_src, uint32_t ld_dst, uint32_t cluster) {
    const uint32_t B = 128, P = B + 2, bb = B * P * 2;   /* 4 buffers: src/dst x 2 sets */
    const int dm = flex_is_dm_core();
    const uint32_t nbr = (rows + B - 1) / B, nbc = (cols + B - 1) / B, nblk = nbr * nbc;
    uint32_t next = 0; while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_T_S(s) (SH_ROWOPS_L1_BASE + (s) * 2 * bb)
    #define SH_T_D(s) (SH_ROWOPS_L1_BASE + (s) * 2 * bb + bb)
    #define SH_T_LOAD(bi, s) do { const uint32_t r_ = ((bi) / nbc) * B, c_ = ((bi) % nbc) * B; \
        const uint32_t nr_ = (rows - r_) < B ? rows - r_ : B, nc_ = (cols - c_) < B ? cols - c_ : B; \
        bare_dma_start_2d(local(SH_T_S(s)), src + ((uint64_t)r_ * ld_src + c_) * 2, nc_ * 2, P * 2, ld_src * 2, nr_); } while (0)
    if (next < nblk) {
        if (dm) { SH_T_LOAD(next, 0); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = (blk / nbc) * B, c0 = (blk % nbc) * B;
        const uint32_t nr = (rows - r0) < B ? rows - r0 : B, nc = (cols - c0) < B ? cols - c0 : B;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm && next < nblk) SH_T_LOAD(next, set ^ 1);
        {   /* each core: its share of the block's row pairs */
            const uint16_t *s = (const uint16_t *)local(SH_T_S(set)); uint16_t *d = (uint16_t *)local(SH_T_D(set));
            uint32_t lo, hi; sh_share(nr, 2, &lo, &hi);
            uint32_t r = lo;
            for (; r + 2 <= hi; r += 2) {
                const uint32_t *s0 = (const uint32_t *)(s + r * P), *s1 = (const uint32_t *)(s + (r + 1) * P);
                uint32_t *d0 = (uint32_t *)(d + r);   /* column c of d at d0 + c*P/2 words */
                uint32_t c = 0;
                for (; c + 2 <= nc; c += 2) {
                    const uint32_t w0 = s0[c >> 1], w1 = s1[c >> 1];        /* w0 = {x[r][c+1], x[r][c]} */
                    d0[(c * P) >> 1] = (w0 & 0xFFFFu) | (w1 << 16);
                    d0[((c + 1) * P) >> 1] = (w0 >> 16) | (w1 & 0xFFFF0000u);
                }
                for (; c < nc; ++c) { d[c * P + r] = s[r * P + c]; d[c * P + r + 1] = s[(r + 1) * P + c]; }
            }
            for (; r < hi; ++r) for (uint32_t c = 0; c < nc; ++c) d[c * P + r] = s[r * P + c];
        }
        flex_intra_cluster_sync();
        if (dm) {
            bare_dma_wait_all();
            for (uint32_t c = 0; c < nc; ++c) bare_dma_start_1d(dst + ((uint64_t)(c0 + c) * ld_dst + r0) * 2, local(SH_T_D(set) + c * P * 2), nr * 2);
        }
        flex_intra_cluster_sync();
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_T_S
    #undef SH_T_D
    #undef SH_T_LOAD
    sh_end_op(cluster);
}
