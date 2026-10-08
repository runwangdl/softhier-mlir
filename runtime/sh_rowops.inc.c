/* Row-wise / elementwise fp16 ops on HBM tensors. Row blocks are staged into TCDM by the DM core
 * (double-buffered: block i+1 streams in while block i is computed), ALL THREE cores of the cluster
 * compute a share of every block in fp32, and the DM core streams the result back. `cluster` selects
 * the executing cluster, or SH_ALL to deal the blocks round-robin over all clusters (ends with a
 * global barrier). fp16 <-> fp32 goes through the Zfh register conversions (sh_h2f / sh_f2h in
 * sh_ops.h: integer lhu/sh + fmv + fcvt; scalar flh/fsh are broken in gvsoc, docs/SIMULATOR_NOTES.md).
 * Kernels are unrolled 4-way with independent accumulators: the Snitch FPU model only overlaps
 * independent instructions (a dependent chain costs ~18 cycles per element, 4 chains ~8). */
#define SH_ROWOPS_L1_BASE 0x1000u       /* staging area: keep clear of TCDM 0 (NULL for the compiler, SDK L1 allocator at 0x10) */
#define SH_ROWOPS_L1_BYTES 0x80000u     /* params + 2 x (x [+b] + y) */
#define SH_ROWOPS_PARAM_BYTES 0x4000u   /* two broadcast parameter rows (<= 4096 cols each) */
#define SH_UNROLL 4

static inline int sh_my_block(uint32_t blk, uint32_t cluster) {
    if (cluster == SH_ALL) return (blk % (ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y)) == flex_get_cluster_id();
    return flex_get_cluster_id() == cluster;
}
static inline void sh_end_op(uint32_t cluster) {
    flex_intra_cluster_sync();
    if (cluster == SH_ALL) flex_global_barrier_xy();
}
/* this core's contiguous share [lo, hi) of n items (rows or elements), in units of `q` */
static inline void sh_share(uint32_t n, uint32_t q, uint32_t *lo, uint32_t *hi) {
    const uint32_t c = flex_get_core_id(), nc = ARCH_NUM_CORE_PER_CLUSTER, nq = (n + q - 1) / q;
    *lo = (nq * c / nc) * q; *hi = (nq * (c + 1) / nc) * q; if (*hi > n) *hi = n; if (*lo > n) *lo = n;
}
static inline float sh_fmaxf(float a, float b) { float r; __asm__ ("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
static inline float sh_fminf(float a, float b) { float r; __asm__ ("fmin.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
static inline float sh_floorf_fast(float x) { int k; float r; __asm__ ("fcvt.w.s %0, %1, rdn" : "=r"(k) : "f"(x)); __asm__ ("fcvt.s.w %0, %1" : "=f"(r) : "r"(k)); return r; }
static inline float sh_exp2i(float kf) {       /* 2^k for integer-valued kf in [-126, 127] */
    int k; float r; __asm__ ("fcvt.w.s %0, %1, rtz" : "=r"(k) : "f"(kf));
    __asm__ ("fmv.w.x %0, %1" : "=f"(r) : "r"((uint32_t)(k + 127) << 23)); return r;
}
/* exp(x): 2^k * 2^f, k = floor(x log2 e), f in [0,1) by a degree-4 polynomial (rel err < 4e-6).
 * x is clamped to [-126 ln2, 88]; 9 FPU ops, no branches. */
static inline float sh_expf(float x) {
    float t = sh_fminf(sh_fmaxf(x * 1.44269504f, -126.f), 127.f);
    float kf = sh_floorf_fast(t), f = t - kf;
    float p = 1.f + f * (0.69304348f + f * (0.24124981f + f * (0.05235283f + f * 0.01334154f)));
    return p * sh_exp2i(kf);
}
static inline float sh_tanhf(float x) {           /* rational approx, |err| ~1e-4 */
    x = sh_fminf(sh_fmaxf(x, -9.f), 9.f);
    float x2 = x * x;
    float a = x * (135135.f + x2 * (17325.f + x2 * (378.f + x2)));
    float b = 135135.f + x2 * (62370.f + x2 * (3150.f + x2 * 28.f));
    return sh_fminf(sh_fmaxf(a / b, -1.f), 1.f);
}
static inline float sh_rsqrtf(float x) { union { float f; uint32_t u; } u; u.f = x; u.u = 0x5f3759df - (u.u >> 1); float y = u.f; y = y * (1.5f - 0.5f * x * y * y); y = y * (1.5f - 0.5f * x * y * y); return y; }

/* One staged block as seen by a kernel. p0/p1: the two broadcast parameter rows (cols each). */
typedef struct {
    uint16_t *y; const uint16_t *x; const uint16_t *b; const uint16_t *p0; const uint16_t *p1;
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
    if (dm) {
        if (p0) bare_dma_start_1d(local(SH_ROWOPS_L1_BASE), p0, rowb);
        if (p1) bare_dma_start_1d(local(SH_ROWOPS_L1_BASE + SH_ROWOPS_PARAM_BYTES / 2), p1, rowb);
    }
    const uint16_t *pp0 = (const uint16_t *)local(SH_ROWOPS_L1_BASE), *pp1 = (const uint16_t *)local(SH_ROWOPS_L1_BASE + SH_ROWOPS_PARAM_BYTES / 2);
    const uint32_t nblk = (rows + rpb - 1) / rpb;
    uint32_t next = 0;                                       /* next block index to stage */
    while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_SET_X(s)  (SH_ROWOPS_L1_BASE + pbytes + (s) * setb)
    #define SH_SET_B(s)  (SH_SET_X(s) + rpb * rowb)
    #define SH_SET_Y(s)  (SH_SET_X(s) + (nbuf - 1) * rpb * rowb)
    if (dm && next < nblk) {                                 /* prefetch the first block */
        const uint32_t r0 = next * rpb, nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        bare_dma_start_2d(local(SH_SET_X(0)), x + (uint64_t)r0 * ldx * 2, rowb, rowb, ldx * 2, nr);
        if (b) bare_dma_start_2d(local(SH_SET_B(0)), b + (uint64_t)r0 * ldb * 2, rowb, rowb, ldb * 2, nr);
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = blk * rpb, nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm) bare_dma_wait_all();                         /* block `blk` landed (and older stores drained) */
        flex_intra_cluster_sync();
        if (dm && next < nblk) {                             /* stream block `next` into the other set */
            const uint32_t n0 = next * rpb, nn = (rows - n0) < rpb ? (rows - n0) : rpb;
            bare_dma_start_2d(local(SH_SET_X(set ^ 1)), x + (uint64_t)n0 * ldx * 2, rowb, rowb, ldx * 2, nn);
            if (b) bare_dma_start_2d(local(SH_SET_B(set ^ 1)), b + (uint64_t)n0 * ldb * 2, rowb, rowb, ldb * 2, nn);
        }
        sh_blk k = { (uint16_t *)local(SH_SET_Y(set)), (const uint16_t *)local(SH_SET_X(set)),
                     b ? (const uint16_t *)local(SH_SET_B(set)) : 0, pp0, pp1, nr, cols, arg };
        fn(&k);
        flex_intra_cluster_sync();
        if (dm) for (uint32_t r = 0; r < nr; ++r) bare_dma_start_1d(y + (uint64_t)(r0 + r) * ldy * 2, local(SH_SET_Y(set) + r * rowb), rowb);
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_SET_X
    #undef SH_SET_B
    #undef SH_SET_Y
    sh_end_op(cluster);
}

/* ---- kernels on one L1 block (each core: its share of rows / elements) ---------------------- */
static inline float sh_row_sum(const uint16_t *xr, uint32_t cols) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f; uint32_t i = 0;
    for (; i + 4 <= cols; i += 4) { s0 += sh_h2f(xr[i]); s1 += sh_h2f(xr[i + 1]); s2 += sh_h2f(xr[i + 2]); s3 += sh_h2f(xr[i + 3]); }
    for (; i < cols; ++i) s0 += sh_h2f(xr[i]);
    return (s0 + s1) + (s2 + s3);
}
static inline float sh_row_sqdev(const uint16_t *xr, uint32_t cols, float mean) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f; uint32_t i = 0;
    for (; i + 4 <= cols; i += 4) {
        float d0 = sh_h2f(xr[i]) - mean, d1 = sh_h2f(xr[i + 1]) - mean, d2 = sh_h2f(xr[i + 2]) - mean, d3 = sh_h2f(xr[i + 3]) - mean;
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    }
    for (; i < cols; ++i) { float d = sh_h2f(xr[i]) - mean; s0 += d * d; }
    return (s0 + s1) + (s2 + s3);
}
static void sh_k_layernorm(const sh_blk *k) {
    const float eps = *(const float *)k->arg; const uint16_t *g = k->p0, *be = k->p1; const uint32_t cols = k->cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols;
        const float mean = sh_row_sum(xr, cols) / (float)cols;
        const float rs = sh_rsqrtf(sh_row_sqdev(xr, cols, mean) / (float)cols + eps);
        uint32_t i = 0;
        for (; i + 4 <= cols; i += 4) {
            float a0 = (sh_h2f(xr[i]) - mean) * (rs * sh_h2f(g[i])) + sh_h2f(be[i]);
            float a1 = (sh_h2f(xr[i + 1]) - mean) * (rs * sh_h2f(g[i + 1])) + sh_h2f(be[i + 1]);
            float a2 = (sh_h2f(xr[i + 2]) - mean) * (rs * sh_h2f(g[i + 2])) + sh_h2f(be[i + 2]);
            float a3 = (sh_h2f(xr[i + 3]) - mean) * (rs * sh_h2f(g[i + 3])) + sh_h2f(be[i + 3]);
            yr[i] = (uint16_t)sh_f2h(a0); yr[i + 1] = (uint16_t)sh_f2h(a1); yr[i + 2] = (uint16_t)sh_f2h(a2); yr[i + 3] = (uint16_t)sh_f2h(a3);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h((sh_h2f(xr[i]) - mean) * (rs * sh_h2f(g[i])) + sh_h2f(be[i]));
    }
}
static void sh_k_softmax(const sh_blk *k) {
    const float scale = *(const float *)k->arg; const uint32_t cols = k->cols;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols;
        float m0 = -3.0e38f, m1 = m0, m2 = m0, m3 = m0; uint32_t i = 0;
        for (; i + 4 <= cols; i += 4) { m0 = sh_fmaxf(m0, sh_h2f(xr[i]) * scale); m1 = sh_fmaxf(m1, sh_h2f(xr[i + 1]) * scale); m2 = sh_fmaxf(m2, sh_h2f(xr[i + 2]) * scale); m3 = sh_fmaxf(m3, sh_h2f(xr[i + 3]) * scale); }
        for (; i < cols; ++i) m0 = sh_fmaxf(m0, sh_h2f(xr[i]) * scale);
        const float m = sh_fmaxf(sh_fmaxf(m0, m1), sh_fmaxf(m2, m3));
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (i = 0; i + 4 <= cols; i += 4) {
            float e0 = sh_expf(sh_h2f(xr[i]) * scale - m), e1 = sh_expf(sh_h2f(xr[i + 1]) * scale - m);
            float e2 = sh_expf(sh_h2f(xr[i + 2]) * scale - m), e3 = sh_expf(sh_h2f(xr[i + 3]) * scale - m);
            yr[i] = (uint16_t)sh_f2h(e0); yr[i + 1] = (uint16_t)sh_f2h(e1); yr[i + 2] = (uint16_t)sh_f2h(e2); yr[i + 3] = (uint16_t)sh_f2h(e3);
            s0 += e0; s1 += e1; s2 += e2; s3 += e3;
        }
        for (; i < cols; ++i) { float e = sh_expf(sh_h2f(xr[i]) * scale - m); yr[i] = (uint16_t)sh_f2h(e); s0 += e; }
        const float inv = 1.f / ((s0 + s1) + (s2 + s3));
        for (i = 0; i + 4 <= cols; i += 4) {
            float a0 = sh_h2f(yr[i]) * inv, a1 = sh_h2f(yr[i + 1]) * inv, a2 = sh_h2f(yr[i + 2]) * inv, a3 = sh_h2f(yr[i + 3]) * inv;
            yr[i] = (uint16_t)sh_f2h(a0); yr[i + 1] = (uint16_t)sh_f2h(a1); yr[i + 2] = (uint16_t)sh_f2h(a2); yr[i + 3] = (uint16_t)sh_f2h(a3);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h(sh_h2f(yr[i]) * inv);
    }
}
static inline float sh_gelu1(float x) { float t = sh_tanhf(0.7978845608f * (x + 0.044715f * x * x * x)); return 0.5f * x * (1.f + t); }
static void sh_k_gelu(const sh_blk *k) {
    uint32_t lo, hi; sh_share(k->nr * k->cols, SH_UNROLL, &lo, &hi);
    const uint16_t *x = k->x; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        float a0 = sh_gelu1(sh_h2f(x[i])), a1 = sh_gelu1(sh_h2f(x[i + 1])), a2 = sh_gelu1(sh_h2f(x[i + 2])), a3 = sh_gelu1(sh_h2f(x[i + 3]));
        y[i] = (uint16_t)sh_f2h(a0); y[i + 1] = (uint16_t)sh_f2h(a1); y[i + 2] = (uint16_t)sh_f2h(a2); y[i + 3] = (uint16_t)sh_f2h(a3);
    }
    for (; i < hi; ++i) y[i] = (uint16_t)sh_f2h(sh_gelu1(sh_h2f(x[i])));
}
static void sh_k_add(const sh_blk *k) {          /* y = x + b, b per row */
    uint32_t lo, hi; sh_share(k->nr * k->cols, SH_UNROLL, &lo, &hi);
    const uint16_t *x = k->x, *b = k->b; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        float a0 = sh_h2f(x[i]) + sh_h2f(b[i]), a1 = sh_h2f(x[i + 1]) + sh_h2f(b[i + 1]), a2 = sh_h2f(x[i + 2]) + sh_h2f(b[i + 2]), a3 = sh_h2f(x[i + 3]) + sh_h2f(b[i + 3]);
        y[i] = (uint16_t)sh_f2h(a0); y[i + 1] = (uint16_t)sh_f2h(a1); y[i + 2] = (uint16_t)sh_f2h(a2); y[i + 3] = (uint16_t)sh_f2h(a3);
    }
    for (; i < hi; ++i) y[i] = (uint16_t)sh_f2h(sh_h2f(x[i]) + sh_h2f(b[i]));
}
static void sh_k_add_bias(const sh_blk *k) {     /* y = x + p0 (one row broadcast) */
    const uint32_t cols = k->cols; const uint16_t *bias = k->p0;
    uint32_t lo, hi; sh_share(k->nr, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const uint16_t *xr = k->x + r * cols; uint16_t *yr = k->y + r * cols; uint32_t i = 0;
        for (; i + 4 <= cols; i += 4) {
            float a0 = sh_h2f(xr[i]) + sh_h2f(bias[i]), a1 = sh_h2f(xr[i + 1]) + sh_h2f(bias[i + 1]), a2 = sh_h2f(xr[i + 2]) + sh_h2f(bias[i + 2]), a3 = sh_h2f(xr[i + 3]) + sh_h2f(bias[i + 3]);
            yr[i] = (uint16_t)sh_f2h(a0); yr[i + 1] = (uint16_t)sh_f2h(a1); yr[i + 2] = (uint16_t)sh_f2h(a2); yr[i + 3] = (uint16_t)sh_f2h(a3);
        }
        for (; i < cols; ++i) yr[i] = (uint16_t)sh_f2h(sh_h2f(xr[i]) + sh_h2f(bias[i]));
    }
}
static void sh_k_scale(const sh_blk *k) {
    const float s = *(const float *)k->arg;
    uint32_t lo, hi; sh_share(k->nr * k->cols, SH_UNROLL, &lo, &hi);
    const uint16_t *x = k->x; uint16_t *y = k->y; uint32_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        float a0 = sh_h2f(x[i]) * s, a1 = sh_h2f(x[i + 1]) * s, a2 = sh_h2f(x[i + 2]) * s, a3 = sh_h2f(x[i + 3]) * s;
        y[i] = (uint16_t)sh_f2h(a0); y[i + 1] = (uint16_t)sh_f2h(a1); y[i + 2] = (uint16_t)sh_f2h(a2); y[i + 3] = (uint16_t)sh_f2h(a3);
    }
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
 * loads/stores so each core moves 4 elements per 2 loads + 2 stores). */
void sh_transpose(uint64_t dst, uint64_t src, uint32_t rows, uint32_t cols, uint32_t ld_src, uint32_t ld_dst, uint32_t cluster) {
    const uint32_t B = 128, bb = B * B * 2;            /* 4 buffers: src/dst x 2 sets = 128 KB */
    const int dm = flex_is_dm_core();
    const uint32_t nbr = (rows + B - 1) / B, nbc = (cols + B - 1) / B, nblk = nbr * nbc;
    uint32_t next = 0; while (next < nblk && !sh_my_block(next, cluster)) ++next;
    uint32_t set = 0;
    #define SH_T_S(s) (SH_ROWOPS_L1_BASE + (s) * 2 * bb)
    #define SH_T_D(s) (SH_ROWOPS_L1_BASE + (s) * 2 * bb + bb)
    if (dm && next < nblk) {
        const uint32_t r0 = (next / nbc) * B, c0 = (next % nbc) * B;
        const uint32_t nr = (rows - r0) < B ? rows - r0 : B, nc = (cols - c0) < B ? cols - c0 : B;
        bare_dma_start_2d(local(SH_T_S(0)), src + ((uint64_t)r0 * ld_src + c0) * 2, nc * 2, B * 2, ld_src * 2, nr);
    }
    while (next < nblk) {
        const uint32_t blk = next, r0 = (blk / nbc) * B, c0 = (blk % nbc) * B;
        const uint32_t nr = (rows - r0) < B ? rows - r0 : B, nc = (cols - c0) < B ? cols - c0 : B;
        ++next; while (next < nblk && !sh_my_block(next, cluster)) ++next;
        if (dm) bare_dma_wait_all();
        flex_intra_cluster_sync();
        if (dm && next < nblk) {
            const uint32_t n_r0 = (next / nbc) * B, n_c0 = (next % nbc) * B;
            const uint32_t n_nr = (rows - n_r0) < B ? rows - n_r0 : B, n_nc = (cols - n_c0) < B ? cols - n_c0 : B;
            bare_dma_start_2d(local(SH_T_S(set ^ 1)), src + ((uint64_t)n_r0 * ld_src + n_c0) * 2, n_nc * 2, B * 2, ld_src * 2, n_nr);
        }
        {   /* each core: its share of the block's row pairs */
            const uint16_t *s = (const uint16_t *)local(SH_T_S(set)); uint16_t *d = (uint16_t *)local(SH_T_D(set));
            uint32_t lo, hi; sh_share(nr, 2, &lo, &hi);
            uint32_t r = lo;
            for (; r + 2 <= hi; r += 2) {
                const uint32_t *s0 = (const uint32_t *)(s + r * B), *s1 = (const uint32_t *)(s + (r + 1) * B);
                uint32_t c = 0;
                for (; c + 2 <= nc; c += 2) {
                    const uint32_t w0 = s0[c >> 1], w1 = s1[c >> 1];        /* w0 = {x[r][c+1], x[r][c]} */
                    *(uint32_t *)(d + c * B + r) = (w0 & 0xFFFFu) | (w1 << 16);
                    *(uint32_t *)(d + (c + 1) * B + r) = (w0 >> 16) | (w1 & 0xFFFF0000u);
                }
                for (; c < nc; ++c) { d[c * B + r] = s[r * B + c]; d[c * B + r + 1] = s[(r + 1) * B + c]; }
            }
            for (; r < hi; ++r) for (uint32_t c = 0; c < nc; ++c) d[c * B + r] = s[r * B + c];
        }
        flex_intra_cluster_sync();
        if (dm) for (uint32_t c = 0; c < nc; ++c) bare_dma_start_1d(dst + ((uint64_t)(c0 + c) * ld_dst + r0) * 2, local(SH_T_D(set) + c * B * 2), nr * 2);
        set ^= 1;
    }
    if (dm) bare_dma_wait_all();
    #undef SH_T_S
    #undef SH_T_D
    sh_end_op(cluster);
}
