/* Row-wise / elementwise fp16 ops on HBM tensors, computed in fp32 on the first core after staging
 * row blocks into TCDM with the DM core. `cluster` selects the executing cluster, or SH_ALL to split
 * the rows round-robin over all clusters (ends with a global barrier). Scalar fp16 glue uses software
 * conversion (see sh_l1_add_fp16); Spatz / fixed flh are later optimisations. */
#define SH_ROWOPS_L1_BYTES 0x40000u     /* staging area at TCDM offset 0 (ops run sequentially) */

static inline int sh_my_block(uint32_t blk, uint32_t cluster) {
    if (cluster == SH_ALL) return (blk % (ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y)) == flex_get_cluster_id();
    return flex_get_cluster_id() == cluster;
}
static inline void sh_end_op(uint32_t cluster) {
    flex_intra_cluster_sync();
    if (cluster == SH_ALL) flex_global_barrier_xy();
}
static inline float sh_tanhf(float x) {           /* no libm: rational approx good to ~1e-4 */
    if (x > 9.f) return 1.f; if (x < -9.f) return -1.f;
    float x2 = x * x;
    float a = x * (135135.f + x2 * (17325.f + x2 * (378.f + x2)));
    float b = 135135.f + x2 * (62370.f + x2 * (3150.f + x2 * 28.f));
    float t = a / b; return t > 1.f ? 1.f : (t < -1.f ? -1.f : t);
}
static inline float sh_expf(float x) {            /* exp via 2^k * poly, |err| ~1e-6 rel */
    if (x < -87.f) return 0.f; if (x > 88.f) x = 88.f;
    float k = (float)(int)(x * 1.44269504f + (x >= 0 ? 0.5f : -0.5f));
    float r = x - k * 0.69314718f;
    float p = 1.f + r * (1.f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
    union { float f; uint32_t u; } u; u.u = (uint32_t)((int)k + 127) << 23;
    return p * u.f;
}
static inline float sh_rsqrtf(float x) { union { float f; uint32_t u; } u; u.f = x; u.u = 0x5f3759df - (u.u >> 1); float y = u.f; y = y * (1.5f - 0.5f * x * y * y); y = y * (1.5f - 0.5f * x * y * y); return y; }

/* Generic driver: stage `rpb` rows of x (and optionally a second input b) into L1, let the first core
 * apply `fn` on the block, write the block back to y. All in fp16 bits in L1. */
typedef void (*sh_rowfn_t)(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nrows, uint32_t cols, const void *arg);
#ifdef SH_DEBUG_SOFTMAX
static void sh_k_softmax(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a);
#endif

static void sh_rowop(uint64_t y, uint64_t x, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ldy, uint32_t ldx, uint32_t ldb,
                     uint32_t b_rows, sh_rowfn_t fn, const void *arg, uint32_t cluster) {
    const uint32_t rowb = cols * 2;
    const uint32_t nin = b ? 2 : 1;
    uint32_t rpb = SH_ROWOPS_L1_BYTES / (rowb * (nin + 1)); if (rpb == 0) rpb = 1; if (rpb > rows) rpb = rows;
    if (cluster == SH_ALL) {   /* enough blocks to keep every cluster busy */
        const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, want = (rows + P - 1) / P;
        if (rpb > want) rpb = want ? want : 1;
    }
    const uint32_t xo = 0, bo = rpb * rowb, yo = bo + (b ? rpb * rowb : 0);
    const int dm = flex_is_dm_core(), first = flex_is_first_core();
    for (uint32_t r0 = 0, blk = 0; r0 < rows; r0 += rpb, ++blk) {
        if (!sh_my_block(blk, cluster)) continue;
        const uint32_t nr = (rows - r0) < rpb ? (rows - r0) : rpb;
        if (dm) {
            bare_dma_start_2d(local(xo), x + (uint64_t)r0 * ldx * 2, rowb, rowb, ldx * 2, nr);
            if (b) {
                if (b_rows == 1) bare_dma_start_1d(local(bo), b, rowb);                       /* broadcast row (bias) */
                else bare_dma_start_2d(local(bo), b + (uint64_t)r0 * ldb * 2, rowb, rowb, ldb * 2, nr);
            }
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
        if (first) fn((uint16_t *)local(yo), (const uint16_t *)local(xo), b ? (const uint16_t *)local(bo) : 0, nr, cols, arg);
#ifdef SH_DEBUG_SOFTMAX        /* bisect aid: verify the softmax block in L1 right after it was computed (rows sum to 1, no negatives) */
        if (first && fn == sh_k_softmax) {
            uint32_t nbad = 0;
            for (uint32_t r = 0; r < nr; ++r) {
                const uint16_t *yr = (const uint16_t *)local(yo + r * rowb);
                float sum = 0.f; uint32_t neg = 0;
                for (uint32_t i = 0; i < cols; ++i) { sum += sh_fp16_to_f32(yr[i]); neg += (yr[i] >> 15) & (yr[i] != 0x8000u); }
                union { float f; uint32_t u; } su; su.f = sum;
                if (sum < 0.9f || sum > 1.1f || neg) { if (nbad < 4) sh_printf("[sh_softmax] cl %u blk %u row %u sum %08x neg %u\n", flex_get_cluster_id(), blk, r0 + r, su.u, neg); nbad++; }
            }
            if (nbad) sh_printf("[sh_softmax] cl %u blk %u: %u bad rows of %u\n", flex_get_cluster_id(), blk, nbad, nr);
        }
#endif
        flex_intra_cluster_sync();
        if (dm) { for (uint32_t r = 0; r < nr; ++r) bare_dma_start_1d(y + (uint64_t)(r0 + r) * ldy * 2, local(yo + r * rowb), rowb); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
    }
    sh_end_op(cluster);
}

/* ---- kernels on one L1 block -------------------------------------------------------------- */
typedef struct { float eps; const uint16_t *gamma; const uint16_t *beta; } sh_ln_arg;
static void sh_k_layernorm(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a) {
    const sh_ln_arg *p = (const sh_ln_arg *)a; (void)bb;
    for (uint32_t r = 0; r < nr; ++r) {
        const uint16_t *xr = xb + r * cols; uint16_t *yr = yb + r * cols;
        float mean = 0.f; for (uint32_t i = 0; i < cols; ++i) mean += sh_fp16_to_f32(xr[i]); mean /= (float)cols;
        float var = 0.f; for (uint32_t i = 0; i < cols; ++i) { float d = sh_fp16_to_f32(xr[i]) - mean; var += d * d; } var /= (float)cols;
        float rs = sh_rsqrtf(var + p->eps);
        for (uint32_t i = 0; i < cols; ++i)
            yr[i] = sh_f32_to_fp16((sh_fp16_to_f32(xr[i]) - mean) * rs * sh_fp16_to_f32(p->gamma[i]) + sh_fp16_to_f32(p->beta[i]));
    }
}
static void sh_k_softmax(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a) {
    const float scale = *(const float *)a; (void)bb;
    for (uint32_t r = 0; r < nr; ++r) {
        const uint16_t *xr = xb + r * cols; uint16_t *yr = yb + r * cols;
        float m = -3.0e38f; for (uint32_t i = 0; i < cols; ++i) { float v = sh_fp16_to_f32(xr[i]) * scale; if (v > m) m = v; }
        float s = 0.f;
        for (uint32_t i = 0; i < cols; ++i) { float e = sh_expf(sh_fp16_to_f32(xr[i]) * scale - m); yr[i] = sh_f32_to_fp16(e); s += e; }
        float inv = 1.f / s;
        for (uint32_t i = 0; i < cols; ++i) yr[i] = sh_f32_to_fp16(sh_fp16_to_f32(yr[i]) * inv);
    }
}
static void sh_k_gelu(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a) {
    (void)bb; (void)a;
    for (uint32_t i = 0; i < nr * cols; ++i) {
        float x = sh_fp16_to_f32(xb[i]);
        float t = sh_tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
        yb[i] = sh_f32_to_fp16(0.5f * x * (1.f + t));
    }
}
static void sh_k_add(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a) {
    const int bias = *(const int *)a;   /* 1: b is a single row broadcast over rows */
    for (uint32_t r = 0; r < nr; ++r) {
        const uint16_t *br = bias ? bb : bb + r * cols;
        for (uint32_t i = 0; i < cols; ++i) yb[r * cols + i] = sh_f32_to_fp16(sh_fp16_to_f32(xb[r * cols + i]) + sh_fp16_to_f32(br[i]));
    }
}
static void sh_k_scale(uint16_t *yb, const uint16_t *xb, const uint16_t *bb, uint32_t nr, uint32_t cols, const void *a) {
    const float s = *(const float *)a; (void)bb;
    for (uint32_t i = 0; i < nr * cols; ++i) yb[i] = sh_f32_to_fp16(sh_fp16_to_f32(xb[i]) * s);
}

/* ---- public ops ---------------------------------------------------------------------------- */
void sh_layernorm(uint64_t y, uint64_t x, uint64_t gamma, uint64_t beta, uint32_t rows, uint32_t cols, uint32_t ld, float eps, uint32_t cluster) {
    /* gamma/beta are read directly from HBM by the first core (cols elements, read once per block row) */
    sh_ln_arg a = { eps, (const uint16_t *)(uintptr_t)gamma, (const uint16_t *)(uintptr_t)beta };
    sh_rowop(y, x, 0, rows, cols, ld, ld, 0, 0, sh_k_layernorm, &a, cluster);
}
void sh_softmax_rows(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float scale, uint32_t cluster) {
    sh_rowop(y, x, 0, rows, cols, ld, ld, 0, 0, sh_k_softmax, &scale, cluster);
}
void sh_gelu(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    sh_rowop(y, x, 0, rows, cols, ld, ld, 0, 0, sh_k_gelu, 0, cluster);
}
void sh_add(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    int bias = 0; sh_rowop(y, a, b, rows, cols, ld, ld, ld, rows, sh_k_add, &bias, cluster);
}
void sh_add_bias(uint64_t y, uint64_t x, uint64_t bias_row, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster) {
    int bias = 1; sh_rowop(y, x, bias_row, rows, cols, ld, ld, cols, 1, sh_k_add, &bias, cluster);
}
void sh_scale(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float s, uint32_t cluster) {
    sh_rowop(y, x, 0, rows, cols, ld, ld, 0, 0, sh_k_scale, &s, cluster);
}

/* dst[cols, rows] = src[rows, cols]^T, in 64x64 blocks through TCDM (block round-robin over clusters). */
void sh_transpose(uint64_t dst, uint64_t src, uint32_t rows, uint32_t cols, uint32_t ld_src, uint32_t ld_dst, uint32_t cluster) {
    const uint32_t B = 64, so = 0, dofs = B * B * 2;
    const int dm = flex_is_dm_core(), first = flex_is_first_core();
    uint32_t blk = 0;
    for (uint32_t r0 = 0; r0 < rows; r0 += B)
    for (uint32_t c0 = 0; c0 < cols; c0 += B, ++blk) {
        if (!sh_my_block(blk, cluster)) continue;
        const uint32_t nr = (rows - r0) < B ? rows - r0 : B, nc = (cols - c0) < B ? cols - c0 : B;
        if (dm) { bare_dma_start_2d(local(so), src + ((uint64_t)r0 * ld_src + c0) * 2, nc * 2, B * 2, ld_src * 2, nr); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
        if (first) {
            const uint16_t *s = (const uint16_t *)local(so); uint16_t *d = (uint16_t *)local(dofs);
            for (uint32_t r = 0; r < nr; ++r) for (uint32_t c = 0; c < nc; ++c) d[c * B + r] = s[r * B + c];
        }
        flex_intra_cluster_sync();
        if (dm) { for (uint32_t c = 0; c < nc; ++c) bare_dma_start_1d(dst + ((uint64_t)(c0 + c) * ld_dst + r0) * 2, local(dofs + c * B * 2), nr * 2); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
    }
    sh_end_op(cluster);
}
