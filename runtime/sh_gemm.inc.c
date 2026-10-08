/* sh_gemm: single-cluster tiled GEMM on RedMulE.
 *
 * RedMulE convention (LightRedmule / flex_redmule_config(m, n, k)): X is m x n, W is n x k,
 * Y is m x k, all row-major and packed in TCDM; the contraction is `n`. Y accumulates in
 * place. So for our Z[M,N] = X[M,K] W[K,N] with tile (tm, tn, tk):
 *      flex_redmule_config(tm, tk, tn)
 *
 * TCDM scratch layout (from cfg->l1_base):  X0 | X1 | W0 | W1 | Y
 * (X1/W1 only when pipeline=1). Loads are 2-D iDMA from HBM (row-major sub-block);
 * the Y store is a per-row 1-D loop because the iDMA model does not do 2-D into HBM.
 */
#define SH_T_DEFAULT 256

static const redmule_compute_format_t sh_redmule_fmt[4] = {
    REDMULE_FP_16, REDMULE_FP_8, REDMULE_INT_16, REDMULE_INT_8 };

static inline void sh_cfg_fill(sh_gemm_cfg *c, const sh_gemm_cfg *in) {
    if (in) *c = *in; else { c->tm = c->tn = c->tk = 0; c->pipeline = 1; c->accumulate = 0; c->fmt = 0; c->l1_base = 0; }
    if (!c->tm) c->tm = SH_T_DEFAULT;
    if (!c->tn) c->tn = SH_T_DEFAULT;
    if (!c->tk) c->tk = SH_T_DEFAULT;
}

uint32_t sh_gemm_l1_bytes(uint32_t M, uint32_t N, uint32_t K, const sh_gemm_cfg *cfg) {
    (void)M; (void)N; (void)K;
    sh_gemm_cfg c; sh_cfg_fill(&c, cfg);
    uint32_t xb = c.tm * c.tk * 2, wb = c.tk * c.tn * 2, yb = c.tm * c.tn * 2;
    uint32_t nbuf = c.pipeline ? 2 : 1;
    return c.l1_base + nbuf * (xb + wb) + yb;
}

/* Zero `bytes` of TCDM at `off` using the read-as-zero region (ZOMEM is ARCH_CLUSTER_ZOMEM_SIZE bytes). */
static inline void sh_l1_zero_dm(uint32_t off, uint32_t bytes) {
    while (bytes) {
        uint32_t chunk = bytes < ARCH_CLUSTER_ZOMEM_SIZE ? bytes : ARCH_CLUSTER_ZOMEM_SIZE;
        bare_dma_start_1d(local(off), zomem(0), chunk);
        off += chunk; bytes -= chunk;
    }
    bare_dma_wait_all();
}

/* Issue (no wait) a 2-D load of a rows x cols fp16 sub-block (HBM, leading dim ld) into packed TCDM. */
static inline void sh_load_block_async(uint32_t l1_off, uint64_t hbm, uint32_t rows, uint32_t cols, uint32_t ld) {
    bare_dma_start_2d(local(l1_off), hbm, cols * 2, cols * 2, ld * 2, rows);
}
/* Store packed TCDM block to an HBM sub-block, row by row (sync). */
static inline void sh_store_block_sync(uint64_t hbm, uint32_t l1_off, uint32_t rows, uint32_t cols, uint32_t ld) {
    for (uint32_t r = 0; r < rows; ++r)
        bare_dma_start_1d(hbm + (uint64_t)r * ld * 2, local(l1_off + r * cols * 2), cols * 2);
    bare_dma_wait_all();
}

int sh_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
            uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg, uint32_t cluster) {
    if (flex_get_cluster_id() != cluster) return 0;
    sh_gemm_cfg c; sh_cfg_fill(&c, cfg);
    const int first = flex_is_first_core(), dm = flex_is_dm_core();

    if (M % c.tm || N % c.tn || K % c.tk) {
        if (first) sh_printf("[sh_gemm] shape %ux%ux%u not divisible by tile %ux%ux%u\n", M, N, K, c.tm, c.tn, c.tk);
        return -1;
    }
    const uint32_t need = sh_gemm_l1_bytes(M, N, K, &c);
    if (need > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_gemm] L1 need %u > %u\n", need, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -2;
    }
    const uint32_t xb = c.tm * c.tk * 2, wb = c.tk * c.tn * 2, yb = c.tm * c.tn * 2;
    const uint32_t x0 = c.l1_base, x1 = c.pipeline ? x0 + xb : x0;
    const uint32_t w0 = x1 + xb,  w1 = c.pipeline ? w0 + wb : w0;
    const uint32_t y  = w1 + wb;
    const uint32_t MT = M / c.tm, NT = N / c.tn, KT = K / c.tk;
    const redmule_compute_format_t fmt = sh_redmule_fmt[c.fmt & 3];

    if (first) flex_redmule_config(c.tm, c.tk, c.tn);

    for (uint32_t r = 0; r < MT; ++r)
    for (uint32_t col = 0; col < NT; ++col) {
        const uint64_t zt = z + ((uint64_t)r * c.tm * ldz + col * c.tn) * 2;
        if (dm) {
            if (c.accumulate) { sh_load_block_async(y, zt, c.tm, c.tn, ldz); bare_dma_wait_all(); }
            else sh_l1_zero_dm(y, yb);
            /* prologue: K-tile 0 into buffer 0 */
            sh_load_block_async(x0, x + ((uint64_t)r * c.tm * ldx) * 2, c.tm, c.tk, ldx);
            sh_load_block_async(w0, w + ((uint64_t)col * c.tn) * 2, c.tk, c.tn, ldw);
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
        for (uint32_t kk = 0; kk < KT; ++kk) {
            const uint32_t xcur = (kk & 1) ? x1 : x0, wcur = (kk & 1) ? w1 : w0;
            const uint32_t xnxt = (kk & 1) ? x0 : x1, wnxt = (kk & 1) ? w0 : w1;
            if (c.pipeline) {
                if (dm && kk + 1 < KT) {   /* prefetch K-tile kk+1 while RedMulE runs kk */
                    sh_load_block_async(xnxt, x + ((uint64_t)r * c.tm * ldx + (kk + 1) * c.tk) * 2, c.tm, c.tk, ldx);
                    sh_load_block_async(wnxt, w + ((uint64_t)(kk + 1) * c.tk * ldw + col * c.tn) * 2, c.tk, c.tn, ldw);
                }
                if (first) { flex_redmule_trigger(xcur, wcur, y, fmt); flex_redmule_wait(); }
                if (dm && kk + 1 < KT) bare_dma_wait_all();
            } else {
                if (first) { flex_redmule_trigger(x0, w0, y, fmt); flex_redmule_wait(); }
                flex_intra_cluster_sync();
                if (dm && kk + 1 < KT) {
                    sh_load_block_async(x0, x + ((uint64_t)r * c.tm * ldx + (kk + 1) * c.tk) * 2, c.tm, c.tk, ldx);
                    sh_load_block_async(w0, w + ((uint64_t)(kk + 1) * c.tk * ldw + col * c.tn) * 2, c.tk, c.tn, ldw);
                    bare_dma_wait_all();
                }
            }
            flex_intra_cluster_sync();
        }
        if (dm) sh_store_block_sync(zt, y, c.tm, c.tn, ldz);
        flex_intra_cluster_sync();
    }
    return 0;
}
