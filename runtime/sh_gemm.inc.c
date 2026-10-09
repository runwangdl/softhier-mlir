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
    const uint32_t cid = flex_get_cluster_id(), P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    if (cluster != SH_ALL && cid != cluster) return 0;
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
        if (cluster == SH_ALL && ((r * NT + col) % P) != cid) continue;   /* output tiles dealt round-robin */
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
    if (cluster == SH_ALL) flex_global_barrier_xy();
    return 0;
}
#ifndef SH_NO_GEMM_MESH   /* programs that never call it can drop it (64 KB instruction memory; see sh_ops_lean) */

/* ---- mesh-wide output-stationary SUMMA ---------------------------------------------------------
 * Cluster (px,py) owns output tile Z[py*T .. , px*T ..] (T = tm = tn). Diagonal clusters load the
 * X row-panel / W column-panel K-tile from HBM and broadcast along their mesh row / column
 * (in-network multicast, line rate, independent of the number of receivers); every cluster
 * accumulates its tile on RedMulE over K. Requires a square mesh P x P, M = N = P*T, K % tk == 0.
 * pipeline=1 double-buffers X/W and prefetches K-tile k+1 under RedMulE.
 * TCDM: X0 [X1] W0 [W1] Y.  Must be called by all cores of all clusters. */
int sh_gemm_mesh(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                 uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg) {
    sh_gemm_cfg c; sh_cfg_fill(&c, cfg);
    const uint32_t P = ARCH_NUM_CLUSTER_X;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t cid = flex_get_cluster_id(), px = cid % P, py = cid / P;
    const uint32_t T = c.tm;
    if (ARCH_NUM_CLUSTER_Y != P || c.tn != T || M != P * T || N != P * T || K % c.tk) {
        if (cid == 0 && first) sh_printf("[sh_gemm_mesh] need square mesh, tm==tn, M==N==P*tm, K%%tk==0 (got %ux%ux%u tile %ux%ux%u mesh %ux%u)\n",
                                         M, N, K, c.tm, c.tn, c.tk, (uint32_t)ARCH_NUM_CLUSTER_X, (uint32_t)ARCH_NUM_CLUSTER_Y);
        return -1;
    }
    if (sh_gemm_l1_bytes(M, N, K, &c) > ARCH_CLUSTER_TCDM_SIZE) {
        if (cid == 0 && first) sh_printf("[sh_gemm_mesh] L1 budget exceeded\n");
        return -2;
    }
    const uint32_t xb = T * c.tk * 2, wb = c.tk * T * 2, yb = T * T * 2;
    const uint32_t x0 = c.l1_base, x1 = c.pipeline ? x0 + xb : x0;
    const uint32_t w0 = x1 + xb,  w1 = c.pipeline ? w0 + wb : w0;
    const uint32_t y  = w1 + wb;
    const uint32_t KT = K / c.tk;
    const redmule_compute_format_t fmt = sh_redmule_fmt[c.fmt & 3];
    GridSyncGroupInfo grp = grid_sync_group_init(P, P);
    /* collective masks are AND-masks on the cluster coordinates: ~(dim-1) = wildcard, dim-1 = exact */
    const uint16_t row_wild = (uint16_t)grp.wakeup_row_mask, col_wild = (uint16_t)grp.wakeup_col_mask;
    const uint16_t row_exact = (uint16_t)(ARCH_NUM_CLUSTER_X - 1), col_exact = (uint16_t)(ARCH_NUM_CLUSTER_Y - 1);
    const uint64_t zt = z + ((uint64_t)py * T * ldz + px * T) * 2;

    /* diagonal cluster: load K-tile kk of its X row-panel + W column-panel, multicast row / column */
    #define SH_DIAG_LOAD(xdst, wdst, kk) do { \
        sh_load_block_async(xdst, x + ((uint64_t)py * T * ldx + (kk) * c.tk) * 2, T, c.tk, ldx); \
        sh_load_block_async(wdst, w + ((uint64_t)(kk) * c.tk * ldw + px * T) * 2, c.tk, T, ldw); \
        bare_dma_wait_all(); \
        /* one collective in flight at a time: two outstanding broadcasts from one DM core crash the \
           gvsoc NoC model (segfault, checked 2026-10-08) */ \
        sh_dma_bcast_1d(xdst, xdst, xb, row_wild, col_exact);  /* along my row    */ \
        flex_dma_async_wait_all(); \
        sh_dma_bcast_1d(wdst, wdst, wb, row_exact, col_wild);  /* along my column */ \
        flex_dma_async_wait_all(); } while (0)

    flex_global_barrier_xy();
    if (first) flex_redmule_config(T, c.tk, T);
    if (dm) { if (c.accumulate) { sh_load_block_async(y, zt, T, T, ldz); bare_dma_wait_all(); } else sh_l1_zero_dm(y, yb); }
    if (dm && px == py) SH_DIAG_LOAD(x0, w0, 0);
    grid_sync_group_barrier_xy(&grp);
    for (uint32_t kk = 0; kk < KT; ++kk) {
        const uint32_t xcur = (kk & 1) ? x1 : x0, wcur = (kk & 1) ? w1 : w0;
        const uint32_t xnxt = (kk & 1) ? x0 : x1, wnxt = (kk & 1) ? w0 : w1;
        if (c.pipeline) {
            if (dm && px == py && kk + 1 < KT) SH_DIAG_LOAD(xnxt, wnxt, kk + 1);
            if (first) { flex_redmule_trigger(xcur, wcur, y, fmt); flex_redmule_wait(); }
        } else {
            if (first) { flex_redmule_trigger(x0, w0, y, fmt); flex_redmule_wait(); }
            grid_sync_group_barrier_xy(&grp);
            if (dm && px == py && kk + 1 < KT) SH_DIAG_LOAD(x0, w0, kk + 1);
        }
        grid_sync_group_barrier_xy(&grp);
    }
    #undef SH_DIAG_LOAD
    if (dm) sh_store_block_sync(zt, y, T, T, ldz);
    flex_global_barrier_xy();
    return 0;
}
#endif /* SH_NO_GEMM_MESH */

#ifndef SH_NO_GEMM_XMCAST
/* ---- small-M GEMM with the activation panel multicast (docs/XPANEL_MCAST.md) --------------------------------------
 * sh_gemm(SH_ALL) deals output tiles round-robin, so every tile re-reads its X row panel from HBM: for M <= a few hundred
 * rows (the expert's 50 N rows, the 241-row prefix) the X traffic is (N / tn) x M K 2 bytes. Here X crosses HBM once:
 *   - cluster c owns the column slice [c Nc, c Nc + nc) of Z (Nc = ceil(N / P) rounded up to the granule g = cfg->tn or
 *     4; clusters past N own nothing but still take part in the barriers) and streams only its own W columns from HBM,
 *     K-panel by K-panel (tk rows), double-buffered (weight-stationary per step);
 *   - the X K-panel (rows x tk) is loaded from HBM by ONE cluster and multicast to all clusters (sh_dma_bcast_1d, full
 *     mesh masks, <= 32 KB per collective, one collective in flight: SIMULATOR_NOTES #4 / #15);
 *   - Y (rows x nc) stays in TCDM over the whole K loop; one RedMulE trigger per K-panel.
 * Two schedules (mode): SH_XM_PANEL streams X panels: panel kk is loaded + multicast by cluster kk % P during the
 * RedMulE of panel kk - 1 (double-buffered X, one global barrier per panel); SH_XM_WHOLE multicasts the whole X row
 * block once (stored panel-major, KT panels of rows x tk) and then runs the K loop with local syncs only.
 * SH_XM_AUTO = WHOLE when its scratch fits, else PANEL. Rows: blocks of tm (cfg->tm, 0 = M); W is re-streamed per row
 * block. tk: cfg->tk, 0 = the largest divisor of K <= 256 whose scratch fits under SH_XM_L1_LIMIT. fp16 only.
 * TCDM (from cfg->l1_base): X0 [X1 | whole X] W0 W1 Y. Call from all cores of all clusters. */
#ifndef SH_XM_L1_LIMIT
#define SH_XM_L1_LIMIT 0x90000u     /* stay below the KV-stationary resident region (sh_flow.inc.c) */
#endif
#define SH_XM_CHUNK 32768u

typedef struct { uint32_t rows, Nc, tk, KT, whole, xb, x0, x1, w0, w1, y, end; } sh_xm_plan;

static uint32_t sh_xm_need(uint32_t rows, uint32_t K, uint32_t Nc, uint32_t tk, uint32_t whole) {
    return (whole ? rows * K * 2 : 2 * rows * tk * 2) + 2 * tk * Nc * 2 + rows * Nc * 2;
}
/* 0 or a negative reason; fills p (cfg fields 0 = auto, see above) */
static int sh_xm_make_plan(sh_xm_plan *p, uint32_t M, uint32_t N, uint32_t K, const sh_gemm_cfg *c, uint32_t mode) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, g = c->tn ? c->tn : 4;
    p->rows = c->tm ? c->tm : M;
    if (M % p->rows) return -1;
    p->Nc = ((N + P - 1) / P + g - 1) / g * g;
    if (c->l1_base >= SH_XM_L1_LIMIT) return -2;
    const uint32_t lim = SH_XM_L1_LIMIT - c->l1_base;
    for (int w = (mode == SH_XM_PANEL ? 0 : 1); w >= 0; --w) {
        if (mode == SH_XM_WHOLE && !w) break;
        uint32_t tk = 0;
        if (c->tk) { if (K % c->tk == 0 && sh_xm_need(p->rows, K, p->Nc, c->tk, w) <= lim) tk = c->tk; }
        else for (uint32_t d = K < 256 ? K : 256; d; --d)
            if (K % d == 0 && sh_xm_need(p->rows, K, p->Nc, d, w) <= lim) { tk = d; break; }
        if (!tk) continue;
        p->tk = tk; p->KT = K / tk; p->whole = (uint32_t)w;
        p->xb = p->rows * tk * 2;
        p->x0 = c->l1_base; p->x1 = w ? p->x0 : p->x0 + p->xb;
        p->w0 = w ? p->x0 + p->rows * K * 2 : p->x1 + p->xb;
        p->w1 = p->w0 + tk * p->Nc * 2;
        p->y  = p->w1 + tk * p->Nc * 2;
        p->end = p->y + p->rows * p->Nc * 2;
        return 0;
    }
    return -2;
}
uint32_t sh_gemm_xmcast_l1_bytes(uint32_t M, uint32_t N, uint32_t K, const sh_gemm_cfg *cfg, uint32_t mode) {
    sh_gemm_cfg c;
    if (cfg) c = *cfg; else { c.tm = c.tn = c.tk = 0; c.pipeline = 1; c.accumulate = 0; c.fmt = 0; c.l1_base = 0; }
    sh_xm_plan p;
    return sh_xm_make_plan(&p, M, N, K, &c, mode) ? 0xFFFFFFFFu : p.end;
}

static inline void sh_xm_redmule_fp16(uint32_t x, uint32_t w, uint32_t y) {   /* one asm statement: SIMULATOR_NOTES #12 */
    __asm__ volatile ("mv t0, %0\n\tmv t1, %1\n\tmv t2, %2\n\t.word 0x386281aa" :: "r"(x), "r"(w), "r"(y) : "t0", "t1", "t2", "memory");
}
/* multicast `bytes` of my TCDM at off to the same offset of every cluster (me included), one collective in flight */
static inline void sh_xm_mcast(uint32_t off, uint32_t bytes) {
    const uint16_t rx = (uint16_t)~(ARCH_NUM_CLUSTER_X - 1u), ry = (uint16_t)~(ARCH_NUM_CLUSTER_Y - 1u);
    for (uint32_t b = 0; b < bytes; b += SH_XM_CHUNK) {
        sh_dma_bcast_1d(off + b, off + b, bytes - b < SH_XM_CHUNK ? bytes - b : SH_XM_CHUNK, rx, ry);
        flex_dma_async_wait_all();
    }
}

int sh_gemm_xmcast_ex(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                      uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg, uint32_t mode) {
    const uint32_t cid = flex_get_cluster_id(), P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    sh_gemm_cfg c;
    if (cfg) c = *cfg; else { c.tm = c.tn = c.tk = 0; c.pipeline = 1; c.accumulate = 0; c.fmt = 0; c.l1_base = 0; }
    sh_xm_plan p;
    const int rc = (c.fmt & 3) ? -3 : sh_xm_make_plan(&p, M, N, K, &c, mode);
    if (rc) {
        if (cid == 0 && first) sh_printf("[sh_gemm_xmcast] %ux%ux%u tile %u,%u,%u mode %u: no plan (%d)\n", M, N, K, c.tm, c.tn, c.tk, mode, rc);
        return rc;
    }
    const uint32_t c0 = cid * p.Nc, nc = c0 < N ? (N - c0 < p.Nc ? N - c0 : p.Nc) : 0;
    const uint32_t rows = p.rows, tk = p.tk, KT = p.KT;
    const uint64_t wc = w + (uint64_t)c0 * 2;
    #define SH_XM_W(dst, kk) sh_load_block_async(dst, wc + (uint64_t)(kk) * tk * ldw * 2, tk, nc, ldw)

    flex_global_barrier_xy();               /* every cluster's scratch is free (the previous op finished everywhere) */
    for (uint32_t r0 = 0; r0 < M; r0 += rows) {
        const uint64_t xr = x + (uint64_t)r0 * ldx * 2, zr = z + ((uint64_t)r0 * ldz + c0) * 2;
        if (first && nc) flex_redmule_config(rows, tk, nc);
        if (dm && nc) {
            if (c.accumulate) { sh_load_block_async(p.y, zr, rows, nc, ldz); bare_dma_wait_all(); }
            else sh_l1_zero_dm(p.y, rows * nc * 2);
        }
        if (p.whole) {
            /* cluster 0 loads the row block panel-major (panel kk: rows x tk, packed) and multicasts it once; meanwhile
               every cluster's DM has W panel 0 in flight */
            if (dm && nc) SH_XM_W(p.w0, 0);
            if (dm && cid == 0) {
                for (uint32_t kk = 0; kk < KT; ++kk) sh_load_block_async(p.x0 + kk * p.xb, xr + (uint64_t)kk * tk * 2, rows, tk, ldx);
                bare_dma_wait_all();
                sh_xm_mcast(p.x0, KT * p.xb);
            }
            if (dm) bare_dma_wait_all();
            flex_global_barrier_xy();
            for (uint32_t kk = 0; kk < KT; ++kk) {
                const uint32_t wcur = (kk & 1) ? p.w1 : p.w0, wnxt = (kk & 1) ? p.w0 : p.w1;
                if (dm && nc && kk + 1 < KT) SH_XM_W(wnxt, kk + 1);
                if (first && nc) { sh_xm_redmule_fp16(p.x0 + kk * p.xb, wcur, p.y); flex_redmule_wait(); }
                if (dm) bare_dma_wait_all();
                flex_intra_cluster_sync();
            }
        } else {
            /* panel kk is loaded + multicast by cluster kk % P one iteration ahead; one global barrier per panel */
            if (dm && cid == 0) { sh_load_block_async(p.x0, xr, rows, tk, ldx); bare_dma_wait_all(); sh_xm_mcast(p.x0, p.xb); }
            if (dm && nc) SH_XM_W(p.w0, 0);
            if (dm) bare_dma_wait_all();
            flex_global_barrier_xy();
            for (uint32_t kk = 0; kk < KT; ++kk) {
                const uint32_t xcur = (kk & 1) ? p.x1 : p.x0, wcur = (kk & 1) ? p.w1 : p.w0;
                const uint32_t xnxt = (kk & 1) ? p.x0 : p.x1, wnxt = (kk & 1) ? p.w0 : p.w1;
                if (dm && kk + 1 < KT) {
                    const int src = ((kk + 1) % P) == cid;
                    if (src) sh_load_block_async(xnxt, xr + (uint64_t)(kk + 1) * tk * 2, rows, tk, ldx);
                    if (nc) SH_XM_W(wnxt, kk + 1);
                    if (src) { bare_dma_wait_all(); sh_xm_mcast(xnxt, p.xb); }
                }
                if (first && nc) { sh_xm_redmule_fp16(xcur, wcur, p.y); flex_redmule_wait(); }
                if (dm) bare_dma_wait_all();
                flex_global_barrier_xy();   /* X[kk+1] has landed everywhere; nobody reads X[kk] any more */
            }
        }
        if (dm && nc) sh_store_block_sync(zr, p.y, rows, nc, ldz);
        flex_intra_cluster_sync();
        if (r0 + rows < M) flex_global_barrier_xy();   /* the next row block's multicast overwrites X everywhere */
    }
    #undef SH_XM_W
    flex_global_barrier_xy();
    return 0;
}
int sh_gemm_xmcast(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg) {
    return sh_gemm_xmcast_ex(x, w, z, M, N, K, ldx, ldw, ldz, cfg, SH_XM_AUTO);
}
#endif /* SH_NO_GEMM_XMCAST */
