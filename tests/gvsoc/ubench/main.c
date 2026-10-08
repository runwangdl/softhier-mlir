/* Micro-benchmarks for the DSE cost model: time one library primitive per case with the
 * core-local cycle counter, many cases per simulation. cases.h (written by
 * softhier_mlir/dse/calibrate.py or by hand) defines UB_CASES as a list of {kind, {p0..p7}}:
 *
 *   UB_BARRIER                       global barrier only (the overhead every case includes)
 *   UB_REDMULE tm tn tk              one RedMulE trigger on TCDM-resident tiles (incl. config)
 *   UB_LOAD2D  rows cols ld nact     2-D HBM->TCDM load; clusters 0..nact-1 load concurrently
 *   UB_STORE   rows cols ld nact     per-row TCDM->HBM store; clusters 0..nact-1 concurrently
 *   UB_LN/SM/GELU/ADD/BIAS/SCALE/T rows cols mode   row op; mode 0 = cluster 0, 1 = SH_ALL
 *   UB_GEMM    M N K tm tn tk pipe mode             sh_gemm; mode 0 = cluster 0, 1 = SH_ALL
 *   UB_SUMMA   M N K T tk pipe                      sh_gemm_mesh
 *   UB_ZERO    bytes                                sh_l1_zero (ZOMEM -> TCDM) on cluster 0
 *
 * Rules that keep the numbers honest (both cost ~5-10k cycles per case when violated):
 *  - everything a case needs (its parameters, HBM addresses) is copied to the stack BEFORE the
 *    timed region: the case table lives in L3 and every access to it is a slow NoC read;
 *  - the row-op input regions are filled with a fixed fp16 pattern at start-up: the software
 *    fp16<->fp32 conversion is data dependent, so uninitialised HBM gives run-to-run noise.
 * GEMM / DMA timing is data independent (the big GEMM inputs are left uninitialised).
 * Output: "[ub] <idx> <kind> <p0..p7> cycles=<n>" per case, plus the ROI over all cases. */
#include "sh_ops.h"
#include "cases.h"

enum { UB_BARRIER, UB_REDMULE, UB_LOAD2D, UB_STORE, UB_LN, UB_SM, UB_GELU, UB_ADD, UB_BIAS, UB_SCALE, UB_T, UB_GEMM, UB_SUMMA, UB_ZERO };
typedef struct { int kind; uint32_t p[8]; } ub_case_t;
static const ub_case_t ub_cases[] = { UB_CASES };
#define NCASES (sizeof(ub_cases) / sizeof(ub_cases[0]))
#define INIT_BYTES 0x00200000u   /* bytes of each row-op input region initialised at start-up (2 MB) */

typedef struct { uint64_t h0, h1, h2, h3; uint32_t cid; } ub_env_t;

static void run_case(const ub_case_t *c, const ub_env_t *e) {
    switch (c->kind) {
    case UB_BARRIER: break;
    case UB_REDMULE: {
        const uint32_t tm = c->p[0], tn = c->p[1], tk = c->p[2];
        const uint32_t x = 0, w = tm * tk * 2, y = w + tk * tn * 2;
        if (e->cid == 0) sh_redmule(x, w, y, tm, tn, tk, SH_FP16);
        break; }
    case UB_LOAD2D:
        if (e->cid < c->p[3]) sh_dma_load_2d(0, e->h0 + (uint64_t)e->cid * c->p[0] * c->p[2] * 2, c->p[0], c->p[1], c->p[2]);
        break;
    case UB_STORE:
        if (e->cid < c->p[3]) sh_dma_store_rows(e->h0 + (uint64_t)e->cid * c->p[0] * c->p[2] * 2, 0, c->p[0], c->p[1], c->p[2]);
        break;
    case UB_LN:    sh_layernorm(e->h2, e->h0, e->h3, e->h3 + 8192, c->p[0], c->p[1], c->p[1], 1e-5f, c->p[2] ? SH_ALL : 0); break;
    case UB_SM:    sh_softmax_rows(e->h2, e->h0, c->p[0], c->p[1], c->p[1], 0.125f, c->p[2] ? SH_ALL : 0); break;
    case UB_GELU:  sh_gelu(e->h2, e->h0, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_ADD:   sh_add(e->h2, e->h0, e->h1, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_BIAS:  sh_add_bias(e->h2, e->h0, e->h3, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_SCALE: sh_scale(e->h2, e->h0, c->p[0], c->p[1], c->p[1], 0.3f, c->p[2] ? SH_ALL : 0); break;
    case UB_T:     sh_transpose(e->h2, e->h0, c->p[0], c->p[1], c->p[1], c->p[0], c->p[2] ? SH_ALL : 0); break;
    case UB_GEMM: {
        sh_gemm_cfg cfg = { .tm = c->p[3], .tn = c->p[4], .tk = c->p[5], .pipeline = c->p[6], .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
        sh_gemm(e->h0, e->h1, e->h2, c->p[0], c->p[1], c->p[2], c->p[2], c->p[1], c->p[1], &cfg, c->p[7] ? SH_ALL : 0);
        break; }
    case UB_SUMMA: {
        sh_gemm_cfg cfg = { .tm = c->p[3], .tn = c->p[3], .tk = c->p[4], .pipeline = c->p[5], .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
        sh_gemm_mesh(e->h0, e->h1, e->h2, c->p[0], c->p[1], c->p[2], c->p[2], c->p[1], c->p[1], &cfg);
        break; }
    case UB_ZERO:   /* p0 bytes of TCDM cleared through the ZOMEM iDMA path (what sh_gemm does per output tile) */
        if (e->cid == 0) sh_l1_zero(0, c->p[0]);
        break;
    }
}

/* Fill [hbm, hbm+bytes) with a 4 KB fp16 pattern (values k/64, k in [-32, 32)) from TCDM, DMA only. */
static void fill_region(uint64_t hbm, uint32_t bytes, uint32_t seed) {
    const uint32_t blk = 4096;
    if (sh_is_first_core()) {
        volatile uint16_t *p = (volatile uint16_t *)sh_l1_addr(0);
        uint32_t s = seed;
        for (uint32_t i = 0; i < blk / 2; ++i) { s = s * 1103515245u + 12345u; p[i] = sh_f32_to_fp16((float)((int)((s >> 16) & 63) - 32) / 64.0f); }
    }
    sh_barrier_cluster();
    for (uint32_t off = 0; off < bytes; off += blk) sh_dma_copy(hbm + off, (uint64_t)sh_l1_addr(0), blk);
}

int main(void) {
    sh_init();
    ub_env_t env = { sh_hbm_addr(0), sh_hbm_addr(0x01000000), sh_hbm_addr(0x02000000), sh_hbm_addr(0x03000000), sh_cluster_id() };
    const int keeper = (env.cid == 0 && sh_is_first_core());
    if (env.cid == 0) { fill_region(env.h0, INIT_BYTES, 1); fill_region(env.h1, INIT_BYTES, 2); fill_region(env.h3, 0x4000, 3); }
    sh_barrier_global();
    if (keeper) sh_timer_start();
    for (uint32_t i = 0; i < NCASES; ++i) {
        const ub_case_t c = ub_cases[i];          /* L3 -> stack, outside the timed region */
        sh_barrier_global();
        const uint32_t t0 = sh_cycles();
        run_case(&c, &env);
        sh_barrier_global();
        const uint32_t t1 = sh_cycles();
        if (keeper) sh_printf("[ub] %u %d %u %u %u %u %u %u %u %u cycles=%u\n", i, c.kind,
                              c.p[0], c.p[1], c.p[2], c.p[3], c.p[4], c.p[5], c.p[6], c.p[7], t1 - t0);
    }
    sh_barrier_global();
    if (keeper) sh_timer_end();
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
