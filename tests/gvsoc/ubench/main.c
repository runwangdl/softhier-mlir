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
 *
 * HBM is never initialised: timing is data-independent and the fills would dominate the run.
 * Output: "[ub] <idx> <kind> <p0..p7> cycles=<n>" per case, plus the ROI over all cases. */
#include "sh_ops.h"
#include "cases.h"

enum { UB_BARRIER, UB_REDMULE, UB_LOAD2D, UB_STORE, UB_LN, UB_SM, UB_GELU, UB_ADD, UB_BIAS, UB_SCALE, UB_T, UB_GEMM, UB_SUMMA };
typedef struct { int kind; uint32_t p[8]; } ub_case_t;
static const ub_case_t ub_cases[] = { UB_CASES };
#define NCASES (sizeof(ub_cases) / sizeof(ub_cases[0]))

static void run_case(const ub_case_t *c) {
    const uint32_t cid = sh_cluster_id();
    const uint64_t h0 = sh_hbm_addr(0), h1 = sh_hbm_addr(0x01000000), h2 = sh_hbm_addr(0x02000000), h3 = sh_hbm_addr(0x03000000);
    switch (c->kind) {
    case UB_BARRIER: break;
    case UB_REDMULE: {
        const uint32_t tm = c->p[0], tn = c->p[1], tk = c->p[2];
        const uint32_t x = 0, w = tm * tk * 2, y = w + tk * tn * 2;
        if (cid == 0) sh_redmule(x, w, y, tm, tn, tk, SH_FP16);
        break; }
    case UB_LOAD2D:
        if (cid < c->p[3]) sh_dma_load_2d(0, h0 + (uint64_t)cid * c->p[0] * c->p[2] * 2, c->p[0], c->p[1], c->p[2]);
        break;
    case UB_STORE:
        if (cid < c->p[3]) sh_dma_store_rows(h0 + (uint64_t)cid * c->p[0] * c->p[2] * 2, 0, c->p[0], c->p[1], c->p[2]);
        break;
    case UB_LN:    sh_layernorm(h2, h0, h3, h3 + 8192, c->p[0], c->p[1], c->p[1], 1e-5f, c->p[2] ? SH_ALL : 0); break;
    case UB_SM:    sh_softmax_rows(h2, h0, c->p[0], c->p[1], c->p[1], 0.125f, c->p[2] ? SH_ALL : 0); break;
    case UB_GELU:  sh_gelu(h2, h0, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_ADD:   sh_add(h2, h0, h1, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_BIAS:  sh_add_bias(h2, h0, h3, c->p[0], c->p[1], c->p[1], c->p[2] ? SH_ALL : 0); break;
    case UB_SCALE: sh_scale(h2, h0, c->p[0], c->p[1], c->p[1], 0.3f, c->p[2] ? SH_ALL : 0); break;
    case UB_T:     sh_transpose(h2, h0, c->p[0], c->p[1], c->p[1], c->p[0], c->p[2] ? SH_ALL : 0); break;
    case UB_GEMM: {
        sh_gemm_cfg cfg = { .tm = c->p[3], .tn = c->p[4], .tk = c->p[5], .pipeline = c->p[6], .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
        sh_gemm(h0, h1, h2, c->p[0], c->p[1], c->p[2], c->p[2], c->p[1], c->p[1], &cfg, c->p[7] ? SH_ALL : 0);
        break; }
    case UB_SUMMA: {
        sh_gemm_cfg cfg = { .tm = c->p[3], .tn = c->p[3], .tk = c->p[4], .pipeline = c->p[5], .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
        sh_gemm_mesh(h0, h1, h2, c->p[0], c->p[1], c->p[2], c->p[2], c->p[1], c->p[1], &cfg);
        break; }
    }
}

int main(void) {
    sh_init();
    const int keeper = (sh_cluster_id() == 0 && sh_is_first_core());
    sh_barrier_global();
    if (keeper) sh_timer_start();
    for (uint32_t i = 0; i < NCASES; ++i) {
        const ub_case_t *c = &ub_cases[i];
        sh_barrier_global();
        uint32_t t0 = sh_cycles();
        run_case(c);
        sh_barrier_global();
        uint32_t t1 = sh_cycles();
        if (keeper) sh_printf("[ub] %u %d %u %u %u %u %u %u %u %u cycles=%u\n", i, c->kind,
                              c->p[0], c->p[1], c->p[2], c->p[3], c->p[4], c->p[5], c->p[6], c->p[7], t1 - t0);
    }
    sh_barrier_global();
    if (keeper) sh_timer_end();
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
