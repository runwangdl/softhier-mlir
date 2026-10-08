/* gvsoc test: sh_gemm on one cluster, data generated on device, sampled self-check.
 * Shape/tiling come from shape.h (written by tests/gvsoc/run.py). */
#include "sh_ops.h"
#include "shape.h"      /* GEMM_M GEMM_N GEMM_K TILE_M TILE_N TILE_K PIPELINE ACCUMULATE CLUSTER NSAMPLES */

int main(void) {
    sh_init();
    const uint32_t M = GEMM_M, N = GEMM_N, K = GEMM_K;
#ifndef X_OFF   /* HBM byte offsets (node = offset / 64 MB; run.py gemm --offsets places X/W/Z in different nodes) */
#define X_OFF 0x00000000
#define W_OFF 0x01000000
#define Z_OFF 0x02000000
#endif
    const uint64_t x = sh_hbm_addr(X_OFF), w = sh_hbm_addr(W_OFF), z = sh_hbm_addr(Z_OFF);
    uint32_t bad = 0;
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
#ifdef REAL_DATA   /* probabilities-like X, small real W: exercises fp16 rounding paths */
        sh_test_fill_fp16(x, M, K, K, 1, 0, 64, 1.0f / 4096);
        sh_test_fill_fp16(w, K, N, N, 2, -16, 16, 0.125f);
#else
        sh_test_fill_int_fp16(x, M, K, K, 1, -1, 1);
        sh_test_fill_int_fp16(w, K, N, N, 2, -2, 2);
#endif
        sh_test_fill_int_fp16(z, M, N, N, 3, ACCUMULATE ? 3 : 0, ACCUMULATE ? 3 : 0);  /* Z0 = 3.0 or 0 */
    }
    sh_barrier_global();
    sh_gemm_cfg cfg = { .tm = TILE_M, .tn = TILE_N, .tk = TILE_K, .pipeline = PIPELINE,
                        .accumulate = ACCUMULATE, .fmt = SH_FP16, .l1_base = 0 };
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        sh_printf("[gemm] %ux%ux%u tile %ux%ux%u pipeline=%d acc=%d cluster=%s l1=%u B\n", M, N, K, TILE_M ? TILE_M : 256, TILE_N ? TILE_N : 256, TILE_K ? TILE_K : 256,
                  PIPELINE, ACCUMULATE, CLUSTER == SH_ALL ? "all" : "0", sh_gemm_l1_bytes(M, N, K, &cfg));
    }
    /* The printf and the timer register share the slow virtual interconnect: starting the timer
     * right after a printf lands the start stamp late and shortens the ROI (256^3: 9.9 us instead
     * of 14.7 us). Drain with a barrier first. */
    sh_barrier_global();
    if (sh_cluster_id() == 0 && sh_is_first_core()) sh_timer_start();
    int rc = sh_gemm(x, w, z, M, N, K, K, N, N, &cfg, CLUSTER);
    sh_barrier_global();
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        sh_timer_end();
        if (rc) sh_printf("[gemm] rc=%d GEMM_FAIL\n", rc);
        else {
            bad = sh_test_check_gemm(x, w, z, M, N, K, K, N, N, NSAMPLES, 0.5f, ACCUMULATE ? 3.0f : 0.0f, "[gemm]");
            sh_printf("[gemm] %s\n", bad ? "GEMM_FAIL" : "GEMM_PASS");
        }
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
