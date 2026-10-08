/* gvsoc test: row-wise ops of the library vs a host numpy reference (sampled dump). Inputs: LCG matrices,
 * preloaded by the host (SH_PRELOAD = the image's sentinel offset) or generated here (--data device). */
#include "sh_ops.h"
#include "shape.h"   /* ROWS COLS CLUSTER NSAMPLES HBM_START [SH_PRELOAD] */

int main(void) {
    sh_init();
    const uint32_t R = ROWS, C = COLS;
    const uint32_t mb = R * C * 2;
    const uint64_t h0 = sh_hbm_addr(HBM_START);   /* layout mirrored in run.py run_rowops */
    const uint64_t x = h0, b = h0 + mb, g = h0 + 2 * mb, be = h0 + 2 * mb + 4096;
    const uint64_t ln = h0 + 3 * mb, sm = h0 + 4 * mb, ge = h0 + 5 * mb, ad = h0 + 6 * mb,
                   bi = h0 + 7 * mb, sc = h0 + 8 * mb, tr = h0 + 9 * mb;
#ifdef SH_PRELOAD
    sh_preload_wait(sh_hbm_addr(SH_PRELOAD));
#else
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        sh_test_fill_fp16(x, R, C, C, 11, -16, 16, 0.125f);
        sh_test_fill_fp16(b, R, C, C, 12, -16, 16, 0.125f);
        sh_test_fill_fp16(g, 1, C, C, 13, 1, 8, 0.25f);
        sh_test_fill_fp16(be, 1, C, C, 14, -4, 4, 0.25f);
    }
    sh_barrier_global();
#endif
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    /* one ROI per op: every sh_timer_end() prints a period and restarts the timer */
    if (lead) sh_timer_start();
    sh_layernorm(ln, x, g, be, R, C, C, 1e-5f, CLUSTER);      if (lead) sh_timer_end();
    sh_softmax_rows(sm, x, R, C, C, 0.5f, CLUSTER);           if (lead) sh_timer_end();
    sh_gelu(ge, x, R, C, C, CLUSTER);                         if (lead) sh_timer_end();
    sh_add(ad, x, b, R, C, C, CLUSTER);                       if (lead) sh_timer_end();
    sh_add_bias(bi, x, be, R, C, C, CLUSTER);                 if (lead) sh_timer_end();
    sh_scale(sc, x, R, C, C, 0.3f, CLUSTER);                  if (lead) sh_timer_end();
    sh_transpose(tr, x, R, C, C, R, CLUSTER);                 if (lead) sh_timer_end();
    sh_barrier_global();
    if (lead) {
        sh_test_dump_samples(ln, R, C, C, 101, NSAMPLES, "LN");
        sh_test_dump_samples(sm, R, C, C, 102, NSAMPLES, "SM");
        sh_test_dump_samples(ge, R, C, C, 103, NSAMPLES, "GELU");
        sh_test_dump_samples(ad, R, C, C, 104, NSAMPLES, "ADD");
        sh_test_dump_samples(bi, R, C, C, 105, NSAMPLES, "BIAS");
        sh_test_dump_samples(sc, R, C, C, 106, NSAMPLES, "SCALE");
        sh_test_dump_samples(tr, C, R, R, 107, NSAMPLES, "T");
        sh_printf("ROWOPS_DONE\n");
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
