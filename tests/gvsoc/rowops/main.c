/* gvsoc test: row-wise ops of the library vs a host numpy reference (sampled dump). */
#include "sh_ops.h"
#include "shape.h"   /* ROWS COLS CLUSTER NSAMPLES */

int main(void) {
    sh_init();
    const uint32_t R = ROWS, C = COLS;
    const uint32_t mb = R * C * 2;
    const uint64_t x = sh_hbm_addr(0), b = sh_hbm_addr(mb), g = sh_hbm_addr(2 * mb), be = sh_hbm_addr(2 * mb + 4096);
    const uint64_t ln = sh_hbm_addr(3 * mb), sm = sh_hbm_addr(4 * mb), ge = sh_hbm_addr(5 * mb), ad = sh_hbm_addr(6 * mb),
                   bi = sh_hbm_addr(7 * mb), sc = sh_hbm_addr(8 * mb), tr = sh_hbm_addr(9 * mb);
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        sh_test_fill_fp16(x, R, C, C, 11, -16, 16, 0.125f);
        sh_test_fill_fp16(b, R, C, C, 12, -16, 16, 0.125f);
        sh_test_fill_fp16(g, 1, C, C, 13, 1, 8, 0.25f);
        sh_test_fill_fp16(be, 1, C, C, 14, -4, 4, 0.25f);
    }
    sh_barrier_global();
    if (sh_cluster_id() == 0 && sh_is_first_core()) sh_timer_start();
    sh_layernorm(ln, x, g, be, R, C, C, 1e-5f, CLUSTER);
    sh_softmax_rows(sm, x, R, C, C, 0.5f, CLUSTER);
    sh_gelu(ge, x, R, C, C, CLUSTER);
    sh_add(ad, x, b, R, C, C, CLUSTER);
    sh_add_bias(bi, x, be, R, C, C, CLUSTER);
    sh_scale(sc, x, R, C, C, 0.3f, CLUSTER);
    sh_transpose(tr, x, R, C, C, R, CLUSTER);
    sh_barrier_global();
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        sh_timer_end();
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
