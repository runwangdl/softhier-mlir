/* gvsoc test: multi-head attention o = softmax(q k^T / sqrt(dh)) v on LCG data, sampled outputs compared on
 * the host against numpy (tests/gvsoc/run.py attention). shape.h: SEQ D_MODEL N_HEADS CLUSTER NSAMPLES COMPOSED
 *   COMPOSED=0  fused sh_attention (one head per cluster, all in TCDM)
 *   COMPOSED=1  the library-call path: sh_transpose(K) + per head sh_gemm / sh_softmax_rows / sh_gemm via HBM
 * The ROI covers only the attention (same inputs, same output buffer, same sampling). */
#include "sh_ops.h"
#include "shape.h"

#define DH (D_MODEL / N_HEADS)

int main(void) {
    sh_init();
    const uint32_t S = SEQ, D = D_MODEL, H = N_HEADS;
    const uint32_t mb = (S * D * 2 + 4095) & ~4095u;
    const uint64_t q = sh_hbm_addr(0), k = sh_hbm_addr(mb), v = sh_hbm_addr(2 * mb), o = sh_hbm_addr(3 * mb);
    const uint64_t kT = sh_hbm_addr(4 * mb), sc = sh_hbm_addr(5 * mb);   /* composed path only: K^T, H score matrices */
    const float scale = 1.0f / 8.0f;   /* 1/sqrt(64); run.py uses the same constant */
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    if (lead) {
        sh_test_fill_fp16(q, S, D, D, 21, -8, 8, 0.125f);
        sh_test_fill_fp16(k, S, D, D, 22, -8, 8, 0.125f);
        sh_test_fill_fp16(v, S, D, D, 23, -16, 16, 0.125f);
        sh_test_fill_fp16(o, S, D, D, 24, 7, 7, 1.0f);   /* poison */
        sh_printf("[attention] S=%u D=%u H=%u dh=%u cluster=%s composed=%d l1=%u B\n", S, D, H, DH,
                  CLUSTER == SH_ALL ? "all" : "0", COMPOSED, sh_attention_l1_bytes(S, DH));
    }
    sh_barrier_global();
    if (lead) sh_timer_start();
    int rc = 0;
#if COMPOSED
    sh_gemm_cfg qk = { .tm = 256, .tn = 256, .tk = DH,  .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    sh_gemm_cfg pv = { .tm = 256, .tn = DH,  .tk = 256, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    sh_transpose(kT, k, S, D, D, S, CLUSTER);
    const uint32_t P = sh_num_clusters();
    for (uint32_t hd = 0; hd < H; ++hd) {
        const uint32_t cl = (CLUSTER == SH_ALL) ? hd % P : CLUSTER;
        const uint64_t s_h = sc + (uint64_t)hd * S * S * 2;
        rc |= sh_gemm(q + hd * DH * 2, kT + (uint64_t)hd * DH * S * 2, s_h, S, S, DH, D, S, S, &qk, cl);
        sh_softmax_rows(s_h, s_h, S, S, S, scale, cl);
        rc |= sh_gemm(s_h, v + hd * DH * 2, o + hd * DH * 2, S, DH, S, S, D, D, &pv, cl);
    }
    sh_barrier_global();
#else
    rc = sh_attention(q, k, v, o, S, D, H, D, D, D, D, scale, CLUSTER);
    if (CLUSTER != SH_ALL) sh_barrier_global();
#endif
    if (lead) {
        sh_timer_end();
        if (rc) sh_printf("[attention] rc=%d ATTENTION_FAIL\n", rc);
#if !COMPOSED
        if (CLUSTER != SH_ALL || H <= sh_num_clusters()) {   /* cluster 0's last head: per-phase cycles */
            static const char *ph[6] = { "stage+kT", "-", "scores", "softmax", "pv", "norm+store" };
            uint32_t prev = sh_attention_profile(S, DH, 0);
            for (uint32_t i = 1; i <= 6; ++i) {
                uint32_t c = sh_attention_profile(S, DH, i);
                if (i != 2) sh_printf("[attention] phase %-10s %u cycles\n", ph[i - 1], c - prev);
                prev = c;
            }
            sh_printf("[attention] total %u cycles (cluster 0, last head)\n", sh_attention_profile(S, DH, 6) - sh_attention_profile(S, DH, 0));
        }
#endif
        sh_test_dump_samples(o, S, DH, D, 301, NSAMPLES, "O0");
        sh_test_dump_samples(o, S, D, D, 302, NSAMPLES, "O");
        sh_printf("ATTENTION_DONE\n");
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
