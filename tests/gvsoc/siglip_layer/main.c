/* One SigLIP (ViT) encoder layer on SoftHier, composed from softhier-ops, random LCG weights,
 * sampled outputs compared on the host against numpy (softhier_mlir/testing/siglip_ref.py).
 *   h   = x + Wo(attn(LN1(x)))      attn: per head softmax(q k^T / sqrt(dh)) v
 *   out = h + W2(gelu(W1(LN2(h))))
 * Weights are [in, out] (X . W convention), biases are rows. shape.h: SEQ D_MODEL D_FF N_HEADS CLUSTER NSAMPLES */
#include "sh_ops.h"
#include "shape.h"

#define DH (D_MODEL / N_HEADS)
/* Bump allocator on the CALLER'S STACK: every core runs main(), and a global in .bss would be
 * incremented 48 times concurrently (each core would see different addresses). */
static uint64_t alloc(uint64_t *next, uint32_t bytes) { uint64_t a = *next; *next += (bytes + 4095) & ~4095u; return a; }
#define ALLOC(bytes) alloc(&next_buf, (bytes))

int main(void) {
    sh_init();
    const uint32_t S = SEQ, D = D_MODEL, F = D_FF, H = N_HEADS;
    uint64_t next_buf = sh_hbm_addr(0);
    /* activations */
    const uint64_t x = ALLOC(S * D * 2), ln1 = ALLOC(S * D * 2), q = ALLOC(S * D * 2), k = ALLOC(S * D * 2), v = ALLOC(S * D * 2);
    const uint64_t kT = ALLOC(D * S * 2), sc = ALLOC(H * S * S * 2), o = ALLOC(S * D * 2), ao = ALLOC(S * D * 2), h = ALLOC(S * D * 2);
    const uint64_t ln2 = ALLOC(S * D * 2), f1 = ALLOC(S * F * 2), g = ALLOC(S * F * 2), f2 = ALLOC(S * D * 2), out = ALLOC(S * D * 2);
    /* parameters */
    const uint64_t wq = ALLOC(D * D * 2), wk = ALLOC(D * D * 2), wv = ALLOC(D * D * 2), wo = ALLOC(D * D * 2);
    const uint64_t w1 = ALLOC(D * F * 2), w2 = ALLOC(F * D * 2);
    const uint64_t bq = ALLOC(D * 2), bk = ALLOC(D * 2), bv = ALLOC(D * 2), bo = ALLOC(D * 2), b1 = ALLOC(F * 2), b2 = ALLOC(D * 2);
    const uint64_t g1 = ALLOC(D * 2), be1 = ALLOC(D * 2), g2 = ALLOC(D * 2), be2 = ALLOC(D * 2);
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    if (lead) {
        sh_test_fill_fp16(x, S, D, D, 1, -16, 16, 0.125f);
        sh_test_fill_fp16(wq, D, D, D, 2, -8, 8, 1.0f / 128); sh_test_fill_fp16(wk, D, D, D, 3, -8, 8, 1.0f / 128);
        sh_test_fill_fp16(wv, D, D, D, 4, -8, 8, 1.0f / 128); sh_test_fill_fp16(wo, D, D, D, 5, -8, 8, 1.0f / 128);
        sh_test_fill_fp16(w1, D, F, F, 6, -8, 8, 1.0f / 128); sh_test_fill_fp16(w2, F, D, D, 7, -8, 8, 1.0f / 256);
        sh_test_fill_fp16(bq, 1, D, D, 8, -4, 4, 0.0625f); sh_test_fill_fp16(bk, 1, D, D, 9, -4, 4, 0.0625f);
        sh_test_fill_fp16(bv, 1, D, D, 10, -4, 4, 0.0625f); sh_test_fill_fp16(bo, 1, D, D, 11, -4, 4, 0.0625f);
        sh_test_fill_fp16(b1, 1, F, F, 12, -4, 4, 0.0625f); sh_test_fill_fp16(b2, 1, D, D, 13, -4, 4, 0.0625f);
        sh_test_fill_fp16(g1, 1, D, D, 14, 2, 6, 0.25f); sh_test_fill_fp16(be1, 1, D, D, 15, -4, 4, 0.125f);
        sh_test_fill_fp16(g2, 1, D, D, 16, 2, 6, 0.25f); sh_test_fill_fp16(be2, 1, D, D, 17, -4, 4, 0.125f);
    }
    sh_barrier_global();
    if (lead) sh_timer_start();

    sh_gemm_cfg big = { .tm = 256, .tn = 256, .tk = 256, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    sh_gemm_cfg qk  = { .tm = 256, .tn = 256, .tk = DH,  .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    sh_gemm_cfg pv  = { .tm = 256, .tn = DH,  .tk = 256, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };

    sh_layernorm(ln1, x, g1, be1, S, D, D, 1e-6f, CLUSTER);
    sh_gemm(ln1, wq, q, S, D, D, D, D, D, &big, CLUSTER); sh_add_bias(q, q, bq, S, D, D, CLUSTER);
    sh_gemm(ln1, wk, k, S, D, D, D, D, D, &big, CLUSTER); sh_add_bias(k, k, bk, S, D, D, CLUSTER);
    sh_gemm(ln1, wv, v, S, D, D, D, D, D, &big, CLUSTER); sh_add_bias(v, v, bv, S, D, D, CLUSTER);
    sh_transpose(kT, k, S, D, D, S, CLUSTER);
    /* attention: heads dealt over clusters when CLUSTER == SH_ALL (each head's GEMMs are small) */
    const uint32_t P = sh_num_clusters();
    for (uint32_t hd = 0; hd < H; ++hd) {
        const uint32_t cl = (CLUSTER == SH_ALL) ? hd % P : CLUSTER;
        const uint64_t s_h = sc + (uint64_t)hd * S * S * 2;
        int r1 = sh_gemm(q + hd * DH * 2, kT + (uint64_t)hd * DH * S * 2, s_h, S, S, DH, D, S, S, &qk, cl);
        sh_softmax_rows(s_h, s_h, S, S, S, 0.125f, cl);
        int r2 = sh_gemm(s_h, v + hd * DH * 2, o + hd * DH * 2, S, DH, S, S, D, D, &pv, cl);
        if ((r1 || r2) && sh_cluster_id() == cl && sh_is_first_core()) sh_printf("[head %u] rc %d %d\n", hd, r1, r2);
    }
    sh_barrier_global();
    sh_gemm(o, wo, ao, S, D, D, D, D, D, &big, CLUSTER); sh_add_bias(ao, ao, bo, S, D, D, CLUSTER);
    sh_add(h, x, ao, S, D, D, CLUSTER);
    sh_layernorm(ln2, h, g2, be2, S, D, D, 1e-6f, CLUSTER);
    sh_gemm(ln2, w1, f1, S, F, D, D, F, F, &big, CLUSTER); sh_add_bias(f1, f1, b1, S, F, F, CLUSTER);
    sh_gelu(g, f1, S, F, F, CLUSTER);
    sh_gemm(g, w2, f2, S, D, F, F, D, D, &big, CLUSTER); sh_add_bias(f2, f2, b2, S, D, D, CLUSTER);
    sh_add(out, h, f2, S, D, D, CLUSTER);

    sh_barrier_global();
    if (lead) {
        sh_timer_end();
        sh_test_dump_samples(x, S, D, D, 200, NSAMPLES, "X");
        sh_test_dump_samples(ln1, S, D, D, 206, NSAMPLES, "LN1");
        sh_test_dump_samples(q, S, D, D, 201, NSAMPLES, "Q");
        sh_test_dump_samples(k, S, D, D, 207, NSAMPLES, "K");
        sh_test_dump_samples(kT, D, S, S, 208, NSAMPLES, "KT");
        sh_test_dump_samples(sc, S, S, S, 209, NSAMPLES, "P0");
        sh_test_dump_samples(v, S, D, D, 211, NSAMPLES, "V");
        sh_test_dump_samples(o, S, DH, D, 210, NSAMPLES, "O0");
        sh_test_dump_samples(o + 6 * DH * 2, S, DH, D, 212, NSAMPLES, "O6");
        sh_test_dump_samples(o, S, D, D, 202, NSAMPLES, "O");
        sh_test_dump_samples(h, S, D, D, 203, NSAMPLES, "H");
        sh_test_dump_samples(g, S, F, F, 204, NSAMPLES, "G");
        sh_test_dump_samples(out, S, D, D, 205, NSAMPLES, "OUT");
        sh_printf("SIGLIP_LAYER_DONE\n");
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
