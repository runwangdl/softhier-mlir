/* One SigLIP (ViT) encoder layer on SoftHier, composed from softhier-ops, random LCG weights,
 * sampled outputs compared on the host against numpy (softhier_mlir/testing/siglip_ref.py).
 *   h   = x + Wo(attn(LN1(x)))      attn: per head softmax(q k^T / sqrt(dh)) v
 *   out = h + W2(gelu(W1(LN2(h))))
 * Weights are [in, out] (X . W convention), biases are rows. shape.h: SEQ D_MODEL D_FF N_HEADS CLUSTER NSAMPLES HBM_START
 * [SH_PRELOAD]: the LCG inputs are preloaded by the host (run.py mirrors the ALLOC layout below) or generated here. */
#include "sh_ops.h"
#include "shape.h"

#define DH (D_MODEL / N_HEADS)
/* Bump allocator on the CALLER'S STACK: every core runs main(), and a global in .bss would be
 * incremented 48 times concurrently (each core would see different addresses). */
static uint64_t alloc(uint64_t *next, uint32_t bytes) { uint64_t a = *next; *next += (bytes + 4095) & ~4095u; return a; }
#define ALLOC(bytes) alloc(&next_buf, (bytes))
#ifdef ATTN_CANARY             /* bisect: checksums of head-0 P rows 0..15 and of the O0 slice at several points */
static uint32_t cksum(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld) {
    uint32_t s = 0;
    for (uint32_t r = 0; r < rows; ++r) { const volatile uint16_t *p = (const volatile uint16_t *)(uintptr_t)(a + (uint64_t)r * ld * 2); for (uint32_t c = 0; c < cols; ++c) s = s * 31u + p[c]; }
    return s;
}
static void scan_p0(uint64_t p, uint32_t rows, uint32_t cols, const char *when) {   /* rows must sum to 1 and be >= 0 */
    uint32_t nbad = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const volatile uint16_t *x = (const volatile uint16_t *)(uintptr_t)(p + (uint64_t)r * cols * 2);
        float sum = 0.f; int neg = 0; for (uint32_t c = 0; c < cols; ++c) { float v = sh_fp16_to_f32(x[c]); sum += v; if (v < 0.f) neg++; }
        if (sum < 0.9f || sum > 1.1f || neg) { if (nbad < 12) sh_printf("[canary %s] P0 row %u sum %f neg %d\n", when, r, sum, neg); nbad++; }
    }
    sh_printf("[canary %s] P0 bad rows %u/%u\n", when, nbad, rows);
}
#endif

int main(void) {
    sh_init();
    const uint32_t S = SEQ, D = D_MODEL, F = D_FF, H = N_HEADS;
    uint64_t next_buf = sh_hbm_addr(HBM_START);
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
#ifdef SH_PRELOAD
    sh_preload_wait(sh_hbm_addr(SH_PRELOAD));
#else
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
#endif
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
#ifdef ATTN_CLUSTER            /* bisect: pin every head to one cluster */
        const uint32_t cl = ATTN_CLUSTER;
#else
        const uint32_t cl = (CLUSTER == SH_ALL) ? hd % P : CLUSTER;
#endif
        const uint64_t s_h = sc + (uint64_t)hd * S * S * 2;
        int r1 = sh_gemm(q + hd * DH * 2, kT + (uint64_t)hd * DH * S * 2, s_h, S, S, DH, D, S, S, &qk, cl);
        sh_softmax_rows(s_h, s_h, S, S, S, 0.125f, cl);
        int r2 = sh_gemm(s_h, v + hd * DH * 2, o + hd * DH * 2, S, DH, S, S, D, D, &pv, cl);
        if ((r1 || r2) && sh_cluster_id() == cl && sh_is_first_core()) sh_printf("[head %u] rc %d %d\n", hd, r1, r2);
#ifdef ATTN_CANARY
        if (hd == 0 && lead) sh_printf("[canary head0] P0 %08x O0 %08x\n", cksum(sc, 16, S, S), cksum(o, 16, DH, D));
#endif
#ifdef ATTN_DEBUG              /* bisect: the producing cluster checks its own slice + its L1 tiles right away */
        if (sh_cluster_id() == cl && sh_is_first_core()) {
            char tag[12] = "[hdbg 00]"; tag[6] = '0' + hd / 10; tag[7] = '0' + hd % 10;
            sh_test_check_gemm(s_h, v + hd * DH * 2, o + hd * DH * 2, S, DH, S, S, D, D, 128, 0.05f, 0.f, tag);
            /* L1 layout of the pv cfg: X0 = 0 (P tile 256x256), W0 = 2*xb = 256 KB (V tile 256x64), Y = 320 KB */
            const volatile uint16_t *lp = (const volatile uint16_t *)(uintptr_t)sh_l1_addr(0);
            const volatile uint16_t *lv = (const volatile uint16_t *)(uintptr_t)sh_l1_addr(256 * 1024);
            const volatile uint16_t *ly = (const volatile uint16_t *)(uintptr_t)sh_l1_addr(320 * 1024);
            uint32_t badp = 0, badv = 0, bady = 0, firstp = 0xFFFFFFFFu, firstv = 0xFFFFFFFFu, firsty = 0xFFFFFFFFu;
            for (uint32_t r = 0; r < S; ++r) {
                const volatile uint16_t *hp = (const volatile uint16_t *)(uintptr_t)(s_h + (uint64_t)r * S * 2);
                for (uint32_t c = 0; c < S; ++c) if (lp[r * S + c] != hp[c]) { if (badp == 0) firstp = r * S + c; badp++; }
                const volatile uint16_t *hv = (const volatile uint16_t *)(uintptr_t)(v + hd * DH * 2 + (uint64_t)r * D * 2);
                for (uint32_t c = 0; c < DH; ++c) if (lv[r * DH + c] != hv[c]) { if (badv == 0) firstv = r * DH + c; badv++; }
                const volatile uint16_t *ho = (const volatile uint16_t *)(uintptr_t)(o + hd * DH * 2 + (uint64_t)r * D * 2);
                for (uint32_t c = 0; c < DH; ++c) if (ly[r * DH + c] != ho[c]) { if (bady == 0) firsty = r * DH + c; bady++; }
            }
            sh_printf("%s L1 vs HBM: P bad=%u (first %u) V bad=%u (first %u) Y-vs-O bad=%u (first %u)\n", tag, badp, firstp, badv, firstv, bady, firsty);
        }
#endif
#ifdef ATTN_SERIAL             /* bisect: one head at a time */
        sh_barrier_global();
#endif
    }
    sh_barrier_global();
#ifdef ATTN_CANARY
    if (lead) sh_printf("[canary loop-end] P0 %08x O0 %08x\n", cksum(sc, 16, S, S), cksum(o, 16, DH, D));
#endif
#ifdef ATTN_DUMP_EARLY         /* bisect: sample head-0 output before anything else touches `o` */
    if (lead) sh_test_dump_samples(o, S, DH, D, 210, NSAMPLES, "O0a");
    sh_barrier_global();
#endif
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
#ifdef ATTN_CANARY
        sh_printf("[canary end] P0 %08x O0 %08x\n", cksum(sc, 16, S, S), cksum(o, 16, DH, D));
        scan_p0(sc, S, S, "end");
        for (uint32_t r = 0; r < 2; ++r) for (uint32_t c = 0; c < S; c += 16) {
            const volatile uint16_t *x = (const volatile uint16_t *)(uintptr_t)(sc + ((uint64_t)r * S + c) * 2);
            sh_printf("P0ROW %u %u %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x\n", r, c,
                      x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
        }
        for (uint32_t c = 0; c < DH; c += 16) {
            const volatile uint16_t *x = (const volatile uint16_t *)(uintptr_t)(o + c * 2);
            sh_printf("O0ROW 0 %u %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x\n", c,
                      x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
        }
#endif
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
