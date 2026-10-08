/* Multi-cluster repro: NH clusters each produce one 64-column slice of a 256 x 768 output `z`
 * (ldz = 768) at the same time -- the P.V stage of multi-head attention. shape.h: MODE NH.
 *   MODE 0  sh_gemm per cluster, concurrently (tile 256x64x256, 1-D row stores of 128 B)
 *   MODE 1  DMA only: first core fills the L1 tile with a pattern, DM core stores it row by row
 *   MODE 2  scalar stores from the first core straight into HBM
 *   MODE 3  as 0 but serialized: global barrier between the heads
 *   MODE 4  as 0 but each head writes a private contiguous 256x64 buffer (ldz = 64)
 *   MODE 5  as 0 but the scores GEMM (256x256x64) + in-place softmax run first (the real sequence)
 *   MODE 6  as 5, then right away a 16-cluster GEMM that reads z (z2 = z . w2, like the Wo projection)
 *   MODE 7  as 6 but the lead core samples z (a few thousand cycles of delay) before the follow-up GEMM
 *   MODE 8  as 6 with int data only (no scores/softmax)
 * Expected value of slice h at (r, c): MODE 1/2: pattern(h, r, c); else the int GEMM. Cluster 0 checks. */
#include "sh_ops.h"
#include "shape.h"
#ifndef NH
#define NH 12
#endif
#define S 256
#define DH 64
#define D 768
static inline uint16_t pattern(uint32_t h, uint32_t r, uint32_t c) { return (uint16_t)(0x4000 | (h << 8) | ((r & 15) << 4) | (c & 15)); }

int main(void) {
    sh_init();
    const uint64_t xs = sh_hbm_addr(0x000000);                /* NH matrices 256x256 (P-like), 128 KB each */
    const uint64_t w  = sh_hbm_addr(0x400000);                /* 256 x 768 (V-like) */
    const uint64_t z  = sh_hbm_addr(0x500000);                /* 256 x 768 output */
    const uint64_t zp = sh_hbm_addr(0x600000);                /* MODE 4: NH private 256x64 buffers, 32 KB each */
    const uint64_t qh = sh_hbm_addr(0x800000);                /* MODE 5: q 256x768, kT 768x256 */
    const uint64_t kT = sh_hbm_addr(0x900000);
    const uint64_t w2 = sh_hbm_addr(0xA00000), z2 = sh_hbm_addr(0xC00000);   /* MODE 6-8: 768x768 weights, 256x768 output */
    const uint32_t cid = sh_cluster_id(), first = sh_is_first_core();
    const int lead = (cid == 0 && first);
    if (lead) {
        for (uint32_t h = 0; h < NH; ++h) sh_test_fill_int_fp16(xs + (uint64_t)h * S * S * 2, S, S, S, 100 + h, -1, 1);
        sh_test_fill_int_fp16(w, S, D, D, 2, -2, 2);
        sh_test_fill_int_fp16(z, S, D, D, 3, 7, 7);           /* poison */
        sh_test_fill_int_fp16(zp, S, NH * DH, NH * DH, 3, 7, 7);
#if MODE == 5 || MODE == 6 || MODE == 7
        sh_test_fill_fp16(qh, S, D, D, 4, -32, 32, 0.125f);
        sh_test_fill_fp16(kT, D, S, S, 5, -32, 32, 0.125f);
#endif
#if MODE >= 6
        sh_test_fill_int_fp16(w2, D, D, D, 6, -1, 1);
#endif
    }
    sh_barrier_global();
    if (lead) sh_timer_start();
    sh_gemm_cfg pv = { .tm = 256, .tn = DH, .tk = 256, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    sh_gemm_cfg qk = { .tm = 256, .tn = 256, .tk = DH, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
    const uint32_t P = sh_num_clusters();
    for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t cl = h % P;
        const uint64_t xh = xs + (uint64_t)h * S * S * 2;
        const uint64_t zh = z + h * DH * 2;
#if MODE == 0 || MODE == 3
        sh_gemm(xh, w + h * DH * 2, zh, S, DH, S, S, D, D, &pv, cl);
#if MODE == 3
        sh_barrier_global();
#endif
#elif MODE == 4
        sh_gemm(xh, w + h * DH * 2, zp + (uint64_t)h * S * DH * 2, S, DH, S, S, D, DH, &pv, cl);
#elif MODE == 5 || MODE == 6 || MODE == 7
        sh_gemm(qh + h * DH * 2, kT + (uint64_t)h * DH * S * 2, xh, S, S, DH, D, S, S, &qk, cl);
        sh_softmax_rows(xh, xh, S, S, S, 0.125f, cl);
        sh_gemm(xh, w + h * DH * 2, zh, S, DH, S, S, D, D, &pv, cl);
#elif MODE == 8
        sh_gemm(xh, w + h * DH * 2, zh, S, DH, S, S, D, D, &pv, cl);
#elif MODE == 1
        if (cid == cl) {
            const uint32_t y = 320 * 1024;   /* same L1 offset the gemm's Y tile uses */
            if (first) { volatile uint16_t *p = (volatile uint16_t *)(uintptr_t)sh_l1_addr(y); for (uint32_t r = 0; r < S; ++r) for (uint32_t c = 0; c < DH; ++c) p[r * DH + c] = pattern(h, r, c); }
            sh_barrier_cluster();
            for (uint32_t r = 0; r < S; ++r) sh_dma_copy(zh + (uint64_t)r * D * 2, sh_l1_addr(y + r * DH * 2), DH * 2);  /* DM core, sync each */
            sh_barrier_cluster();
        }
#elif MODE == 2
        if (cid == cl && first) for (uint32_t r = 0; r < S; ++r) { volatile uint16_t *p = (volatile uint16_t *)(uintptr_t)(zh + (uint64_t)r * D * 2); for (uint32_t c = 0; c < DH; ++c) p[c] = pattern(h, r, c); }
#endif
    }
    sh_barrier_global();
#if MODE >= 6
#if MODE == 7
    if (lead) { sh_test_dump_samples(z, S, D, D, 210, 64, "Zs"); sh_test_dump_samples(xs, S, S, S, 211, 64, "Ps"); sh_test_dump_samples(w, S, D, D, 212, 64, "Ws"); }
    sh_barrier_global();
#endif
    sh_gemm_cfg big = { .tm = 256, .tn = 256, .tk = 256, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };
#ifndef NO_GEMM2
    sh_gemm(z, w2, z2, S, D, D, D, D, D, &big, SH_ALL);
#endif
#if MODE == 7
    if (lead) { sh_test_dump_samples(z, S, D, D, 210, 64, "Ze"); sh_test_dump_samples(xs, S, S, S, 211, 64, "Pe"); sh_test_dump_samples(w, S, D, D, 212, 64, "We"); }
#endif
#endif
    if (lead) {
        sh_timer_end();
        uint32_t bad = 0;
#if MODE >= 6 && !defined(NO_Z2CHECK)
        bad += sh_test_check_gemm(z, w2, z2, S, D, D, D, D, D, 128, MODE == 8 ? 0.5f : 2.0f, 0.f, "[z2]");
#endif
        for (uint32_t h = 0; h < NH; ++h) {
            const uint64_t xh = xs + (uint64_t)h * S * S * 2;
            char tag[16]; tag[0] = '['; tag[1] = 'h'; tag[2] = '0' + h / 10; tag[3] = '0' + h % 10; tag[4] = ']'; tag[5] = 0;
#if MODE == 1 || MODE == 2
            uint32_t b = 0; for (uint32_t r = 0; r < S; ++r) { const volatile uint16_t *p = (const volatile uint16_t *)(uintptr_t)(z + h * DH * 2 + (uint64_t)r * D * 2);
                for (uint32_t c = 0; c < DH; ++c) if (p[c] != pattern(h, r, c)) { if (b < 3) sh_printf("  %s z[%u,%u] got %04x want %04x\n", tag, r, h * DH + c, p[c], pattern(h, r, c)); b++; } }
            sh_printf("%s bad=%u %s\n", tag, b, b ? "FAIL" : "PASS"); bad += b;
#elif MODE == 4
            bad += sh_test_check_gemm(xh, w + h * DH * 2, zp + (uint64_t)h * S * DH * 2, S, DH, S, S, D, DH, 128, 0.5f, 0.f, tag);
#elif MODE == 5 || MODE == 6 || MODE == 7
            bad += sh_test_check_gemm(xh, w + h * DH * 2, z + h * DH * 2, S, DH, S, S, D, D, 128, 0.05f, 0.f, tag);
#else
            bad += sh_test_check_gemm(xh, w + h * DH * 2, z + h * DH * 2, S, DH, S, S, D, D, 128, 0.5f, 0.f, tag);
#endif
        }
        sh_printf("[mesh_slices] MODE %d NH %d %s\n", MODE, NH, bad ? "MESH_FAIL" : "MESH_PASS");
    }
    sh_barrier_global(); sh_eoc(0); return 0;
}
