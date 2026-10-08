/* Two GEMMs with different RedMulE tile shapes back to back on one cluster (attention pattern):
 *   A: 256x256x64  (scores: contraction 64)      B: 256x64x256 (P.V: output width 64)   then C: 256x256x256 */
#include "sh_ops.h"
int main(void) {
    sh_init();
    const uint64_t xa = sh_hbm_addr(0x000000), wa = sh_hbm_addr(0x100000), za = sh_hbm_addr(0x200000);
    const uint64_t xb = sh_hbm_addr(0x300000), wb = sh_hbm_addr(0x400000), zb = sh_hbm_addr(0x500000);
    const uint64_t xc = sh_hbm_addr(0x600000), wc = sh_hbm_addr(0x700000), zc = sh_hbm_addr(0x800000);
    const uint64_t xd = sh_hbm_addr(0x900000), wd = sh_hbm_addr(0xA00000), zd = sh_hbm_addr(0xB00000);  /* D: 64-col slices of 768-wide W/Z */
    const uint64_t xe = sh_hbm_addr(0xC00000), we = sh_hbm_addr(0xD00000), ze = sh_hbm_addr(0xE00000);  /* E: softmax(P) . V slice, real data */
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    if (lead) {
        sh_test_fill_int_fp16(xa, 256, 64, 64, 1, -1, 1);  sh_test_fill_int_fp16(wa, 64, 256, 256, 2, -2, 2);
        sh_test_fill_int_fp16(xb, 256, 256, 256, 3, -1, 1); sh_test_fill_int_fp16(wb, 256, 64, 64, 4, -2, 2);
        sh_test_fill_int_fp16(xc, 256, 256, 256, 5, -1, 1); sh_test_fill_int_fp16(wc, 256, 256, 256, 6, -2, 2);
        sh_test_fill_int_fp16(xd, 256, 256, 256, 7, -1, 1); sh_test_fill_int_fp16(wd, 256, 768, 768, 8, -2, 2);
        sh_test_fill_int_fp16(zd, 256, 768, 768, 9, 7, 7);  /* poison */
        sh_test_fill_fp16(xe, 256, 256, 256, 10, -32, 32, 0.125f);   /* scores-like */
        sh_test_fill_fp16(we, 256, 768, 768, 11, -16, 16, 0.125f);  /* V-like */
    }
    sh_barrier_global();
    sh_softmax_rows(xe, xe, 256, 256, 256, 0.125f, 0);   /* P = softmax(scores/8), in place, cluster 0 */
    sh_gemm_cfg a = { .tm = 256, .tn = 256, .tk = 64, .pipeline = 1 }, b = { .tm = 256, .tn = 64, .tk = 256, .pipeline = 1 }, c = { .pipeline = 1 };
    sh_gemm(xa, wa, za, 256, 256, 64, 64, 256, 256, &a, 0);
    sh_gemm(xb, wb, zb, 256, 64, 256, 256, 64, 64, &b, 0);
    sh_gemm(xc, wc, zc, 256, 256, 256, 256, 256, 256, &c, 0);
    sh_gemm(xd, wd + 6 * 64 * 2, zd + 6 * 64 * 2, 256, 64, 256, 256, 768, 768, &b, 0);   /* head-6 slice */
    sh_gemm(xe, we + 0 * 64 * 2, ze + 0 * 64 * 2, 256, 64, 256, 256, 768, 768, &b, 0);   /* P.V head-0 slice, real data */
    sh_barrier_global();
    if (lead) {
        uint32_t bad = 0;
        bad += sh_test_check_gemm(xa, wa, za, 256, 256, 64, 64, 256, 256, 128, 0.5f, 0.f, "[A 256x256x64]");
        bad += sh_test_check_gemm(xb, wb, zb, 256, 64, 256, 256, 64, 64, 128, 0.5f, 0.f, "[B 256x64x256]");
        bad += sh_test_check_gemm(xc, wc, zc, 256, 256, 256, 256, 256, 256, 128, 0.5f, 0.f, "[C 256x256x256]");
        bad += sh_test_check_gemm(xd, wd + 6 * 64 * 2, zd + 6 * 64 * 2, 256, 64, 256, 256, 768, 768, 128, 0.5f, 0.f, "[D slice ld=768]");
        bad += sh_test_check_gemm(xe, we, ze, 256, 64, 256, 256, 768, 768, 128, 0.05f, 0.f, "[E softmax.V real]");
        sh_printf("[seq] %s\n", bad ? "GEMM_FAIL" : "GEMM_PASS");
        sh_timer_start(); sh_timer_end();
    }
    sh_barrier_global(); sh_eoc(0); return 0;
}
