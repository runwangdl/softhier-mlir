/* gvsoc micro-test: the hardware fp16<->fp32 register path (sh_h2f / sh_f2h, sh_ops.h) against the
 * software converters, bit for bit, plus a timing of both over N elements in TCDM. Cluster 0, core 0. */
#include "sh_ops.h"
#ifndef N_ELEMS
#define N_ELEMS 4096
#endif

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }
static int is_nan16(uint32_t h) { return (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu); }
static int is_nan32(uint32_t u) { return (u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu); }

int main(void) {
    sh_init();
    if (sh_cluster_id() == 0 && sh_is_first_core()) {
        uint32_t bad_h2f = 0, bad_f2h = 0, n = 0;
        /* 1. every fp16 code -> f32: hw vs sw */
        for (uint32_t h = 0; h < 0x10000u; ++h) {
            union { float f; uint32_t u; } a, b; a.f = sh_h2f(h); b.f = sh_fp16_to_f32((uint16_t)h);
            if (is_nan16(h)) { if (!is_nan32(a.u)) bad_h2f++; }
            else if (a.u != b.u) { if (bad_h2f < 8) sh_printf("h2f h=%04x hw=%08x sw=%08x\n", h, a.u, b.u); bad_h2f++; }
        }
        sh_printf("[fp16cvt] h2f all 65536 codes: bad=%u\n", bad_h2f);
        /* 2. random f32 (all exponents) + ties + subnormal/overflow edges -> fp16: hw vs sw */
        uint32_t s = 0x1234567u;
        for (n = 0; n < 200000; ++n) {
            union { float f; uint32_t u; } a; a.u = (lcg(&s) << 8) ^ lcg(&s);
            if (n & 1) a.u = (a.u & 0x807fffffu) | (((lcg(&s) % 40) + 100) << 23);   /* exponents around the fp16 range */
            if ((n & 7) == 2) a.u &= ~0x0fffu;                                        /* exact / tie patterns */
            if ((n & 7) == 3) a.u = (a.u & ~0x1fffu) | 0x1000u;                       /* exact tie at bit 12 */
            uint32_t hw = sh_f2h(a.f), sw = sh_f32_to_fp16(a.f);
            if (is_nan32(a.u)) { if (!is_nan16(hw)) bad_f2h++; }
            else if (hw != sw) { if (bad_f2h < 8) sh_printf("f2h f=%08x hw=%04x sw=%04x\n", a.u, hw, sw); bad_f2h++; }
        }
        sh_printf("[fp16cvt] f2h %u random floats: bad=%u\n", n, bad_f2h);
        /* 3. timing: y = x * 1.5 + 0.25 over N_ELEMS halves in TCDM, sw path then hw path */
        volatile uint16_t *x = (volatile uint16_t *)sh_l1_addr(0), *y = (volatile uint16_t *)sh_l1_addr(N_ELEMS * 2);
        for (uint32_t i = 0; i < N_ELEMS; ++i) x[i] = sh_f32_to_fp16((float)(int)(lcg(&s) % 64) * 0.125f - 4.f);
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; ++i) y[i] = sh_f32_to_fp16(sh_fp16_to_f32(x[i]) * 1.5f + 0.25f);
        sh_timer_end();      /* ROI #1 printed by the simulator: software path */
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; ++i) y[i] = (uint16_t)sh_f2h(sh_h2f(x[i]) * 1.5f + 0.25f);
        sh_timer_end();      /* ROI #2: hardware path */
        uint32_t bad = 0;
        for (uint32_t i = 0; i < N_ELEMS; ++i) if (y[i] != sh_f32_to_fp16(sh_fp16_to_f32(x[i]) * 1.5f + 0.25f)) bad++;
        sh_printf("[fp16cvt] timed loop N=%u bad=%u\n", (uint32_t)N_ELEMS, bad);
        sh_printf("FP16CVT_%s\n", (bad_h2f | bad_f2h | bad) ? "FAIL" : "PASS");
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
