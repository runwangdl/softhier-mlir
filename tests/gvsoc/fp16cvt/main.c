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
            uint32_t hw = sh_f2h(a.f) & 0xFFFFu, sw = sh_f32_to_fp16(a.f);
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
        /* 4. ILP probes: 4-way unrolled elementwise (ROI #3); sum with 1 accumulator (ROI #4) vs 4 (ROI #5) */
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; i += 4) {
            float a0 = sh_h2f(x[i]), a1 = sh_h2f(x[i + 1]), a2 = sh_h2f(x[i + 2]), a3 = sh_h2f(x[i + 3]);
            a0 = a0 * 1.5f + 0.25f; a1 = a1 * 1.5f + 0.25f; a2 = a2 * 1.5f + 0.25f; a3 = a3 * 1.5f + 0.25f;
            y[i] = (uint16_t)sh_f2h(a0); y[i + 1] = (uint16_t)sh_f2h(a1); y[i + 2] = (uint16_t)sh_f2h(a2); y[i + 3] = (uint16_t)sh_f2h(a3);
        }
        sh_timer_end();
        float s1 = 0.f;
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; ++i) s1 += sh_h2f(x[i]);
        sh_timer_end();
        float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; i += 4) { q0 += sh_h2f(x[i]); q1 += sh_h2f(x[i + 1]); q2 += sh_h2f(x[i + 2]); q3 += sh_h2f(x[i + 3]); }
        sh_timer_end();
        for (uint32_t i = 0; i < N_ELEMS; ++i) if (y[i] != sh_f32_to_fp16(sh_fp16_to_f32(x[i]) * 1.5f + 0.25f)) bad++;
        /* ROI #6: 4-wide interleaved asm conversions (sh_h2f4 / sh_f2h4); ROI #7: 8-wide (two blocks) */
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; i += 4) {
            float a0, a1, a2, a3; sh_h2f4((const uint16_t *)x + i, &a0, &a1, &a2, &a3);
            a0 = a0 * 1.5f + 0.25f; a1 = a1 * 1.5f + 0.25f; a2 = a2 * 1.5f + 0.25f; a3 = a3 * 1.5f + 0.25f;
            sh_f2h4((uint16_t *)y + i, a0, a1, a2, a3);
        }
        sh_timer_end();
        sh_timer_start();
        for (uint32_t i = 0; i < N_ELEMS; i += 8) {
            float a0, a1, a2, a3, b0, b1, b2, b3;
            sh_h2f4((const uint16_t *)x + i, &a0, &a1, &a2, &a3); sh_h2f4((const uint16_t *)x + i + 4, &b0, &b1, &b2, &b3);
            a0 = a0 * 1.5f + 0.25f; a1 = a1 * 1.5f + 0.25f; a2 = a2 * 1.5f + 0.25f; a3 = a3 * 1.5f + 0.25f;
            b0 = b0 * 1.5f + 0.25f; b1 = b1 * 1.5f + 0.25f; b2 = b2 * 1.5f + 0.25f; b3 = b3 * 1.5f + 0.25f;
            sh_f2h4((uint16_t *)y + i, a0, a1, a2, a3); sh_f2h4((uint16_t *)y + i + 4, b0, b1, b2, b3);
        }
        sh_timer_end();
        for (uint32_t i = 0; i < N_ELEMS; ++i) if (y[i] != sh_f32_to_fp16(sh_fp16_to_f32(x[i]) * 1.5f + 0.25f)) bad++;
        sh_printf("[fp16cvt] ilp probes: sum1=%d sum4=%d (x1000) bad=%u\n", (int)(s1 * 1000.f), (int)((q0 + q1 + q2 + q3) * 1000.f), bad);
        /* 5. division probe: 8 independent back-to-back fdiv.s vs one at a time (bit compare, fp32 in TCDM) */
        {
            volatile float *fx = (volatile float *)sh_l1_addr(0x10000), *fy = (volatile float *)sh_l1_addr(0x18000), *fz = (volatile float *)sh_l1_addr(0x20000);
            for (uint32_t i = 0; i < 1024; ++i) fx[i] = (float)(int)(lcg(&s) % 200 - 100) * 0.03125f;
            for (uint32_t i = 0; i < 1024; ++i) { float v = fx[i]; fy[i] = v / (1.f + v * v); }
            for (uint32_t i = 0; i < 1024; i += 8) {
                float v0 = fx[i], v1 = fx[i + 1], v2 = fx[i + 2], v3 = fx[i + 3], v4 = fx[i + 4], v5 = fx[i + 5], v6 = fx[i + 6], v7 = fx[i + 7];
                float d0 = 1.f + v0 * v0, d1 = 1.f + v1 * v1, d2 = 1.f + v2 * v2, d3 = 1.f + v3 * v3, d4 = 1.f + v4 * v4, d5 = 1.f + v5 * v5, d6 = 1.f + v6 * v6, d7 = 1.f + v7 * v7;
                float r0, r1, r2, r3, r4, r5, r6, r7;
                __asm__ ("fdiv.s %0, %8, %16\n\tfdiv.s %1, %9, %17\n\tfdiv.s %2, %10, %18\n\tfdiv.s %3, %11, %19\n\t"
                         "fdiv.s %4, %12, %20\n\tfdiv.s %5, %13, %21\n\tfdiv.s %6, %14, %22\n\tfdiv.s %7, %15, %23"
                         : "=&f"(r0), "=&f"(r1), "=&f"(r2), "=&f"(r3), "=&f"(r4), "=&f"(r5), "=&f"(r6), "=&f"(r7)
                         : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7),
                           "f"(d0), "f"(d1), "f"(d2), "f"(d3), "f"(d4), "f"(d5), "f"(d6), "f"(d7));
                fz[i] = r0; fz[i + 1] = r1; fz[i + 2] = r2; fz[i + 3] = r3; fz[i + 4] = r4; fz[i + 5] = r5; fz[i + 6] = r6; fz[i + 7] = r7;
            }
            uint32_t bad_div = 0;
            for (uint32_t i = 0; i < 1024; ++i) {
                union { float f; uint32_t u; } a, b; a.f = fy[i]; b.f = fz[i];
                if (a.u != b.u) { if (bad_div < 6) sh_printf("div i=%u x=%08x seq=%08x par=%08x\n", i, ((union { float f; uint32_t u; }){ .f = fx[i] }).u, a.u, b.u); bad_div++; }
            }
            sh_printf("[fp16cvt] div probe: 1024 elems, 8 back-to-back fdiv.s vs sequential: bad=%u\n", bad_div);
            bad += bad_div;
            /* 6. FP->int->FP hazards, 8 independent chains back to back:
             *    A: floor via fcvt.w.s rdn + fcvt.s.w;  B: 2^k via fcvt.w.s + int ops + fmv.w.x;
             *    C: pure-FP round (t + 1.5*2^23 - 1.5*2^23) + fmv.x.w/slli/add/fmv.w.x. Reference: scalar loop. */
            for (uint32_t i = 0; i < 1024; ++i) fx[i] = (float)(int)(lcg(&s) % 2000 - 1000) * 0.0625f;
            uint32_t badA = 0, badB = 0, badC = 0;
            #define FLOOR1(t, kf) do { int k_; __asm__ ("fcvt.w.s %0, %1, rdn" : "=r"(k_) : "f"(t)); __asm__ ("fcvt.s.w %0, %1" : "=f"(kf) : "r"(k_)); } while (0)
            #define EXP2I1(kf, r) do { int k_; __asm__ ("fcvt.w.s %0, %1, rtz" : "=r"(k_) : "f"(kf)); __asm__ ("fmv.w.x %0, %1" : "=f"(r) : "r"((uint32_t)(k_ + 127) << 23)); } while (0)
            #define RND1(t, kf, r) do { float m_ = (t) + 12582912.f; uint32_t b_; __asm__ ("fmv.x.w %0, %1" : "=r"(b_) : "f"(m_)); kf = m_ - 12582912.f; __asm__ ("fmv.w.x %0, %1" : "=f"(r) : "r"((b_ << 23) + ((127u - 0x4B400000u) << 23))); } while (0)
            for (uint32_t i = 0; i < 1024; ++i) { float kf; FLOOR1(fx[i], kf); fy[i] = kf; }
            for (uint32_t i = 0; i < 1024; i += 8) {
                float t0 = fx[i], t1 = fx[i + 1], t2 = fx[i + 2], t3 = fx[i + 3], t4 = fx[i + 4], t5 = fx[i + 5], t6 = fx[i + 6], t7 = fx[i + 7];
                float k0, k1, k2, k3, k4, k5, k6, k7;
                FLOOR1(t0, k0); FLOOR1(t1, k1); FLOOR1(t2, k2); FLOOR1(t3, k3); FLOOR1(t4, k4); FLOOR1(t5, k5); FLOOR1(t6, k6); FLOOR1(t7, k7);
                fz[i] = k0; fz[i + 1] = k1; fz[i + 2] = k2; fz[i + 3] = k3; fz[i + 4] = k4; fz[i + 5] = k5; fz[i + 6] = k6; fz[i + 7] = k7;
            }
            for (uint32_t i = 0; i < 1024; ++i) if (fy[i] != fz[i]) { if (badA < 4) sh_printf("A i=%u t=%d/16 seq=%d par=%d\n", i, (int)(fx[i] * 16.f), (int)fy[i], (int)fz[i]); badA++; }
            for (uint32_t i = 0; i < 1024; ++i) { float kf = fy[i] < -126.f ? -126.f : fy[i], r; EXP2I1(kf, r); fz[i] = r; fy[i] = kf; }
            for (uint32_t i = 0; i < 1024; i += 8) {
                float r0, r1, r2, r3, r4, r5, r6, r7;
                EXP2I1(fy[i], r0); EXP2I1(fy[i + 1], r1); EXP2I1(fy[i + 2], r2); EXP2I1(fy[i + 3], r3); EXP2I1(fy[i + 4], r4); EXP2I1(fy[i + 5], r5); EXP2I1(fy[i + 6], r6); EXP2I1(fy[i + 7], r7);
                if (r0 != fz[i]) badB++; if (r1 != fz[i + 1]) badB++; if (r2 != fz[i + 2]) badB++; if (r3 != fz[i + 3]) badB++;
                if (r4 != fz[i + 4]) badB++; if (r5 != fz[i + 5]) badB++; if (r6 != fz[i + 6]) badB++; if (r7 != fz[i + 7]) badB++;
            }
            for (uint32_t i = 0; i < 1024; i += 8) {
                float k0, k1, k2, k3, k4, k5, k6, k7, r0, r1, r2, r3, r4, r5, r6, r7;
                RND1(fy[i], k0, r0); RND1(fy[i + 1], k1, r1); RND1(fy[i + 2], k2, r2); RND1(fy[i + 3], k3, r3);
                RND1(fy[i + 4], k4, r4); RND1(fy[i + 5], k5, r5); RND1(fy[i + 6], k6, r6); RND1(fy[i + 7], k7, r7);
                if (k0 != fy[i] || r0 != fz[i]) badC++; if (k1 != fy[i + 1] || r1 != fz[i + 1]) badC++; if (k2 != fy[i + 2] || r2 != fz[i + 2]) badC++; if (k3 != fy[i + 3] || r3 != fz[i + 3]) badC++;
                if (k4 != fy[i + 4] || r4 != fz[i + 4]) badC++; if (k5 != fy[i + 5] || r5 != fz[i + 5]) badC++; if (k6 != fy[i + 6] || r6 != fz[i + 6]) badC++; if (k7 != fy[i + 7] || r7 != fz[i + 7]) badC++;
            }
            sh_printf("[fp16cvt] fp->int hazard probe (8 independent chains): floor bad=%u (known gvsoc hazard, SIMULATOR_NOTES #6) exp2i bad=%u pure-fp-round bad=%u\n", badA, badB, badC);
            bad += badB + badC;   /* the fcvt.w.s -> fcvt.s.w pair is known-broken in the model; the library avoids it */
            /* 7. TCDM access cost (ROI #8..#10): word copy; 2x2 transpose micro-tiles with 256 B and 260 B pitch */
            volatile uint32_t *wx = (volatile uint32_t *)sh_l1_addr(0x10000), *wy = (volatile uint32_t *)sh_l1_addr(0x20000);
            sh_timer_start();
            for (uint32_t i = 0; i < 4096; ++i) wy[i] = wx[i];
            sh_timer_end();
            {
                const uint16_t *ts = (const uint16_t *)sh_l1_addr(0x10000); uint16_t *td = (uint16_t *)sh_l1_addr(0x30000);
                sh_timer_start();
                for (uint32_t r = 0; r < 64; r += 2) {
                    const uint32_t *s0 = (const uint32_t *)(ts + r * 128), *s1 = (const uint32_t *)(ts + (r + 1) * 128); uint32_t *d0 = (uint32_t *)(td + r);
                    for (uint32_t c = 0; c < 128; c += 2) { const uint32_t w0 = s0[c >> 1], w1 = s1[c >> 1]; d0[(c * 128) >> 1] = (w0 & 0xFFFFu) | (w1 << 16); d0[((c + 1) * 128) >> 1] = (w0 >> 16) | (w1 & 0xFFFF0000u); }
                }
                sh_timer_end();
                sh_timer_start();
                for (uint32_t r = 0; r < 64; r += 2) {
                    const uint32_t *s0 = (const uint32_t *)(ts + r * 130), *s1 = (const uint32_t *)(ts + (r + 1) * 130); uint32_t *d0 = (uint32_t *)(td + r);
                    for (uint32_t c = 0; c < 128; c += 2) { const uint32_t w0 = s0[c >> 1], w1 = s1[c >> 1]; d0[(c * 130) >> 1] = (w0 & 0xFFFFu) | (w1 << 16); d0[((c + 1) * 130) >> 1] = (w0 >> 16) | (w1 & 0xFFFF0000u); }
                }
                sh_timer_end();
            }
        }
        /* 9. Xfvec fp16 SIMD (4 halves per 64-bit FP register, fld/fsd + `.insn r` encodings with plain "f"
         *    operands): vfadd.h vfmul.h vfmul.r.h vfmac.h vfmax.h vfcpka.h.s vfcpkb.h.s, lane bits vs the scalar
         *    path; then ROI #17: y = 0.3 x over 4096 halves with vfmul.r.h (compare with ROI #3, scalar 8-wide). */
        {
            typedef union { double d; uint16_t h[4]; } v4h;
            #define V4OP(f3, f7, r, a, b) __asm__ (".insn r 0x33, " #f3 ", " #f7 ", %0, %1, %2" : "=f"(r) : "f"(a), "f"(b))
            uint32_t vb_add = 0, vb_mul = 0, vb_mulr = 0, vb_mac = 0, vb_mac1 = 0, vb_max = 0, vb_cpk = 0;
            v4h sv; sv.h[0] = sh_f32_to_fp16(0.3f); sv.h[1] = sv.h[2] = sv.h[3] = 0;
            for (uint32_t i = 0; i < 4096; i += 4) {
                v4h a, b, c, r; double t;
                a.d = *(const double *)(const uint16_t *)(x + i); b.d = *(const double *)(const uint16_t *)(x + ((i + 4) & 4095));
                c.d = *(const double *)(const uint16_t *)(x + ((i + 8) & 4095));
                V4OP(2, 0x41, t, a.d, b.d); r.d = t;
                for (int l = 0; l < 4; ++l) if (r.h[l] != sh_f32_to_fp16(sh_fp16_to_f32(a.h[l]) + sh_fp16_to_f32(b.h[l]))) vb_add++;
                V4OP(2, 0x43, t, a.d, b.d); r.d = t;
                for (int l = 0; l < 4; ++l) if (r.h[l] != sh_f32_to_fp16(sh_fp16_to_f32(a.h[l]) * sh_fp16_to_f32(b.h[l]))) vb_mul++;
                V4OP(6, 0x43, t, a.d, sv.d); r.d = t;
                for (int l = 0; l < 4; ++l) if (r.h[l] != sh_f32_to_fp16(sh_fp16_to_f32(a.h[l]) * sh_fp16_to_f32(sv.h[0]))) { if (vb_mulr < 3) sh_printf("mulr i=%u l=%d a=%04x s=%04x got=%04x want=%04x\n", i, l, a.h[l], sv.h[0], r.h[l], sh_f32_to_fp16(sh_fp16_to_f32(a.h[l]) * sh_fp16_to_f32(sv.h[0]))); vb_mulr++; }
                t = c.d; __asm__ (".insn r 0x33, 2, 0x48, %0, %1, %2" : "+f"(t) : "f"(a.d), "f"(b.d)); r.d = t;   /* vfmac.h: c += a*b */
                for (int l = 0; l < 4; ++l) {
                    uint16_t w = sh_f32_to_fp16(sh_fp16_to_f32(c.h[l]) + sh_fp16_to_f32(a.h[l]) * sh_fp16_to_f32(b.h[l]));
                    int d = (int)r.h[l] - (int)w; if (d < 0) d = -d; if (d > 1) vb_mac++; else if (d == 1) vb_mac1++;
                }
                V4OP(2, 0x46, t, a.d, b.d); r.d = t;
                for (int l = 0; l < 4; ++l) { float p = sh_fp16_to_f32(a.h[l]), q = sh_fp16_to_f32(b.h[l]); if (r.h[l] != (p > q ? a.h[l] : b.h[l])) vb_max++; }
                {
                    float g0 = sh_fp16_to_f32(a.h[0]) * 1.5f + 0.25f, g1 = sh_fp16_to_f32(a.h[1]) * 1.5f + 0.25f;
                    float g2 = sh_fp16_to_f32(a.h[2]) * 1.5f + 0.25f, g3 = sh_fp16_to_f32(a.h[3]) * 1.5f + 0.25f;
                    __asm__ (".insn r 0x33, 2, 0x58, %0, %1, %2" : "=f"(t) : "f"(g0), "f"(g1));       /* vfcpka.h.s: lanes 0,1 */
                    __asm__ (".insn r 0x33, 6, 0x58, %0, %1, %2" : "+f"(t) : "f"(g2), "f"(g3));       /* vfcpkb.h.s: lanes 2,3 */
                    r.d = t;
                    if (r.h[0] != sh_f32_to_fp16(g0) || r.h[1] != sh_f32_to_fp16(g1) || r.h[2] != sh_f32_to_fp16(g2) || r.h[3] != sh_f32_to_fp16(g3)) vb_cpk++;
                }
            }
            sh_printf("[fp16cvt] xfvec probe: add bad=%u mul bad=%u mul.r bad=%u mac bad=%u (1ulp %u) max bad=%u cpk bad=%u\n",
                      vb_add, vb_mul, vb_mulr, vb_mac, vb_mac1, vb_max, vb_cpk);
            bad += vb_add + vb_mul + vb_mulr + vb_mac + vb_max + vb_cpk;
            {   /* timed: y = 0.3 x with vfmul.r.h, 4 elements per fld / .insn / fsd */
                const double *xd = (const double *)(const uint16_t *)x; double *yd = (double *)(uint16_t *)y; const double s4 = sv.d;
                sh_timer_start();
                for (uint32_t i = 0; i < 1024; ++i) { double t; V4OP(6, 0x43, t, xd[i], s4); yd[i] = t; }
                sh_timer_end();
                uint32_t vb_t = 0;
                for (uint32_t i = 0; i < 4096; ++i) if (y[i] != sh_f32_to_fp16(sh_fp16_to_f32(x[i]) * sh_fp16_to_f32(sv.h[0]))) vb_t++;
                sh_printf("[fp16cvt] xfvec timed scale loop: bad=%u\n", vb_t); bad += vb_t;
            }
            #undef V4OP
        }
        sh_printf("FP16CVT_%s\n", (bad_h2f | bad_f2h | bad) ? "FAIL" : "PASS");
    }
    sh_barrier_global();
    /* 8. core scaling on cluster 0 (ROI #11..#16): each loop timed with 1 core running, then with all
     *    3 cores running it concurrently (core 0 times, barriers around). Loops: tight (fits the 32 B
     *    fetch line), straight-line integer (64 adds), 8-wide fp16 scale kernel over 4096 elements. */
    if (sh_cluster_id() == 0) {
        const uint32_t cid = sh_core_id();
        volatile uint32_t *sink = (volatile uint32_t *)sh_l1_addr(0x40000 + cid * 64);
        volatile uint16_t *px = (volatile uint16_t *)sh_l1_addr(0x50000 + cid * 0x4000), *py = (volatile uint16_t *)sh_l1_addr(0x60000 + cid * 0x4000);
        for (uint32_t i = 0; i < 4096; ++i) px[i] = 0x3c00 + (uint16_t)i;
        for (int ncores = 1; ncores <= 3; ncores += 2) {
            const int run = cid < (uint32_t)ncores;
            sh_barrier_cluster(); if (cid == 0) sh_timer_start();
            if (run) { uint32_t acc = 0; for (uint32_t i = 0; i < 16384; ++i) acc += i ^ (acc << 1); *sink = acc; }
            sh_barrier_cluster(); if (cid == 0) sh_timer_end();
            sh_barrier_cluster(); if (cid == 0) sh_timer_start();
            if (run) {
                uint32_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
                for (uint32_t i = 0; i < 1024; ++i) {
                    #define S4 a0 += a1 ^ i; a1 += a2 ^ i; a2 += a3 ^ i; a3 += a0 ^ i;
                    S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 S4
                    #undef S4
                }
                *sink = a0 + a1 + a2 + a3;
            }
            sh_barrier_cluster(); if (cid == 0) sh_timer_end();
            sh_barrier_cluster(); if (cid == 0) sh_timer_start();
            if (run) {
                for (uint32_t i = 0; i < 4096; i += 8) {
                    float a0, a1, a2, a3, b0, b1, b2, b3;
                    sh_h2f4((const uint16_t *)px + i, &a0, &a1, &a2, &a3); sh_h2f4((const uint16_t *)px + i + 4, &b0, &b1, &b2, &b3);
                    a0 *= 0.3f; a1 *= 0.3f; a2 *= 0.3f; a3 *= 0.3f; b0 *= 0.3f; b1 *= 0.3f; b2 *= 0.3f; b3 *= 0.3f;
                    sh_f2h4((uint16_t *)py + i, a0, a1, a2, a3); sh_f2h4((uint16_t *)py + i + 4, b0, b1, b2, b3);
                }
            }
            sh_barrier_cluster(); if (cid == 0) sh_timer_end();
        }
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
