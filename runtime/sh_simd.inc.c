/* Loops must start on a 32-byte boundary: each core fetches through a single 32 B prefetch line,
 * so a 5-instruction loop that straddles two lines refetches both lines every iteration (measured
 * 5.5x slower than the same loop inside one line). Applies to everything compiled after this point. */
#pragma GCC optimize ("align-loops=32")
/* Snitch Xfvec fp16 SIMD: four fp16 lanes in one 64-bit FP register, loaded/stored with fld/fsd
 * (a `double` is only the bit container). The upstream GNU assembler does not know the mnemonics,
 * so the ops are emitted with `.insn r` (opcode OP = 0x33, funct3 2 = .h, 6 = .r.h which replicates
 * lane 0 of rs2). Verified lane-exact against the scalar path by tests/gvsoc/fp16cvt.
 *
 * Why SIMD: the cluster is instruction-fetch bound (one shared 8 B/cycle instruction port for the
 * three cores, 32 B single-line prefetchers, no RVC: ~1 instruction/cycle for the whole cluster on
 * straight-line code, docs/SIMULATOR_NOTES.md #7), so instructions per element is what counts and a
 * 4-lane op is 4x cheaper than the scalar fmv/fcvt path. */
typedef double sh_v4h;
typedef union { sh_v4h v; uint16_t h[4]; uint32_t w[2]; } sh_v4u;
/* Every FP <-> integer round trip through memory (lane extraction, lane assembly) uses a per-core
 * slot in TCDM, never a stack local: the stack lives in a separate memory behind the narrow AXI and
 * an fsd issued by the FP subsystem can still be in flight when the integer core's lw of the same
 * address executes (stale read; seen as wrong softmax row sums, docs/SIMULATOR_NOTES.md #8). TCDM
 * accesses complete synchronously. 64 B per core at TCDM 0x800 (below the row-op staging area). */
#define SH_SIMD_SCRATCH_BASE 0x800u
static inline volatile sh_v4u *sh_v4_scratch(void) { return (volatile sh_v4u *)local(SH_SIMD_SCRATCH_BASE + flex_get_core_id() * 64u); }
/* Ordering fences between the two units (the integer core and its FP subsystem are separate masters
 * on the memory; neither the barrier CSR nor program order waits for the other side's stores):
 *  - FP -> int: an fmv.x.w whose integer result the following loads depend on. The subsystem
 *    completes offloaded instructions in order, so the value arrives only after every earlier fsd
 *    has retired. sh_fp_fence() alone drains the FP subsystem (call it before a barrier that hands
 *    FP-stored data to the DMA or to another core).
 *  - int -> FP: a read-back of the last stored word whose value feeds the fld address, so the fld
 *    cannot issue before the integer store has completed. */
static inline void sh_fp_fence(void) { uint32_t rb; float z = 0.f; __asm__ volatile ("fmv.x.w %0, %1" : "=r"(rb) : "f"(z)); __asm__ volatile ("" :: "r"(rb)); }
static inline volatile sh_v4u *sh_v4_after_fsd(sh_v4h v, volatile sh_v4u *u) {
    uint32_t rb, a; __asm__ volatile ("fmv.x.w %0, %1" : "=r"(rb) : "f"(v));
    __asm__ volatile ("andi %0, %1, 0\n\tadd %0, %0, %2" : "=&r"(a) : "r"(rb), "r"((uint32_t)(uintptr_t)u));
    return (volatile sh_v4u *)(uintptr_t)a;
}
static inline volatile sh_v4u *sh_v4_after_sw(volatile sh_v4u *u, uint32_t last_word_idx) {
    uint32_t rb = u->w[last_word_idx], a;
    __asm__ volatile ("andi %0, %1, 0\n\tadd %0, %0, %2" : "=&r"(a) : "r"(rb), "r"((uint32_t)(uintptr_t)u));
    return (volatile sh_v4u *)(uintptr_t)a;
}
static inline float sh_fmaxf(float a, float b) { float r; __asm__ ("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
static inline float sh_fminf(float a, float b) { float r; __asm__ ("fmin.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }

#define SH_V4_OP(name, f7, f3) \
    static inline sh_v4h name(sh_v4h a, sh_v4h b) { sh_v4h r; __asm__ (".insn r 0x33, " #f3 ", " #f7 ", %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
SH_V4_OP(sh_v4_add, 0x41, 2)  SH_V4_OP(sh_v4_sub, 0x42, 2)  SH_V4_OP(sh_v4_mul, 0x43, 2)  SH_V4_OP(sh_v4_div, 0x44, 2)
SH_V4_OP(sh_v4_min, 0x45, 2)  SH_V4_OP(sh_v4_max, 0x46, 2)
SH_V4_OP(sh_v4_add_r, 0x41, 6) SH_V4_OP(sh_v4_sub_r, 0x42, 6) SH_V4_OP(sh_v4_mul_r, 0x43, 6) SH_V4_OP(sh_v4_div_r, 0x44, 6)
SH_V4_OP(sh_v4_min_r, 0x45, 6) SH_V4_OP(sh_v4_max_r, 0x46, 6)
#undef SH_V4_OP
/* acc += a * b (vfmac.h) */
static inline sh_v4h sh_v4_mac(sh_v4h acc, sh_v4h a, sh_v4h b) { __asm__ (".insn r 0x33, 2, 0x48, %0, %1, %2" : "+f"(acc) : "f"(a), "f"(b)); return acc; }
static inline sh_v4h sh_v4_mac_r(sh_v4h acc, sh_v4h a, sh_v4h b) { __asm__ (".insn r 0x33, 6, 0x48, %0, %1, %2" : "+f"(acc) : "f"(a), "f"(b)); return acc; }
/* pack four fp32 into fp16 lanes (vfcpka.h.s lanes 0,1 + vfcpkb.h.s lanes 2,3), RNE */
static inline sh_v4h sh_v4_pack(float a, float b, float c, float d) {
    sh_v4h r;
    __asm__ (".insn r 0x33, 2, 0x58, %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
    __asm__ (".insn r 0x33, 6, 0x58, %0, %1, %2" : "+f"(r) : "f"(c), "f"(d));
    return r;
}
/* all four lanes = h (fp16 bits) / = fp16(f) */
static inline sh_v4h sh_v4_splat_h(uint32_t h) { volatile sh_v4u *u = sh_v4_scratch(); u->w[0] = u->w[1] = (h & 0xFFFFu) | (h << 16); return sh_v4_after_sw(u, 1)->v; }
static inline sh_v4h sh_v4_splat(float f) { return sh_v4_splat_h(sh_f2h(f)); }
static inline float sh_v4_lane(sh_v4h v, int l) { volatile sh_v4u *u = sh_v4_scratch(); u->v = v; u = sh_v4_after_fsd(v, u); return sh_h2f(u->h[l]); }
static inline float sh_v4_hsum(sh_v4h v) { volatile sh_v4u *u = sh_v4_scratch(); u->v = v; u = sh_v4_after_fsd(v, u); return (sh_h2f(u->h[0]) + sh_h2f(u->h[1])) + (sh_h2f(u->h[2]) + sh_h2f(u->h[3])); }
static inline float sh_v4_hmin(sh_v4h v) { volatile sh_v4u *u = sh_v4_scratch(); u->v = v; u = sh_v4_after_fsd(v, u); return sh_fminf(sh_fminf(sh_h2f(u->h[0]), sh_h2f(u->h[1])), sh_fminf(sh_h2f(u->h[2]), sh_h2f(u->h[3]))); }
static inline float sh_v4_hmax(sh_v4h v) { volatile sh_v4u *u = sh_v4_scratch(); u->v = v; u = sh_v4_after_fsd(v, u); return sh_fmaxf(sh_fmaxf(sh_h2f(u->h[0]), sh_h2f(u->h[1])), sh_fmaxf(sh_h2f(u->h[2]), sh_h2f(u->h[3]))); }

/* 2^t on four lanes for t in [-14, 15] (fp16 normal range), ~1e-3 relative: k = round(t) by the
 * 1.5*2^10 trick, p(f) degree 3 on [-0.5, 0.5], and 2^k assembled from the bits of t + 1536
 * (= 0x6600 + k per lane) with two 32-bit integer ops per lane pair (SWAR). 17 instructions per
 * 4 lanes; no FP<->int conversions (docs/SIMULATOR_NOTES.md #6). */
typedef struct { sh_v4h c1536, c3, c2, c1, one; } sh_v4_exp2_consts;
static inline sh_v4_exp2_consts sh_v4_exp2_init(void) {
    sh_v4_exp2_consts c = { sh_v4_splat_h(0x6600), sh_v4_splat(0.054602623f), sh_v4_splat(0.24192412f), sh_v4_splat(0.69331646f), sh_v4_splat_h(0x3c00) };
    return c;
}
static inline sh_v4h sh_v4_exp2(sh_v4h t, const sh_v4_exp2_consts *c) {
    sh_v4h m = sh_v4_add_r(t, c->c1536);                      /* integer-valued, bits = 0x6600 + k per lane */
    sh_v4h f = sh_v4_sub(t, sh_v4_sub_r(m, c->c1536));        /* t - k in [-0.5, 0.5] */
    sh_v4h p = sh_v4_add_r(sh_v4_mul_r(f, c->c3), c->c2);
    p = sh_v4_add_r(sh_v4_mul(p, f), c->c1);
    p = sh_v4_add_r(sh_v4_mul(p, f), c->one);
    volatile sh_v4u *u = sh_v4_scratch(); u->v = m;           /* 2^k bits = (k + 15) << 10 = (bits - 0x65F1) << 10, both lanes of a word at once */
    u = sh_v4_after_fsd(m, u);
    u->w[0] = (u->w[0] - 0x65F165F1u) << 10; u->w[1] = (u->w[1] - 0x65F165F1u) << 10;
    return sh_v4_mul(p, sh_v4_after_sw(u, 1)->v);
}
/* Same on four vectors at once, in place. The integer round trip (fsd -> lw -> sw -> fld) crosses
 * from the FP subsystem to the integer core and back; each crossing stalls for several cycles, so
 * the four stores are issued before the first load and the four integer stores before the first
 * fld (the asm barriers), and the stall is paid once per batch instead of once per vector. */
static inline void sh_v4_exp2x4(sh_v4h *t, const sh_v4_exp2_consts *c) {
    volatile sh_v4u *u = sh_v4_scratch(); sh_v4h m[4], f[4];
    for (int i = 0; i < 4; ++i) { m[i] = sh_v4_add_r(t[i], c->c1536); u[i].v = m[i]; }
    u = sh_v4_after_fsd(m[3], u);                             /* all four fsd retired */
    for (int i = 0; i < 4; ++i) {
        const uint32_t w0 = u[i].w[0], w1 = u[i].w[1];
        f[i] = sh_v4_sub(t[i], sh_v4_sub_r(m[i], c->c1536));
        u[i].w[0] = (w0 - 0x65F165F1u) << 10; u[i].w[1] = (w1 - 0x65F165F1u) << 10;
    }
    u = sh_v4_after_sw(u, 7);                                 /* all eight sw completed */
    for (int i = 0; i < 4; ++i) {
        sh_v4h q = sh_v4_add_r(sh_v4_mul_r(f[i], c->c3), c->c2);
        q = sh_v4_add_r(sh_v4_mul(q, f[i]), c->c1);
        q = sh_v4_add_r(sh_v4_mul(q, f[i]), c->one);
        t[i] = sh_v4_mul(q, u[i].v);
    }
}
