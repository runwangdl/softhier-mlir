/* sh_attention_head: one head of o = softmax(scale * q k^T) v, fused inside one cluster's TCDM.
 *
 * The composed path (sh_transpose + sh_gemm + sh_softmax_rows + sh_gemm) round-trips HBM after
 * every step and transposes the whole K matrix up front. Here, for S <= 256, everything stays
 * resident in L1 (S=256, dh=64: q 32 KB, k 32 KB, kT 32 KB, scores 128 KB, v 32 KB, o 32 KB,
 * 1 KB row sums = 289 KB of the 1 MB):
 *
 *   DM core   : one 2-D DMA load each for q_h, k_h, v_h; ZOMEM-clear of the two RedMulE outputs;
 *               k_h -> k_h^T in L1 as dh element-granular 2-D DMA transfers (one per kT row)
 *   first core: RedMulE  E[S,S]  += q[S,dh] . kT[dh,S]         config(S, dh, S)
 *   all cores : rows dealt round-robin: m_i = max_j E_ij, E_ij <- exp(scale (E_ij - m_i)) as fp16,
 *               sum_i kept in fp32 (the normalisation is deferred to the S x dh output)
 *   first core: RedMulE  o[S,dh]  += E[S,S] . v[S,dh]          config(S, S, dh)
 *   all cores : o_i <- o_i / sum_i
 *   DM core   : per-row 1-D stores of o_h to HBM (the iDMA model has no 2-D store into HBM)
 *
 * Why the softmax looks the way it does (gvsoc Snitch model, measured 2026-10-08, cycles/element on
 * one core): a dependent TCDM load costs ~13 cycles, every FP->int move (fmv.x.w, fcvt.w.s, flt+branch)
 * is a ~15-cycle round trip, dependent FP ops cost 3-4 cycles and independent ones pipeline. So:
 *  - the row max runs on fmax.s over the fp16 bit patterns placed in the top half of an fp32 (the
 *    sign-magnitude ordering is the same, no conversion, two elements per 32-bit load, 4 chains);
 *  - exp is a branch-free 2^y with the row constants folded into one fmadd, a degree-4 polynomial
 *    (|rel err| < 5e-5, below fp16 resolution) and one fcvt.w.s round trip per element;
 *  - the unnormalised exps go back as fp16 pairs (one 32-bit store per two elements) and the
 *    1/sum_i scaling is applied to o (S x dh) instead of E (S x S);
 *  - fp16 <-> fp32 is fmv.w.x + fcvt.s.h and fcvt.h.s + fmv.x.w: flh/fsh are broken in the simulator
 *    (docs/SIMULATOR_NOTES.md #2), the conversions are not, and this is ~4x cheaper than the
 *    software sh_fp16_to_f32 / sh_f32_to_fp16 pair.
 * Constraints: S <= 256, S % 4 == 0, dh % 2 == 0, L1 budget (sh_attention_l1_bytes).
 */
#define SH_ATTN_MAX_S 256u
#ifndef SH_ATTN_KT_DMA
#define SH_ATTN_KT_DMA 1   /* 1: k^T by element-granular 2-D DMA on the DM core; 0: on the cores */
#endif
#define SH_ATTN_NPROF 8u   /* phase cycle stamps (first core, mcycle) kept in TCDM behind the row sums */

/* sh_h2f / sh_f2h: the hardware Zfh register conversions from sh_ops.h */
static inline float sh_fbits(uint32_t bits) { float f; asm volatile("fmv.w.x %0, %1" : "=f"(f) : "r"(bits)); return f; }
static inline uint32_t sh_bitsf(float f) { uint32_t b; asm volatile("fmv.x.w %0, %1" : "=r"(b) : "f"(f)); return b; }
static inline float sh_fmax(float a, float b) { float r; asm("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }

/* 2^(x*c1 + c0) for x*c1 + c0 <= 0 (c1 = scale*log2(e), c0 = -m*c1). k = round(y) by the 1.5*2^23 trick, so
 * kf = t - 1.5*2^23 is exact and r = y - kf is in [-0.5, 0.5]; 2^r by a degree-4 polynomial; 2^k by building
 * the exponent field on the integer side from the low bits of t. The integer side only ever consumes the FP->int
 * move with ALU ops: in this gvsoc Snitch model an FP instruction that directly reads the integer result of a
 * preceding FP->int move (fcvt.w.s then fcvt.s.w) gets a stale register value. */
static inline float sh_attn_exp2(float x, float c1, float c0) {
    const float y = sh_fmax(x * c1 + c0, -126.f);
    float t = y + 12582912.f;                              /* 1.5 * 2^23: low mantissa bits hold round(y) */
    asm("" : "+f"(t));                                     /* opaque to -ffast-math: no (y + C) - C -> y, */
    float kf = t - 12582912.f;                             /* no y - (t - C) -> (y - t) + C reassociation */
    asm("" : "+f"(kf));
    const float r = y - kf;
    const float p = 1.f + r * (0.69314718f + r * (0.24022651f + r * (0.05550411f + r * 0.00961813f)));
    const uint32_t e = (sh_bitsf(t) - (0x4B400000u - 127u)) << 23;   /* (k + 127) << 23 */
    return p * sh_fbits(e);
}

typedef struct { uint32_t q, k, kt, s, v, o, sum, prof, end; } sh_attn_l1;

static inline sh_attn_l1 sh_attn_layout(uint32_t S, uint32_t dh, uint32_t base) {
    sh_attn_l1 l;
    const uint32_t qb = S * dh * 2, sb = S * S * 2;
    l.q = base; l.k = l.q + qb; l.kt = l.k + qb; l.s = l.kt + qb; l.v = l.s + sb; l.o = l.v + qb;
    l.sum = l.o + qb; l.prof = l.sum + S * 4; l.end = l.prof + SH_ATTN_NPROF * 4;
    return l;
}

uint32_t sh_attention_l1_bytes(uint32_t S, uint32_t dh) { return sh_attn_layout(S, dh, 0).end; }

static inline uint32_t sh_mcycle(void) { uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c; }
#define SH_ATTN_STAMP(i) do { if (first) ((volatile uint32_t *)local(l.prof))[i] = sh_mcycle(); } while (0)
/* Cycle stamps of the last sh_attention_head on the calling cluster: 0 entry, 1 operands staged + k transposed,
 * 2 (same), 3 scores GEMM done, 4 softmax done, 5 P.V GEMM done, 6 o normalised and stored. */
uint32_t sh_attention_profile(uint32_t S, uint32_t dh, uint32_t phase) {
    return ((volatile uint32_t *)local(sh_attn_layout(S, dh, 0).prof))[phase & (SH_ATTN_NPROF - 1)];
}

/* One row of the scores (n fp16, n % 4 == 0): E <- exp(scale (E - max)) in place; returns the fp32 row sum. */
static inline float sh_attn_softmax_row(uint32_t *xw, uint32_t n, float scale) {
    const uint32_t n2 = n >> 1;
    /* pass 1: row max on the fp16 bit patterns reinterpreted as fp32 (same ordering), 4 fmax chains */
    float m0 = sh_fbits(0xFF800000u), m1 = m0, m2 = m0, m3 = m0;
    for (uint32_t i = 0; i < n2; i += 2) {
        const uint32_t w0 = xw[i], w1 = xw[i + 1];
        m0 = sh_fmax(m0, sh_fbits(w0 << 16)); m1 = sh_fmax(m1, sh_fbits(w0 & 0xFFFF0000u));
        m2 = sh_fmax(m2, sh_fbits(w1 << 16)); m3 = sh_fmax(m3, sh_fbits(w1 & 0xFFFF0000u));
    }
    const float m = sh_h2f((uint16_t)(sh_bitsf(sh_fmax(sh_fmax(m0, m1), sh_fmax(m2, m3))) >> 16));
    /* pass 2: e = 2^((x - m) * scale * log2 e), written back as fp16 pairs, 4 partial sums */
    const float c1 = scale * 1.44269504f, c0 = -m * c1;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (uint32_t i = 0; i < n2; i += 2) {
        const uint32_t w0 = xw[i], w1 = xw[i + 1];
        const float e0 = sh_attn_exp2(sh_h2f((uint16_t)w0), c1, c0), e1 = sh_attn_exp2(sh_h2f((uint16_t)(w0 >> 16)), c1, c0);
        const float e2 = sh_attn_exp2(sh_h2f((uint16_t)w1), c1, c0), e3 = sh_attn_exp2(sh_h2f((uint16_t)(w1 >> 16)), c1, c0);
        s0 += e0; s1 += e1; s2 += e2; s3 += e3;
        xw[i] = (sh_f2h(e0) & 0xFFFFu) | (sh_f2h(e1) << 16);
        xw[i + 1] = (sh_f2h(e2) & 0xFFFFu) | (sh_f2h(e3) << 16);
    }
    return (s0 + s1) + (s2 + s3);
}

/* o row (dh fp16, dh % 2 == 0) *= inv */
static inline void sh_attn_scale_row(uint32_t *ow, uint32_t dh, float inv) {
    for (uint32_t i = 0; i < (dh >> 1); ++i) {
        const uint32_t w = ow[i];
        const float a = sh_h2f((uint16_t)w) * inv, b = sh_h2f((uint16_t)(w >> 16)) * inv;
        ow[i] = (sh_f2h(a) & 0xFFFFu) | (sh_f2h(b) << 16);
    }
}

int sh_attention_head(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t dh,
                      uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t cluster) {
    if (cluster != SH_ALL && flex_get_cluster_id() != cluster) return 0;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id(), NC = ARCH_NUM_CORE_PER_CLUSTER;
    const sh_attn_l1 l = sh_attn_layout(S, dh, 0);
    if (S > SH_ATTN_MAX_S || (S & 3) || (dh & 1) || l.end > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_attention_head] S=%u dh=%u: need S <= %u, S %% 4 == 0, dh %% 2 == 0, L1 %u <= %u\n",
                             S, dh, SH_ATTN_MAX_S, l.end, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -1;
    }
    SH_ATTN_STAMP(0);
    /* 1. stage the operands, clear the two RedMulE accumulators, transpose k in L1 by DMA */
    if (dm) {
        sh_load_block_async(l.q, q, S, dh, ldq);
        sh_load_block_async(l.k, k, S, dh, ldk);
        sh_load_block_async(l.v, v, S, dh, ldv);
        sh_l1_zero_dm(l.s, S * S * 2);          /* waits for everything issued so far */
        sh_l1_zero_dm(l.o, S * dh * 2);
#if SH_ATTN_KT_DMA
        for (uint32_t c = 0; c < dh; ++c)        /* kT row c = column c of k: S elements of 2 B, stride dh*2 */
            bare_dma_start_2d(local(l.kt + c * S * 2), local(l.k + c * 2), 2, 2, dh * 2, S);
        bare_dma_wait_all();
#endif
    }
    flex_intra_cluster_sync();
#if !SH_ATTN_KT_DMA
    {   /* k -> k^T on the cores: core c transposes columns c, c+NC, ... (each is one contiguous kT row) */
        const uint16_t *ks = (const uint16_t *)local(l.k); uint16_t *kt = (uint16_t *)local(l.kt);
        for (uint32_t c = core; c < dh; c += NC) {
            uint16_t *row = kt + c * S;
            for (uint32_t r = 0; r < S; ++r) row[r] = ks[r * dh + c];
        }
    }
    flex_intra_cluster_sync();
#endif
    SH_ATTN_STAMP(1); SH_ATTN_STAMP(2);
    /* 2. E = q . kT on RedMulE */
    if (first) { flex_redmule_config(S, dh, S); flex_redmule_trigger(l.q, l.kt, l.s, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_ATTN_STAMP(3);
    /* 3. row softmax numerators on all cores (rows dealt round-robin), row sums to L1 */
    {
        uint32_t *sc = (uint32_t *)local(l.s); float *sum = (float *)local(l.sum);
        for (uint32_t r = core; r < S; r += NC) sum[r] = sh_attn_softmax_row(sc + r * (S >> 1), S, scale);
    }
    flex_intra_cluster_sync();
    SH_ATTN_STAMP(4);
    /* 4. o = E . v on RedMulE */
    if (first) { flex_redmule_config(S, S, dh); flex_redmule_trigger(l.s, l.v, l.o, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
    SH_ATTN_STAMP(5);
    /* 5. o_i /= sum_i, then store o_h */
    {
        uint32_t *ow = (uint32_t *)local(l.o); const float *sum = (const float *)local(l.sum);
        for (uint32_t r = core; r < S; r += NC) sh_attn_scale_row(ow + r * (dh >> 1), dh, 1.f / sum[r]);
    }
    flex_intra_cluster_sync();
    if (dm) sh_store_block_sync(o, l.o, S, dh, ldo);
    flex_intra_cluster_sync();
    SH_ATTN_STAMP(6);
    return 0;
}

/* Multi-head: head h = columns [h*dh, (h+1)*dh) of q/k/v/o (D = H*dh). cluster == SH_ALL deals head h
 * to cluster h % P and ends with a global barrier; otherwise the one cluster runs all heads. */
int sh_attention(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t D, uint32_t H,
                 uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t cluster) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, dh = H ? D / H : 0;
    int rc = 0;
    if (H == 0 || dh * H != D) {
        if (flex_get_cluster_id() == 0 && flex_is_first_core()) sh_printf("[sh_attention] D=%u not divisible by H=%u\n", D, H);
        return -1;
    }
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t cl = (cluster == SH_ALL) ? h % P : cluster;
        const uint64_t off = (uint64_t)h * dh * 2;
        int r = sh_attention_head(q + off, k + off, v + off, o + off, S, dh, ldq, ldk, ldv, ldo, scale, cl);
        if (r) rc = r;
    }
    if (cluster == SH_ALL) flex_global_barrier_xy();
    return rc;
}
