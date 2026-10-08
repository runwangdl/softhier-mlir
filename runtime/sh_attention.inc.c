/* Fused attention: o = softmax(scale * q k^T) v, one (head, q-block) work item at a time inside one
 * cluster's TCDM, every head's K / K^T / V resident while its q blocks are processed.
 *
 * The composed path (sh_transpose + sh_gemm + sh_softmax_rows + sh_gemm) round-trips HBM after every
 * step. Here nothing but q, k, v in and o out touches HBM; the S x S scores never leave L1. A work item
 * is sq rows of one head (sq | S): the scores block is sq x S, so the row softmax sees complete rows (no
 * online rescaling) and S is bounded only by the L1 budget (S = 1024, dh = 64, sq = 256: 961 KB).
 * TCDM layout from SH_ATTN_L1_BASE (above TCDM 0 = NULL and the per-core SIMD scratch at 0x800):
 *   q [sq,dh] | k [S,dh] | kT [dh,S] | s [sq,S] | v [S,dh] | o [sq,dh] | sum [sq] fp32 | profile stamps
 *
 *   DM core   : 2-D DMA loads of k_h, v_h (once per head), k_h -> k_h^T in L1 as dh element-granular 2-D
 *               DMA transfers (one per kT row, ~1 cycle/element: the staging phase of a 256 x 64 head is
 *               18.4 k cycles, 30.9 k with the in-core loop of SH_ATTN_KT_DMA=0, because the cluster is
 *               instruction-fetch bound), q block load, ZOMEM clear of the two RedMulE outputs
 *   first core: RedMulE  s[sq,S]  += q[sq,dh] . kT[dh,S]       config(sq, dh, S)
 *   all cores : rows dealt round-robin: s_i <- 2^(s2 s_i - m_i) in fp16 SIMD (m_i = s2 max_j s_ij,
 *               s2 = scale log2 e), sum_i in fp32; the normalisation is deferred to the sq x dh output
 *   first core: RedMulE  o[sq,dh] += s[sq,S] . v[S,dh]         config(sq, S, dh)
 *   all cores : o_i <- o_i * (1 / sum_i) in fp16 SIMD
 *   DM core   : per-row 1-D stores of the o block (the iDMA model has no 2-D store into HBM)
 *
 * The softmax is the fp16 SIMD row kernel of sh_rowops.inc.c (sh_v4_exp2x4: four fp16 lanes per
 * instruction, 2^k assembled on the integer side, docs/SIMULATOR_NOTES.md #6-#8 for the FP/int fences):
 * the cluster executes ~1 instruction/cycle in total whatever the number of cores (#7), so instructions
 * per element is the cost: ~9 here vs ~30 for the scalar fp32 version this replaces (2.0 M -> 0.6 M
 * cycles per 256 x 256 head).
 *
 * Work items (H heads x S/sq blocks, head-major) are dealt in contiguous chunks over the clusters when
 * cluster == SH_ALL, so a cluster re-stages K/V for at most two heads. sq is a policy knob
 * (sh_attention_q); sh_attention_q_block() is the default rule: the largest sq in {S, 256, 128, 64}
 * that fits L1 and minimises the rows per cluster, ceil(items / P) * sq (S = 256, 12 heads, 16
 * clusters: sq = 64 -> 48 items, 3 per cluster = 192 rows instead of 256 on 12 clusters).
 * Constraints: S % 4 == 0, dh % 4 == 0, sq | S, sq % 4 == 0, L1 budget (sh_attention_l1_bytes_q).
 */
#ifndef SH_ATTN_KT_DMA
#define SH_ATTN_KT_DMA 1   /* 1: k^T by element-granular 2-D DMA on the DM core; 0: on the cores */
#endif
#define SH_ATTN_L1_BASE 0x1000u
#define SH_ATTN_NPROF 8u   /* phase cycle stamps (first core, mcycle) kept in TCDM behind the row sums */

typedef struct { uint32_t q, k, kt, s, v, o, sum, prof, end; } sh_attn_l1;

static inline sh_attn_l1 sh_attn_layout(uint32_t S, uint32_t dh, uint32_t sq, uint32_t base) {
    sh_attn_l1 l;
    const uint32_t kb = S * dh * 2, qb = sq * dh * 2, sb = sq * S * 2;
    l.q = base; l.k = l.q + qb; l.kt = l.k + kb; l.s = l.kt + kb; l.v = l.s + sb; l.o = l.v + kb;
    l.sum = l.o + qb; l.prof = l.sum + sq * 4; l.end = l.prof + SH_ATTN_NPROF * 4;
    return l;
}

uint32_t sh_attention_l1_bytes_q(uint32_t S, uint32_t dh, uint32_t sq) { return sh_attn_layout(S, dh, sq ? sq : S, SH_ATTN_L1_BASE).end; }
uint32_t sh_attention_l1_bytes(uint32_t S, uint32_t dh) { return sh_attention_l1_bytes_q(S, dh, S); }

/* Default q-block rule (the compiler may override it through sh_attention_q): candidates S, 256, 128, 64 */
uint32_t sh_attention_q_block(uint32_t S, uint32_t dh, uint32_t H, uint32_t P) {
    const uint32_t cand[4] = { S, 256, 128, 64 };
    uint32_t best = 0, best_rows = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sq = cand[i];
        if (sq == 0 || sq > S || S % sq || sq % 4 || sh_attention_l1_bytes_q(S, dh, sq) > ARCH_CLUSTER_TCDM_SIZE) continue;
        const uint32_t items = H * (S / sq), rows = ((items + P - 1) / P) * sq;
        if (!best || rows < best_rows) { best = sq; best_rows = rows; }
    }
    return best;   /* 0: nothing fits */
}

static inline uint32_t sh_mcycle(void) { uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c; }
/* Cycle stamps of the last work item on the calling cluster: 0 item start, 1 operands staged (+ k transposed when
 * this item started a new head), 2 (same), 3 scores GEMM done, 4 softmax done, 5 P.V GEMM done, 6 o normalised and
 * stored; 7 = entry of the sh_attention call on this cluster (so 6 - 7 is the cluster's whole time). */
uint32_t sh_attention_profile(uint32_t S, uint32_t dh, uint32_t phase) {
    /* the stamps live behind the row sums of the layout actually used; store their offset at a fixed place */
    (void)S; (void)dh;
    const uint32_t off = *(volatile uint32_t *)local(SH_ATTN_L1_BASE - 4);
    return ((volatile uint32_t *)local(off))[phase & (SH_ATTN_NPROF - 1)];
}

/* softmax constants, built once per kernel call (each splat is a scratch round trip) */
typedef struct { sh_v4_exp2_consts ec; sh_v4h s24, cm14, zero, ninf, pinf; float s2; } sh_attn_consts;

/* One scores row (n = 4 cv fp16, in place): x <- 2^(s2 x - m2), m2 = s2 * max(x) (min for s2 < 0), t clamped
 * to >= -14 (fp16 normal range); returns the fp32 row sum of the numerators. Lanes hold sums of <= 8 terms. */
static inline float sh_attn_softmax_row(sh_v4h *x, uint32_t cv, const sh_attn_consts *c) {
    uint32_t j; float m2;
    if (c->s2 >= 0.f) {
        sh_v4h m0 = c->ninf, m1 = m0;
        for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_max(m0, x[j]); m1 = sh_v4_max(m1, x[j + 1]); }
        if (j < cv) m0 = sh_v4_max(m0, x[j]);
        m2 = sh_v4_hmax(sh_v4_max(m0, m1)) * c->s2;
    } else {
        sh_v4h m0 = c->pinf, m1 = m0;
        for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_min(m0, x[j]); m1 = sh_v4_min(m1, x[j + 1]); }
        if (j < cv) m0 = sh_v4_min(m0, x[j]);
        m2 = sh_v4_hmin(sh_v4_min(m0, m1)) * c->s2;
    }
    const sh_v4h m24 = sh_v4_splat(m2);
    sh_v4h acc = c->zero; float sum = 0.f;
    #define SH_AT_T(v) sh_v4_max_r(sh_v4_sub_r(sh_v4_mul_r(v, c->s24), m24), c->cm14)
    for (j = 0; j + 4 <= cv; j += 4) {
        sh_v4h t[4] = { SH_AT_T(x[j]), SH_AT_T(x[j + 1]), SH_AT_T(x[j + 2]), SH_AT_T(x[j + 3]) };
        sh_v4_exp2x4(t, &c->ec);
        x[j] = t[0]; x[j + 1] = t[1]; x[j + 2] = t[2]; x[j + 3] = t[3];
        acc = sh_v4_add(sh_v4_add(acc, sh_v4_add(t[0], t[1])), sh_v4_add(t[2], t[3]));
        if ((j & 4) == 4) { sum += sh_v4_hsum(acc); acc = c->zero; }
    }
    for (; j < cv; ++j) { sh_v4h e = sh_v4_exp2(SH_AT_T(x[j]), &c->ec); x[j] = e; acc = sh_v4_add(acc, e); }
    #undef SH_AT_T
    return sum + sh_v4_hsum(acc);
}

/* All work items [i0, i1) (item = head * nb + block) on the calling cluster. */
static int sh_attn_items(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t dh, uint32_t sq,
                         uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t i0, uint32_t i1) {
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id(), NC = ARCH_NUM_CORE_PER_CLUSTER, nb = S / sq;
    const sh_attn_l1 l = sh_attn_layout(S, dh, sq, SH_ATTN_L1_BASE);
    volatile uint32_t *prof = (volatile uint32_t *)local(l.prof);
    #define SH_ATTN_STAMP(i) do { if (first) prof[i] = sh_mcycle(); } while (0)
    if (first) {   /* a cluster without items leaves all stamps equal (profile reads 0 cycles) */
        *(volatile uint32_t *)local(SH_ATTN_L1_BASE - 4) = l.prof;
        const uint32_t now = sh_mcycle();
        for (uint32_t i = 0; i < SH_ATTN_NPROF; ++i) prof[i] = now;
    }
    /* constants once per call (not per row): every splat is an FP <-> integer round trip */
    sh_attn_consts c;
    c.ec = sh_v4_exp2_init(); c.s2 = scale * SH_LOG2E;
    c.s24 = sh_v4_splat(c.s2); c.cm14 = sh_v4_splat_h(SH_CM14); c.zero = sh_v4_splat_h(0);
    c.ninf = sh_v4_splat_h(0xFC00u); c.pinf = sh_v4_splat_h(0x7C00u);
    uint32_t cur_head = 0xFFFFFFFFu;
    for (uint32_t it = i0; it < i1; ++it) {
        const uint32_t h = it / nb, r0 = (it % nb) * sq;
        const uint64_t hoff = (uint64_t)h * dh * 2;
        SH_ATTN_STAMP(0);
        if (dm) {
            if (h != cur_head) {                      /* new head: stage k, v and transpose k in L1 */
                sh_load_block_async(l.k, k + hoff, S, dh, ldk);
                sh_load_block_async(l.v, v + hoff, S, dh, ldv);
                bare_dma_wait_all();
#if SH_ATTN_KT_DMA
                for (uint32_t cc = 0; cc < dh; ++cc)   /* kT row cc = column cc of k: S elements of 2 B, stride dh*2 */
                    bare_dma_start_2d(local(l.kt + cc * S * 2), local(l.k + cc * 2), 2, 2, dh * 2, S);
#endif
            }
            sh_load_block_async(l.q, q + hoff + (uint64_t)r0 * ldq * 2, sq, dh, ldq);
            sh_l1_zero_dm(l.s, sq * S * 2);           /* waits for everything issued so far */
            sh_l1_zero_dm(l.o, sq * dh * 2);
        }
        flex_intra_cluster_sync();
#if !SH_ATTN_KT_DMA
        if (h != cur_head) {   /* k -> k^T on the cores: core c transposes columns c, c+NC, ... (each one contiguous kT row) */
            const uint16_t *ks = (const uint16_t *)local(l.k); uint16_t *kt = (uint16_t *)local(l.kt);
            for (uint32_t cc = core; cc < dh; cc += NC) {
                uint16_t *row = kt + cc * S;
                for (uint32_t r = 0; r < S; ++r) row[r] = ks[r * dh + cc];
            }
            flex_intra_cluster_sync();
        }
#endif
        cur_head = h;
        SH_ATTN_STAMP(1); SH_ATTN_STAMP(2);
        /* scores = q . kT on RedMulE */
        if (first) { flex_redmule_config(sq, dh, S); flex_redmule_trigger(l.q, l.kt, l.s, REDMULE_FP_16); flex_redmule_wait(); }
        flex_intra_cluster_sync();
        SH_ATTN_STAMP(3);
        /* row softmax numerators (fp16 SIMD, rows dealt round-robin over the cores), fp32 row sums */
        {
            sh_v4h *sc = (sh_v4h *)local(l.s); float *sum = (float *)local(l.sum);
            const uint32_t cv = S >> 2;
            for (uint32_t r = core; r < sq; r += NC) sum[r] = sh_attn_softmax_row(sc + r * cv, cv, &c);
        }
        sh_fp_fence();                                /* this core's fsd results are in TCDM before RedMulE reads them */
        flex_intra_cluster_sync();
        SH_ATTN_STAMP(4);
        /* o = P . v on RedMulE */
        if (first) { flex_redmule_config(sq, S, dh); flex_redmule_trigger(l.s, l.v, l.o, REDMULE_FP_16); flex_redmule_wait(); }
        flex_intra_cluster_sync();
        SH_ATTN_STAMP(5);
        /* o_i *= 1 / sum_i (fp16 SIMD), then store the o block */
        {
            sh_v4h *ov = (sh_v4h *)local(l.o); const float *sum = (const float *)local(l.sum);
            const uint32_t dv = dh >> 2;
            for (uint32_t r = core; r < sq; r += NC) {
                const sh_v4h inv4 = sh_v4_splat(1.f / sum[r]);
                sh_v4h *row = ov + r * dv;
                for (uint32_t j = 0; j < dv; ++j) row[j] = sh_v4_mul_r(row[j], inv4);
            }
        }
        sh_fp_fence();
        flex_intra_cluster_sync();
        if (dm) sh_store_block_sync(o + hoff + (uint64_t)r0 * ldo * 2, l.o, sq, dh, ldo);
        flex_intra_cluster_sync();
        SH_ATTN_STAMP(6);
    }
    #undef SH_ATTN_STAMP
    return 0;
}

static int sh_attn_check(uint32_t S, uint32_t dh, uint32_t sq, int first) {
    if ((S & 3) || (dh & 3) || sq == 0 || (sq & 3) || S % sq || sh_attention_l1_bytes_q(S, dh, sq) > ARCH_CLUSTER_TCDM_SIZE) {
        if (first) sh_printf("[sh_attention] S=%u dh=%u sq=%u: need S %% 4 == 0, dh %% 4 == 0, sq | S, sq %% 4 == 0, L1 %u <= %u\n",
                             S, dh, sq, sh_attention_l1_bytes_q(S, dh, sq ? sq : 1), (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        return -1;
    }
    return 0;
}

/* One head on one cluster (all q blocks of the head; q block = the largest that fits). */
int sh_attention_head(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t dh,
                      uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t cluster) {
    if (cluster != SH_ALL && flex_get_cluster_id() != cluster) return 0;
    const uint32_t sq = sh_attention_q_block(S, dh, 1, 1);
    if (sh_attn_check(S, dh, sq, flex_is_first_core())) return -1;
    return sh_attn_items(q, k, v, o, S, dh, sq, ldq, ldk, ldv, ldo, scale, 0, S / sq);
}

/* Multi-head: head h = columns [h*dh, (h+1)*dh) of q/k/v/o (D = H*dh), q blocks of sq rows (0 = default rule).
 * cluster == SH_ALL deals the H * S/sq work items in contiguous chunks over the clusters and ends with a global
 * barrier; otherwise the one cluster runs all of them. Call from all cores of all clusters. */
int sh_attention_q(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t D, uint32_t H,
                   uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t cluster, uint32_t sq) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, cid = flex_get_cluster_id(), dh = H ? D / H : 0;
    const int lead = cid == 0 && flex_is_first_core();
    if (H == 0 || dh * H != D) {
        if (lead) sh_printf("[sh_attention] D=%u not divisible by H=%u\n", D, H);
        return -1;
    }
    if (!sq) sq = sh_attention_q_block(S, dh, H, cluster == SH_ALL ? P : 1);
    if (sh_attn_check(S, dh, sq, lead)) return -1;
    const uint32_t n = H * (S / sq);
    int rc = 0;
    if (cluster == SH_ALL) {
        rc = sh_attn_items(q, k, v, o, S, dh, sq, ldq, ldk, ldv, ldo, scale, n * cid / P, n * (cid + 1) / P);
        flex_global_barrier_xy();
    } else if (cid == cluster) {
        rc = sh_attn_items(q, k, v, o, S, dh, sq, ldq, ldk, ldv, ldo, scale, 0, n);
    }
    return rc;
}

int sh_attention(uint64_t q, uint64_t k, uint64_t v, uint64_t o, uint32_t S, uint32_t D, uint32_t H,
                 uint32_t ldq, uint32_t ldk, uint32_t ldv, uint32_t ldo, float scale, uint32_t cluster) {
    return sh_attention_q(q, k, v, o, S, D, H, ldq, ldk, ldv, ldo, scale, cluster, 0);
}
