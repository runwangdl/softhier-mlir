/* Flow-matching dataflow kernels of the SmolVLA action expert (prefix sh_f_); docs/FLOW_DATAFLOW.md.
 *
 * 1. KV-stationary attention (research direction R3). The prefix KV of the 16 expert layers (self layers: the VLM
 *    prefix K/V; cross layers: their once-per-chunk projections) is constant over the 10 flow steps. sh_f_kvs_deal
 *    deals it once per chunk into the clusters' TCDM: cluster c = g * NPART + p (g = kv head, p = key part, NPART = the
 *    GQA group size = 3, 15 clusters) keeps, for every layer, key rows [p * RP, min((p + 1) * RP, Lp)) of kv head g
 *    (RP = Lp / NPART rounded up to 32), K already transposed (kT [dh, Lpad]) and V ([Lpad, dh]); the self layers' own
 *    keys (50 rows, recomputed every step) are appended to the last part. Per step and layer, sh_f_attention_kvs:
 *      a. cluster 0 loads the query block q [Sq, H * dh] once from HBM and multicasts it to every cluster
 *         (in-network broadcast, line rate);
 *      b. cluster (g, p) stacks the GROUP query heads of kv head g ([GROUP * Sq, dh]), computes E = Q kT_p on RedMulE,
 *         a masked row softmax over its key part (row max m and row sum l kept per row), O_p = E V_p (unnormalised);
 *      c. global barrier; cluster (g, p) gathers head (g * GROUP + p)'s rows of the NPART partials (O, m, l) from its
 *         partner clusters' TCDM over the NoC (remote iDMA reads, no HBM), combines them
 *         o = sum_j 2^(m_j - M) O_j / sum_j 2^(m_j - M) l_j  (M = max_j m_j, log2 domain), and stores o to HBM.
 *    HBM per step and layer: q once (96 KB), own k / v (self layers), o; the KV itself never leaves TCDM.
 *    TCDM: the resident KV lives at SH_F_RES_BASE..1 MB (<= 393 KB per cluster for SmolVLA); every other kernel of the
 *    expert program stays below it (row ops < 0x81000, GEMM tiles < 0x67000, this kernel's scratch < 0x40000).
 *
 * 2. Real fp8 (e4m3) steps of the weight GEMMs (research direction R4); see the fp8 section below. */
#define SH_F_L1_BASE 0x1000u
#define SH_F_RES_BASE 0x90000u
#define SH_F_NPROF 8u

static inline uint32_t sh_f_up64(uint32_t b) { return (b + 63u) & ~63u; }

/* RedMulE fp16 trigger in ONE asm statement. The SDK's flex_redmule_trigger sets t0 / t1 / t2 in three separate asm
 * statements that only clobber them; once inlined, GCC is free to use t0 as a temporary between them (seen: the
 * address of a stack array element computed into t0 after `addi t0, x`, so RedMulE read X from a stack address and
 * the TCDM reported an out-of-bound request; docs/SIMULATOR_NOTES.md #12). Clobbered registers cannot hold the
 * inputs, so this form is safe. Encoding = the SDK's REDMULE_FP_16 word (rs1 t0, rs2 t1, rs3 t2, op 011). */
static inline void sh_f_redmule_fp16(uint32_t x, uint32_t w, uint32_t y) {
    __asm__ volatile ("mv t0, %0\n\tmv t1, %1\n\tmv t2, %2\n\t.word 0x386281aa" :: "r"(x), "r"(w), "r"(y) : "t0", "t1", "t2", "memory");
}
static inline uint32_t sh_f_up32(uint32_t n) { return (n + 31u) & ~31u; }

/* ---- geometry of the KV-stationary deal ---------------------------------------------------- */
typedef struct { uint32_t r0, np, own, L, Lpad; } sh_f_part;     /* prefix rows [r0, r0 + np), own keys, L = np + own */
static inline sh_f_part sh_f_part_of(uint32_t p, uint32_t npart, uint32_t Lp, uint32_t So, int self_layer) {
    sh_f_part s; const uint32_t rp = sh_f_up32((Lp + npart - 1) / npart);
    s.r0 = p * rp; if (s.r0 > Lp) s.r0 = Lp;
    s.np = (s.r0 + rp <= Lp) ? rp : Lp - s.r0;
    s.own = (self_layer && p == npart - 1) ? So : 0;
    s.L = s.np + s.own; s.Lpad = (s.L + 15u) & ~15u; if (!s.Lpad) s.Lpad = 16;   /* softmax: cv % 4 == 0 */
    return s;
}
/* TCDM offset of layer l's resident block (kT [dh, Lpad] then V [Lpad, dh]) for part p */
static inline uint32_t sh_f_res_off(uint32_t l, uint32_t p, uint32_t npart, uint32_t Lp, uint32_t So, uint32_t dh, uint32_t selfmask) {
    uint32_t off = SH_F_RES_BASE;
    for (uint32_t i = 0; i < l; ++i) off += 2u * dh * sh_f_part_of(p, npart, Lp, So, (selfmask >> i) & 1u).Lpad * 2u;
    return off;
}
uint32_t sh_f_kvs_resident_bytes(uint32_t nlayers, uint32_t Lp, uint32_t So, uint32_t dh, uint32_t npart, uint32_t selfmask) {
    uint32_t mx = 0;
    for (uint32_t p = 0; p < npart; ++p) { uint32_t b = sh_f_res_off(nlayers, p, npart, Lp, So, dh, selfmask) - SH_F_RES_BASE; if (b > mx) mx = b; }
    return mx;
}

/* scratch of sh_f_attention_kvs (all offsets below SH_F_RES_BASE) */
typedef struct { uint32_t qb, qs, s, o, m, l, kst, tok, vld, c, cm, cl, out, prof, end; } sh_f_kvs_l1;
static inline sh_f_kvs_l1 sh_f_kvs_layout(uint32_t Sq, uint32_t H, uint32_t grp, uint32_t dh, uint32_t Lpmax) {
    sh_f_kvs_l1 a; const uint32_t R = grp * Sq, NC = ARCH_NUM_CORE_PER_CLUSTER;
    a.qb = SH_F_L1_BASE;                        a.qs = a.qb + sh_f_up64(Sq * H * dh * 2);
    a.s = a.qs + sh_f_up64(R * dh * 2);         a.o = a.s + sh_f_up64(R * Lpmax * 2);
    a.m = a.o + sh_f_up64(R * dh * 2);          a.l = a.m + sh_f_up64(R * 4);
    a.kst = a.l + sh_f_up64(R * 4);             a.tok = a.kst + sh_f_up64((Sq > Lpmax ? Sq : Lpmax) * dh * 2);
    a.vld = a.tok + sh_f_up64(Lpmax * 2);       a.c = a.vld + NC * 2 * sh_f_up64(Lpmax * 2);
    a.cm = a.c + sh_f_up64(grp * Sq * dh * 2);  a.cl = a.cm + sh_f_up64(grp * Sq * 4);
    a.out = a.cl + sh_f_up64(grp * Sq * 4);     a.prof = a.out + sh_f_up64(Sq * dh * 2);
    a.end = a.prof + sh_f_up64(SH_F_NPROF * 4);
    return a;
}
uint32_t sh_f_kvs_profile(uint32_t phase) { return ((volatile uint32_t *)local(SH_F_RES_BASE - 64u))[phase & (SH_F_NPROF - 1)]; }
#define SH_F_STAMP(i) do { if (first) ((volatile uint32_t *)local(SH_F_RES_BASE - 64u))[i] = sh_mcycle(); } while (0)

/* kT[:, col0 .. col0 + n) = K^T of n packed rows [n, dh] at TCDM `src` (one element-granular 2-D DMA per kT row) */
static inline void sh_f_transpose_dm(uint32_t kt, uint32_t Lpad, uint32_t col0, uint32_t src, uint32_t n, uint32_t dh) {
    for (uint32_t c = 0; c < dh; ++c)
        bare_dma_start_2d(local(kt + (c * Lpad + col0) * 2), local(src + c * 2), 2, 2, dh * 2, n);
}

/* Once per chunk, all cores of all clusters: deal the prefix KV of `nlayers` layers into the clusters' TCDM.
 * Layer l (self when bit l of selfmask is set): keys at (self ? ks0 : kx0) + (l / 2) * (self ? sstride : xstride), values
 * at vs0 / vx0 + the same, row-major [Lp, Hkv * dh] with leading dimension ld (elements). */
int sh_f_kvs_deal(uint64_t ks0, uint64_t vs0, uint64_t sstride, uint64_t kx0, uint64_t vx0, uint64_t xstride,
                  uint32_t nlayers, uint32_t selfmask, uint32_t Lp, uint32_t So, uint32_t Hkv, uint32_t grp, uint32_t dh, uint32_t ld) {
    const uint32_t P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y, cid = flex_get_cluster_id(), npart = grp;
    const uint32_t res = sh_f_kvs_resident_bytes(nlayers, Lp, So, dh, npart, selfmask);
    if (Hkv * npart > P || SH_F_RES_BASE + res > ARCH_CLUSTER_TCDM_SIZE) {
        if (cid == 0 && flex_is_first_core()) sh_printf("[sh_f_kvs_deal] %u kv heads x %u parts on %u clusters, resident %u B: does not fit\n", Hkv, npart, P, res);
        return -1;
    }
    if (flex_is_first_core()) ((volatile uint32_t *)local(SH_F_RES_BASE - 64u))[7] = 0;     /* call counter of the phase print */
    if (cid == 0 && flex_is_first_core()) sh_printf("[sh_f_kvs_deal] resident KV %u B per cluster (%u layers, %u kv heads x %u key parts)\n", res, nlayers, Hkv, npart);
    if (cid < Hkv * npart && flex_is_dm_core()) {
        const uint32_t g = cid / npart, p = cid % npart, stage = SH_F_L1_BASE;
        for (uint32_t l = 0; l < nlayers; ++l) {
            const int self = (selfmask >> l) & 1u;
            const sh_f_part s = sh_f_part_of(p, npart, Lp, So, self);
            const uint32_t kt = sh_f_res_off(l, p, npart, Lp, So, dh, selfmask), v = kt + dh * s.Lpad * 2;
            const uint64_t kb = (self ? ks0 : kx0) + (uint64_t)(l / 2) * (self ? sstride : xstride);
            const uint64_t vb = (self ? vs0 : vx0) + (uint64_t)(l / 2) * (self ? sstride : xstride);
            const uint64_t col = (uint64_t)g * dh * 2, row = (uint64_t)s.r0 * ld * 2;
            sh_l1_zero_dm(kt, 2 * dh * s.Lpad * 2);                                  /* kT and V incl. padding */
            sh_load_block_async(stage, kb + row + col, s.np, dh, ld);
            sh_load_block_async(v, vb + row + col, s.np, dh, ld);
            bare_dma_wait_all();
            sh_f_transpose_dm(kt, s.Lpad, 0, stage, s.np, dh);
            bare_dma_wait_all();
        }
    }
    flex_global_barrier_xy();
    return 0;
}

/* One score row (cv vectors): x <- 2^(s2 x - m2) on valid columns, 0 elsewhere; returns the fp32 row sum and m2 */
static inline float sh_f_softmax_row(sh_v4h *x, const sh_v4h *vld, const sh_v4h *nb, uint32_t cv, float s2,
                                     const sh_v4_exp2_consts *ec, sh_v4h s24, sh_v4h cm14, float *m2o) {
    sh_v4h m0 = sh_v4_splat_h(0xFC00u), m1 = m0; uint32_t j;
    for (j = 0; j + 2 <= cv; j += 2) { m0 = sh_v4_max(m0, sh_v4_mac(nb[j], x[j], vld[j])); m1 = sh_v4_max(m1, sh_v4_mac(nb[j + 1], x[j + 1], vld[j + 1])); }
    const float m2 = sh_v4_hmax(sh_v4_max(m0, m1)) * s2;
    const sh_v4h m24 = sh_v4_splat(m2);
    sh_v4h acc = sh_v4_splat_h(0); float sum = 0.f;
    #define SH_F_T(j) sh_v4_max_r(sh_v4_sub_r(sh_v4_mul_r(sh_v4_mac(nb[j], x[j], vld[j]), s24), m24), cm14)
    for (j = 0; j + 4 <= cv; j += 4) {   /* cv % 4 == 0 (Lpad % 16); acc flushed every 8 vectors and at the end */
        sh_v4h t[4] = { SH_F_T(j), SH_F_T(j + 1), SH_F_T(j + 2), SH_F_T(j + 3) };
        sh_v4_exp2x4(t, ec);
        t[0] = sh_v4_mul(t[0], vld[j]); t[1] = sh_v4_mul(t[1], vld[j + 1]); t[2] = sh_v4_mul(t[2], vld[j + 2]); t[3] = sh_v4_mul(t[3], vld[j + 3]);
        x[j] = t[0]; x[j + 1] = t[1]; x[j + 2] = t[2]; x[j + 3] = t[3];
        acc = sh_v4_add(sh_v4_add(acc, sh_v4_add(t[0], t[1])), sh_v4_add(t[2], t[3]));
        if ((j & 4) == 4) { sum += sh_v4_hsum(acc); acc = sh_v4_splat_h(0); }
    }
    #undef SH_F_T
    *m2o = sh_v4_lane(m24, 0);           /* the fp16-rounded max actually subtracted */
    return sum + sh_v4_hsum(acc);
}

/* Per step and layer, all cores of all clusters (needs sh_f_kvs_deal of the same geometry earlier in the chunk).
 * q [Sq, H * dh] (ldq), own keys / values ko / vo [Sq, Hkv * dh] (self layers; 0 otherwise), tok the prefix token-class
 * row (0: all valid), o [Sq, H * dh] (ldo). */
int sh_f_attention_kvs(uint64_t q, uint64_t ko, uint64_t vo, uint64_t tok, uint64_t o, uint32_t layer, uint32_t selfmask,
                       uint32_t Sq, uint32_t Lp, uint32_t So, uint32_t H, uint32_t Hkv, uint32_t dh,
                       uint32_t ldq, uint32_t ldko, uint32_t ldvo, uint32_t ldo, float scale) {
    const uint32_t cid = flex_get_cluster_id(), grp = H / Hkv, npart = grp, R = grp * Sq;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t core = flex_get_core_id();
    const int self = (selfmask >> layer) & 1u;
    const uint32_t Lpmax = sh_f_up32(sh_f_up32((Lp + npart - 1) / npart) + So);
    const sh_f_kvs_l1 a = sh_f_kvs_layout(Sq, H, grp, dh, Lpmax);
    if ((dh & 3) || H % Hkv || a.end > SH_F_RES_BASE - 64u) {
        if (cid == 0 && first) sh_printf("[sh_f_attention_kvs] H=%u Hkv=%u dh=%u: scratch %u > %u or bad shape\n", H, Hkv, dh, a.end, SH_F_RES_BASE - 64u);
        return -1;
    }
    SH_F_STAMP(0);
    /* a. the query block: cluster 0 loads it once and multicasts it (one collective in flight at a time) */
    if (cid == 0 && dm) {
        const uint32_t qbytes = Sq * H * dh * 2, CH = 32768u;
        sh_load_block_async(a.qb, q, Sq, H * dh, ldq);
        bare_dma_wait_all();
        const uint16_t rx = (uint16_t)~(ARCH_NUM_CLUSTER_X - 1u), ry = (uint16_t)~(ARCH_NUM_CLUSTER_Y - 1u);
        for (uint32_t b = 0; b < qbytes; b += CH) {
            flex_dma_async_broadcast(a.qb + b, a.qb + b, qbytes - b < CH ? qbytes - b : CH, rx, ry);
            flex_dma_async_wait_all();
        }
    }
    flex_global_barrier_xy();
    SH_F_STAMP(1);
    const int active = cid < Hkv * npart;
    const uint32_t g = cid / npart, p = cid % npart;
    const sh_f_part s = sh_f_part_of(p, npart, Lp, So, self);
    const uint32_t kt = sh_f_res_off(layer, p, npart, Lp, So, dh, selfmask), vr = kt + dh * s.Lpad * 2;
    const uint32_t Lpad = s.Lpad, cv = Lpad >> 2, vb = sh_f_up64(Lpad * 2);
    uint32_t lo = 0, hi = 0;
    if (active) {
        /* b. stack the group's query heads, own keys / values into the resident part, token classes, zero S / O */
        if (dm) {
            for (uint32_t j = 0; j < grp; ++j)
                bare_dma_start_2d(local(a.qs + j * Sq * dh * 2), local(a.qb + (g * grp + j) * dh * 2), dh * 2, dh * 2, H * dh * 2, Sq);
            if (s.own) {
                sh_load_block_async(a.kst, ko + (uint64_t)g * dh * 2, Sq, dh, ldko);
                sh_load_block_async(vr + s.np * dh * 2, vo + (uint64_t)g * dh * 2, Sq, dh, ldvo);
            }
            if (tok) bare_dma_start_1d(local(a.tok), tok + (uint64_t)s.r0 * 2, s.np * 2);
            sh_l1_zero_dm(a.s, R * Lpad * 2);         /* waits for everything issued so far */
            sh_l1_zero_dm(a.o, R * dh * 2);
            if (s.own) { sh_f_transpose_dm(kt, Lpad, s.np, a.kst, Sq, dh); bare_dma_wait_all(); }
        }
        flex_intra_cluster_sync();
        SH_F_STAMP(2);
        if (first) { flex_redmule_config(R, dh, Lpad); sh_f_redmule_fp16(a.qs, kt, a.s); flex_redmule_wait(); }
        flex_intra_cluster_sync();
        SH_F_STAMP(3);
        /* masked row softmax over this key part; per-core validity rows (prefix from tok, own causal, padding 0) */
        sh_share(R, 1, &lo, &hi);
        volatile uint16_t *vld = (volatile uint16_t *)local(a.vld + core * 2 * vb), *nb = vld + (vb >> 1);
        {
            const volatile uint16_t *tr = (const volatile uint16_t *)local(a.tok);
            for (uint32_t j = 0; j < Lpad; ++j) {
                uint32_t v = (j < s.np) ? (tok ? (tr[j] != SH_LLM_PAD) : 1u) : 0u;
                vld[j] = v ? 0x3C00u : 0u; nb[j] = v ? 0u : 0xFBFFu;
            }
        }
        const sh_v4_exp2_consts ec = sh_v4_exp2_init();
        const float s2 = scale * SH_LOG2E;
        const sh_v4h s24 = sh_v4_splat(s2), cm14 = sh_v4_splat_h(SH_CM14);
        float *msum = (float *)local(a.l), *mmax = (float *)local(a.m);
        const sh_v4h *vv = (const sh_v4h *)(uintptr_t)vld, *nn = (const sh_v4h *)(uintptr_t)nb;
        vv = sh_x_after_sh(nb + Lpad - 1, vv);
        for (uint32_t r = lo; r < hi; ++r) {
            const uint32_t qi = r % Sq;
            if (s.own) {               /* own column np + j valid iff j <= qi: rebuild at a head boundary, else enable one */
                if (r == lo || qi == 0) { for (uint32_t j = 0; j < s.own; ++j) { vld[s.np + j] = j <= qi ? 0x3C00u : 0u; nb[s.np + j] = j <= qi ? 0u : 0xFBFFu; } }
                else { vld[s.np + qi] = 0x3C00u; nb[s.np + qi] = 0u; }
                vv = sh_x_after_sh(nb + s.np + s.own - 1, vv);
            }
            float m2;
            msum[r] = sh_f_softmax_row(SH_V4P(local(a.s + r * Lpad * 2)), vv, nn, cv, s2, &ec, s24, cm14, &m2);
            mmax[r] = m2;
        }
        sh_fp_fence();
        flex_intra_cluster_sync();
        SH_F_STAMP(4);
        if (first) { flex_redmule_config(R, Lpad, dh); sh_f_redmule_fp16(a.s, vr, a.o); flex_redmule_wait(); }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();                  /* every partial (O, m, l) is in its cluster's TCDM */
    SH_F_STAMP(5);
    /* c. gather head h = g * grp + p's rows of the npart partials over the NoC and combine */
    if (active) {
        const uint32_t h = g * grp + p;
        if (dm) {
            for (uint32_t j = 0; j < npart; ++j) {
                const uint32_t pc = g * npart + j;
                bare_dma_start_1d(local(a.c + j * Sq * dh * 2), remote_cid(pc, a.o + p * Sq * dh * 2), Sq * dh * 2);
                bare_dma_start_1d(local(a.cm + j * Sq * 4), remote_cid(pc, a.m + p * Sq * 4), Sq * 4);
                bare_dma_start_1d(local(a.cl + j * Sq * 4), remote_cid(pc, a.l + p * Sq * 4), Sq * 4);
            }
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
        sh_share(Sq, 1, &lo, &hi);
        const float *cm = (const float *)local(a.cm), *cl = (const float *)local(a.cl);
        const uint32_t dv = dh >> 2;
        for (uint32_t r = lo; r < hi; ++r) {
            float M = cm[r];
            for (uint32_t j = 1; j < npart; ++j) M = sh_fmaxf(M, cm[j * Sq + r]);
            float w[4], L = 0.f;                                   /* npart <= 4 */
            for (uint32_t j = 0; j < npart; ++j) { w[j] = sh_exp2_clamped(sh_fmaxf(cm[j * Sq + r] - M, -126.f)); L += w[j] * cl[j * Sq + r]; }
            const float inv = 1.f / L;
            sh_v4h *ov = SH_V4P(local(a.out + r * dh * 2));
            const sh_v4h *o0 = SH_V4CP(local(a.c + r * dh * 2));
            const sh_v4h c0 = sh_v4_splat(w[0] * inv);
            for (uint32_t e = 0; e < dv; ++e) ov[e] = sh_v4_mul_r(o0[e], c0);
            for (uint32_t j = 1; j < npart; ++j) {
                const sh_v4h *oj = SH_V4CP(local(a.c + (j * Sq + r) * dh * 2));
                const sh_v4h cj = sh_v4_splat(w[j] * inv);
                for (uint32_t e = 0; e < dv; ++e) ov[e] = sh_v4_mac_r(ov[e], oj[e], cj);
            }
        }
        sh_fp_fence();
        flex_intra_cluster_sync();
        if (dm) sh_store_block_sync(o + (uint64_t)h * dh * 2, a.out, Sq, dh, ldo);
        flex_intra_cluster_sync();
    }
    SH_F_STAMP(6);
    flex_global_barrier_xy();
    {   /* phase profile: calls 0, 1 on cluster 0 (part 0), calls 2, 3 on cluster npart - 1 (last part, own keys on self layers) */
        volatile uint32_t *t = (volatile uint32_t *)local(SH_F_RES_BASE - 64u);   /* t[7]: this cluster's call count */
        if (first && ((cid == 0 && t[7] < 2) || (cid == npart - 1 && t[7] >= 2 && t[7] < 4)))
            sh_printf("[sh_f_attention_kvs] cluster %u layer %u: q multicast %u, stage %u, QK %u, softmax %u, PV+barrier %u, gather+combine %u cycles\n",
                      cid, layer, t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3], t[5] - t[4], t[6] - t[5]);
        if (first) t[7] = t[7] + 1;
    }
    return 0;
}

/* ---- 2. fp8 (e4m3) steps of the weight GEMMs ------------------------------------------------------------------
 * Quantiser and its numpy twin: softhier_mlir/frontend/fp8.py. Weights live in HBM as one e4m3 byte per element
 * (per-tensor power-of-two scale 2^e_w, k = 8 + e_w stored as an int16 next to them). An fp8 GEMM step:
 *   1. sh_f_rn4: x <- RN4(x) * 2^k in place (round to 4 significant bits = the e4m3 value, Veltkamp split in fp16 SIMD:
 *      c = 129 x, hi = c - (c - x); 4 fp16 SIMD ops per 4 elements; x is consumed by this GEMM only);
 *   2. the GEMM with W' = fp16 bits ((b & 0x7F) << 7) | ((b & 0x80) << 8) = e4m3(b) * 2^-8, so X' W' = x_q w_q exactly,
 *      fp16 RedMulE (fp16 accumulation, fp16 output). Three ways to get W' into TCDM:
 *        mode 0  the DMA streams the bytes (half the fp16 traffic), the three cores expand them in TCDM (software, real);
 *        mode 1  the DMA streams the bytes and nothing expands them: the timing of a cast in the DMA back-end at line
 *                rate (hypothetical hardware); RedMulE then reads a stale tile, so the NUMBERS OF MODE 1 ARE INVALID;
 *        mode 2  the DMA streams a host-expanded fp16 copy of W' (2 B per element): the numbers of mode 0 / 1 hardware
 *                bit for bit (same fp16 operands into the same RedMulE), at fp16 traffic. */
static void sh_fk_rn4(const sh_blk *k) {
    const sh_v4h c129 = sh_v4_splat_h(0x5808u), sc = sh_v4_splat(*(const float *)k->arg);   /* 129.0, 2^k */
    uint32_t lo, hi; sh_share(k->nr * k->cols, 16, &lo, &hi);
    const sh_v4h *x = SH_V4CP(k->x); sh_v4h *y = SH_V4P(k->y);
    for (uint32_t i = lo >> 2, e = hi >> 2; i < e; ++i) {
        const sh_v4h c = sh_v4_mul(x[i], c129);
        y[i] = sh_v4_mul(sh_v4_sub(c, sh_v4_sub(c, x[i])), sc);
    }
}
void sh_f_rn4(uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, int32_t kexp, uint32_t cluster) {
    float sc = 1.f; for (int32_t i = 0; i < kexp; ++i) sc *= 2.f; for (int32_t i = 0; i > kexp; --i) sc *= 0.5f;
    sh_rowop(x, x, 0, 0, 0, rows, cols, ld, ld, 0, sh_fk_rn4, &sc, cluster);
}

/* W' tile (fp16) from n bytes of e4m3 codes, split over the three cores (n % 16 == 0) */
static inline void sh_f_expand(uint32_t w16, uint32_t b8, uint32_t n) {
    uint32_t lo, hi; sh_share(n >> 2, 4, &lo, &hi);
    const volatile uint32_t *s = (const volatile uint32_t *)local(b8); volatile uint32_t *d = (volatile uint32_t *)local(w16);
    for (uint32_t i = lo; i < hi; ++i) {
        const uint32_t w = s[i];
        const uint32_t s0 = (w & 0xFFu) | ((w & 0xFF00u) << 8), s1 = ((w >> 16) & 0xFFu) | ((w >> 8) & 0xFF0000u);
        d[2 * i] = ((s0 & 0x007F007Fu) << 7) | ((s0 & 0x00800080u) << 8);
        d[2 * i + 1] = ((s1 & 0x007F007Fu) << 7) | ((s1 & 0x00800080u) << 8);
    }
}

uint32_t sh_f_gemm_w8_l1_bytes(const sh_gemm_cfg *cfg) {
    sh_gemm_cfg c; sh_cfg_fill(&c, cfg);
    return c.l1_base + 2 * (c.tm * c.tk * 2 + c.tk * c.tn + c.tk * c.tn * 2) + c.tm * c.tn * 2;
}

/* Z[M,N] = X'[M,K] W'[K,N], W given as e4m3 bytes (ldw in elements = bytes). Output tiles dealt round-robin like
 * sh_gemm; per K-tile: DMA X / byte tile k+1 while RedMulE runs tile k and the cores expand bytes k+1 (expand = 1). */
int sh_f_gemm_w8(uint64_t x, uint64_t w8, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                 uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg, uint32_t expand, uint32_t cluster) {
    const uint32_t cid = flex_get_cluster_id(), P = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    if (cluster != SH_ALL && cid != cluster) return 0;
    sh_gemm_cfg c; sh_cfg_fill(&c, cfg);
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t need = sh_f_gemm_w8_l1_bytes(&c);
    if (M % c.tm || N % c.tn || K % c.tk || (c.tk * c.tn) % 16 || need > SH_F_RES_BASE) {
        if (first) sh_printf("[sh_f_gemm_w8] %ux%ux%u tile %ux%ux%u: not divisible or L1 %u > %u\n", M, N, K, c.tm, c.tn, c.tk, need, SH_F_RES_BASE);
        return -1;
    }
    const uint32_t xb = c.tm * c.tk * 2, bb = c.tk * c.tn, wb = c.tk * c.tn * 2, yb = c.tm * c.tn * 2;
    const uint32_t X[2] = { c.l1_base, c.l1_base + xb }, B[2] = { X[1] + xb, X[1] + xb + bb }, W[2] = { B[1] + bb, B[1] + bb + wb };
    const uint32_t y = W[1] + wb, MT = M / c.tm, NT = N / c.tn, KT = K / c.tk;
    if (first) flex_redmule_config(c.tm, c.tk, c.tn);
    for (uint32_t r = 0; r < MT; ++r)
    for (uint32_t col = 0; col < NT; ++col) {
        if (cluster == SH_ALL && ((r * NT + col) % P) != cid) continue;
        const uint64_t zt = z + ((uint64_t)r * c.tm * ldz + col * c.tn) * 2;
        #define SH_F_LOAD(s, kk) do { sh_load_block_async(X[s], x + ((uint64_t)r * c.tm * ldx + (kk) * c.tk) * 2, c.tm, c.tk, ldx); \
            bare_dma_start_2d(local(B[s]), w8 + (uint64_t)(kk) * c.tk * ldw + col * c.tn, c.tn, c.tn, ldw, c.tk); } while (0)
        if (dm) { sh_l1_zero_dm(y, yb); SH_F_LOAD(0, 0); bare_dma_wait_all(); }
        flex_intra_cluster_sync();
        if (expand) sh_f_expand(W[0], B[0], bb);
        flex_intra_cluster_sync();
        for (uint32_t kk = 0; kk < KT; ++kk) {
            const uint32_t cur = kk & 1, nxt = cur ^ 1;
            if (dm && kk + 1 < KT) SH_F_LOAD(nxt, kk + 1);
            if (first) sh_f_redmule_fp16(X[cur], W[cur], y);
            if (dm && kk + 1 < KT) bare_dma_wait_all();
            flex_intra_cluster_sync();
            if (expand && kk + 1 < KT) sh_f_expand(W[nxt], B[nxt], bb);
            if (first) flex_redmule_wait();
            flex_intra_cluster_sync();
        }
        #undef SH_F_LOAD
        if (dm) sh_store_block_sync(zt, y, c.tm, c.tn, ldz);
        flex_intra_cluster_sync();
    }
    if (cluster == SH_ALL) flex_global_barrier_xy();
    return 0;
}

/* One weight GEMM of the flow loop at the step's format: fmt != SH_FP8 -> sh_gemm on the fp16 weights w16; SH_FP8 ->
 * x <- RN4(x) 2^k (k = the int16 at kexp), then mode 0 / 1: sh_f_gemm_w8 on the bytes w8 (expand / timing only),
 * mode 2: sh_gemm on the expanded copy wq16. */
int sh_f_gemm_step(uint64_t x, uint64_t w16, uint64_t w8, uint64_t wq16, uint64_t kexp, uint64_t z,
                   uint32_t M, uint32_t N, uint32_t K, uint32_t ldx, uint32_t ldw, uint32_t ldz,
                   const sh_gemm_cfg *cfg, uint32_t fmt, uint32_t mode, uint32_t cluster) {
    if (fmt != SH_FP8) return sh_gemm(x, w16, z, M, N, K, ldx, ldw, ldz, cfg, cluster);
    const int32_t k = *(const volatile int16_t *)(uintptr_t)kexp;
    sh_f_rn4(x, M, K, ldx, k, cluster);
    if (mode == 2) return sh_gemm(x, wq16, z, M, N, K, ldx, ldw, ldz, cfg, cluster);
    return sh_f_gemm_w8(x, w8, z, M, N, K, ldx, ldw, ldz, cfg, mode == 0, cluster);
}
