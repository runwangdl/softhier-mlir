/* On-chip RSSM imagination (docs/WORLD_MODEL.md): a DreamerV3-style RSSM (world-model-on-edge wm/rssm.py: MLP
 * encoder, LayerNorm-GRU, categorical latent with unimix, actor) as a weight-stationary kernel. The whole weight blob
 * is staged once into the TCDM of every participating cluster; K trajectories are dealt over the clusters (SH_ALL:
 * cluster c takes rows [c Kc, (c+1) Kc)) and each cluster advances its rows H steps without touching HBM except for
 * the recorded states. GEMMs run on RedMulE with m = rows of the chunk; LayerNorm, the GRU gates, the per-group
 * softmax + unimix and the actor's tanh run as fp16 SIMD row kernels on the three cores (sh_simd.inc.c).
 *
 * One step (rssm.py; x | y = column concatenation, which is free: the buffers are laid out so that every concat is
 * a column range of one packed RedMulE operand):
 *   posterior only:  e  = relu(LN(relu(LN(obs We1)) We2))                         -> HE[:, D:D+U]
 *   x  = relu(LN([z | a] Win))                                                    -> XH[:, 0:Hd]
 *   g  = LN([x | h] Wg) (update-gate bias - 1 folded on the host);  r = sig(g_r), c = tanh(r g_c), u = sig(g_u)
 *   h' = h + u (c - h)                                                            -> XH[:, Hd:], F[:, S:], HE[:, 0:D]
 *   prior:      l = relu(LN(h' Wio)) Wis + bis      posterior: l = relu(LN([h' | e] Woo)) Wos + bos
 *   z' = 0.99 softmax_16(l) + 0.01 / 16 per group of `classes`                    -> F[:, 0:S], ZA[:, 0:S]
 *   a' = tanh((relu(LN(relu(LN([z' | h'] Wa1)) Wa2)) Wao + bao)[:, 0:act])        -> ZA[:, S:S+act]
 *
 * Weight blob (host: softhier_mlir.frontend.wm_rssm.pack): a table of 32 uint32 byte offsets (SH_WM_*; table[31] =
 * blob bytes), then fp16 matrices [in, out] row-major and fp16 vectors, each 64-byte aligned. Matrices whose input is
 * padded (a: act -> AP = act rounded up to 4, obs -> OP) have zero rows for the padding; Wao has AO = 2 act rounded
 * up to 4 columns. A blob without the posterior entries (offset 0) only supports posterior = 0. */
#define SH_WM_L1_BASE 0x1000u
enum { SH_WM_W_IN, SH_WM_G_IN, SH_WM_B_IN, SH_WM_W_G, SH_WM_G_G, SH_WM_B_G, SH_WM_W_IO, SH_WM_G_IO, SH_WM_B_IO,
       SH_WM_W_IS, SH_WM_B_IS, SH_WM_W_A1, SH_WM_G_A1, SH_WM_B_A1, SH_WM_W_A2, SH_WM_G_A2, SH_WM_B_A2, SH_WM_W_AO,
       SH_WM_B_AO, SH_WM_W_E1, SH_WM_G_E1, SH_WM_B_E1, SH_WM_W_E2, SH_WM_G_E2, SH_WM_B_E2, SH_WM_W_OO, SH_WM_G_OO,
       SH_WM_B_OO, SH_WM_W_OS, SH_WM_B_OS, SH_WM_NTAB };
#define SH_WM_TABLE_BYTES 128u
#define SH_WM_LN_EPS 1e-3f

static inline uint32_t sh_wm_up64(uint32_t b) { return (b + 63u) & ~63u; }

/* ---- TCDM GEMM: y[m, k] = x[m, n] w[n, k] (packed operands, TCDM offsets) ---------------------------------- */
static void sh_wm_mm(uint32_t x, uint32_t w, uint32_t y, uint32_t m, uint32_t n, uint32_t k, int first, int dm) {
    if (dm) sh_l1_zero_dm(y, m * k * 2);                  /* RedMulE accumulates into y */
    flex_intra_cluster_sync();
    if (first) { flex_redmule_config(m, n, k); flex_redmule_trigger(x, w, y, REDMULE_FP_16); flex_redmule_wait(); }
    flex_intra_cluster_sync();
}

/* ---- LayerNorm (+ ReLU) over `cols` of each row, rows split over the cores; y may alias x ---------------------
 * mean from fp16 SIMD partial sums, variance from squared deviations prescaled by 1/16 (no overflow up to |x - mean|
 * of 4095); a small variance (< 1e-3: every |x - mean| < 0.62 sqrt(cols / 384)) is recounted with a x16 prescale in
 * SIMD instead of the library's scalar fallback, so the result keeps fp16 resolution for nearly constant rows. */
static void sh_wm_ln(uint32_t y, uint32_t ldy, uint32_t x, uint32_t ldx, uint32_t rows, uint32_t cols, uint32_t g, uint32_t b, int relu) {
    const uint32_t cv = cols >> 2;
    const sh_v4h *gg = SH_V4CP(local(g)), *bb = SH_V4CP(local(b));
    const sh_v4h q4 = sh_v4_splat(0.0625f), q16 = sh_v4_splat(16.f), z4 = sh_v4_splat_h(0);
    uint32_t lo, hi; sh_share(rows, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const sh_v4h *xv = SH_V4CP(local(x + r * ldx * 2)); sh_v4h *yv = SH_V4P(local(y + r * ldy * 2));
        const float mean = sh_v4_row_sum(xv, cv) / (float)cols;
        const uint32_t mh = sh_f2h(mean);
        const sh_v4h mh4 = sh_v4_splat_h(mh), ml4 = sh_v4_splat(mean - sh_h2f(mh));
        float var = sh_v4_row_sqdev(xv, cv, mh4, ml4, q4) * (256.f / (float)cols);
        if (var < 1e-3f) var = sh_v4_row_sqdev(xv, cv, mh4, ml4, q16) * (1.f / (256.f * (float)cols));
        const sh_v4h rs4 = sh_v4_splat(sh_rsqrtf(var + SH_WM_LN_EPS));
        if (relu) for (uint32_t j = 0; j < cv; ++j) yv[j] = sh_v4_max(sh_v4_mac(bb[j], sh_v4_mul_r(sh_v4_sub_r(sh_v4_sub_r(xv[j], mh4), ml4), rs4), gg[j]), z4);
        else      for (uint32_t j = 0; j < cv; ++j) yv[j] = sh_v4_mac(bb[j], sh_v4_mul_r(sh_v4_sub_r(sh_v4_sub_r(xv[j], mh4), ml4), rs4), gg[j]);
    }
    sh_fp_fence();
}

/* ---- GRU gates: p = LN'd [reset | cand | update] (D each), h in place; h' also to f and he -----------------------
 * sig(v) = 1 / (1 + 2^(-v log2 e)), tanh(v) = (1 - 2^(-2 v log2 e)) / (1 + 2^(-2 v log2 e)), exponents clamped to the
 * fp16 range [-14, 15]; work items = (row, 16 columns), split over the cores (a cluster may own only a few rows). */
static void sh_wm_gru(uint32_t p, uint32_t ldp, uint32_t h, uint32_t ldh, uint32_t f, uint32_t ldf, uint32_t he, uint32_t ldhe,
                      uint32_t rows, uint32_t D) {
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h nl = sh_v4_splat(-SH_LOG2E), nl2 = sh_v4_splat(-2.f * SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), c15 = sh_v4_splat_h(SH_C15);
    const uint32_t dv = D >> 2, per = D >> 4;
    uint32_t lo, hi; sh_share(rows * per, 1, &lo, &hi);
    #define SH_WM_CL(v) sh_v4_min_r(sh_v4_max_r(v, cm14), c15)
    for (uint32_t it = lo; it < hi; ++it) {
        const uint32_t r = it / per, j = (it % per) * 4;
        const sh_v4h *pv = SH_V4CP(local(p + r * ldp * 2));
        sh_v4h *hv = SH_V4P(local(h + r * ldh * 2)), *fv = SH_V4P(local(f + r * ldf * 2)), *ev = SH_V4P(local(he + r * ldhe * 2));
        sh_v4h t[4], rr[4], cc[4];
        for (int i = 0; i < 4; ++i) t[i] = SH_WM_CL(sh_v4_mul_r(pv[j + i], nl));
        sh_v4_exp2x4(t, &ec);
        for (int i = 0; i < 4; ++i) { rr[i] = sh_v4_div(ec.one, sh_v4_add_r(t[i], ec.one)); t[i] = SH_WM_CL(sh_v4_mul_r(sh_v4_mul(rr[i], pv[dv + j + i]), nl2)); }
        sh_v4_exp2x4(t, &ec);
        for (int i = 0; i < 4; ++i) { cc[i] = sh_v4_div(sh_v4_sub(ec.one, t[i]), sh_v4_add_r(t[i], ec.one)); t[i] = SH_WM_CL(sh_v4_mul_r(pv[2 * dv + j + i], nl)); }
        sh_v4_exp2x4(t, &ec);
        for (int i = 0; i < 4; ++i) {
            const sh_v4h u = sh_v4_div(ec.one, sh_v4_add_r(t[i], ec.one)), hold = hv[j + i];
            const sh_v4h hn = sh_v4_mac(hold, u, sh_v4_sub(cc[i], hold));
            hv[j + i] = hn; fv[j + i] = hn; ev[j + i] = hn;
        }
    }
    #undef SH_WM_CL
    sh_fp_fence();
}

/* ---- categorical latent: z = (1 - mix) softmax(l + b) + mix / C per group of C classes (C % 4 == 0) -------------
 * work items = (row, group) over the cores; group max by one lane reduction per group. Writes z to two places. */
static void sh_wm_softmax(uint32_t l, uint32_t ldl, uint32_t bias, uint32_t z1, uint32_t ld1, uint32_t z2, uint32_t ld2,
                          uint32_t rows, uint32_t S, uint32_t C, float mix) {
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h l2e = sh_v4_splat(SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), mix4 = sh_v4_splat(mix / (float)C);
    const uint32_t cv = C >> 2, groups = S / C;
    const sh_v4h *bv = SH_V4CP(local(bias));
    uint32_t lo, hi; sh_share(rows * groups, 1, &lo, &hi);
    sh_v4h e[8];
    for (uint32_t it = lo; it < hi; ++it) {
        const uint32_t r = it / groups, g0 = (it % groups) * cv;
        const sh_v4h *lv = SH_V4CP(local(l + r * ldl * 2));
        sh_v4h *o1 = SH_V4P(local(z1 + r * ld1 * 2)), *o2 = SH_V4P(local(z2 + r * ld2 * 2));
        for (uint32_t i = 0; i < cv; ++i) e[i] = sh_v4_add(lv[g0 + i], bv[g0 + i]);
        sh_v4h m = e[0];
        for (uint32_t i = 1; i < cv; ++i) m = sh_v4_max(m, e[i]);
        const sh_v4h m4 = sh_v4_splat(sh_v4_hmax(m) * SH_LOG2E);
        for (uint32_t i = 0; i < cv; ++i) e[i] = sh_v4_max_r(sh_v4_sub_r(sh_v4_mul_r(e[i], l2e), m4), cm14);
        uint32_t i = 0;
        for (; i + 4 <= cv; i += 4) sh_v4_exp2x4(e + i, &ec);
        for (; i < cv; ++i) e[i] = sh_v4_exp2(e[i], &ec);
        sh_v4h s = e[0];
        for (i = 1; i < cv; ++i) s = sh_v4_add(s, e[i]);
        const sh_v4h inv4 = sh_v4_splat((1.f - mix) / sh_v4_hsum(s));
        for (i = 0; i < cv; ++i) { const sh_v4h pz = sh_v4_mac_r(mix4, e[i], inv4); o1[g0 + i] = pz; o2[g0 + i] = pz; }
    }
    sh_fp_fence();
}

/* ---- actor head: a = tanh(o[:, 0:4] + b[0:4]) -> dst (the 4th lane is the std logit's tanh; it multiplies the zero
 * padding row of Win, the host ignores it) ------------------------------------------------------------------------ */
static void sh_wm_tanh_head(uint32_t o, uint32_t ldo, uint32_t bias, uint32_t dst, uint32_t ldd, uint32_t rows) {
    const sh_v4_exp2_consts ec = sh_v4_exp2_init();
    const sh_v4h nl2 = sh_v4_splat(-2.f * SH_LOG2E), cm14 = sh_v4_splat_h(SH_CM14), c15 = sh_v4_splat_h(SH_C15);
    const sh_v4h b4 = *SH_V4CP(local(bias));
    uint32_t lo, hi; sh_share(rows, 1, &lo, &hi);
    for (uint32_t r = lo; r < hi; ++r) {
        const sh_v4h v = sh_v4_add(*SH_V4CP(local(o + r * ldo * 2)), b4);
        const sh_v4h t = sh_v4_exp2(sh_v4_min_r(sh_v4_max_r(sh_v4_mul_r(v, nl2), cm14), c15), &ec);
        *SH_V4P(local(dst + r * ldd * 2)) = sh_v4_div(sh_v4_sub(ec.one, t), sh_v4_add_r(t, ec.one));
    }
    sh_fp_fence();
}

/* ---- the kernel ------------------------------------------------------------------------------------------------ */
typedef struct { uint32_t za, xh, p, f, he, t1, t2, o, ob, end, wza, wxh, wf, whe; } sh_wm_l1;

static inline sh_wm_l1 sh_wm_layout(const sh_wm_cfg *c, uint32_t base, uint32_t kb) {
    const uint32_t D = c->deter, S = c->stoch * c->classes, AP = (c->act + 3u) & ~3u, OP = (c->obs + 3u) & ~3u, AO = (2u * c->act + 3u) & ~3u;
    const uint32_t Hd = c->hidden, U = c->units, T = Hd > U ? Hd : U;
    sh_wm_l1 l;
    l.wza = S + AP; l.wxh = Hd + D; l.wf = S + D; l.whe = c->posterior ? D + U : D;
    const uint32_t wp = 3u * D > S ? 3u * D : S;
    l.za = base;                                l.xh = l.za + sh_wm_up64(kb * l.wza * 2);
    l.p = l.xh + sh_wm_up64(kb * l.wxh * 2);    l.f = l.p + sh_wm_up64(kb * wp * 2);
    l.he = l.f + sh_wm_up64(kb * l.wf * 2);     l.t1 = l.he + sh_wm_up64(kb * l.whe * 2);
    l.t2 = l.t1 + sh_wm_up64(kb * T * 2);       l.o = l.t2 + sh_wm_up64(kb * T * 2);
    l.ob = l.o + sh_wm_up64(kb * AO * 2);       l.end = l.ob + (c->posterior ? sh_wm_up64(kb * OP * 2) : 0);
    return l;
}

static volatile uint32_t *sh_wm_prof(void) { return (volatile uint32_t *)local(SH_WM_L1_BASE); }   /* SH_WM_NPROF words */

uint32_t sh_wm_profile(uint32_t i) { return sh_wm_prof()[i % SH_WM_NPROF]; }

int sh_wm_rssm(const sh_wm_cfg *c, uint64_t wblob, uint64_t h0, uint64_t z0, uint64_t a0, uint64_t obs,
               uint64_t hs, uint64_t zs, uint64_t as, uint32_t cluster) {
    const uint32_t cid = flex_get_cluster_id(), NCL = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    if (cluster != SH_ALL && cid != cluster) return 0;
    const int first = flex_is_first_core(), dm = flex_is_dm_core();
    const uint32_t P = cluster == SH_ALL ? NCL : 1, me = cluster == SH_ALL ? cid : 0;
    const uint32_t D = c->deter, C = c->classes, S = c->stoch * C, Hd = c->hidden, U = c->units;
    const uint32_t AP = (c->act + 3u) & ~3u, OP = (c->obs + 3u) & ~3u, AO = (2u * c->act + 3u) & ~3u;
    const uint32_t Kc = (c->K + P - 1) / P, r0 = me * Kc, nr = r0 >= c->K ? 0 : (c->K - r0 < Kc ? c->K - r0 : Kc);
    volatile uint32_t *prof = sh_wm_prof();
    uint32_t tl = 0;
    /* debug: step 0 of the first chunk, every intermediate (packed TCDM buffer, m rows) to dbg + slot * 64 KB */
    #define SH_WM_DBG(slot, off, bytes) do { if (c->dbg && t == 0 && c0 == 0) { if (dm) { bare_dma_start_1d(c->dbg + (slot) * 0x10000u, local(off), bytes); bare_dma_wait_all(); } flex_intra_cluster_sync(); } } while (0)
    #define SH_WM_T(i) do { if (first) { const uint32_t now_ = sh_mcycle(); prof[i] += now_ - tl; tl = now_; } } while (0)
    if (first) { for (uint32_t i = 0; i < SH_WM_NPROF; ++i) prof[i] = 0; tl = sh_mcycle(); }
    /* 1. the weight blob: table first (its last word is the blob size), then everything */
    const uint32_t wb = SH_WM_L1_BASE + 128u;
    if (dm) { bare_dma_start_1d(local(wb), wblob, SH_WM_TABLE_BYTES); bare_dma_wait_all(); }
    flex_intra_cluster_sync();
    const volatile uint32_t *tab = (const volatile uint32_t *)local(wb);
    const uint32_t blob = tab[31];
    #define W(i) (wb + tab[i])
    uint32_t kb = c->kb ? c->kb : nr;
    if (kb > nr) kb = nr;
    const uint32_t abase = wb + sh_wm_up64(blob);
    while (kb > 1 && sh_wm_layout(c, abase, kb).end > ARCH_CLUSTER_TCDM_SIZE) kb = (kb + 1) / 2;
    const sh_wm_l1 l = sh_wm_layout(c, abase, kb ? kb : 1);
    if ((D & 15) || (S % C) || (C & 3) || C > 32 || (Hd & 3) || (U & 3) || l.end > ARCH_CLUSTER_TCDM_SIZE ||
        (c->posterior && tab[SH_WM_W_E1] == 0)) {
        if (first && me == 0) sh_printf("[sh_wm_rssm] unsupported: D=%u S=%u C=%u Hd=%u U=%u L1 %u > %u or no posterior weights\n",
                                        D, S, C, Hd, U, l.end, (uint32_t)ARCH_CLUSTER_TCDM_SIZE);
        if (cluster == SH_ALL) flex_global_barrier_xy();
        return -1;
    }
    if (dm) { bare_dma_start_1d(local(wb + SH_WM_TABLE_BYTES), wblob + SH_WM_TABLE_BYTES, blob - SH_WM_TABLE_BYTES); bare_dma_wait_all(); }
    flex_intra_cluster_sync();
    SH_WM_T(SH_WM_P_LOAD);
    for (uint32_t c0 = 0; c0 < nr; c0 += kb) {
        const uint32_t m = nr - c0 < kb ? nr - c0 : kb, row = r0 + c0;
        if (dm) {   /* initial state of the chunk: h -> XH[:, Hd:], z -> ZA[:, 0:S], a -> ZA[:, S:S+AP] */
            sh_l1_zero_dm(l.za, m * l.wza * 2);
            bare_dma_start_2d(local(l.xh + Hd * 2), h0 + (uint64_t)row * D * 2, D * 2, l.wxh * 2, D * 2, m);
            bare_dma_start_2d(local(l.za), z0 + (uint64_t)row * S * 2, S * 2, l.wza * 2, S * 2, m);
            bare_dma_start_2d(local(l.za + S * 2), a0 + (uint64_t)row * AP * 2, AP * 2, l.wza * 2, AP * 2, m);
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
        SH_WM_T(SH_WM_P_STORE);
        for (uint32_t t = 0; t < c->H; ++t) {
            if (c->posterior) {     /* embed = enc(obs_t) -> HE[:, D:D+U] */
                if (dm) { bare_dma_start_1d(local(l.ob), obs + ((uint64_t)t * c->K + row) * OP * 2, m * OP * 2); bare_dma_wait_all(); }
                flex_intra_cluster_sync();
                SH_WM_T(SH_WM_P_STORE);
                sh_wm_mm(l.ob, W(SH_WM_W_E1), l.t1, m, OP, U, first, dm);                           SH_WM_T(SH_WM_P_GEMM);
                sh_wm_ln(l.t1, U, l.t1, U, m, U, W(SH_WM_G_E1), W(SH_WM_B_E1), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
                sh_wm_mm(l.t1, W(SH_WM_W_E2), l.t2, m, U, U, first, dm);                            SH_WM_T(SH_WM_P_GEMM);
                sh_wm_ln(l.he + D * 2, l.whe, l.t2, U, m, U, W(SH_WM_G_E2), W(SH_WM_B_E2), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
            }
            /* x = relu(LN([z | a] Win)) -> XH[:, 0:Hd] */
            sh_wm_mm(l.za, W(SH_WM_W_IN), l.t1, m, l.wza, Hd, first, dm);                           SH_WM_T(SH_WM_P_GEMM);
            SH_WM_DBG(0, l.za, m * l.wza * 2); SH_WM_DBG(1, l.t1, m * Hd * 2);
            sh_wm_ln(l.xh, l.wxh, l.t1, Hd, m, Hd, W(SH_WM_G_IN), W(SH_WM_B_IN), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
            /* GRU */
            SH_WM_DBG(2, l.xh, m * l.wxh * 2);
            sh_wm_mm(l.xh, W(SH_WM_W_G), l.p, m, l.wxh, 3 * D, first, dm);                          SH_WM_T(SH_WM_P_GEMM);
            SH_WM_DBG(3, l.p, m * 3 * D * 2);
            sh_wm_ln(l.p, 3 * D, l.p, 3 * D, m, 3 * D, W(SH_WM_G_G), W(SH_WM_B_G), 0); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
            SH_WM_DBG(4, l.p, m * 3 * D * 2);
            sh_wm_gru(l.p, 3 * D, l.xh + Hd * 2, l.wxh, l.f + S * 2, l.wf, l.he, l.whe, m, D); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_GRU);
            SH_WM_DBG(5, l.he, m * l.whe * 2);
            /* latent head -> z' */
            if (c->posterior) {
                sh_wm_mm(l.he, W(SH_WM_W_OO), l.t1, m, l.whe, Hd, first, dm);                       SH_WM_T(SH_WM_P_GEMM);
                sh_wm_ln(l.t1, Hd, l.t1, Hd, m, Hd, W(SH_WM_G_OO), W(SH_WM_B_OO), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
                sh_wm_mm(l.t1, W(SH_WM_W_OS), l.p, m, Hd, S, first, dm);                            SH_WM_T(SH_WM_P_GEMM);
                sh_wm_softmax(l.p, S, W(SH_WM_B_OS), l.f, l.wf, l.za, l.wza, m, S, C, 0.01f);
            } else {
                sh_wm_mm(l.he, W(SH_WM_W_IO), l.t1, m, l.whe, Hd, first, dm);                       SH_WM_T(SH_WM_P_GEMM);
                sh_wm_ln(l.t1, Hd, l.t1, Hd, m, Hd, W(SH_WM_G_IO), W(SH_WM_B_IO), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
                sh_wm_mm(l.t1, W(SH_WM_W_IS), l.p, m, Hd, S, first, dm);                            SH_WM_T(SH_WM_P_GEMM);
                sh_wm_softmax(l.p, S, W(SH_WM_B_IS), l.f, l.wf, l.za, l.wza, m, S, C, 0.01f);
            }
            flex_intra_cluster_sync();                                                              SH_WM_T(SH_WM_P_SOFTMAX);
            SH_WM_DBG(6, l.p, m * S * 2); SH_WM_DBG(7, l.f, m * l.wf * 2);
            /* actor -> a' */
            sh_wm_mm(l.f, W(SH_WM_W_A1), l.t1, m, l.wf, U, first, dm);                              SH_WM_T(SH_WM_P_GEMM);
            sh_wm_ln(l.t1, U, l.t1, U, m, U, W(SH_WM_G_A1), W(SH_WM_B_A1), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
            sh_wm_mm(l.t1, W(SH_WM_W_A2), l.t2, m, U, U, first, dm);                                SH_WM_T(SH_WM_P_GEMM);
            sh_wm_ln(l.t2, U, l.t2, U, m, U, W(SH_WM_G_A2), W(SH_WM_B_A2), 1); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_LN);
            sh_wm_mm(l.t2, W(SH_WM_W_AO), l.o, m, U, AO, first, dm);                                SH_WM_T(SH_WM_P_GEMM);
            SH_WM_DBG(8, l.o, m * AO * 2);
            sh_wm_tanh_head(l.o, AO, W(SH_WM_B_AO), l.za + S * 2, l.wza, m); flex_intra_cluster_sync(); SH_WM_T(SH_WM_P_HEAD);
            SH_WM_DBG(9, l.za, m * l.wza * 2);
            /* record h', z', a' (every step, or the last one) */
            if (c->record || t + 1 == c->H) {
                if (dm) {
                    const uint64_t ti = c->record ? t : 0;
                    for (uint32_t i = 0; i < m; ++i) {
                        const uint64_t gi = ti * c->K + row + i;
                        bare_dma_start_1d(hs + gi * D * 2, local(l.xh + (i * l.wxh + Hd) * 2), D * 2);
                        bare_dma_start_1d(zs + gi * S * 2, local(l.za + i * l.wza * 2), S * 2);
                        bare_dma_start_1d(as + gi * AP * 2, local(l.za + (i * l.wza + S) * 2), AP * 2);
                    }
                    bare_dma_wait_all();
                }
                flex_intra_cluster_sync();
                SH_WM_T(SH_WM_P_STORE);
            }
        }
    }
    #undef W
    #undef SH_WM_T
    #undef SH_WM_DBG
    if (first) prof[SH_WM_P_ROWS] = nr, prof[SH_WM_P_KB] = kb;
    if (cluster == SH_ALL) flex_global_barrier_xy();
    if (first && me == 0)
        sh_printf("[sh_wm_rssm] K=%u H=%u clusters=%u rows/cluster=%u chunk=%u posterior=%u L1=%u blob=%u | cluster0 cycles: load %u gemm %u ln %u gru %u softmax %u head %u io %u\n",
                  c->K, c->H, P, nr, kb, c->posterior, l.end, blob, prof[SH_WM_P_LOAD], prof[SH_WM_P_GEMM], prof[SH_WM_P_LN],
                  prof[SH_WM_P_GRU], prof[SH_WM_P_SOFTMAX], prof[SH_WM_P_HEAD], prof[SH_WM_P_STORE]);
    return 0;
}
