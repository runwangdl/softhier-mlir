/* TCDM-resident ops (single cluster: run on the calling cluster, intra-cluster sync only,
 * so generated code can call them from every cluster without deadlock). The elementwise ones
 * split the range over all 3 cores; fp16 math goes through the Zfh register conversions
 * (sh_h2f / sh_f2h: integer lhu/sh + fmv + fcvt, since scalar flh/fsh are broken in gvsoc,
 * docs/SIMULATOR_NOTES.md #2) and the fp32 FPU. */
uint32_t sh_l1_addr(uint32_t off) { return local(off); }

static inline void sh_l1_share(uint32_t n, uint32_t *lo, uint32_t *hi) {   /* this core's [lo, hi) of n, 4-aligned */
    const uint32_t c = flex_get_core_id(), nc = ARCH_NUM_CORE_PER_CLUSTER, nq = (n + 3) / 4;
    *lo = (nq * c / nc) * 4; *hi = (nq * (c + 1) / nc) * 4; if (*hi > n) *hi = n; if (*lo > n) *lo = n;
}

void sh_l1_zero(uint32_t off, uint32_t bytes) {
    if (flex_is_dm_core()) sh_l1_zero_dm(off, bytes);
    flex_intra_cluster_sync();
}
void sh_l1_fill_fp16(uint32_t off, uint32_t n, uint16_t bits) {
    uint32_t lo, hi; sh_l1_share(n, &lo, &hi);
    uint16_t *p = (uint16_t *)local(off);
    for (uint32_t i = lo; i < hi; ++i) p[i] = bits;
    flex_intra_cluster_sync();
}
void sh_l1_relu_fp16(uint32_t off, uint32_t n) {
    uint32_t lo, hi; sh_l1_share(n, &lo, &hi);
    uint16_t *p = (uint16_t *)local(off);
    for (uint32_t i = lo; i < hi; ++i) if (p[i] & 0x8000u) p[i] = 0;
    flex_intra_cluster_sync();
}
void sh_l1_add_fp16(uint32_t dst, uint32_t src, uint32_t n) {
    uint32_t lo, hi; sh_l1_share(n, &lo, &hi);
    uint16_t *d = (uint16_t *)local(dst); const uint16_t *s = (const uint16_t *)local(src);
    uint32_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        float a0 = sh_h2f(d[i]) + sh_h2f(s[i]), a1 = sh_h2f(d[i + 1]) + sh_h2f(s[i + 1]);
        float a2 = sh_h2f(d[i + 2]) + sh_h2f(s[i + 2]), a3 = sh_h2f(d[i + 3]) + sh_h2f(s[i + 3]);
        d[i] = (uint16_t)sh_f2h(a0); d[i + 1] = (uint16_t)sh_f2h(a1); d[i + 2] = (uint16_t)sh_f2h(a2); d[i + 3] = (uint16_t)sh_f2h(a3);
    }
    for (; i < hi; ++i) d[i] = (uint16_t)sh_f2h(sh_h2f(d[i]) + sh_h2f(s[i]));
    flex_intra_cluster_sync();
}
/* y[m,n] += x[m,k] . w[k,n], packed fp16 tiles in TCDM (RedMulE: config(m, k, n)). */
void sh_redmule(uint32_t x, uint32_t w, uint32_t y, uint32_t m, uint32_t n, uint32_t k, uint32_t fmt) {
    if (flex_is_first_core()) {
        flex_redmule_config(m, k, n);
        flex_redmule_trigger(x, w, y, sh_redmule_fmt[fmt & 3]);
        flex_redmule_wait();
    }
    flex_intra_cluster_sync();
}
/* 1-D copy between any two addresses (TCDM/HBM), DM core, synchronous. */
void sh_dma_copy(uint64_t dst, uint64_t src, uint32_t bytes) {
    if (flex_is_dm_core()) { bare_dma_start_1d(dst, src, bytes); bare_dma_wait_all(); }
    flex_intra_cluster_sync();
}
