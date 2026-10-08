/* TCDM-resident ops (single cluster: run on the calling cluster, intra-cluster sync only,
 * so generated code can call them from every cluster without deadlock). */
uint32_t sh_l1_addr(uint32_t off) { return local(off); }

void sh_l1_zero(uint32_t off, uint32_t bytes) {
    if (flex_is_dm_core()) sh_l1_zero_dm(off, bytes);
    flex_intra_cluster_sync();
}
void sh_l1_fill_fp16(uint32_t off, uint32_t n, uint16_t bits) {
    if (flex_is_first_core()) { volatile uint16_t *p = (volatile uint16_t *)local(off); for (uint32_t i = 0; i < n; ++i) p[i] = bits; }
    flex_intra_cluster_sync();
}
void sh_l1_relu_fp16(uint32_t off, uint32_t n) {
    if (flex_is_first_core()) { volatile uint16_t *p = (volatile uint16_t *)local(off); for (uint32_t i = 0; i < n; ++i) if (p[i] & 0x8000u) p[i] = 0; }
    flex_intra_cluster_sync();
}
/* Scalar fp16 glue goes through integer loads + software fp16<->fp32 conversion + fp32 FPU math:
 * the stock gvsoc Snitch model's flh/fsh (Zfh) use the integer register file (see
 * AI_AGENT/SoftHier/dse/patches/gvsoc_zfh_flh_fsh.patch); this path works on both. */
void sh_l1_add_fp16(uint32_t dst, uint32_t src, uint32_t n) {
    if (flex_is_first_core()) {
        volatile uint16_t *d = (volatile uint16_t *)local(dst); volatile uint16_t *s = (volatile uint16_t *)local(src);
        for (uint32_t i = 0; i < n; ++i) d[i] = sh_f32_to_fp16(sh_fp16_to_f32(d[i]) + sh_fp16_to_f32(s[i]));
    }
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
