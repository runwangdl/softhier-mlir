/* Runtime wrappers around the flex_ SDK. */
void sh_init(void) {
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}
void     sh_barrier_global(void)  { flex_global_barrier_xy(); }
void     sh_barrier_cluster(void) { flex_intra_cluster_sync(); }
uint32_t sh_cluster_id(void)      { return flex_get_cluster_id(); }
uint32_t sh_core_id(void)         { return flex_get_core_id(); }
int      sh_is_first_core(void)   { return flex_is_first_core() != 0; }
int      sh_is_dm_core(void)      { return flex_is_dm_core() != 0; }
uint32_t sh_num_clusters(void)    { return ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y; }
void     sh_timer_start(void)     { flex_timer_start(); }
void     sh_timer_end(void)       { flex_timer_end(); }
uint32_t sh_cycles(void)          { uint32_t c; __asm__ volatile("csrr %0, mcycle" : "=r"(c)); return c; }
void     sh_eoc(uint32_t v)       { flex_eoc(v); }
uint64_t sh_hbm_addr(uint64_t o)  { return hbm_addr(o); }
uint64_t sh_hbm_malloc(uint32_t b){ return (uint64_t)(uintptr_t)flex_hbm_malloc(b); }
uint32_t sh_l1_size(void)         { return ARCH_CLUSTER_TCDM_SIZE; }

void sh_printf(const char *fmt, ...) {
    char buf[256];
    va_list va; va_start(va, fmt);
    vsnprintf(buf, sizeof buf, fmt, va);
    va_end(va);
    printf("%s", buf);
}

float sh_fp16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else { /* subnormal */
            e = 1; while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff; bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) bits = (s << 31) | 0x7f800000 | (m << 13);
    else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    union { uint32_t u; float f; } u; u.u = bits; return u.f;
}
uint16_t sh_f32_to_fp16(float f) {
    union { uint32_t u; float f; } u; u.f = f;
    uint32_t s = (u.u >> 16) & 0x8000, e = (u.u >> 23) & 0xff, m = u.u & 0x7fffff;
    if (e == 255) return s | 0x7c00 | (m ? 0x200 : 0);
    int ne = (int)e - 127 + 15;
    if (ne >= 31) return s | 0x7c00;
    if (ne <= 0) {
        if (ne < -10) return s;
        m |= 0x800000; uint32_t shift = 14 - ne;
        uint32_t r = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return s | r;
    }
    uint32_t r = (ne << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return s | r;
}
/* Core-local cycle counter: the mcycle CSR reads the gvsoc clock, so several regions can be timed in one run. */
uint32_t sh_cycles(void) { uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c; }
