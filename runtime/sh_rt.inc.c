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
/* HBM preload: `hbm_preload_done` is raised when the loader has issued its last 64 KB write, and the data then
 * crosses the NoC from cluster (0,0) at link bandwidth (tens of us for 16 MB, ms for the full SmolVLA image), so
 * the first global barrier does NOT imply the data is there. The image writer puts a 64 B sentinel (one NoC flit)
 * as the last segment; cluster 0 spins on it, waits a margin for in-flight chunks on longer mesh paths, and
 * everyone meets at a global barrier. */
#define SH_PRELOAD_MAGIC 0x5EEDC0DEu
void sh_preload_wait(uint64_t sentinel) {
    if (flex_get_cluster_id() == 0 && flex_is_first_core()) {
        const volatile uint32_t *w = (const volatile uint32_t *)(uintptr_t)sentinel;
        const uint32_t t0 = sh_cycles();
        while (w[0] != SH_PRELOAD_MAGIC || w[15] != SH_PRELOAD_MAGIC) {
            if ((uint32_t)(sh_cycles() - t0) > 400000000u) { sh_printf("[sh_preload_wait] no sentinel after 400 ms, continuing\n"); break; }
        }
        const uint32_t t1 = sh_cycles();
        while ((uint32_t)(sh_cycles() - t1) < 8192u) {}
        sh_printf("[sh_preload_wait] image visible after %u cycles\n", (uint32_t)(t1 - t0));
    }
    flex_global_barrier_xy();
}

/* Private per-core stacks (see sh_ops.h): top = stack memory end - 4 KB - core * bytes. The trampoline saves the old
 * sp on the new stack, calls fn, restores sp. Everything caller-saved is declared clobbered. */
void sh_call_on_core_stack(void (*fn)(void), uint32_t bytes_per_core) {
    uint32_t top = (ARCH_CLUSTER_STACK_BASE + ARCH_CLUSTER_STACK_SIZE - 0x1000u - flex_get_core_id() * bytes_per_core) & ~15u;
    __asm__ volatile (
        "mv   t0, sp\n\t"
        "mv   sp, %0\n\t"
        "addi sp, sp, -16\n\t"
        "sw   t0, 0(sp)\n\t"
        "jalr %1\n\t"
        "lw   t0, 0(sp)\n\t"
        "mv   sp, t0\n\t"
        : : "r"(top), "r"(fn)
        : "ra", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "memory");
}
