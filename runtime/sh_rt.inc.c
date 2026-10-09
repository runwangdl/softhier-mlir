/* Runtime wrappers around the flex_ SDK. */
SH_FAR void sh_init(void) {
    flex_barrier_xy_init();
    flex_global_barrier_xy();
#ifndef SH_NO_ALLOC
    flex_alloc_init();
#endif
    if (flex_is_dm_core()) {   /* cluster-set barrier words (sh_group_barrier): gvsoc fills the sync memory with 0x57 */
        volatile uint32_t *w = (volatile uint32_t *)(ARCH_SYNC_BASE + flex_get_cluster_id() * ARCH_SYNC_SIZE + 64u);
        w[0] = 0u; w[1] = 0u;
    }
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

#ifndef SH_TINY_PRINTF
SH_FAR void sh_printf(const char *fmt, ...) {
    char buf[256];
    va_list va; va_start(va, fmt);
    vsnprintf(buf, sizeof buf, fmt, va);
    va_end(va);
    printf("%s", buf);
}
#else
SH_FAR __attribute__((optimize("Os"))) static char *sh_tp_u(char *o, uint32_t v, uint32_t base, int width) {
    char t[12]; int n = 0;
    do { const uint32_t d = v % base; t[n++] = (char)(d < 10 ? '0' + d : 'a' + d - 10); v /= base; } while (v);
    while (n < width) t[n++] = '0';
    while (n) *o++ = t[--n];
    return o;
}
SH_FAR __attribute__((optimize("Os"))) void sh_printf(const char *fmt, ...) {
    char buf[256], *o = buf, *const end = buf + sizeof buf - 24;
    va_list va; va_start(va, fmt);
    for (const char *f = fmt; *f && o < end; ++f) {
        if (*f != '%') { *o++ = *f; continue; }
        int width = 0; ++f;
        while (*f >= '0' && *f <= '9') width = width * 10 + (*f++ - '0');
        if (*f == '.') { ++f; while (*f >= '0' && *f <= '9') ++f; }
        while (*f == 'l') ++f;
        if (*f == 's') { const char *a = va_arg(va, const char *); while (*a && o < end) *o++ = *a++; }
        else if (*f == 'c') *o++ = (char)va_arg(va, int);
        else if (*f == 'u') o = sh_tp_u(o, va_arg(va, uint32_t), 10, width);
        else if (*f == 'x') o = sh_tp_u(o, va_arg(va, uint32_t), 16, width);
        else if (*f == 'd') { int32_t v = va_arg(va, int32_t); if (v < 0) { *o++ = '-'; v = -v; } o = sh_tp_u(o, (uint32_t)v, 10, width); }
        else if (*f == 'f') { double v = va_arg(va, double); if (v < 0) { *o++ = '-'; v = -v; }
                              const uint32_t ip = (uint32_t)v; o = sh_tp_u(o, ip, 10, 0); *o++ = '.';
                              o = sh_tp_u(o, (uint32_t)((v - (double)ip) * 1e6), 10, 6); }
        else if (*f == '%') *o++ = '%';
        else { *o++ = '?'; if (!*f) break; }
    }
    va_end(va);
    volatile uint32_t *uart = (volatile uint32_t *)(ARCH_SOC_REGISTER_EOC + 16);
    for (const char *c = buf; c < o; ++c) *uart = (uint32_t)*c;
}
#endif

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
SH_FAR void sh_preload_wait(uint64_t sentinel) {
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

/* In-network multicast: the SDK's bare_dma_start_1d_broadcast loads dst/src/size into register variables a0..a4 and
 * THEN calls bare_dma_set_mask(); when that call is not inlined (sh_train.inc.c declares the SDK helpers `extern inline`
 * to keep -Os code small) the call clobbers a0..a4 and the DMA goes to the mask value (gvsoc: "No entry found for burst
 * (base: 0xfffc00000000fffc)"). This version sets the mask first, then binds the registers. */
static inline __attribute__((always_inline)) void sh_dma_bcast_1d(uint64_t dst_off, uint64_t src_off, uint32_t size,
                                                                  uint16_t row_mask, uint16_t col_mask) {
    {
        register uint32_t reg_mask asm("a5") = ((uint32_t)col_mask << 16) | row_mask;
        asm volatile(".word %0\n" ::"i"(R_TYPE_ENCODE(DMMASK_FUNCT7, 15, 15, XDMA_FUNCT3, 15, OP_CUSTOM1)), "r"(reg_mask) : "memory");
    }
    const uint64_t dst = remote_pos(get_pos(flex_get_cluster_id()), dst_off), src = local(src_off);
    register uint32_t reg_dst_low asm("a0") = (uint32_t)(dst >> 0);
    register uint32_t reg_dst_high asm("a1") = (uint32_t)(dst >> 32);
    register uint32_t reg_src_low asm("a2") = (uint32_t)(src >> 0);
    register uint32_t reg_src_high asm("a3") = (uint32_t)(src >> 32);
    register uint32_t reg_size asm("a4") = size;
    asm volatile(".word %0\n" ::"i"(R_TYPE_ENCODE(DMSRC_FUNCT7, 13, 12, XDMA_FUNCT3, 0, OP_CUSTOM1)), "r"(reg_src_high), "r"(reg_src_low));
    asm volatile(".word %0\n" ::"i"(R_TYPE_ENCODE(DMDST_FUNCT7, 11, 10, XDMA_FUNCT3, 0, OP_CUSTOM1)), "r"(reg_dst_high), "r"(reg_dst_low));
    register uint32_t reg_txid asm("a0");
    asm volatile(".word %1\n" : "=r"(reg_txid) : "i"(R_TYPE_ENCODE(DMCPYC_FUNCT7, 0b00001, 14, XDMA_FUNCT3, 10, OP_CUSTOM1)), "r"(reg_size) : "memory");
    (void)reg_txid;
}

/* ---- cluster sets (sh_ops.h: SH_GROUP) -------------------------------------------------------------------------------
 * Internal helpers used by every op's dealing code: P = sh_set_P(cluster) clusters share the work, the calling cluster has
 * rank sh_set_r(cluster) among them (SH_SET_NONE: not a member). For SH_ALL these are the old (P = all, rank = id). */
#define SH_SET_NONE 0xFFFFFFFFu
#define SH_NCL (ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y)
static inline uint32_t sh_popc16(uint32_t m) {
    m = m - ((m >> 1) & 0x5555u); m = (m & 0x3333u) + ((m >> 2) & 0x3333u); m = (m + (m >> 4)) & 0x0F0Fu; return (m + (m >> 8)) & 0x1Fu;
}
static inline int sh_set_multi(uint32_t c) { return c == SH_ALL || SH_IS_GROUP(c); }
static inline uint32_t sh_set_P(uint32_t c) { return c == SH_ALL ? SH_NCL : SH_IS_GROUP(c) ? sh_popc16(c & 0xFFFFu) : 1u; }
static inline uint32_t sh_set_r(uint32_t c) {
    const uint32_t cid = flex_get_cluster_id();
    if (c == SH_ALL) return cid;
    if (SH_IS_GROUP(c)) return ((c >> cid) & 1u) ? sh_popc16(c & ((1u << cid) - 1u)) : SH_SET_NONE;
    return c == cid ? 0u : SH_SET_NONE;
}
/* the cluster id of member i (mod the set size): head / item i of a set-dealt op */
static inline uint32_t sh_set_nth(uint32_t c, uint32_t i) {
    if (c == SH_ALL) return i % SH_NCL;
    if (!SH_IS_GROUP(c)) return c;
    uint32_t m = c & 0xFFFFu, k = i % sh_popc16(m), id = 0;
    for (;; ++id) if ((m >> id) & 1u) { if (k == 0) return id; --k; }
}
#define SH_SET_SYNC_CNT 64u
#define SH_SET_SYNC_ITER 68u
/* barrier of the clusters in `mask`: all cores of every member call it */
static void sh_group_barrier(uint32_t mask) {
    flex_intra_cluster_sync();
    if (flex_is_dm_core()) {
        flex_annotate_barrier(0);
        uint32_t root = 0; while (!((mask >> root) & 1u)) ++root;
        volatile uint32_t *cnt  = (volatile uint32_t *)(ARCH_SYNC_BASE + root * ARCH_SYNC_SIZE + SH_SET_SYNC_CNT);
        volatile uint32_t *iter = (volatile uint32_t *)(ARCH_SYNC_BASE + root * ARCH_SYNC_SIZE + SH_SET_SYNC_ITER);
        const uint32_t n = sh_popc16(mask), prev = *iter;
        if (__atomic_fetch_add((uint32_t *)cnt, 1u, __ATOMIC_RELAXED) == n - 1u) {
            *cnt = 0u;
            __atomic_fetch_add((uint32_t *)iter, 1u, __ATOMIC_RELAXED);
        } else {
            while (*iter == prev) { const uint32_t t0 = sh_cycles(); while ((uint32_t)(sh_cycles() - t0) < 32u) {} }   /* back off: fewer NoC polls */
        }
        flex_annotate_barrier(0);
    }
    flex_intra_cluster_sync();
}
/* end of a set-dealt op */
static inline void sh_set_end(uint32_t c) {
    if (c == SH_ALL) flex_global_barrier_xy();
    else if (SH_IS_GROUP(c)) { if ((c >> flex_get_cluster_id()) & 1u) sh_group_barrier(c & 0xFFFFu); }
}
uint32_t sh_set_size(uint32_t c)   { return sh_set_P(c); }
int      sh_set_member(uint32_t c) { return sh_set_r(c) != SH_SET_NONE; }
uint32_t sh_set_leader(uint32_t c) { return c == SH_ALL ? 0u : SH_IS_GROUP(c) ? sh_set_nth(c, 0) : c; }
void     sh_set_barrier(uint32_t c) { if (sh_set_multi(c)) sh_set_end(c); else flex_intra_cluster_sync(); }
