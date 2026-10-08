/* softhier-ops: the SoftHier operator library that softhier-mlir generated code calls.
 *
 * Design rules
 *  - This header is the ONLY thing generated code (and tests) include. The SoftHier SDK
 *    headers define non-static functions, so they may be included by exactly one
 *    translation unit: that unit is sh_ops.c (a unity build of sh_*.inc.c).
 *  - Every operator is called by ALL cores of a cluster (SPMD); the operator dispatches
 *    the DM core (iDMA), the first core (RedMulE / scalar math) and syncs internally.
 *  - Tensors are fp16 row-major in HBM, addressed by 64-bit byte addresses.
 *  - Operators take an explicit tiling/config struct so the compiler (decide) and the
 *    library (apply) stay separate; 0 means "library default".
 */
#ifndef SH_OPS_H
#define SH_OPS_H
#include <stdint.h>
#include <stddef.h>

/* ---- runtime wrappers (so callers never include the SDK) ---------------------------- */
void     sh_init(void);                 /* barrier init + allocator init; call first, all cores */
void     sh_barrier_global(void);       /* all clusters, all cores */
void     sh_barrier_cluster(void);      /* the 3 cores of this cluster */
uint32_t sh_cluster_id(void);
uint32_t sh_core_id(void);
int      sh_is_first_core(void);        /* compute core: RedMulE + scalar FP */
int      sh_is_dm_core(void);           /* data-mover core: all iDMA */
uint32_t sh_num_clusters(void);
void     sh_timer_start(void);          /* global timer: call from ONE core only */
void     sh_timer_end(void);
void     sh_eoc(uint32_t val);
uint32_t sh_cycles(void);               /* this core's cycle counter (mcycle CSR = the gvsoc clock) */
void     sh_printf(const char *fmt, ...);
uint64_t sh_hbm_addr(uint64_t byte_offset);   /* HBM base + offset */
uint64_t sh_hbm_malloc(uint32_t bytes);       /* first core of each cluster only (SDK allocator) */
uint32_t sh_l1_size(void);                    /* TCDM bytes per cluster */

#define SH_ALL 0xFFFFFFFFu   /* `cluster` argument: split the work over all clusters (global barrier at the end) */

/* ---- formats ------------------------------------------------------------------------ */
enum sh_fmt { SH_FP16 = 0, SH_FP8 = 1, SH_INT16 = 2, SH_INT8 = 3 };

/* ---- GEMM:  Z[M,N] (+)= X[M,K] . W[K,N]   (fp16 row-major, leading dims in elements) ---- */
typedef struct {
    uint32_t tm, tn, tk;     /* RedMulE tile; 0 -> 256. X tile tm x tk, W tile tk x tn, Z tile tm x tn */
    uint32_t pipeline;       /* 1: double-buffer X/W and prefetch K-tile k+1 during compute */
    uint32_t accumulate;     /* 1: Z += X.W (Z tile loaded first); 0: Z = X.W (tile zeroed via ZOMEM) */
    uint32_t fmt;            /* enum sh_fmt for the RedMulE datapath (default SH_FP16) */
    uint32_t l1_base;        /* TCDM byte offset of the scratch area (default 0) */
} sh_gemm_cfg;

/* Tiled GEMM. `cluster` = the one cluster that runs it, or SH_ALL: output tiles are dealt
 * round-robin over all clusters (each streams its own panels from HBM; global barrier at the end).
 * Requires M%tm==0, N%tn==0, K%tk==0 and the 5-tile scratch to fit in TCDM.
 * Returns 0 on success, <0 on a constraint violation (reason printed by the first core). */
int sh_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
            uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg, uint32_t cluster);

/* Mesh-wide SUMMA GEMM over all P x P clusters: cluster (px,py) owns Z tile [py,px]; diagonal
 * clusters stream the X/W panels from HBM and multicast them along their row/column. Requires a
 * square mesh, tm == tn, M == N == P*tm, K % tk == 0. Call from all cores of all clusters. */
int sh_gemm_mesh(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                 uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg);

/* Bytes of TCDM the given cfg needs (so a compiler can check the budget without running). */
uint32_t sh_gemm_l1_bytes(uint32_t M, uint32_t N, uint32_t K, const sh_gemm_cfg *cfg);

/* ---- row-wise / elementwise ops on fp16 HBM tensors (rows x cols, leading dim ld) -------------
 * `cluster`: executing cluster id, or SH_ALL to split row blocks over all clusters. */
void sh_layernorm(uint64_t y, uint64_t x, uint64_t gamma, uint64_t beta, uint32_t rows, uint32_t cols, uint32_t ld, float eps, uint32_t cluster);
void sh_softmax_rows(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float scale, uint32_t cluster); /* softmax(scale*x) per row */
void sh_gelu(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster);                    /* tanh approximation */
void sh_add(uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster);         /* y = a + b */
void sh_add_bias(uint64_t y, uint64_t x, uint64_t bias_row, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t cluster); /* y = x + bias (per column) */
void sh_scale(uint64_t y, uint64_t x, uint32_t rows, uint32_t cols, uint32_t ld, float s, uint32_t cluster);
void sh_transpose(uint64_t dst, uint64_t src, uint32_t rows, uint32_t cols, uint32_t ld_src, uint32_t ld_dst, uint32_t cluster); /* dst[cols,rows] */

/* ---- TCDM-resident ops (calling cluster only; intra-cluster sync inside) ------------------ */
uint32_t sh_l1_addr(uint32_t off);                                   /* TCDM byte offset -> address */
void     sh_l1_zero(uint32_t off, uint32_t bytes);                   /* via ZOMEM iDMA */
void     sh_l1_fill_fp16(uint32_t off, uint32_t n, uint16_t bits);
void     sh_l1_relu_fp16(uint32_t off, uint32_t n);
void     sh_l1_add_fp16(uint32_t dst, uint32_t src, uint32_t n);     /* dst += src */
void     sh_redmule(uint32_t x, uint32_t w, uint32_t y, uint32_t m, uint32_t n, uint32_t k, uint32_t fmt); /* y[m,n] += x[m,k].w[k,n] */
void     sh_dma_load_2d(uint32_t l1_off, uint64_t hbm, uint32_t rows, uint32_t cols, uint32_t ld);   /* HBM sub-block -> packed TCDM (2-D iDMA), DM core, sync */
void     sh_dma_store_rows(uint64_t hbm, uint32_t l1_off, uint32_t rows, uint32_t cols, uint32_t ld); /* packed TCDM -> HBM sub-block (per-row 1-D), DM core, sync */
void     sh_dma_copy(uint64_t dst, uint64_t src, uint32_t bytes);    /* 1-D, DM core, sync */

/* ---- test helpers (on-device data generation + self-check, no host round trip) --------- */
/* Fill rows x cols fp16 matrix (ld elements) with integers in [lo, hi] from an LCG(seed), times `scale`
 * (sh_test_fill_int_fp16 = scale 1). Twin implementation: softhier_mlir/testing/lcg.py. First core. */
void sh_test_fill_int_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, int lo, int hi);
void sh_test_fill_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, int lo, int hi, float scale);
/* Print nsamples fp16 codes of a matrix at LCG(seed) positions: "<tag> r c hex" lines, for host comparison. */
void sh_test_dump_samples(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, uint32_t nsamples, const char *tag);
/* Check `nsamples` pseudo-random positions of Z against z0 + a scalar fp32 dot product of X,W
 * (z0 = the constant Z was pre-filled with when testing accumulate=1). Returns number of mismatches (|diff| > tol). First core. Prints a summary. */
uint32_t sh_test_check_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                            uint32_t ldx, uint32_t ldw, uint32_t ldz, uint32_t nsamples, float tol,
                            float z0, const char *tag);
/* Constant fills / checks (cluster 0 does the work; global barrier inside, so call from all clusters). */
void     sh_test_fill_const_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t bits);
void     sh_test_fill_colparity_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t even, uint32_t odd);
uint32_t sh_test_check_const_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t bits, uint32_t tol, const char *tag);
uint32_t sh_test_check_const_l1_fp16(uint32_t off, uint32_t n, uint32_t bits, uint32_t tol, const char *tag);
/* fp16 <-> fp32 on the host-side convention (IEEE binary16), software (any core, any target). */
float    sh_fp16_to_f32(uint16_t h);
uint16_t sh_f32_to_fp16(float f);

/* Hardware fp16 <-> fp32 (Zfh register conversions). The gvsoc Snitch model's scalar flh/fsh are
 * broken (docs/SIMULATOR_NOTES.md #2) but fcvt.s.h / fcvt.h.s between registers work: load the half
 * with an integer lhu, NaN-box it into an FP register with fmv.w.x, convert; and back with
 * fcvt.h.s (RNE) + fmv.x.w + sh. ~1 FPU op each instead of ~60 integer ops. Verified bit-exact
 * against the software pair by tests/gvsoc/fp16cvt (NaN payloads excepted).
 * sh_f2h returns the half in the LOW 16 bits; the high 16 bits are the NaN-box (0xFFFF), so store
 * it through a uint16_t (one `sh`) or mask it. */
static inline float sh_h2f(uint32_t h) {
    float f;
    __asm__ ("fmv.w.x %0, %1\n\tfcvt.s.h %0, %0" : "=f"(f) : "r"(h | 0xFFFF0000u));
    return f;
}
static inline uint32_t sh_f2h(float f) {
    uint32_t h; float t;
    __asm__ ("fcvt.h.s %1, %2, rne\n\tfmv.x.w %0, %1" : "=r"(h), "=&f"(t) : "f"(f));
    return h;
}
/* 4-wide variants: one asm block each, so the four independent fmv/fcvt chains are issued
 * interleaved (the Snitch FPU model overlaps independent instructions but GCC schedules the
 * dependent pairs back to back). p must hold 4 halves. */
static inline void sh_h2f4(const uint16_t *p, float *a0, float *a1, float *a2, float *a3) {
    const uint32_t h0 = p[0] | 0xFFFF0000u, h1 = p[1] | 0xFFFF0000u, h2 = p[2] | 0xFFFF0000u, h3 = p[3] | 0xFFFF0000u;
    __asm__ ("fmv.w.x %0, %4\n\tfmv.w.x %1, %5\n\tfmv.w.x %2, %6\n\tfmv.w.x %3, %7\n\t"
             "fcvt.s.h %0, %0\n\tfcvt.s.h %1, %1\n\tfcvt.s.h %2, %2\n\tfcvt.s.h %3, %3"
             : "=&f"(*a0), "=&f"(*a1), "=&f"(*a2), "=&f"(*a3) : "r"(h0), "r"(h1), "r"(h2), "r"(h3));
}
static inline void sh_f2h4(uint16_t *p, float a0, float a1, float a2, float a3) {
    uint32_t h0, h1, h2, h3; float t0, t1, t2, t3;
    __asm__ ("fcvt.h.s %4, %8, rne\n\tfcvt.h.s %5, %9, rne\n\tfcvt.h.s %6, %10, rne\n\tfcvt.h.s %7, %11, rne\n\t"
             "fmv.x.w %0, %4\n\tfmv.x.w %1, %5\n\tfmv.x.w %2, %6\n\tfmv.x.w %3, %7"
             : "=&r"(h0), "=&r"(h1), "=&r"(h2), "=&r"(h3), "=&f"(t0), "=&f"(t1), "=&f"(t2), "=&f"(t3)
             : "f"(a0), "f"(a1), "f"(a2), "f"(a3));
    p[0] = (uint16_t)h0; p[1] = (uint16_t)h1; p[2] = (uint16_t)h2; p[3] = (uint16_t)h3;
}

#endif /* SH_OPS_H */
