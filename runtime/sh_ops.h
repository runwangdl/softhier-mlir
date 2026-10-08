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
void     sh_printf(const char *fmt, ...);
uint64_t sh_hbm_addr(uint64_t byte_offset);   /* HBM base + offset */
uint64_t sh_hbm_malloc(uint32_t bytes);       /* first core of each cluster only (SDK allocator) */
uint32_t sh_l1_size(void);                    /* TCDM bytes per cluster */

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

/* Single-cluster tiled GEMM, runs on cluster `cluster` (others return at once).
 * Requires M%tm==0, N%tn==0, K%tk==0 and the 5-tile scratch to fit in TCDM.
 * Returns 0 on success, <0 on a constraint violation (reason printed by the first core). */
int sh_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
            uint32_t ldx, uint32_t ldw, uint32_t ldz, const sh_gemm_cfg *cfg, uint32_t cluster);

/* Bytes of TCDM the given cfg needs (so a compiler can check the budget without running). */
uint32_t sh_gemm_l1_bytes(uint32_t M, uint32_t N, uint32_t K, const sh_gemm_cfg *cfg);

/* ---- test helpers (on-device data generation + self-check, no host round trip) --------- */
/* Fill rows x cols fp16 matrix (ld elements) with integers in [lo, hi] from an LCG(seed). First core. */
void sh_test_fill_int_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, int lo, int hi);
/* Check `nsamples` pseudo-random positions of Z against z0 + a scalar fp32 dot product of X,W
 * (z0 = the constant Z was pre-filled with when testing accumulate=1). Returns number of mismatches (|diff| > tol). First core. Prints a summary. */
uint32_t sh_test_check_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                            uint32_t ldx, uint32_t ldw, uint32_t ldz, uint32_t nsamples, float tol,
                            float z0, const char *tag);
/* fp16 <-> fp32 on the host-side convention (IEEE binary16). */
float    sh_fp16_to_f32(uint16_t h);
uint16_t sh_f32_to_fp16(float f);

#endif /* SH_OPS_H */
