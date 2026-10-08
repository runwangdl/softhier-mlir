/* On-device test data + self-check (no host round trip). All run on the calling core. */
static inline uint32_t sh_lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }

void sh_test_fill_int_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t seed, int lo, int hi) {
    uint32_t s = seed ^ 0x9e3779b9u;
    const uint32_t range = (uint32_t)(hi - lo + 1);
    for (uint32_t r = 0; r < rows; ++r) {
        volatile uint16_t *row = (volatile uint16_t *)(uintptr_t)(a + (uint64_t)r * ld * 2);
        for (uint32_t cc = 0; cc < cols; ++cc) {
            int v = lo + (int)(sh_lcg(&s) % range);
            row[cc] = sh_f32_to_fp16((float)v);
        }
    }
}

uint32_t sh_test_check_gemm(uint64_t x, uint64_t w, uint64_t z, uint32_t M, uint32_t N, uint32_t K,
                            uint32_t ldx, uint32_t ldw, uint32_t ldz, uint32_t nsamples, float tol,
                            float z0, const char *tag) {
    uint32_t s = 0xC0FFEEu, bad = 0;
    float maxdiff = 0.f;
    for (uint32_t n = 0; n < nsamples; ++n) {
        /* cover every tile region: stride samples over a grid, then jitter */
        uint32_t i = (n * 7919u + sh_lcg(&s)) % M, j = (n * 104729u + sh_lcg(&s)) % N;
        const volatile uint16_t *xr = (const volatile uint16_t *)(uintptr_t)(x + (uint64_t)i * ldx * 2);
        const volatile uint16_t *wc = (const volatile uint16_t *)(uintptr_t)(w + (uint64_t)j * 2);
        float acc = z0;
        for (uint32_t k = 0; k < K; ++k) acc += sh_fp16_to_f32(xr[k]) * sh_fp16_to_f32(wc[(uint64_t)k * ldw]);
        float got = sh_fp16_to_f32(((const volatile uint16_t *)(uintptr_t)(z + (uint64_t)i * ldz * 2))[j]);
        float d = got - acc; if (d < 0) d = -d;
        if (d > maxdiff) maxdiff = d;
        if (d > tol) { if (bad < 4) sh_printf("  mismatch z[%u,%u] got %f want %f\n", i, j, got, acc); bad++; }
    }
    sh_printf("%s samples=%u bad=%u maxdiff=%f %s\n", tag, nsamples, bad, maxdiff, bad ? "FAIL" : "PASS");
    return bad;
}

/* ---- constant fills / checks used by the MLIR examples (cluster 0 does the work) ----------- */
#define SH_SCRATCH_BYTES 0x10000u   /* top 64 KB of TCDM, used only by the test helpers */
static inline uint32_t sh_scratch_off(void) { return ARCH_CLUSTER_TCDM_SIZE - SH_SCRATCH_BYTES; }

/* HBM rows x cols (ld elements) = bits; even/odd column variant when odd_bits != 0xFFFFFFFF. */
static void sh_fill_rows(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t even, uint32_t odd) {
    if (flex_get_cluster_id() == 0) {
        const uint32_t rowb = cols * 2, scr = sh_scratch_off();
        if (flex_is_first_core()) {
            volatile uint16_t *p = (volatile uint16_t *)local(scr);
            for (uint32_t i = 0; i < cols; ++i) p[i] = (uint16_t)((odd != 0xFFFFFFFFu && (i & 1)) ? odd : even);
        }
        flex_intra_cluster_sync();
        if (flex_is_dm_core()) {
            for (uint32_t r = 0; r < rows; ++r) bare_dma_start_1d(a + (uint64_t)r * ld * 2, local(scr), rowb);
            bare_dma_wait_all();
        }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
}
void sh_test_fill_const_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t bits) {
    sh_fill_rows(a, rows, cols, ld, bits, 0xFFFFFFFFu);
}
void sh_test_fill_colparity_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t even, uint32_t odd) {
    sh_fill_rows(a, rows, cols, ld, even, odd);
}
/* Count HBM elements within tol (in fp16 ulps of the raw code) of bits; prints <tag>_CHECK / _PASS|_FAIL. */
uint32_t sh_test_check_const_fp16(uint64_t a, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t bits, uint32_t tol, const char *tag) {
    uint32_t ok = 0;
    flex_global_barrier_xy();   /* producer may be another cluster */
    if (flex_get_cluster_id() == 0) {
        const uint32_t rowb = cols * 2, scr = sh_scratch_off();
        const uint32_t rows_per = SH_SCRATCH_BYTES / rowb ? SH_SCRATCH_BYTES / rowb : 1;
        for (uint32_t r0 = 0; r0 < rows; r0 += rows_per) {
            const uint32_t nr = (rows - r0) < rows_per ? (rows - r0) : rows_per;
            if (flex_is_dm_core()) {
                bare_dma_start_2d(local(scr), a + (uint64_t)r0 * ld * 2, rowb, rowb, ld * 2, nr);
                bare_dma_wait_all();
            }
            flex_intra_cluster_sync();
            if (flex_is_first_core()) {
                volatile uint16_t *p = (volatile uint16_t *)local(scr);
                for (uint32_t i = 0; i < nr * cols; ++i) { int d = (int)p[i] - (int)bits; if (d < 0) d = -d; if ((uint32_t)d <= tol) ok++; }
            }
            flex_intra_cluster_sync();
        }
        if (flex_is_first_core())
            sh_printf("%s_CHECK ok=%u/%u %s_%s\n", tag, ok, rows * cols, tag, ok == rows * cols ? "PASS" : "FAIL");
    }
    flex_global_barrier_xy();
    return ok;
}
uint32_t sh_test_check_const_l1_fp16(uint32_t off, uint32_t n, uint32_t bits, uint32_t tol, const char *tag) {
    uint32_t ok = 0;
    if (flex_get_cluster_id() == 0 && flex_is_first_core()) {
        volatile uint16_t *p = (volatile uint16_t *)local(off);
        for (uint32_t i = 0; i < n; ++i) { int d = (int)p[i] - (int)bits; if (d < 0) d = -d; if ((uint32_t)d <= tol) ok++; }
        sh_printf("%s_CHECK ok=%u/%u %s_%s\n", tag, ok, n, tag, ok == n ? "PASS" : "FAIL");
    }
    flex_intra_cluster_sync();
    return ok;
}
