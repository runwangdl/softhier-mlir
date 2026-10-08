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
