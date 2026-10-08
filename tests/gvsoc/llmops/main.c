/* gvsoc test: the Llama-style ops of the library (runtime/sh_llm.inc.c) vs a host numpy twin
 * (tests/gvsoc/run.py llmops). shape.h: SEQ D_MODEL N_HEADS N_KV_HEADS CLUSTER NSAMPLES.
 * One ROI per op (every sh_timer_end prints a period and restarts the timer): RMS, ROPE, SILU, SMA
 * (masked softmax, SmolVLA-style prefix mask), SMC (causal mask), O (GQA attention with the prefix mask),
 * OU (GQA attention without a mask). */
#include "sh_ops.h"
#include "shape.h"

#define DH 64
#define DKV (N_KV_HEADS * DH)

/* The SmolVLA-style prefix token classes at sequence length S (host twin: run.py llm_tokens):
 * image tokens [0, S-64) -> 0, language [S-64, S-16): the first 5 valid (0), the rest PAD, state token
 * S-16 -> 1, tail PAD. causal: tok[j] = j. */
static uint32_t tok_prefix(uint32_t j, uint32_t S) {
    const uint32_t img = S - 64;
    if (j < img) return 0;
    if (j < img + 5) return 0;
    if (j < img + 48) return SH_LLM_PAD;
    if (j == img + 48) return 1;
    return SH_LLM_PAD;
}

int main(void) {
    sh_init();
    const uint32_t S = SEQ, D = D_MODEL;
    const uint32_t mb = (S * D * 2 + 4095) & ~4095u, sb = (S * S * 2 + 4095) & ~4095u;
    uint64_t off = 0;
    #define ALLOC(bytes) (off += (bytes), sh_hbm_addr(off - (bytes)))
    const uint64_t x = ALLOC(mb), b = ALLOC(mb), g = ALLOC(4096), tab = ALLOC((S * DH * 2 + 4095) & ~4095u);
    const uint64_t rms = ALLOC(mb), rope = ALLOC(mb), silu = ALLOC(mb);
    const uint64_t sc = ALLOC(sb), sma = ALLOC(sb), smc = ALLOC(sb);
    const uint64_t q = ALLOC(mb), k = ALLOC(mb), v = ALLOC(mb), o = ALLOC(mb), ou = ALLOC(mb);
    const uint64_t toka = ALLOC(4096), tokc = ALLOC(4096);
    #undef ALLOC
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    if (lead) {
        sh_test_fill_fp16(x, S, D, D, 11, -16, 16, 0.125f);
        sh_test_fill_fp16(b, S, D, D, 12, -16, 16, 0.125f);
        sh_test_fill_fp16(g, 1, D, D, 13, 1, 8, 0.25f);
        sh_test_fill_fp16(tab, S, DH, DH, 15, -16, 16, 0.0625f);
        sh_test_fill_fp16(sc, S, S, S, 16, -32, 32, 0.25f);
        sh_test_fill_fp16(q, S, D, D, 21, -8, 8, 0.125f);
        sh_test_fill_fp16(k, S, DKV, DKV, 22, -8, 8, 0.125f);
        sh_test_fill_fp16(v, S, DKV, DKV, 23, -16, 16, 0.125f);
        sh_test_fill_fp16(o, S, D, D, 24, 7, 7, 1.0f);    /* poison */
        sh_test_fill_fp16(ou, S, D, D, 24, 7, 7, 1.0f);
        volatile uint16_t *ta = (volatile uint16_t *)(uintptr_t)toka, *tc = (volatile uint16_t *)(uintptr_t)tokc;
        for (uint32_t j = 0; j < S; ++j) { ta[j] = (uint16_t)tok_prefix(j, S); tc[j] = (uint16_t)j; }
        sh_printf("[llmops] S=%u D=%u H=%u Hkv=%u cluster=%s\n", S, D, (uint32_t)N_HEADS, (uint32_t)N_KV_HEADS, CLUSTER == SH_ALL ? "all" : "0");
    }
    sh_barrier_global();
    int rc = 0;
    if (lead) sh_timer_start();
    sh_rmsnorm(rms, x, g, S, D, D, 1e-5f, CLUSTER);                                  if (lead) sh_timer_end();
    sh_rope(rope, x, tab, S, D, D, DH, DH, CLUSTER);                                 if (lead) sh_timer_end();
    sh_silu_mul(silu, x, b, S, D, D, CLUSTER);                                       if (lead) sh_timer_end();
    sh_softmax_masked(sma, sc, S, S, S, 0.5f, toka, toka, CLUSTER);                  if (lead) sh_timer_end();
    sh_softmax_masked(smc, sc, S, S, S, 0.5f, tokc, tokc, CLUSTER);                  if (lead) sh_timer_end();
    rc |= sh_attention_gqa(q, k, v, o, S, D, N_HEADS, N_KV_HEADS, D, DKV, DKV, D, 0.125f, toka, CLUSTER);
    if (CLUSTER != SH_ALL) sh_barrier_global();
    if (lead) sh_timer_end();
    rc |= sh_attention_gqa(q, k, v, ou, S, D, N_HEADS, N_KV_HEADS, D, DKV, DKV, D, 0.125f, 0, CLUSTER);
    if (CLUSTER != SH_ALL) sh_barrier_global();
    if (lead) sh_timer_end();
    sh_barrier_global();
    if (lead) {
        if (rc) sh_printf("[llmops] attention rc=%d LLMOPS_FAIL\n", rc);
        static const char *ph[6] = { "stage+kT", "-", "scores", "softmax", "pv", "norm+store" };
        uint32_t prev = sh_attention_profile(S, DH, 0);
        for (uint32_t i = 1; i <= 6; ++i) {
            uint32_t c = sh_attention_profile(S, DH, i);
            if (i != 2) sh_printf("[llmops] attention phase %-10s %u cycles\n", ph[i - 1], c - prev);
            prev = c;
        }
        sh_test_dump_samples(rms, S, D, D, 101, NSAMPLES, "RMS");
        sh_test_dump_samples(rope, S, D, D, 102, NSAMPLES, "ROPE");
        sh_test_dump_samples(silu, S, D, D, 103, NSAMPLES, "SILU");
        sh_test_dump_samples(sma, S, S, S, 104, NSAMPLES, "SMA");
        sh_test_dump_samples(smc, S, S, S, 105, NSAMPLES, "SMC");
        sh_test_dump_samples(o, S, D, D, 106, NSAMPLES, "O");
        sh_test_dump_samples(ou, S, D, D, 107, NSAMPLES, "OU");
        sh_printf("LLMOPS_DONE\n");
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
