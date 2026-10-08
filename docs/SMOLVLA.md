# SmolVLA on SoftHier: result ledger

What runs on the gvsoc, against which reference, with which tolerances and simulated times. The vision
tower (SigLIP, `tests/gvsoc/run.py smolvla`) is summarised in the README; this file covers the **VLM text
prefix** (`run.py smolvla-vlm`, branch `agent/vlm-prefix`), i.e. everything the action expert's
cross-attention consumes.

## The model, as SmolVLA actually uses it (`lerobot` `modeling_smolvla.py` / `smolvlm_with_expert.py`)

Checkpoint `/app/models/smolvla_base` (`config.json`: `vlm_model_name HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
`num_vlm_layers 16`, `attention_mode cross_attn`, `self_attn_every_n_layers 2`, `expert_width_multiplier 0.75`,
`tokenizer_max_length 48`, `pad_language_to max_length`, `resize_imgs_with_padding 512x512`, 3 cameras).

| | |
|---|---|
| text tower | SmolVLM2's Llama text model (`model.vlm_with_expert.vlm.model.text_model.*`) **cut to the first 16 of 32 layers**; hidden 960, 15 query heads x 64, **5 key/value heads** (GQA group 3, `k_proj`/`v_proj` are 320 wide), MLP 2560 SiLU-gated (`down(silu(gate x) * up x)`), RMSNorm eps 1e-5, **no biases anywhere**, `norm.weight` final norm |
| connector | `pixel_shuffle(scale 4)` of the 1024 SigLIP tokens -> 64 tokens x 12288, then `modality_projection.proj` [960, 12288] (no bias) |
| prefix | per camera: 64 image tokens x sqrt(960); then 48 language tokens `embed_tokens[ids] * sqrt(960)` (the task string + `\n`, padded to 48 with `<\|im_end\|>` = id 2, `lang_mask` 0 on the padding); then **1 state token** `state_proj(pad(state, 32))` (bias, no scaling). 3 cameras: 192 + 48 + 1 = **241 tokens** |
| RoPE | lerobot's own `apply_rope(max_wavelength=10_000)`, rotate-half convention (`y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin`, frequencies `10000^(-2i/64)`) -- **not** the checkpoint's `rope_theta = 100000`. Positions = `cumsum(pad_mask) - 1`: **language padding does not advance the position**, so the state token sits at position `n_img + n_valid_lang`, not 240 |
| attention mask | `make_att_2d_masks(pad, att)` with `att = [0]*n_img + [0]*48 + [1]`: `att2d[i, j] = cumsum(att)[j] <= cumsum(att)[i] & pad[i] & pad[j]`. So image + language tokens attend **all valid image + language tokens bidirectionally and never the state token**; the state token attends everything valid; padding is never a key, and a padding query row is fully masked (torch gives it uniform probabilities) |
| what the expert reads | `past_key_values[layer]["key_states" / "value_states"]` = the layer's k / v **after RoPE**, `[B, S, 5, 64]`; the expert (hidden 720) projects its queries to 960 = 15 x 64 and attends these with the same group mapping. The final `norm` output of the prefix is **not** used by the expert |

Surprises worth knowing: the RoPE base (10000 vs the config's 100000; lerobot never reads the config's value),
the padding-aware positions, the state token being invisible to the image/language tokens (so its own
row is the only one that differs from a plain bidirectional encoder), `embed_tokens` scaled by sqrt(960)
(as in PaliGemma, not in SmolVLM itself), and the residual stream carrying "massive activations": |x| up to
2200 in a few channels of the image tokens from the embeddings on (fp16 ulp 1-2 there), while K/V/OUT stay
below 20.

## Library ops (`runtime/sh_llm.inc.c`, fp16 HBM tensors, fp16 SIMD on the three cores, every op SPMD)

| op | call | notes |
|---|---|---|
| RMSNorm | `sh_rmsnorm(y, x, gamma, rows, cols, ld, eps, cluster)` | sum of squares in fp16 lanes of x/16, folded to fp32 every 16 elements (fp32 recount for tiny rows), output pass `x * rs * gamma` |
| RoPE | `sh_rope(y, x, cos_sin, rows, cols, ld, ld_table, dh, cluster)` | the host preloads `[S, dh]` fp16 rows `cos[dh/2] \| sin[dh/2]` for each row's position; every head of the row; 5 SIMD ops per 4 pairs |
| SiLU-gate | `sh_silu_mul(y, a, b, ...)` | `a b / (1 + 2^(-a log2 e))` with the shared `sh_v4_exp2x4`, exponent clamped to [-14, 15] |
| masked softmax | `sh_softmax_masked(y, x, rows, cols, ld, scale, tok_q, tok_k, cluster)` | **mask = one uint16 per token**: `tok[j]` = cumulative attention-block id (`cumsum(att_masks)`), `SH_LLM_PAD` (0xFFFF) for padding; key j allowed for query i iff `tok[j] <= tok[i]` (unsigned, so padding is never attended); padding query rows -> uniform. Per distinct query class the kernel builds two fp16 rows over the keys (`keep` 1/0, `nb` 0/-inf), cached per core (4 slots): `e = 2^(s (x + nb) - m) * keep`. Causal = `tok[j] = j`, bidirectional = all 0, SmolVLA = 0 / PAD / 1 |
| GQA attention | `sh_attention_gqa(q, k, v, o, S, D, H, Hkv, ldq, ldk, ldv, ldo, scale, tok, cluster)` | query head h uses kv head `h / (H/Hkv)`; one head per cluster in TCDM (`sh_attention_head_masked`: the `sh_attention_head` structure with the SIMD masked softmax, `tok == 0` = no mask); `S <= 256, S % 4 == 0` |
| pixel shuffle | `sh_pixel_shuffle(dst, src, grid, D, s, cluster)` | iDMA gather: output token (gr, gb) = its 4x4 patch block row-major, 4 loads + 1 store of 24 KB per token (checked equal to `SmolVLMConnector.pixel_shuffle` on the host) |

Dialect: `softhier.rmsnorm`, `softhier.rope {head_dim}`, `softhier.silu_mul`, `softhier.scale`, `softhier.pixel_shuffle {scale}`,
and the optional `mask` operand (`memref<S x i16>`) on `softhier.softmax` / `softhier.attention` plus `kv_heads` on
`softhier.attention` (-> `sh_attention_gqa`; without either it still lowers to `sh_attention`). `arith.divui` / `remui`
over indices are lowered too (two-region weight families). FileCheck: `tests/filecheck/llm_translate.mlir` (15/15 pass).

### `tests/gvsoc/run.py llmops` (LCG data, numpy twin, 256 samples per op, atol = rtol = 2e-2)

S = 256, D = 960, 15 query / 5 kv heads; the softmax inputs are 256 x 256 with the SmolVLA-style mask (`SMA`: 192
image, 5 valid + 43 padded language, state) and the causal mask (`SMC`); `O` = GQA attention with the SmolVLA mask,
`OU` without a mask. 2026-10-08, gvsoc at 1 GHz, ideal HBM:

| op | 16 clusters (ns) | cluster 0, S = 128 (ns) | max abs err (16 cl / cl 0) |
|---|---|---|---|
| RMS (256 x 960) | 124 114 | 875 505 | 0.0020 / 0.0016 |
| ROPE (256 x 960, 15 heads) | 61 762 | 394 437 | 0.0000 / 0.0000 |
| SILU (256 x 960) | 150 207 | 1 059 988 | 0.0017 / 0.0018 |
| SMA (256 x 256, prefix mask) | 76 965 | 164 873 | 0.0000 / 0.0001 |
| SMC (256 x 256, causal) | 143 114 | 509 168 | 0.0002 / 0.0001 |
| O (GQA + mask, 15 heads) | 762 216 | 2 414 762 | 0.0007 / 0.0012 |
| OU (GQA, no mask) | 692 404 | 2 836 563 | 0.0009 / 0.0006 |

All PASS (0 bad samples). One masked head at S = 256 costs 674 k cycles on its cluster: stage + k^T 18 k, scores
GEMM 2 k, **softmax 628 k** (the cluster is instruction-fetch bound, docs/SIMULATOR_NOTES.md #7: ~9 cycles per
score element even in 4-lane SIMD), P.V 2.6 k, normalise + store 23 k. The causal softmax is 2x the prefix one
because every row is its own mask class (keep/-inf rows rebuilt per row).

GEMM shapes of the tower on RedMulE (`run.py gemm`, fp16, tiles tm x tn x tk): 256x960x960 and 128x960x960 with
320 x 320 tiles, 256x320x960, and the connector 64x960x12288 / 192x960x12288 with 320 x 256 tiles all PASS
(`--real` data: K = 12288 accumulated in fp16 gives max abs 0.09-0.10 on sums of ~2; this is the RedMulE fp16
accumulator, visible below as the EMB error). The old `gemm` test placed Z at a fixed 32 MB, under a 23.6 MB W:
fixed.

## The prefix on the device (`run.py smolvla-vlm`)

`python3 -m softhier_mlir.frontend.smolvla prepare-vlm --cams {1,3} --out /app/models/smolvla_base/vlm_c{1,3}.npz`
builds: the text-tower weights in library layout (fp16, 323 MiB), the host-prepared inputs (fp32 SigLIP outputs
of `test_image(seed)` per camera through the HF `SiglipVisionModel`, the language rows `embed_tokens[ids]` for
`"pick up the cube\n"` = ids `[18188, 614, 260, 20636, 198]` padded with 2, the state row `state_proj(state)`, the
token classes, the RoPE table), the fp32 reference with lerobot's semantics (`prefix_reference`, cross-checked
against `transformers.LlamaModel` with `rope_theta=10000` and the 2-D mask passed as a 4-D additive mask: **max
|diff| 3.05e-05** over two layers) and the fp16-floor twin (`numpy_reference_vlm`: fp32 math, every stored
intermediate rounded to fp16, the device's op order).

The program (`emit_vlm`): per camera `softhier.pixel_shuffle` + the connector GEMM straight into rows of the
residual buffer, `softhier.scale` sqrt(960) over the image + language rows (the language / state rows are
preloaded raw), then **one `scf.for` over the 16 layers** (rmsnorm, q/k/v GEMMs with **k and v written straight
into the per-layer KV cache**, rope on q and on the cached k, masked GQA attention one head per cluster, o-proj +
residual, rmsnorm, gate/up GEMMs, silu_mul, down GEMM + residual), final rmsnorm. Sequence padded to S_pad = 128
(113 tokens) / 256 (241 tokens) with PAD classes; program 49 KB (the instruction memory is 64 KB). Weights of
the 16 layers (300 MiB) do not fit the 256 MiB west HBM edge with the activations: layers >= `n_west` live at the
south edge (offset 0x30000000) and the layer loop selects the region with `arith.divui` (one address expression,
no second loop body).

**KV cache layout (for the expert agent)**: `kv_base` (printed by the run, 0x711000 for the 1-camera program),
`kv_stride = 2 * S_pad * 320 * 2` bytes per layer; layer L's keys at `kv_base + L * kv_stride` as `[S_pad, 320]`
fp16 row-major (kv head g = columns `[64g, 64g+64)`, **RoPE already applied**, rows >= n are padding), its values
at `+ S_pad * 320 * 2` with the same shape. The token-class array (`memref<S_pad x i16>`) and the RoPE table
`[S_pad, 64]` are preloaded next to the inputs; the expert must use the same classes (its own tokens get class 2
and attend everything) and positions continuing from `n_img + n_valid_lang + 1`.

Tolerance (as for the vision tower, `report_smolvla`): a tensor passes when every sampled element is within
`atol = max(0.05, 3 % of max |ref|) + 5 % relative` of the fp32 reference; the per-tensor max / median errors are
what matters, the atol is loose on the residual stream because of the massive activations (|x| ~ 2200, fp16
ulp 2).

### 113 tokens (1 camera: 64 image + 48 language + 1 state), 16 layers, 16 clusters

Run as two 8-layer programs (`--layers 8` and `--layer0 8 --layers 8`; host memory, see above), 128 samples per
tensor, 16 clusters, gvsoc at 1 GHz, ideal HBM. The second program starts from the fp32 reference's L8 rounded to
fp16, so its errors are those of layers 9-16 alone (the connector's error, below, does not carry over).
Preload image 174 MiB (8 layers + connector + inputs) / 150 MiB; program 49 KB; wall 117 s + 118 s.

Simulated time: preload wait 2.8 ms (not timed), connector + embedding scaling 0.344 ms, then

| layer | attention (ms) | proj + MLP (ms) | total (ms) | | layer | attention | proj + MLP | total |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.400 | 0.568 | 0.968 | | 9 | 0.357 | 0.569 | 0.926 |
| 2 | 0.356 | 0.570 | 0.926 | | 10 | 0.357 | 0.508 | 0.865 |
| 3 | 0.357 | 0.508 | 0.865 | | 11 | 0.357 | 0.508 | 0.865 |
| 4 | 0.357 | 0.508 | 0.865 | | 12 | 0.357 | 0.507 | 0.865 |
| 5 | 0.357 | 0.507 | 0.865 | | 13 | 0.357 | 0.565 | 0.923 |
| 6 | 0.357 | 0.565 | 0.923 | | 14 | 0.362 | 0.569 | 0.931 |
| 7 | 0.362 | 0.569 | 0.931 | | 15 | 0.362 | 0.569 | 0.931 |
| 8 | 0.362 | 0.569 | 0.931 | | 16 | 0.362 | 0.569 | 0.931 |

**0.909 ms per layer** (mean, 16 layers), **14.9 ms for the 16-layer prefix + 0.34 ms connector** at 113 tokens.
Attention = 15 heads on 15 clusters, each ~360 k cycles of which ~300 k are the fetch-bound softmax; proj + MLP =
7 GEMMs (M = 128) + 2 rmsnorm + silu_mul + 2 adds.

Accuracy (rows that are read: 64 image + 5 language + state; padded language rows excluded, see run.py):

| tensor | samples | max abs vs fp32 ref | median | max \|ref\| | max / max\|ref\| | vs fp16 floor max abs | |
|---|---|---|---|---|---|---|---|
| EMB | 79 | 6.5388 | 0.6482 | 2212.90 | 2.95e-03 | 6.7500 | PASS |
| L1 | 79 | 6.8927 | 0.6208 | 2255.60 | 3.06e-03 | 7.0000 | PASS |
| K1 | 82 | 0.0627 | 0.0055 | 11.57 | 5.42e-03 | 0.0615 | PASS |
| V1 | 84 | 0.0023 | 0.0002 | 0.66 | 3.46e-03 | 0.0023 | PASS |
| L2 | 79 | 6.9321 | 0.5769 | 2325.65 | 2.98e-03 | 7.1250 | PASS |
| K2 | 82 | 0.0553 | 0.0095 | 18.20 | 3.04e-03 | 0.0547 | PASS |
| V2 | 84 | 0.0142 | 0.0040 | 1.74 | 8.17e-03 | 0.0144 | PASS |
| L3 | 79 | 6.9372 | 0.5997 | 2358.33 | 2.94e-03 | 7.2500 | PASS |
| K3 | 82 | 0.0365 | 0.0105 | 17.11 | 2.13e-03 | 0.0371 | PASS |
| V3 | 84 | 0.0207 | 0.0045 | 2.50 | 8.26e-03 | 0.0210 | PASS |
| L4 | 79 | 6.9009 | 0.6004 | 2364.65 | 2.92e-03 | 7.2500 | PASS |
| K4 | 82 | 0.0682 | 0.0093 | 12.64 | 5.39e-03 | 0.0684 | PASS |
| V4 | 84 | 0.0191 | 0.0057 | 2.58 | 7.38e-03 | 0.0192 | PASS |
| L5 | 79 | 6.9612 | 0.6040 | 2374.72 | 2.93e-03 | 7.2500 | PASS |
| K5 | 82 | 0.0700 | 0.0120 | 11.97 | 5.85e-03 | 0.0693 | PASS |
| V5 | 84 | 0.0339 | 0.0058 | 2.70 | 1.26e-02 | 0.0334 | PASS |
| L6 | 79 | 6.9665 | 0.6108 | 2358.66 | 2.95e-03 | 7.2500 | PASS |
| K6 | 82 | 0.0525 | 0.0101 | 15.23 | 3.45e-03 | 0.0527 | PASS |
| V6 | 84 | 0.0342 | 0.0078 | 3.19 | 1.07e-02 | 0.0342 | PASS |
| L7 | 79 | 6.9631 | 0.6528 | 2337.71 | 2.98e-03 | 7.2500 | PASS |
| K7 | 82 | 0.0734 | 0.0092 | 13.98 | 5.25e-03 | 0.0781 | PASS |
| V7 | 84 | 0.0428 | 0.0073 | 2.90 | 1.48e-02 | 0.0420 | PASS |
| L8 | 79 | 6.8209 | 0.6517 | 2319.41 | 2.94e-03 | 7.0000 | PASS |
| K8 | 82 | 0.0768 | 0.0082 | 15.05 | 5.10e-03 | 0.0742 | PASS |
| V8 | 84 | 0.0314 | 0.0073 | 3.10 | 1.01e-02 | 0.0322 | PASS |
| L9 | 79 | 0.1902 | 0.0214 | 2321.03 | 8.19e-05 | 0.3750 | PASS |
| K9 | 82 | 0.0327 | 0.0028 | 14.17 | 2.31e-03 | 0.0312 | PASS |
| V9 | 84 | 0.0139 | 0.0018 | 3.60 | 3.86e-03 | 0.0137 | PASS |
| L10 | 79 | 0.2180 | 0.0234 | 2367.68 | 9.21e-05 | 0.3750 | PASS |
| K10 | 82 | 0.0367 | 0.0025 | 14.00 | 2.62e-03 | 0.0391 | PASS |
| V10 | 84 | 0.0223 | 0.0016 | 4.11 | 5.43e-03 | 0.0215 | PASS |
| L11 | 79 | 0.2916 | 0.0296 | 2398.00 | 1.22e-04 | 0.3750 | PASS |
| K11 | 82 | 0.0235 | 0.0042 | 13.54 | 1.73e-03 | 0.0195 | PASS |
| V11 | 84 | 0.0132 | 0.0019 | 3.66 | 3.60e-03 | 0.0137 | PASS |
| L12 | 79 | 0.2640 | 0.0286 | 2397.12 | 1.10e-04 | 0.3750 | PASS |
| K12 | 82 | 0.0554 | 0.0036 | 14.11 | 3.93e-03 | 0.0625 | PASS |
| V12 | 84 | 0.0220 | 0.0023 | 4.11 | 5.36e-03 | 0.0234 | PASS |
| L13 | 79 | 0.3342 | 0.0330 | 2383.96 | 1.40e-04 | 0.5000 | PASS |
| K13 | 82 | 0.0363 | 0.0028 | 16.32 | 2.22e-03 | 0.0391 | PASS |
| V13 | 84 | 0.0187 | 0.0020 | 3.19 | 5.85e-03 | 0.0195 | PASS |
| L14 | 79 | 0.5952 | 0.0443 | 2471.67 | 2.41e-04 | 0.5000 | PASS |
| K14 | 82 | 0.0342 | 0.0041 | 16.24 | 2.10e-03 | 0.0352 | PASS |
| V14 | 84 | 0.0167 | 0.0017 | 2.92 | 5.70e-03 | 0.0161 | PASS |
| L15 | 79 | 0.5748 | 0.0539 | 2477.17 | 2.32e-04 | 0.5000 | PASS |
| K15 | 82 | 0.0294 | 0.0024 | 16.71 | 1.76e-03 | 0.0273 | PASS |
| V15 | 84 | 0.0185 | 0.0019 | 4.14 | 4.46e-03 | 0.0156 | PASS |
| L16 | 79 | 1.0029 | 0.0620 | 2488.67 | 4.03e-04 | 0.5000 | PASS |
| K16 | 82 | 0.0239 | 0.0033 | 18.11 | 1.32e-03 | 0.0200 | PASS |
| V16 | 84 | 0.0320 | 0.0027 | 4.29 | 7.46e-03 | 0.0312 | PASS |
| OUT | 82 | 0.0097 | 0.0006 | 24.76 | 3.93e-04 | 0.0078 | PASS |

Reading: the residual stream's max error (6.5-7.0 from EMB through L8) is the **connector GEMM** (K = 12288
accumulated in fp16 on RedMulE; the fp16 floor twin accumulates in fp32, which is why "vs floor" is the same 7):
it sits in the massive-activation channels (|x| ~ 2200, fp16 ulp 2, i.e. ~3 ulp) of a few image tokens and is
carried unchanged by the residual adds; the layers themselves add 0.2-1.0 (L9-L16 restart from an exact L8).
Everything the expert consumes is at the fp16 floor: **K within 0.08 (0.2-0.6 % of its max 12-18), V within
0.043 (0.4-1.5 % of its max 0.7-4.3)**, the final norm within 0.01 of 24.8. Median errors are 2-10x smaller.

### 241 tokens (3 cameras), 16 layers, 16 clusters

RESULTS_C3
