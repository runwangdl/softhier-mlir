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

RESULTS_C1

### 241 tokens (3 cameras), 16 layers, 16 clusters

RESULTS_C3
