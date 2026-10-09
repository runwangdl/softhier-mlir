# SmolVLA action expert + flow-matching loop on SoftHier (ledger)

Branch `agent/action-expert`. Code: `runtime/sh_expert.inc.c` (kernels, prefix `sh_x_`), `softhier_mlir/dialects/softhier.py`
(ops `cross_attention`, `axpy`, `dump_all`, `silu_mul` with optional `b`, `gemm step/fmt_steps`; `rmsnorm` / `rope` shared
with agent/vlm-prefix), `softhier_mlir/backend/emit_c.py`, `softhier_mlir/frontend/smolvla_expert.py` (programs + numpy
twin), `softhier_mlir/frontend/smolvla_expert_ref.py` (lerobot reference), `tests/gvsoc/expert.py` (runner: `layer`,
`op`, `flow`), `tests/filecheck/expert_translate.mlir`.

## 1. What the model does at inference (read from lerobot 0.4.4 `modeling_smolvla.py` / `smolvlm_with_expert.py` and the checkpoint)

| item | value (checkpoint `lerobot/smolvla_base`) |
|---|---|
| expert | 16 Llama decoder layers, hidden 720 (`0.75 x 960`), intermediate 2048 (`get_intermediate_size(720)`), RMSNorm eps 1e-5, no biases |
| attention | 15 query heads x 64 (`q_proj [960, 720]`), 5 kv heads x 64 (GQA, query head h uses kv head h // 3), scale 1/8, eager fp32 softmax, `big_neg` mask |
| layer parity | `self_attn_every_n_layers = 2`: even layers are **self-attention** layers (`k/v_proj [320, 720]` on the expert's own tokens; keys/values = [VLM prefix KV of that layer ; own]), odd layers are **cross-attention** layers (`k/v_proj [320, 320]` applied to the VLM prefix KV *cache*, no own keys) |
| RoPE | lerobot's own `apply_rope` (half-split == HF rotate-half, base 10 000, fp32), not the HF model's theta 100 000. Self layers: q and own k at positions `n_valid + i` (`prefix_offsets + cumsum - 1`); cross layers: q at positions `i` (`expert_position_id - min`); the prefix keys are the cached, already rotated VLM keys; the projected cross keys get no RoPE |
| masks | prefix columns: `prefix_pad_masks` (language padding to 48 tokens is masked; images / state always valid); own columns (self layers only): causal `j <= i` (`make_att_2d_masks` with att_mask 1 on every action token) |
| prefix | 3 x 64 image tokens + 48 language tokens + 1 state token = 241; `n_valid = 204` for the test prompt (11 language tokens) |
| suffix embedding | `e = action_in_proj(x_t)`; `time_emb = sinusoidal(t, 720, 4e-3, 4.0)` (`[sin | cos]`, float64, fed fp32 `t`); `h = action_time_mlp_out(silu(action_time_mlp_in([e | time_emb])))` |
| flow | `num_steps = 10`, `dt = -0.1`, `t_s = 1 - 0.1 s`, `x_0 = noise ~ N(0, 1)` of shape `[50, 32]`, `x_{s+1} = x_s + dt * v_s`, `v_s = action_out_proj(final_rmsnorm(h_16))`; the action chunk is `x_10` |
| KV cache | `past_key_values[l] = {key_states, value_states}` `[B, 241, 5, 64]` per VLM layer, keys post-RoPE; the expert reads it flattened `[241, 320]` head-major |

Exact algebra used by the device program (no approximation beyond fp16):
* `[Wq^T | Wk^T | Wv^T]` fused into one `[720, 1600]` GEMM on self layers; `[Wgate^T | Wup^T]` fused into `[720, 4096]`;
  the SwiGLU product is written in place over the gate half.
* The time half of `action_time_mlp_in` is folded on the host into a per-step bias table `tb[s] = time_emb(t_s) @ W_in[:, 720:]^T + b_in`
  (`[10, 720]`, lerobot's own `time_emb` values; the numpy twin reproduces them to 3e-8 when fed fp32 `t`).
* The cross layers' `K_l @ Wk_l^T`, `V_l @ Wv_l^T` are step-invariant: computed **once per chunk** on the device before the step loop
  (16 GEMMs `[241, 320] x [320, 320]`, 0.345 ms), not once per step as lerobot's code does.

The numpy twin of exactly this program (fp32 math, fp16 rounding at every stored tensor) differs from lerobot's fp32 run by
at most 1.8e-3 on `x_t` at every step (0.0016 on the final actions): the algebra above is the model.

## 2. Device program

```
preload_wait
for p in 0..7:   kx_c[p] = kp[2p+1] @ wkx[2p+1];  vx_c[p] = vp[2p+1] @ wvx[2p+1]      (cross layers, once per chunk)
x = x0
for s in 0..9:                                                     (scf.for; GEMM fmt = fmt_steps[s] if given)
    e  = x @ wa + ba;  e1 = silu(e @ wti + tb[s]);  h = e1 @ wto + bto
    for p in 0..7:                                                 (scf.for over (self, cross) layer pairs)
        self  layer 2p  : xn = rmsnorm(h); qkv = xn @ wqkv; rope(q, tab_self); rope(k, tab_self)
                          o = cross_attention(q, kp[2p], vp[2p], own k, v, mask tok); h += o @ wo
                          xn = rmsnorm(h); gu = xn @ wgu; g = silu(g) * u; h += g @ wd
        cross layer 2p+1: xn = rmsnorm(h); q = xn @ wq; rope(q, tab_cross)
                          o = cross_attention(q, kx_c[p], vx_c[p], mask tok); h += o @ wo; ... same MLP
    fin = rmsnorm(h, gf);  v = fin @ wout + bout;  x = x + (-0.1) v        (axpy)
    mark step<s>; dump x
```
Program size is independent of the depth (two loops; per-pair parameter slabs at a constant HBM stride; 52.7 KB of code).
HBM image: 194 MiB fp16 (weights 189.5 MiB, prefix KV 4.7 MiB padded to 256 rows, tables).

### Prefix KV layout consumed (= agent/vlm-prefix `emit_vlm`'s)
`kv_base` + `L * kv_stride` (`kv_stride = 2 * S_pad * 320 * 2 = 0x50000` at `S_pad = 256`): layer `L` keys `[S_pad, 320]` fp16
row-major, kv head `g` = columns `[64g, 64g + 64)`, RoPE already applied (base 10 000, rotate-half); values at
`+ S_pad * 640`, same shape; rows `>= 241` and padded-language rows are padding. The mask is the prefix's uint16 token-class
array `tok[241]` (0 image / language, 1 state, 0xFFFF padding); the expert's queries are class 2, so lerobot's
"key j allowed iff tok[j] <= tok[i]" reduces to "not padding" on the prefix columns, plus causal own keys on the self
layers. `emit_flow(kv_base=..., kv_stride=..., s_pad=...)` points the program at the VLM program's region (nothing
preloaded then); without `kv_base` it allocates the region in this layout and preloads the host's KV. The expert reads
only the first 241 rows of each block (`memref<241x320xf16>` views at `kv_base + (2p [+1]) * kv_stride`). RoPE positions of
the self layers' q / own k: `n_valid + i`, with `n_valid = n_img + n_valid_lang + 1` (204 here).

### Cluster mapping (R3 "KV-stationary")
`sh_x_attention` deals query head `h` to cluster `h % 16` (15 heads -> 15 clusters busy; the three heads of a kv group each
stage their own copy of the group's K/V slice: 2 x 241 x 64 x 2 B = 62 KB per head per layer). Everything of a head stays
in that cluster's TCDM (q, K staged + transposed by element-granular 2-D DMA, scores `[50, 320]`, V, o: ~170 KB). The KV
itself lives in HBM and is re-streamed per step: keeping all 16 layers' KV resident would need 1.2 MB per cluster at head
granularity (> 1 MB TCDM), so a resident variant needs layer- or kv-group-granular dealing (not done). GEMMs: output tiles
dealt round-robin over the 16 clusters (`tm = 50`; `tn` 48-256 so that 15-16 clusters get a tile; `tk` = whole K or
240-512; double-buffered), see `TILES` in the frontend (overridable: `expert.py flow --tiles qkv=50,320,720;...`).

## 3. Kernels and op sharing with agent/vlm-prefix (`runtime/sh_llm.inc.c`)
| op | lowered to | status |
|---|---|---|
| `softhier.rmsnorm` | `sh_rmsnorm` (sh_llm) | shared; the expert's duplicate was deleted |
| `softhier.rope {head_dim}` | `sh_rope` (sh_llm; `[rows, head_dim]` cos/sin table) | shared; same table for q and own k |
| `softhier.silu_mul %a, %b` | `sh_silu_mul` (sh_llm) when y / a / b have one leading dim | shared (the MLP: in place over the gate half of `[50, 4096]`) |
| `softhier.silu_mul %a` (no b) or strided operands | `sh_x_silu_mul` (per-operand ld, plain SiLU) | expert-only (the time MLP's SiLU) |
| `softhier.axpy` | `sh_x_axpy` | expert-only (Euler update) |
| `softhier.cross_attention` | `sh_x_attention` | expert-only: `Sq` queries over `Lp + So` keys from **two** sources (stationary prefix block + own causal keys), GQA, token-class mask. `sh_attention_gqa` covers `S x S` self-attention of one sequence; unifying needs its K/V length decoupled from the query length and a second key source (or the own k/v written behind the prefix rows of a `[Lp + S]` buffer) |
| `softhier.gemm step/fmt_steps` | `sh_gemm` with `.fmt = table[step]` | the R4 hook |
| `softhier.dump_all` | `sh_test_dump_all` | test output |

`sh_x_attention_head`: K rows (prefix, then own) staged and transposed by DMA into `kT[64, Lpad]` (`Lpad` = L rounded up to
32, zero padding), RedMulE `E = q kT` (`config(50, 64, Lpad)`), masked fp16 SIMD row softmax on the 3 cores (per-core
validity rows: a 1/0 multiplier `vld` and a 0/-65504 bias `nb`, `xm = nb + x vld`; masked and padding columns are exactly 0
after `exp * vld`; the causal own column `Lp + i` is enabled one entry per row), RedMulE `o = E V`, `1 / rowsum` on `o`,
per-row stores. Phase stamps via `sh_x_attention_profile`. All row kernels run on the `sh_rowop` driver (double-buffered
row blocks, 3 cores, fp16 SIMD); the expert's ones take a leading dimension per operand (the shared ones take one `ld`).

## 4. Results (gvsoc at 1 GHz, 16 clusters, ideal HBM, fast ISS libraries)

### Step 1: one self layer + one cross layer, LCG data, vs the fp16-floor numpy twin (`expert.py layer`)
All 7 dumped tensors pass (128 samples each, atol 1 % of the tensor max + 3 %): H0 0.023 of 6.2, O0 0.003 of 1.2, QKV0 (q, k
rotated) 0.022 of 4.9, KX1 0.000, O1 0.0004 of 0.19, M1 0.039 of 10.3, H1 0.030 of 7.1. 0.74 ms self layer, 1.96 ms cross
layer (random weights: the two layers have different tile counts), 21 s wall.

Op tests (`expert.py op --which attn|xattn|rmsnorm|rope|silu|silu_view|axpy`): each at the fp16 floor (silu 0.0013 of 3.5,
axpy 0.0008, ...).

### Step 2: 10-step flow, real weights, host prefix KV (VLM in bf16, expert fp32 in lerobot), vs lerobot (`expert.py flow`)
| quantity | device vs lerobot fp32 | device vs fp16-floor twin | twin vs lerobot |
|---|---|---|---|
| final actions `x_10` (all 1600 elements) | **max abs 0.0093, mean 0.0009** (max \|a\| 1.71) | 0.0107 | 0.0016 |
| `x_t` after step s = 0..9 (all elements) | 0.0029 0.0032 0.0042 0.0055 0.0059 0.0046 0.0054 0.0071 0.0095 0.0093 | 0.002-0.011 | 0.0013-0.0018 |
| step-0 suffix embedding | 0.017 of 3.5 | 0.018 | |
| step-0 residual after layer l (128 samples) | 0.011 (l=0) ... 0.027 (l=15, max 9.4) | same | |
| step-0 attention output of layer l | 0.003-0.011 (max 0.9-2.7) | same | |

The device sits on the fp16 floor of its own program (the device-vs-twin and twin-vs-lerobot distances are the same order
as device-vs-lerobot); fp16 operands, RedMulE fp16 accumulation and the fp16 `x_t` Euler state are the floor. The same
numbers were obtained with the expert's own row kernels and with the shared sh_llm ones, with the host-side KV buffers
and with the VLM prefix program's KV layout (`kv_base + L * kv_stride`, token-class mask).

### Step 3: timing (`--profile`, default tiles, shared sh_llm row kernels)
**9.96 ms per flow step, 99.6 ms per chunk** of 50 actions (+ 0.30 ms KV projection once per chunk, + 3.2 ms preload of
the 194 MiB image), 10 steps x 16 layers on 16 clusters; 522 s wall. Per-op breakdown (160 layer instances per op):

| op (mark segment) | total | share | per call |
|---|---|---|---|
| attention (15 heads on 15 clusters, 50 x 291 / 50 x 241 keys, GQA) | 38.7 ms | 38.9 % | 242 us |
| rmsnorm + gate/up GEMM (`[50,720] x [720,4096]`, tn 256, tk 240; 147 MMAC, 5.9 MB) | 15.4 ms | 15.5 % | 96 us |
| silu * up (`[50, 2048]`, in place) | 12.6 ms | 12.6 % | 78 us |
| down GEMM (`[50,2048] x [2048,720]`, tn 48, tk 512; 74 MMAC, 2.9 MB) + residual | 10.6 ms | 10.6 % | 66 us |
| rmsnorm + q / qkv GEMM (`x [720,1600]` self: 58 MMAC, 2.3 MB; `x [720,960]` cross; tn 64, tk 720) | 10.3 ms | 10.3 % | 64 us |
| o_proj GEMM (`[50,960] x [960,720]`, tn 48, tk 960; 35 MMAC, 1.4 MB) + residual | 6.6 ms | 6.6 % | 41 us |
| rope (q, and own k on self layers) | 3.6 ms | 3.6 % | 22 us |
| suffix embedding (3 small GEMMs, biases, SiLU) | 1.4 ms | 1.4 % | 143 us / step |
| step tail (final rmsnorm, `x [720,32]`, bias, axpy) | 0.4 ms | 0.4 % | 42 us / step |

What dominates: not the weight GEMMs (41 % together: the four of them stream 12.5 MB of fp16 weights per layer in ~270 us,
~46 B/ns aggregate, i.e. HBM/NoC bound as R2 predicts; per-call RedMulE work is 2-5 us) but the Snitch-core work:
the attention softmax (39 %: 15 heads x 50 x 320 scores, fetch-bound fp16 SIMD on 3 cores, ~16 cycles per score element,
plus q/K/V staging and the K transpose by DMA) and the SwiGLU product (13 %: 50 x 2048 elements through the row driver).
The whole chunk is 5.2 GMAC of RedMulE work = 80 us at the 16-cluster peak, so the expert runs at ~1 % of peak: it is a
latency/row-op problem at M = 50, as the research notes' R2/R3 expected (26 MAC/B). Levers in order: fuse the softmax
into fewer, longer fp16-SIMD passes or move it to RedMulE-friendly form; run the gate | up product inside the down GEMM's
X staging; fuse rope into the q GEMM's epilogue; tile shapes matter little (`qkv=50,320,240` saves 3.5 us per layer).

A first profile with the expert's own `sh_x_rmsnorm` (before sharing the sh_llm kernel) read 25.4 ms per step with the
q/qkv segment at 961 us: that kernel's fp16 sum-of-squares fell back to its scalar fp32 recount on real activations.
Deleted; the shared `sh_rmsnorm` costs ~10 us. The micro-benchmarks of the GEMM shapes alone (`run.py gemm ... all`):
50x1600x720 54 us (tn 64, tk 720) / 32 us (tn 320, tk 240), 50x960x720 35 us / 22 us (tn 192, tk 240).

### Per-step RedMulE format (R4 hook)
`expert.py flow --fmt fp8,fp16,int8,...` (one entry per step) sets `fmt_steps` on every weight GEMM inside the step loop; the
emitter turns it into `.fmt = ((const uint32_t[]){SH_FP8, SH_FP16, ...})[i_step]` (`tests/filecheck/expert_translate.mlir`).
Real fp8 steps (fp8 weight copies, activation cast, measured schedules): docs/FLOW_DATAFLOW.md. The `fmt_steps` path below is plumbing only: the operands stay fp16 in memory, so a step run with `SH_FP8` / `SH_INT8` computes on
misinterpreted bytes (its `x_t` is garbage by design; the following fp16 steps run normally on it). A real fp8 step needs
fp8 weight copies in HBM (half the weight traffic, R2) and an fp16 -> fp8 cast of the activations in the GEMM's X staging.

## 5. Findings on the way
* **Per-hart stacks are 1 KB apart** (`flex_start.s`): the generated kernel function + `sh_printf` on core 0 overran core 1's
  stack and clobbered its saved return address (illegal instruction at the address of an HBM buffer, long after the fault).
  Fixed in the runtime for every generated program: `sh_call_on_core_stack` (private 40 KB stacks per core); SIMULATOR_NOTES #11.
* The library row ops take one leading dimension for all operands; the expert's strided views (gate | up halves) need one
  per operand (`sh_x_silu_mul`, `sh_x_axpy`), or equal-ld layouts (the activation written in place over the gate half so the
  shared `sh_silu_mul` applies).
* `sh_gemm` with partial RedMulE tiles (m = 50, m = 241, k = 32 / 48 / 64) is exact (self-checked on gvsoc).
* The host (7 GB) OOM-kills gvsoc and the reference under load: gvsoc deaths without output are kills, not model bugs; the
  lerobot reference is built on the meta device and run in two phases (VLM prefix in bf16 = lerobot's deployment dtype of
  the frozen VLM; expert fp32).
* `tests/gvsoc/rowops` placed the bias row inside the LN output for matrices < 8 KB (test layout, fixed).
