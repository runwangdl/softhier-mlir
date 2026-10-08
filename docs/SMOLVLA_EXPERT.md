# SmolVLA action expert + flow-matching loop on SoftHier (ledger)

Branch `agent/action-expert`. Code: `runtime/sh_expert.inc.c` (kernels, prefix `sh_x_`),
`softhier_mlir/dialects/softhier.py` (ops `rmsnorm`, `rope`, `silu_mul`, `axpy`, `cross_attention`, `dump_all`,
`gemm step/fmt_steps`), `softhier_mlir/backend/emit_c.py`, `softhier_mlir/frontend/smolvla_expert.py` (programs +
numpy twin), `softhier_mlir/frontend/smolvla_expert_ref.py` (lerobot reference), `tests/gvsoc/expert.py` (runner),
`tests/filecheck/expert_translate.mlir`.

## 1. What the model does at inference (read from lerobot 0.4.4 `modeling_smolvla.py` / `smolvlm_with_expert.py` and the checkpoint)

| item | value (checkpoint `lerobot/smolvla_base`) |
|---|---|
| expert | 16 Llama decoder layers, hidden 720 (`0.75 x 960`), intermediate 2048 (`get_intermediate_size(720)`), RMSNorm eps 1e-5, no biases |
| attention | 15 query heads x 64 (`q_proj [960, 720]`), 5 kv heads x 64 (GQA, query head h uses kv head h // 3), scale 1/8, eager fp32 softmax, `big_neg` mask |
| layer parity | `self_attn_every_n_layers = 2`: even layers are **self-attention** layers (`k/v_proj [320, 720]` on the expert's own tokens, keys/values = [VLM prefix KV of that layer ; own]), odd layers are **cross-attention** layers (`k/v_proj [320, 320]` applied to the VLM prefix KV *cache*, no own keys) |
| RoPE | lerobot's own `apply_rope` (half-split, base 10 000, fp32), **not** the HF model's rope (theta 100 000). Self layers: q and own k at positions `n_valid + i` (`prefix_offsets + cumsum - 1`); cross layers: q at positions `i` (`expert_position_id - min`), the prefix keys are the cached *already rotated* VLM keys, the projected cross keys get no RoPE |
| masks | prefix columns: `prefix_pad_masks` (language padding to 48 tokens is masked, images / state always valid); own columns (self layers only): causal `j <= i` (`make_att_2d_masks` with att_mask 1 on every action token) |
| prefix | 3 x 64 image tokens + 48 language tokens + 1 state token = 241; `n_valid` = 241 - language padding |
| suffix embedding | `e = action_in_proj(x_t)`; `time_emb = sinusoidal(t, 720, 4e-3, 4.0)` (`[sin | cos]`, float64); `h = action_time_mlp_out(silu(action_time_mlp_in([e | time_emb])))` |
| flow | `num_steps = 10`, `dt = -0.1`, `t_s = 1 - 0.1 s`, `x_0 = noise ~ N(0, 1)` of shape `[50, 32]`, `x_{s+1} = x_s + dt * v_s`, `v_s = action_out_proj(final_rmsnorm(h_16))`; the action chunk is `x_10` (the first 6 of 32 dims are real) |
| KV cache | `past_key_values[l] = {key_states, value_states}` of shape `[B, 241, 5, 64]` per VLM layer, keys post-RoPE; the expert reads it flattened to `[241, 320]` head-major |

Exact algebra used by the device program (no approximation beyond fp16):
* `[Wq^T | Wk^T | Wv^T]` fused into one `[720, 1600]` GEMM on self layers; `[Wgate^T | Wup^T]` fused into `[720, 4096]`.
* The time half of `action_time_mlp_in` is folded on the host into a per-step bias table `tb[s] = time_emb(t_s) @ W_in[:, 720:]^T + b_in` (`[10, 720]`, preloaded).
* The cross layers' `K_l @ Wk_l^T`, `V_l @ Wv_l^T` are step-invariant: computed **once per chunk** on the device before the step loop (`[241, 320] x [320, 320]`, 16 GEMMs), not once per step as lerobot's code does.

## 2. Device program

```
preload_wait
for p in 0..7:   kx_c[p] = kp_c[p] @ wkx_c[p];  vx_c[p] = vp_c[p] @ wvx_c[p]        (cross layers 2p+1)
x = x0
for s in 0..9:                                                     (scf.for; GEMM fmt = fmt_steps[s] if given)
    e  = x @ wa + ba;  e1 = silu(e @ wti + tb[s]);  h = e1 @ wto + bto
    for p in 0..7:                                                 (scf.for over (self, cross) layer pairs)
        self  layer 2p  : xn = rmsnorm(h); qkv = xn @ wqkv; rope(q, tab_self); rope(k, tab_self[:, :320])
                          o = cross_attention(q, kp, vp, own k, v, valid); h += o @ wo
                          xn = rmsnorm(h); gu = xn @ wgu; m = silu(g) * u; h += m @ wd
        cross layer 2p+1: xn = rmsnorm(h); q = xn @ wq; rope(q, tab_cross)
                          o = cross_attention(q, kx, vx, valid); h += o @ wo; ... same MLP
    fin = rmsnorm(h, gf);  v = fin @ wout + bout;  x = x + (-0.1) v        (axpy)
    mark step<s>; dump x
```
Program size is independent of the depth (two loops, per-pair parameter slabs at a constant HBM stride).
HBM image: 194 MiB fp16 (weights 189 MiB, prefix KV 4.9 MiB, tables); all buffers below 200 MiB.

### KV layout assumed from the VLM prefix (to be matched by agent/vlm-prefix)
Per VLM layer `l = 0..15`, fp16 row-major in HBM:
* `kp_l [241, 320]`: `key_states` **after** lerobot's RoPE at positions `cumsum(prefix_pad) - 1`, column `j = kv_head * 64 + d`;
* `vp_l [241, 320]`: `value_states`, same column layout;
* one `valid [1, 241]` fp16 row (1.0 real token, 0.0 padding) and `n_valid` (used for the self layers' RoPE table).
The expert consumes `kp_l / vp_l` of **every** layer: raw on the even (self) layers, through its own 320 -> 320
`k/v_proj` on the odd (cross) layers (done on the device once per chunk). The prefix tokens are ordered
`[image0 (64) | image1 (64) | image2 (64) | language (48) | state (1)]`.

### Cluster mapping (R3 "KV-stationary")
`sh_x_attention` deals query head `h` to cluster `h % 16` (15 heads -> 15 clusters busy, the three heads of a kv group
each stage their own copy of the group's K/V slice: 2 x 241 x 64 x 2 B = 62 KB per head per layer). Everything of a
head stays in that cluster's TCDM (q, K staged + transposed by element-granular 2-D DMA, scores `[50, 320]`, V, o:
~170 KB). The KV itself lives in HBM and is re-streamed per step; keeping the 16 layers' KV resident in TCDM would
need 1.2 MB per cluster at head granularity, more than the 1 MB TCDM, so a resident variant needs layer- or
kv-group-granular dealing (not done). GEMMs: output tiles dealt round-robin over the 16 clusters
(`tm = 50`, `tn` 48-256 so that 15-16 clusters get a tile, `tk` = whole K or 240-512, double-buffered).

## 3. Kernels (`runtime/sh_expert.inc.c`)
* `sh_x_rmsnorm`, `sh_x_silu_mul` (also plain SiLU with `b = 0`), `sh_x_axpy`, `sh_x_rope`: `sh_rowop` driver
  (double-buffered row blocks, 3 cores, fp16 SIMD). RoPE takes a host-built per-row table `[cos | sin]` per head.
* `sh_x_attention_head`: K rows (prefix, then own) staged and transposed by DMA into `kT[64, Lpad]` (`Lpad` = L
  rounded up to 32, zero padding), RedMulE `E = q kT` (`config(50, 64, Lpad)`), masked fp16 SIMD row softmax on the
  3 cores (per-core validity rows: a 1/0 multiplier `vld` and a 0/-65504 bias `nb`, so `xm = nb + x vld`; masked and
  padding columns are exactly 0 after `exp * vld`; the causal own column `Lp + i` is enabled one entry per row),
  RedMulE `o = E V`, `1/rowsum` on `o`, per-row stores. Phase stamps via `sh_x_attention_profile`.
* Everything else is the existing library (`sh_gemm`, `sh_add`, `sh_add_bias`).

To unify with `runtime/sh_llm.inc.c` (agent/vlm-prefix) at merge time: RMSNorm, RoPE (they may use an on-device
cos/sin computation instead of a table), SiLU-gated activation, the masked softmax / GQA attention. The expert's
attention differs by having two key sources (stationary prefix + own causal keys) and a validity row; one kernel with
`So = 0` covers the prefix-only case and could cover the VLM's own prefill if a causal/prefix-LM mask over the
*prefix* columns is added (the VLM prefix is bidirectional over images/language, so `valid`-style masking suffices there).

## 4. Results

(filled in as the steps are verified; all numbers are gvsoc at 1 GHz, 16 clusters, ideal HBM)

### Step 1: one self layer + one cross layer, LCG data, vs the fp16-floor numpy twin
pending

### Step 2: 10-step flow with real weights and the host's prefix KV, vs lerobot fp32
pending

### Step 3: timing / per-step format plumbing
pending
