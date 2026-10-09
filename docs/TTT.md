# Test-time adaptation on SoftHier: one LoRA step on the SmolVLA action expert (ledger)

Branch `agent/w4-ttt-backward` (research plan W4). One SGD / Adam step of a LoRA adapter on the action expert, gradients
never entering the frozen VLM, every number simulated on gvsoc (16 clusters, 1 GHz, ideal HBM, `models_fast`) and
checked against float64 torch autograd on the same fp16 inputs.

Code: `runtime/sh_train.inc.c` (kernels, prefix `sh_t_`), `softhier_mlir/dialects/softhier.py` + `backend/emit_c.py`
(ops below), `softhier_mlir/frontend/smolvla_ttt.py` (programs + torch references), `tests/gvsoc/ttt.py`
(`ops | layer | expert | reduce | dp`), `tests/filecheck/train_translate.mlir`.

## 1. What is trained, what is computed

LoRA r = 16, alpha = 32 (s = 2) on q_proj, o_proj and down_proj of all 16 expert layers: 98 048 parameters per layer,
1.57 M in total, kept as one flat arena (fp16 copy read by the GEMMs, fp32 master, fp16 gradients, fp32 Adam moments).
B is initialised non-zero so both A and B receive a gradient at step 1.

Step = forward of one flow step (t = 1, x_t = noise) with LoRA -> v -> MSE(v, u) with u = noise - a (a = lerobot's own
action chunk for this observation: a stand-in target, A2 decides the real loss) -> backward through the final norm,
action_out_proj and 16 layers into the LoRA only -> optimizer on the fp32 masters. Prefix KV (VLM) and the cross
layers' projected KV are constants: no gradient is formed for them. Gradient scaling: the MSE gradient is multiplied by
a loss scale (2^16 for one sample, chosen from the torch gradient magnitudes so every activation gradient sits in the
fp16 normal range) and the optimizer divides it out in fp32.

Per layer, forward (saved to HBM for the backward: h, xn, s xA_q, q|k|v after RoPE, o, s oA_o, h1, g|u, m, s mA_d):
```
xn = rmsnorm(h);  q|k|v = xn Wqkv;  q += s (xn Aq) Bq;  rope;  o = attention(q, [kp; k], [vp; v])
ao = o Wo + s (o Ao) Bo;  h1 = h + ao;  g|u = rmsnorm(h1) Wgu;  m = silu(g) u;  h' = h1 + m Wd + s (m Ad) Bd
```
Backward (one `softhier.linear_bwd` per projection; every frozen-weight product is formed transposed, dX^T = W dY^T,
so RedMulE reads W in its stored [in, out] layout and only 50-row activations are transposed):
```
dm, dAd, dBd = linear_bwd(dh', Wd, LoRA d)       dg, du = silu_mul_bwd(g, u, dm)
dxn2 = linear_bwd([dg|du], Wgu)                  dh1 = rmsnorm_bwd(h1, g2, dxn2) + dh'
do, dAo, dBo = linear_bwd(dh1, Wo, LoRA o)       dq, dk, dv = attention_bwd(q, K, V, o, do)   (cross layers: dq)
dq, dk = rope^T (table with -sin)                dxn, dAq, dBq = linear_bwd(dq|dk|dv, Wqkv, LoRA q)
dh = rmsnorm_bwd(h, g1, dxn) + dh1
```

## 2. Primitives (step 1: library + dialect + emit_c + FileCheck; `ttt.py ops`)

| op (dialect) | library | what |
|---|---|---|
| `gemm_t {trans_x, trans_w}` | `sh_t_gemm_tr` | GEMM with physically transposed operands: `sh_transpose` into an HBM scratch, then `sh_gemm` |
| `rmsnorm_bwd [res]` | `sh_t_rmsnorm_bwd` | fp32 statistics, fp16 SIMD output pass, residual gradient fused |
| `silu_mul_bwd` | `sh_t_silu_mul_bwd` | fp16 SIMD (the forward's exp2), strided gate/up views |
| `softmax_bwd` | `sh_t_softmax_bwd` | dx = scale y (dy - rowdot(y, dy)) |
| `attention_bwd [own] [mask] [grads, scratch]` | `sh_t_attention_bwd` | per head in one cluster's TCDM: S = qK^T recomputed, P, D = rowdot(dO, O), dP = dO V^T, dS = scale P (dP - D), dq = dS K, own dk = dS^T q, dv = P^T dO (5 RedMulE GEMMs, K^T / V^T / dS^T / P^T by element-granular DMA); per-head dk/dv summed over each kv group |
| `mse_grad [loss]` | `sh_t_mse_grad` | dy = gscale (pred - tgt), fp32 per-row loss |
| `optim_step {sgd, adam} [moments]` | `sh_t_optim` | fp32 master + fp16 copy, Adam moments fp32, grads fp16 or fp32 times 1/loss scale |
| `grad_allreduce {mode}` | `sh_t_allreduce` | data-parallel gradient sum with the in-network REDADD (section 5) |
| `lora_fwd`, `linear_bwd`, `copy`, `add {train}`, `cross_attention {train}`, `cluster_id`, `cluster = -2` | `sh_t_lora_fwd`, `sh_t_linear_bwd`, `sh_t_copy`, `sh_t_add`, `sh_t_attention_fwd`, `SH_SELF` | the compositions the programs use (section 6 explains why they exist) |

All on the multi-stream row-op driver `sh_t_rowop` (sh_rowop / sh_llm_rowop generalised to 4 input + 4 output streams
with their own width, leading dimension and element size: the fp32 HBM path the optimizer needs).

`ttt.py ops` (expert shapes, LCG data, float64 torch reference, 128 samples each, all PASS):

| op | shape | time (16 clusters) | error vs float64 |
|---|---|---|---|
| gemm_t, transposed X (dA = x^T u) | 720x16, K = 50 | 19.6 us | exact (integer data) |
| gemm_t, transposed W | 50x720, K = 16 | 10.7 us | exact |
| sh_transpose of one frozen weight | 2048 x 720 | **145 us** | exact |
| rmsnorm_bwd + residual | 50 x 720 | 75.5 us (126.6 us all-fp32) | 3.6e-4 of max |
| silu_mul_bwd | 50 x 2048 (strided) | 99.4 us | 3.5e-4 |
| softmax_bwd | 50 x 320 | 40.8 us | 1.7e-4 |
| mse_grad + loss | 50 x 32 | 8.9 us | exact |
| attention_bwd, self (241 prefix + 50 causal own keys, 15 q / 5 kv heads) | | 425 us (forward 269 us) | dq\|dk\|dv 5.9e-4 |
| attention_bwd, cross (241 keys) | | 323 us (forward 217 us) | dq 2.2e-3 |
| SGD / Adam | 4096 params | 10.3 / 19.5 us | 0 / 4.7e-7 (fp32), fp16 copy 3.8e-5 |

**Transposed operands.** Transposing the frozen weights each step would cost 145 us per 1.47 M elements, i.e. ~0.6 ms
per layer and ~9.8 ms per step for the 16 layers' 100 M weight elements (or 190 MB of extra HBM for transposed copies).
`linear_bwd` avoids both: dX^T = W dY^T needs only dY^T and the result transposed back (50-row activations, ~0.5 M
elements per layer, ~11x fewer than the weights). The weight-gradient GEMMs of the LoRA are formed as (u^T x)^T for the
same reason.

## 3. One layer fwd + bwd + update (step 2, `ttt.py layer`)

A self-attention layer followed by a cross-attention layer, LCG weights / input, the output gradient given (loss =
sum(h_out * R)), SGD lr 1e-2. All 12 LoRA gradients (A, B of q, o, down, both layers), 128 samples each vs float64 torch:

| | max abs err / max \|ref\| | rel-L2 |
|---|---|---|
| dA, dB, self layer | 0.43 % - 1.04 % | 0.98 % - 1.28 % |
| dA, dB, cross layer | 0.35 % - 0.71 % | 0.66 % - 1.06 % |
| updated A, B (SGD, lr 1e-2) | = lr x the gradient error (fp32 master update exact to 1e-7) | |

Adam step 1 (lr 1e-3) moves every weight by ~lr sign(g); elements whose gradient is inside the fp16 noise (|g| ~ 0) can
flip sign (2 lr = 2e-3 error on those elements; all others within 1.2e-8).

Timing (self / cross layer): forward 0.78 / 0.70 ms, backward 1.30 / 1.16 ms.

**Tolerance = the RedMulE fp16 accumulator.** The gradients come out of GEMMs with K = 50 ... 4096 accumulated in fp16
(RedMulE has no wider accumulator). A host emulation of one fp16-accumulating dot product (fused FMA, RNE) gives rel-L2
1.0e-3 (K = 50), 3.9e-3 (720), 6.4e-3 (2048), 9.3e-3 (4096) against the exact product, while rounding the exact result
to fp16 costs 2e-4: the 0.3-1.5 % gradient errors are that floor, compounded over the chain.

## 4. Full expert, one TTT step (step 3, `ttt.py expert --layers 16 --profile`)

Real weights (`expert.npz`), loss 0.0725 on device vs 0.0713 in torch, loss scale 2^16:

| quantity | device vs float64 torch |
|---|---|
| v (all 1600 elements sampled 512) | max abs 0.029 (\|v\| max 4.13, rel-L2 5.1e-3): the inference program's floor (docs/SMOLVLA_EXPERT.md: 0.029 on v at step 0) |
| LoRA gradients, layers 0 / 7 / 15, all six tensors | rel-L2 0.33 % - 1.54 %, max abs <= 1.1 % of max \|g\| |
| updated masters (SGD lr 1e-2) | lr x the gradient error: <= 6.3e-7 abs (the update itself is <= 6.4e-5) |

Simulated time on 16 clusters (one sample, all clusters on it; marks after every op group):

| phase | time | per layer | notes |
|---|---|---|---|
| cross-layer KV projection | 0.36 ms | | once per chunk, not per step |
| forward: suffix embedding | 0.11 ms | | |
| forward: 16 layers | 10.61 ms | self 0.70, cross 0.63 | the inference expert: 0.62 ms per layer; + 3 LoRA branches, residuals in a separate add |
| head + loss + dL/dh_16 | 0.13 ms | | |
| backward: down proj (linear_bwd, LoRA d) | 1.88 ms | 0.117 | |
| backward: silu_mul | 1.62 ms | 0.101 | |
| backward: gate/up (linear_bwd) + rmsnorm_bwd | 3.85 ms | 0.240 | K = 4096 |
| backward: o proj (linear_bwd, LoRA o) | 1.47 ms | 0.092 | |
| backward: attention | 5.97 ms | 0.373 | 33 % of the backward |
| backward: rope^T + q|k|v proj (LoRA q) + rmsnorm_bwd | 3.22 ms | 0.201 | |
| **backward total** | **18.1 ms** | 1.13 | |
| **SGD, 1.57 M params** | **1.34 ms** | | Adam: 0.95 ms per 196 k params, ~7.6 ms for all (scalar fp32 + rsqrt) |
| **step (fwd + bwd + SGD)** | **30.6 ms** | | = 3.1 inference flow steps (9.96 ms), 0.31 of a 10-step chunk |

**fwd : bwd : update = 1 : 1.63 : 0.12** (SGD; Adam 1 : 1.63 : 0.69).

Activations kept in HBM for the backward: 1.11 MB per layer (gate|up 400 KB and m 200 KB are 54 % of it), 17.8 MB for
16 layers + the final residual (22.4 MB as allocated, 4 KB-aligned and m stored with the ld of gate|up), plus 1.7 MB of
backward scratch. That is above the 16 MB of TCDM of the whole chip, i.e. the backward cannot stay on chip without
recomputing g|u and m from h1 (one gate/up GEMM per layer, ~0.25 ms, -> ~7 MB). The plan's 7.9 MB estimate counted ~5
tensors per layer.

## 5. Data-parallel: 16 samples, in-network REDADD of the LoRA gradients

`sh_t_allreduce`: every cluster holds its full gradient arena; chunk i (16 KB) is owned by cluster i mod 16
(reduce-scatter, all 16 owners issue their REDADD at once), the owner writes the sum to the shared HBM arena (no
all-gather: HBM is shared and the optimizer reads it). Two modes:
* mode 0: `COLLECTIVE_REDADD_FP_16` on the fp16 gradients;
* mode 1: exact: one REDMAX for a global power-of-two scale, every element as a 20-bit fixed-point integer split into a
  signed high and an unsigned 12-bit low limb, two integer REDADDs that cannot overflow for 16 clusters, recombined in
  fp32 by the owner.

`ttt.py reduce`: 16 real per-sample gradient arenas (16 noise samples, full 16-layer expert, torch float64, x 2^16,
fp16; 3.0 MiB each, sum up to 1.2e4, 0.003 % subnormal inputs), 2048 samples vs the float64 sum of the same fp16 inputs:

| mode | time | rel-L2 | max abs (\|sum\| <= 1.2e4) |
|---|---|---|---|
| fp16 REDADD | **0.71 ms** | **1.0e-3** | 5.75 |
| fp16 floor = fp16(exact sum) | | 2.1e-4 | 1.75 |
| exact two-limb integer REDADD (fp32 result) | 95-103 ms | 4.5e-6 | 7.1e-3 |

The fp16 REDADD loses 5x the fp16 rounding floor (the gvsoc NoC converts each partial sum back to fp16 by truncation)
but stays 10x below the per-sample gradients' own error (~1e-2, section 3), and nothing is flushed to zero at this
loss scale: **plain fp16 REDADD is enough; the exact scheme is correct but core-bound** (each cluster converts its whole
1.57 M-element arena, ~65 cycles per element). A hi/lo split of fp16 halves does not apply here: the gradients are born
fp16 (RedMulE output), so the low half is zero; the error is in the accumulation, which only a wider accumulator (the
NoC adding in fp32, as `test_time_adaptation.md` proposes) or integer limbs fix.

The 0.71 ms is 14x the line-rate model (127 + 3.1 MB / 64 B/cycle = 49 us): every cluster first streams its 3 MB arena
from HBM into TCDM slots (50 MB of HBM reads in total), plus the burst-buffer flush (below). Gradients that are
accumulated per chunk in TCDM would remove the staging.

**End to end** (`ttt.py dp --layers 4 --loss-scale 16384`): every cluster runs the whole step on its own sample
(`cluster = SH_SELF`, activations and gradients in a 10 MiB per-cluster HBM region, the program indexes it with
`softhier.cluster_id`), then the fp16 REDADD and SGD with the mean gradient. 4 layers because 16 regions of the full
16-layer step (22 MB each) plus the weights do not fit the 256 MiB the preload reaches.

| quantity | value |
|---|---|
| cluster 0's v vs torch | max abs 0.042 of 5.4 (rel-L2 4.4e-3) |
| mean LoRA gradient (16 samples, after the fp16 REDADD), layers 0 and 3 | rel-L2 0.18 % - 3.7 % (max abs <= 2.7 % of max), 10 of 12 tensors <= 1 % |
| time (cluster 0's marks) | fwd 24.3 ms, loss 0.84 ms, bwd 40.4 ms, REDADD 6.44 ms (incl. waiting for the slowest cluster; the reduction itself ~0.2 ms for 0.75 MiB), SGD 0.54 ms: **72.6 ms for 16 samples x 4 layers** |
| extrapolated to 16 layers | ~0.26 s per 16 samples = **~16 ms per sample** (vs 30.6 ms with all 16 clusters on one sample), latency 0.26 s, 16 x 22 MB of activations |

Data parallelism buys ~1.9x throughput at M = 50 (a single cluster's GEMMs and attention heads run without the
cross-cluster barriers and with whole tiles), at 16x the latency and 16x the activation memory.

## 6. What the simulator / toolchain made necessary (also docs/SIMULATOR_NOTES.md #12-#14)

* **64 KB instruction memory.** The first full program was 83.5 KB. Now 57-63.5 KB: per-projection compositions
  (`lora_fwd`, `linear_bwd`) instead of ~7 ops each, `-Os` for the generated code (func attribute `sh.optimize`) and
  for the library's control code (`SH_T_COLD`), the attention forward of training programs on the backward kernel's
  staging code (`cross_attention {train}`), the unused mesh SUMMA dropped (`SH_NO_GEMM_MESH`; gc-sections keeps it),
  `SH_T_NO_RED_EXACT` for programs that only use the fp16 REDADD.
* **newlib is RVC.** `memset` / `memcpy` from the toolchain's newlib use compressed instructions, which the cores do not
  execute ("illegal instruction ... 0x433d"); GCC emits them for struct zeroing / copies (-Os compound literals). The
  library now provides plain RV32 versions.
* **SDK `inline` helpers under -Os.** `bare_dma_*` are C99 `inline` without `static`; -O3 always inlined them, -Os
  calls them -> undefined references. `extern inline` declarations in the unity build emit the definitions.
* **REDADD accumulates into stale burst buffers.** The gvsoc collective model adds the pulled data into the read-burst
  buffer the root iDMA hands the request (and into each intermediate router's copy); those buffers are a static FIFO pool
  of 256 x 4 KB that keeps old data, so results were right only until the pool wrapped. Workaround: 1 MB of zero-memory
  reads right before the reduction (`sh_t_red_flush`).
* **Accuracy traps fixed on the way**: seeding the residual into an accumulating GEMM (h1 = h + o Wo in one RedMulE
  pass) tripled the v error (every fp16 FMA rounded at |h|): separate fp16 adds again. The packed-block SIMD scale left
  the last 1-3 elements of an odd-width (50) block unscaled (dA 20-50 % off until fixed).

## 7. What a deployment loop still needs

1. **The loss.** u = noise - a(own chunk) is a placeholder that makes TTT self-distillation; the self-supervised target
   (A2: next-frame frozen SigLIP features through a small head, VANE style) adds that head's fwd/bwd.
2. **Training samples.** One flow time (t = 1) per step here; flow matching trains on t ~ Beta and fresh noise, i.e.
   x_t = t noise + (1 - t) a and its own suffix embedding per sample; on-device Gaussian noise; several (x_t, t) per
   step batched as M = 50 B rows (the weight GEMMs are HBM-bound at M = 50, so batching is nearly free up to B ~ 4).
3. **Memory.** 17.8 MB of saved activations per sample exceed the chip's TCDM and, for 16 samples in data parallel, the
   HBM next to the weights (16 x 22 MB): recompute gate|up and m in the backward, keep only h per layer, or batch
   instead of 16-way DP.
4. **Optimizer state.** Adam's step count and bias corrections are compile-time attributes; a loop needs them in HBM.
   Dynamic loss scaling (overflow check on the gradient arena) instead of the host-chosen 2^16.
5. **Scheduling against inference.** A step costs 30.6 ms = 0.31 chunk on all 16 clusters; adapting every chunk needs
   either a cluster split (W3: an adaptation group beside the inference group) or adapting every few chunks.
6. **Speed levers measured here**: attention backward (33 % of the backward; the scores are recomputed), the per-op
   transposes inside `linear_bwd` (~0.5 M elements per layer), rmsnorm_bwd's fp32 statistics pass, Adam's scalar fp32
   loop; the REDADD staging from HBM.

## 8. Reproduce

```bash
.venv/bin/python tests/gvsoc/ttt.py ops                                  # ~15 s wall
.venv/bin/python tests/gvsoc/ttt.py layer [--opt adam --lr 1e-3]         # ~20 s
.venv/bin/python tests/gvsoc/ttt.py expert --layers 16 --profile         # ~3 min (needs /app/models/smolvla_base/expert.npz)
.venv/bin/python tests/gvsoc/ttt.py reduce                               # 16 torch samples (cached) + ~1.5 h wall (mode 1 is core-bound)
.venv/bin/python tests/gvsoc/ttt.py dp --layers 4 --loss-scale 16384     # 16 samples, one per cluster
```
