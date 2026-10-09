# One SmolVLA inference on SoftHier (W1 ledger)

Branch `agent/w1-e2e`. The SmolVLA pipeline runs as one chain on the gvsoc: vision tower per camera, connector,
16-layer VLM prefix that writes the KV cache, and the action expert's 10-step flow that reads that KV cache. The
final action chunk is checked against lerobot's `SmolVLAPolicy` on the same observation. The cost table for the
algorithm side is `docs/dse/e2e_cost.{md,csv}` (section 5).

All numbers come from the SoftHier flex_cluster gvsoc with 16 clusters at 1 GHz and ideal HBM (`SOFTHIER_IDEAL_HBM=1`;
DRAMSys only runs on x86). Every HBM-bound segment is therefore optimistic.

```bash
# lerobot reference (lerobot venv): camera images, SigLIP + connector outputs, KV cache, x_t per step, actions (fp32)
/app/models/lerobot_venv/bin/python -m softhier_mlir.frontend.smolvla_expert_ref --vlm-dtype fp32 --cams 1 --img-size 256 \
    --out /app/models/smolvla_base/e2e_ref_c1_t256.npz
# chain inputs + host tables (system python: torch for the language rows / state projection)
python3 -m softhier_mlir.frontend.smolvla_e2e prepare --ref /app/models/smolvla_base/e2e_ref_c1_t256.npz \
    --out /app/models/smolvla_base/e2e_c1_t256.npz
# the chain on gvsoc (4 programs; --e2e-trace adds HBM bytes / RedMulE / iDMA busy time per segment)
.venv/bin/python tests/gvsoc/run.py smolvla-e2e --npz /app/models/smolvla_base/e2e_c1_t256.npz --e2e-trace
# cost-table runs of single segments
.venv/bin/python tests/gvsoc/run.py smolvla-e2e --e2e-trace --vision-only 1024 --layers 1
.venv/bin/python tests/gvsoc/run.py smolvla-e2e --e2e-trace --prefix-only 3 --layers 8
.venv/bin/python tests/gvsoc/run.py smolvla-e2e --e2e-trace --expert-only /app/models/smolvla_base/e2e_c3_t256.npz --chunk 25
.venv/bin/python tests/gvsoc/run.py smolvla-e2e --e2e-trace --expert-only expert --chunk 50
python -m softhier_mlir.dse.e2e_cost --collect      # -> docs/dse/e2e/measured.json, docs/dse/e2e_cost.{csv,md}
```

## 1. Configuration: what "256 tokens per camera" means

Tokens per camera counts SigLIP tokens. At 256 tokens the camera image is 256 x 256, which gives 16 x 16 patches.
The patches take SmolVLM's bucketed position rows (`vision_position_ids`: patch row/column i maps to bucket
`max(0, 2i - 1)` of the 32 x 32 table, exactly as `SmolVLMVisionEmbeddings` computes them). The 4 x 4 pixel shuffle then
produces **16 image tokens per camera**, so the prefix has **65 tokens for 1 camera and 97 for 3 cameras**
(+ 48 language tokens + 1 state token). Both sizes pad to S_pad = 128.

The 113 / 241-token prefixes need 1024 SigLIP tokens per camera, which is lerobot's 512 x 512 input. In the cost table
these are the "1024" rows (section 5); they are composed from simulated segments.

The task statement paired "256 tokens" with "113-token prefix". These two do not go together: pixel-shuffling 256
SigLIP tokens gives 16 image tokens, not 64. The chain is therefore validated at 65 / 97 tokens, and full resolution is
composed.

lerobot runs the same model on these smaller images: `embed_prefix` is called directly on 256 x 256 images, with no
resize-with-pad to 512.

## 2. The chain: four programs, exact hand-over

| phase | program (frontend) | HBM image | hands over |
|---|---|---|---|
| V | `smolvla_e2e.emit_vision`: one `scf.for` over the cameras around the 12-layer SigLIP loop (+ head loop), post-LN, pixel shuffle per camera, one connector GEMM over all image tokens | 186.6 / 187.4 MiB (1 / 3 cams) | `IMG` = connector output `[n_img, 960]` |
| Pa | `smolvla.emit_vlm` (`x_img`, `dump_state`, `pad_rows_zero`): sqrt(960) scaling, layers 1-8 writing the KV cache | 150.3 MiB | `XS` residual `[n, 960]`, `KC/VC 1..8` `[n, 320]` |
| Pb | `emit_vlm` (`layer0 = 8`, `x_init = XS`): layers 9-16 | 150.3 MiB | `KC/VC 9..16` |
| X | `smolvla_expert.emit_flow` (any prefix length, `chunk`): cross-layer KV projection once per chunk, 10 Euler steps | 192.0 MiB | x_t per step, the action chunk |

**Stitching.** Each phase ends by dumping its hand-over state with `softhier.dump_all`, which prints every element as
fp16 hex. The host reads it back (`smolvla_e2e.full_dump` asserts that the dump is complete) and preloads it into the
next program at the address that program expects. The next program starts from bit-identical HBM contents. The KV
region keeps the VLM layout: `kv_base + L * kv_stride`, `[S_pad, 320]` K then V, RoPE already applied.

The expert therefore reads exactly the bytes the prefix wrote. Expert queries are class 2 (the prefix token classes
are reused), and RoPE positions continue at `n_valid + i`.

Hand-over dumps and preload waits are outside the timed segments. The chain's time is the sum of the four programs'
marked segments.

**Why phases and not one program**

* **HBM capacity.** The fp16 weights total 162 (SigLIP) + 22.5 (connector) + 300 (16 text layers) + 189.5 (expert)
  = 674 MiB. The chip has 512 MiB of populated HBM (4 west + 4 south nodes of 64 MiB). At fp16 a single resident
  program cannot hold SmolVLA on this chip. Fitting it needs fp8 weights, more HBM, or weight streaming between
  phases (which is what the phases do).
* **Host memory.** gvsoc's RSS is roughly 0.4 GB plus 2.5-3x the preload image. The measured maximum child RSS per
  phase was 0.92-1.36 GB with traces on (the expert phase is the largest). The whole 16-layer prefix in one program
  (300 MiB) was OOM-killed before (docs/SMOLVLA.md), so the prefix is split 8 + 8.

Program sizes are 49.6-53.8 KB, all under the 64 KB instruction memory, and they do not grow with the camera count
(camera loop) or the depth (layer loops).

## 3. Accuracy: chained action chunk vs lerobot (fp32 reference, same observation)

| quantity | 1 camera (65 tokens) | 3 cameras (97 tokens) |
|---|---|---|
| SigLIP output (VIS, 128 samples / camera) | max abs 0.237 (max \|ref\| 23.3), median 0.009 | 0.368 (25.6), median 0.011 |
| connector output (IMG, every element) | **2.14** (58.2), median 0.060 | **2.25** (60.0), median 0.064 |
| prefix K, all 16 layers, valid rows (every element) | <= 0.28 (max \|K\| 11-19), median 0.004-0.015 | <= 0.29, median <= 0.018 |
| prefix V, all 16 layers | <= 0.144 (max \|V\| 0.55-5.5) | <= 0.144 |
| x_t after step 0 ... 9 (all 1600 elements) | 0.0044 0.0081 0.0114 0.0141 0.0178 0.0197 0.0230 0.0257 0.0278 0.0317 | 0.0059 0.0123 0.0228 0.0315 0.0388 0.0454 0.0535 0.0576 0.0630 0.0672 |
| **action chunk x_10 vs lerobot** | **max abs 0.0317, mean 0.0023** (max \|a\| 1.36) | **max abs 0.0672, mean 0.0031** (max \|a\| 2.84) |
| device actions vs the expert's fp16 twin on the device KV | 0.0078 | 0.0137 |

**Where the error comes from.** fp16 host twins of the prefix and the expert were fed from different inputs
(`tools/e2e/error_attribution.py <cams>`, run after the chain):

| prefix fed with ... | prefix-twin K error | expert-twin actions vs lerobot (1 cam / 3 cams) |
|---|---|---|
| lerobot KV itself (fp16-rounded) | - | 0.0011 / 0.0039 |
| lerobot's connector output | 0.017 / 0.020 | 0.0016 / 0.0026 |
| the device's connector output (IMG) | 0.275 / 0.275 | 0.0109 / **0.0580** |
| (the device chain) | 0.28 / 0.29 | 0.0317 / 0.0672 |

The device's connector output alone accounts for 0.058 of the 0.067 error at 3 cameras. The connector is
`[n_img, 12288] x [12288, 960]`, and RedMulE accumulates the full K = 12288 in its fp16 accumulator.

Emulating the RedMulE update (one fused fp16 rounding per MAC) on the host's fp16 vision output gives max abs 2.02
against lerobot. The device measures 2.14. With exact accumulation the result is within 0.034.

`tools/e2e/connector_fp16_accumulation.py` emulates the fixes. The cheapest is to accumulate fp16 partials over K blocks of 256 and reduce them in fp32. That emulates to max
abs 0.040, 50x smaller. It needs a split-K with an fp32 reduction in `sh_gemm`, which does not exist yet; the
existing `accumulate` path re-reads fp16 Z.

Everything downstream of the connector (prefix GEMMs, expert) sits at the fp16 floor of its own program.

## 4. Simulated time per segment (16 clusters, 1 GHz, ideal HBM)

| segment | 1 camera x 256 tokens (65-token prefix) | 3 cameras x 256 tokens (97-token prefix) |
|---|---|---|
| vision (12 SigLIP layers, per camera 27.32-27.33 ms: attention 1.195 + proj/FFN 1.04-1.07 per layer) | 27.32 ms | 81.99 ms |
| connector (pixel shuffle + GEMM + sqrt(960) scaling) | 0.280 ms | 0.327 ms |
| prefix (16 layers, 0.888-0.954 / 0.942-1.007 ms per layer; S_pad 128 both) | 14.80 ms | 15.67 ms |
| expert KV projection (once per chunk) | 0.137 ms | 0.164 ms |
| expert per flow step (chunk 50) | 7.657 ms | 8.051 ms |
| expert, 10 steps + KV projection | 76.70 ms | 80.67 ms |
| **total** | **119.10 ms** | **178.66 ms** |
| HBM traffic per inference (iDMA trace) | 4.85 GB | 6.09 GB |
| RedMulE busy, mean over 16 clusters (cluster min-max) | 4.2 % (1.2-7.5) | 3.3 % (0.8-8.3) |

Wall time per chain (traced, shared host) was about 15 min for 1 camera and 20 min for 3 cameras, plus
host-contention outliers. At these sizes the expert is 64 % / 45 % of the inference and vision 23 % / 46 %. Neither
segment is RedMulE-bound: RedMulE is busy 1.6 % in vision and 5 % in the expert, because both are HBM or
Snitch-row-op bound (docs/SMOLVLA_EXPERT.md, docs/DSE.md).

**Composed full-resolution numbers** (1024 SigLIP tokens per camera = 512 x 512, the configuration lerobot runs):

| | 1 camera (113-token prefix) | 3 cameras (241-token prefix) |
|---|---|---|
| vision: embed 0.231 + 12 x (attention 10.811 + proj/FFN 3.749) + post-LN 0.550 ms per camera (one layer simulated) | 175.5 ms | 526.5 ms |
| connector (prefix-only program, simulated) | 0.344 ms | 0.872 ms |
| prefix (8 layers simulated, x 2) | 14.56 ms | 33.05 ms |
| expert KV projection + 10 steps, chunk 50 | 82.6 ms (model: 8.245 ms/step) | **98.4 ms (simulated: 9.813 ms/step, 0.295 ms KV projection)** |
| **total, 10 steps, chunk 50** | **273.0 ms** (expert by model) | **658.9 ms** |

The earlier 12-layer seq-1024 vision run (README) gave 14.86 ms/layer and 190.4 ms. The single layer simulated here
gives 14.56 ms; the 2 % difference was not investigated and does not
change any conclusion.

At full resolution, vision is 80 % of the 3-camera inference. Its per-head attention on 1024 x 1024 scores (10.8 ms of
14.6 ms per layer) is the single largest item. The fused `softhier.attention` op (README: 7.8 ms vs 11.8 ms at
S = 1024) is the obvious lever and is not used in the chain yet.

## 5. Cost table (`python -m softhier_mlir.dse.e2e_cost`)

`docs/dse/e2e_cost.md` (summary) and `docs/dse/e2e_cost.csv` (every column) cover 32 configurations: cameras {1, 3}
x tokens/camera {256, 1024} x flow steps {1, 2, 5, 10} x chunk {25, 50}. Each configuration gives ms per segment
(vision / connector / prefix / expert KV projection / per step / total), HBM MB per segment, GMAC, RedMulE busy %
per segment, and per-cluster RedMulE and iDMA busy % (16 values each, measured and composed rows).

Every row is labelled with its source:

* **measured**: the chained program. For steps < 10, the expert time is the first k per-step times of the 10-step
  run; step cost does not depend on t.
* **composed**: simulated segment programs multiplied out. This covers 1024-token vision (one layer), the
  full-resolution connector + prefix (8 layers), and chunk 25 at 65 / 97 tokens. The chunk-25 runs reuse the
  chain's device KV. They are validated against lerobot's first 25 action rows, which are exact because the expert's
  own keys are causal: 0.0287 / 0.0575.
* **model**: the expert at Lp = 113 (both chunks) and Lp = 241 with chunk 25. The model is
  `step ms = 2.548 + 0.0862 C + 2.45e-4 C Lp`, fitted on the 5 simulated (Lp, C) points. Leave-one-out error is
  <= 0.9 %, including the held-out 241-token point (9.813 simulated vs 9.902 predicted). The KV projection is linear
  in Lp. RedMulE busy is proportional to MACs.

HBM bytes come from the iDMA trace: the sum of transfer sizes with an HBM source or destination, split over segments
by time overlap (`softhier_mlir.sim.trace.segment_stats`).

## 6. What had to change (all additive; defaults keep the earlier programs identical)

* **`emit_vlm` padding rows (bug).** The S_pad rows beyond n (63 of them at 65 tokens) got uniform attention like a
  padding query and evolved as garbage tokens. By layer 5 their V overflowed fp16. The masked keys' `0 x inf` in
  P.V then turned every row into NaN. This was deterministic at 65 tokens and never triggered at 113 / 241.
  `pad_rows_zero=True` zeroes the attention output of those rows every layer (one `softhier.scale` by 0), so they
  stay 0. The chain uses it. The flag is off by default, so the documented 113 / 241 runs are unchanged.
* `emit_vlm`: `img_tokens` (16 per camera at 256 SigLIP tokens), `x_img` (connector output handed over, so the
  connector is skipped), `x_init` (residual from the previous program instead of `ref_L<layer0>`), `dump_state`
  (dump the residual and the program's KV cache in full). `prefix_layout(..., img_tokens)`.
* `emit_flow`: the prefix length comes from `p_tok` (it was fixed at 241; the K/V views, the projected cross KV and
  the `kv` tile follow it). A `chunk` parameter (activations, tables and tm cut to C rows).
* `smolvla_expert_ref.py`: `--cams`, `--img-size`, and dumps of the SigLIP output, connector output, prefix
  embeddings and vision position ids (the manual vision + connector path is asserted equal to `embed_image`).
* **Traced logs.** The simulated program prints character by character, so in a `--trace` run whole trace lines get
  inserted in the middle of `[mark]` and dump lines (`[mark] start 30` + `86454000: 86454: [/chip/...`).
  `sim/trace.program_lines` cuts them out and re-joins the program's lines. `run_sim(program_output_only=True)` uses
  it and does not load multi-GB logs into memory. The mark's mcycle equals simulated ns, checked against trace
  timestamps; the printf line itself appears about 2 us later.
* `sim/trace.segment_stats`: per-mark-segment HBM bytes and RedMulE / iDMA / barrier busy ns per cluster, with
  events clipped to segments. The existing `_DMA` regex only matched lines whose first transfer is `Txn 0`;
  `_DMA_ANY` matches all of them.

## 7. Caveats

* Ideal HBM. All bytes and times assume the ideal HBM model.
* Validation is at 256 tokens per camera, and lerobot was not trained at that resolution. The check is device vs
  lerobot on the same inputs, not task success.
* The vision attention in the chain is the per-head path (12 x 3 ops), as in the validated `smolvla.emit`, not the
  fused op.
* The 1024-token rows are composed. Vision there is 12 x one simulated layer. The expert at 113 tokens is the model.
