# Flow-matching dataflow of the SmolVLA action expert: KV-stationary attention and real fp8 steps (W2)

Branch `agent/w2-flow-dataflow`. Code: `runtime/sh_flow.inc.c` (kernels, prefix `sh_f_`), `softhier_mlir/frontend/fp8.py`
(e4m3 quantiser + numpy twin), `softhier_mlir/frontend/smolvla_expert.py` (`emit_flow(attn=, fp8_mode=)`, op tests
`gemm8<fam>`), `softhier_mlir/dialects/softhier.py` (`softhier.call`), `tests/gvsoc/expert.py` (`--attn kvs`,
`--fp8-mode`, `--save-x`, `--trace-dma`), `tests/gvsoc/dma_bytes.py` (bytes per segment from the gvsoc iDMA trace),
`tests/gvsoc/flow_tables.py`. Baseline program and numbers: `docs/SMOLVLA_EXPERT.md`.

All numbers: gvsoc, 16 clusters, 1 GHz, ideal HBM (DRAMSys is x86-only), the 10-step flow of the expert with real
`lerobot/smolvla_base` weights and the host prefix KV. "Measured" = a full simulation of the stated program;
"composed" = a sum of separately measured parts (labelled where used).

## 1. KV-stationary attention (R3)

### Dealing
The prefix KV the expert reads (self layers: the VLM prefix K / V of that layer; cross layers: `K_l Wk_l`, `V_l Wv_l`,
computed once per chunk) is constant for the whole chunk. `sh_f_kvs_deal` puts it into TCDM once, after the
cross-layer KV projection:

```
cluster c = 3 g + p   (g = kv head 0..4, p = key part 0..2; 15 clusters, cluster 15 holds nothing)
  for every layer l: key rows [96 p, min(96 (p + 1), 241)) of kv head g
     kT  [64, Lpad]   K already transposed (once per chunk instead of once per head per step)
     V   [Lpad, 64]
  self layers, part 2: + the 50 own keys / values, rewritten every step (prefix 49 + own 50 = 99 -> Lpad 112)
  Lpad = rows rounded up to 16: 96 / 96 / 112 (self), 96 / 96 / 64 (cross)
TCDM: resident region 0x90000..1 MB, 393 216 B per cluster for 16 layers; every other kernel of the program
      stays below 0x90000 (row ops < 0x81000, GEMM tiles < 0x67000, this kernel's scratch < 0x40000)
```

Per step and layer, `sh_f_attention_kvs` (all clusters):
1. cluster 0 loads the query block `q [50, 960]` (96 KB) once from HBM and multicasts it to all clusters
   (`flex_dma_async_broadcast`, full-mesh masks, three 32 KB chunks, one in flight at a time);
2. cluster (g, p) stacks the 3 query heads of kv head g into `Q [150, 64]`, RedMulE `E = Q kT_p` (150 x Lpad), masked
   fp16-SIMD row softmax (row max `m` and row sum `l` kept per row, own keys causal), RedMulE `O_p = E V_p` unnormalised;
3. global barrier; cluster (g, p) reads head 3 g + p's rows of the three partials `(O_j, m_j, l_j)` from its two partner
   clusters' TCDM (remote iDMA, no HBM) and combines `o = sum_j 2^(m_j - M) O_j / sum_j 2^(m_j - M) l_j`, stores o.

A GQA group is computed as one 150-row block per key part (the three heads share the K / V part), so no K / V is
replicated and no cluster touches HBM for prefix K / V after the deal.

### Correctness
`expert.py flow --attn kvs`: 2 steps x 4 layers vs the fp16-floor twin 0.0049 (stream: 0.0044); full 10 steps x 16
layers: action chunk vs lerobot fp32 **max 0.0051, mean 0.0008** (stream, same harness: 0.0093 / 0.0009), every x_t
within 0.0081 of the twin.

### Streaming vs resident (per flow step, 16 layers; measured)

| | streaming (`sh_x_attention`) | KV-stationary (`sh_f_attention_kvs`) |
|---|---|---|
| time per step | **9.963 ms** | **10.018 ms** (+0.6 %) |
| attention per layer call | 242.1 us | 245.3 us |
| HBM read, attention | 18.0 MB (15 heads x [q 6.4 KB + K 30.8 + V 30.8 + own k/v 12.8 + tok 0.5]) | **2.1 MB** (q once 96 KB/layer, own k/v 64 KB/self layer, tok) |
| HBM write, attention | 1.5 MB | 1.5 MB |
| HBM read, whole step | 345.7 MB | 329.8 MB (-4.6 %) |
| NoC cluster-to-cluster | 0 | 6.4 MB (402 KB/layer: 96 KB multicast injected + 306 KB partial gather, 204 KB of it remote) |
| TCDM-local DMA (K transposes, stacking) | 8.2 MB | 4.9 MB |
| once per chunk | - | deal 0.12 ms, 4.9 MB HBM read; 393 KB TCDM resident per cluster |

Bytes: gvsoc iDMA trace of a 1-step run (`--trace-dma`, `tests/gvsoc/dma_bytes.py`), every transaction classified by
address. The trace attributes a burst to the segment in which it *finished*; 1.5 MB of the o_proj GEMM's X reads land
in the KV-stationary attention segment (3.6 MB there), the per-step totals are exact. The attention numbers in the
table are the per-op count (q, own k / v, tok, o), which the totals confirm (345.7 - 329.8 = 15.9 MB = 18.0 - 2.1).

Phases of one call (cluster 0 = part 0, cluster 2 = part 2, cycles): q multicast 0.2-4 k, staging 0.6-4 k, QK 1 k,
**softmax 143-237 k**, PV 1-2 k, gather + combine 15 k. The critical path is the softmax of part 2 on self layers
(150 rows x 112 columns = 16.8 k scores, 237 k cycles; a stream head does 50 x 320 = 16 k scores): the work per cluster
is the same as streaming, so the time is the same. Each row now covers 96-112 keys instead of 241-291, so there are 3x
more rows and the per-row overhead (max, fp16 lane extraction, sum flush) grows; the combine adds 15 us.

**Conclusion.** Keeping the prefix KV resident is free in time and removes 88 % of the attention's HBM traffic
(18.0 -> 2.1 MB per step), but attention traffic is only 5 % of the step's HBM traffic, so the step moves 4.6 % fewer
bytes and runs 0.6 % slower: the attention is bound by the fp16-SIMD softmax on 3 cores, not by bytes, exactly as
R3 predicted. The same multicast idea has a larger target in this program: every weight GEMM re-reads its X panel from
HBM once per output tile (`sh_gemm` deals output tiles), ~110 MB of the 346 MB per step (qkv 25 tiles x 72 KB, down
15 x 205 KB); multicasting X once per GEMM would save ~3x what KV residency saves. Time only moves if the softmax gets
cheaper (fewer instructions per score, or longer rows per core).

## 2. Real fp8 (e4m3) steps (R4)

### What the hardware model offers, and what is used instead
SoftHier's RedMulE fp8 mode cannot carry an fp8 step: `matmul_fp8e4m3` rounds every FMA to e4m3 (fp8 accumulation),
and its address generator steps by the arch constant `elem_size = 2` while the fp8 compute indexes bytes, so it reads a
scrambled tile (`light_redmule.cpp`). The fp8 step therefore keeps RedMulE in fp16 and gives it fp8 *values*:

* **Weights** (host, once; `frontend/fp8.py`): e4m3 as the RedMulE model defines it (bias 7, exponent 15 reserved,
  max 240, subnormal step 2^-9), round-to-nearest-even, saturating; per-tensor power-of-two scale
  `s_w = 2^ceil(log2(amax/240))` (all 64 layer GEMM weights: amax 0.0664, e_w = -11). One byte per element in HBM.
  On the device the byte b becomes the fp16 bit pattern `((b & 0x7F) << 7) | ((b & 0x80) << 8)` = `e4m3(b) * 2^-8`
  exactly (normal and subnormal codes; checked for all 240 codes).
* **Activations** (device, per fp8 GEMM, `sh_f_rn4`): `x <- RN4(x) * 2^(8 + e_w)` in place, RN4 = round to 4
  significant bits by the Veltkamp split in fp16 SIMD (`c = 129 x; hi = c - (c - x)`, 3 ops + 1 scale per 4 lanes).
  RN4 equals RNE-to-e4m3 on every fp16 normal below 500 (checked exhaustively); with a per-row power-of-two scale this
  is the e4m3 value of every element above 1/7680 of its row max; smaller ones keep 4 significant bits instead of the
  e4m3 subnormal grid (no row scale is needed: activations stay below 7). Every X is consumed by exactly one GEMM, so
  the cast is in place.
* **GEMM**: fp16 RedMulE on (x_q 2^k, w') = fp16 accumulation of exact fp8 x fp8 products, fp16 output: the numerics
  of an fp8-input / fp16-accumulate datapath. Only the 5 layer GEMM families run in fp8 (qkv / q, o, gate|up, down);
  the action / time / output projections (1 % of the weights) stay fp16.

Three ways to get w' into TCDM (`sh_f_gemm_step`, `--fp8-mode`):

| mode | HBM weight bytes | w' in TCDM by | numbers |
|---|---|---|---|
| 0 | 1 B / element | the 3 cores expand the bytes (SWAR integer ops) | valid |
| 1 | 1 B / element | nobody (zeroed tile): timing of a cast in the DMA back-end at line rate | invalid |
| 2 | 2 B / element (host-expanded copy) | DMA | valid, = mode 0 bit for bit |
| 3 | 1 B / element | as mode 1, and no activation cast pass: bound of a cast fused into the producing op | invalid |

### One GEMM, validated (`expert.py op --which gemm8<fam>`, 50 x K x N, random x, uniform weights)
Mode 0 and mode 2 outputs are **bit-identical**, and both are within fp16-accumulation distance of the numpy fp8
reference (`fp8.gemm_ref`, same quantiser, fp32 accumulation): gate|up 0.023 of 7.1, qkv 0.027 of 6.4, o 0.022 of 7.9,
down 0.070 of 12.4 (the fp16 GEMM vs its fp32 reference: 0.016 / 0.037 / 0.029 / 0.074).

| GEMM (tile) | fp16 | mode 2: cast pass + fp16 W | mode 1: cast pass + bytes | mode 0: cast + bytes + core expansion |
|---|---|---|---|---|
| gate\|up 720 x 4096 (50,256,240) | 91.0 us | 108.1 | 74.1 | 1540 |
| qkv 720 x 1600 (50,64,720) | 53.6 | 70.7 | 59.0 | 796 |
| o 960 x 720 (50,48,960) | 44.5 | 65.4 | 60.3 | 429 |
| down 2048 x 720 (50,48,512) | 85.3 | 123.8 | 112.6 | 887 |

The cast pass costs 17-38 us per GEMM (it is a separate row op: X goes HBM -> TCDM -> HBM once more); the byte
halving saves 5-34 us of the GEMM itself (most on gate|up, whose 256-byte rows halve; the 48-byte rows of o / down
cost nearly as much as 96-byte ones); software expansion costs ~8 cycles per weight per cluster, 10-17x the fp16 GEMM.
Weight bytes in the south HBM region (where the flow program puts them) time identically to the west region.

### Per-step cost of an fp8 step (16 layers; measured, step 0 of a 2-step fp8, fp16 run)

| step variant | time per step | HBM read / write per step | vs fp16 step |
|---|---|---|---|
| fp16 (stream) | 9.963 ms | 345.7 / 22.7 MB | - |
| fp8 mode 1 (bytes + cast pass), stream | 11.253 ms | 256.3 / 29.8 MB | +12.9 % time, -26 % read |
| fp8 mode 3 (bytes, cast fused: bound), stream | 9.821 ms | - | -1.4 % |
| fp8 mode 3 + KV-stationary | 9.851 ms | 233.2 / 22.7 MB (+ 6.4 MB NoC) | -1.1 % time, **-32.5 % read** |
| fp8 mode 2 (numbers path: cast pass + fp16 W) | 11.94 ms | 345.7 MB + cast pass | +19.9 % |
| fp8 mode 0 (core expansion) | 4 layers: 15.66 ms vs 2.90 ms fp16 (5.4x) | | |

A first mode-1 measurement left the unwritten w' tiles as garbage; the garbage activations pushed `sh_rmsnorm` into its
scalar overflow path and made even the *following fp16 step* 0.17 ms slower. Mode 1 / 3 now zero the tiles once per
call (0.6 us).

### Precision schedules (device, mode 2, full 10-step flow, stream attention)

| schedule (fmt of the layer GEMMs per step) | x_10 vs lerobot fp32: max / mean / rms | vs device all-fp16: max / mean | vs own twin: max | chunk time, measured (mode 2) | chunk time, composed (mode 3 + KVS steps) | HBM read per chunk, composed (fp8 bytes path) |
|---|---|---|---|---|---|---|
| all fp16 | 0.0093 / 0.0009 / 0.0014 | - | 0.0107 | 99.6 ms | 100.2 ms (measured) | 3457 MB |
| fp8 steps 0-2 | 0.0423 / 0.0024 / 0.0057 | 0.0483 / 0.0025 | 0.0155 | 105.9 ms | 99.7 ms | 3189 MB |
| fp8 steps 0-5 | 0.0687 / 0.0037 / 0.0096 | 0.0744 / 0.0040 | 0.0194 | 111.7 ms | 99.2 ms | 2921 MB |
| fp8 all steps | 0.1339 / 0.0075 / 0.0137 | 0.1335 / 0.0077 | 0.0251 | 119.5 ms | 98.5 ms | 2563 MB |
| fp8 steps 7-9 (control) | 0.0333 / 0.0042 / 0.0055 | 0.0261 / 0.0043 | 0.0225 | 105.9 ms | 99.7 ms | 3189 MB |

Max |action| is 1.71. "Own twin": the fp16-floor numpy model of the same schedule (the device's distance to it is the
fp16 floor of the fp8 program; it grows from 0.011 to 0.025 with the number of fp8 steps). Measured per-step times in
the mode-2 runs: fp8 step 11.93-11.97 ms, fp16 step 9.99-10.01 ms. The composed columns add measured single steps
(fp8 step: mode 3 + KVS 9.851 ms, 233 MB read with KVS / 256 MB with the cast pass; fp16 step: 10.018 / 9.963 ms,
345.7 MB) and are *not* end-to-end runs: mode 3 has invalid numbers by construction.

numpy twin of the same schedules (`np_flow(fmt_steps=...)`), x_10 vs lerobot max / mean: all fp16 0.0016 / 0.00015;
fp8 0-2 0.056 / 0.0021; fp8 0-5 0.086 / 0.0036; fp8 all 0.159 / 0.0078; fp8 7-9 0.025 / 0.0044; fp8 6-9 0.029 / 0.0046.
int8 (numpy only, W per output channel, X per row, symmetric /127): 0-5 0.023 / 0.0009; all 0.031 / 0.0026;
7-9 0.011 / 0.0020. Not run on the device: RedMulE's int8 mode accumulates in int8, and int8 x int8 products do not
fit an fp16 significand, so the fp16-RedMulE trick used for fp8 does not carry over.

**Conclusion.** Real fp8 steps work on the device and their error is what the quantiser predicts (device within
0.016-0.025 of its twin). Where the fp8 steps go matters, but not the way R4 assumed: three early fp8 steps (0-2) give
*half the mean* error of three late ones (7-9: 0.0024 vs 0.0042) but a *higher max* (0.042 vs 0.033), and all-fp8
reaches 0.134 max (8 % of the largest action); per-tensor e4m3 on these near-uniform weights is coarse (2.6 % rms
weight error), int8 per-channel would be ~4x more accurate (numpy). On today's SoftHier the fp8 step is *slower*:
+20 % with the numbers path, +13 % even with the bytes halved in the DMA (the activation cast pass costs more than the
bytes save), 5.4x with core-side expansion. Only with a cast in the DMA path *and* the activation cast fused into its
producer does an fp8 step reach -1.4 % time at -26 % (stream) / -33 % (with KVS) HBM read: per-step precision is a
traffic / energy lever on this machine, not a latency lever, because the step is bound by core row ops (softmax,
SiLU) and RedMulE runs at ~1 % of its peak (docs/SMOLVLA_EXPERT.md).

## 3. What hardware would make per-step precision cheap

Measured above: the fp8 *bytes* are worth 26-33 % of the step's HBM traffic, but every software place to convert
them costs more than they save (core expansion: 10-17x the GEMM; activation cast pass: 17-38 us per GEMM, more than
the GEMM's byte saving on 3 of 4 GEMMs). What would make the switch free, in order of the measured effect:

1. **Cast in the DMA path** (qlora_dse.md section 6, scheme C, "on-the-fly DMA"): the iDMA back-end writes
   `((b & 0x7F) << 7) | ((b & 0x80) << 8)` (fp8 -> fp16, 2 B out per 1 B in; a per-transfer flag) when it lands a
   weight tile in TCDM. Mode 1 / 3 time exactly this at line rate: the GEMMs keep fp16 RedMulE and stream half the
   weight bytes. It is a few gates per byte lane and serves QLoRA (int4 -> fp16) with the same datapath.
2. **Cast fused into the producer**, i.e. the row op that writes X (RMSNorm, SiLU*up, the attention epilogue) emits
   RN4(x) 2^k: one extra SIMD op per 4 lanes instead of a pass. Mode 1 -> mode 3 measures it: 1.43 ms per step.
3. **An fp8-input / fp16-accumulate RedMulE mode** (the RedMulE RTL's hybrid FP8 path computes in FP16 internally;
   check against PULP's RTL before claiming), with `elem_size` a per-trigger property in the model instead of an arch
   constant. Then TCDM tiles are half the size too (twice the K per tile), and the format becomes a per-trigger field:
   zero-cost per-step switching as R4 assumed.
4. **fp8 KV**: halves the resident KV (393 -> 197 KB per cluster, leaving room for the X multicast buffers) and the
   streaming attention's 18 MB per step; the scores stay fp16 / fp32. Needs (1) or (3) on the QK / PV products.
5. The per-step switch itself already costs nothing: `.fmt` / the table index is a runtime scalar.

## 4. Reproduce
```
python tests/gvsoc/expert.py flow --attn kvs --profile --dumps X A                    # KV-stationary, full flow
python tests/gvsoc/expert.py flow --steps 1 --profile --trace-dma [--attn kvs]        # bytes per segment
python tests/gvsoc/expert.py op --which gemm8gu                                        # fp8 GEMM: modes 0-2 vs numpy
python tests/gvsoc/expert.py flow --fp8-mode 2 --fmt fp8,fp8,fp8,fp16,... --save-x x_fp8_0_2.npz
python tests/gvsoc/flow_tables.py schedules --dir <dir with x_*.npz>
```
