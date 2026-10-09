# X-panel multicast: a small-M GEMM whose activation crosses HBM once (W6)

Branch `agent/w6-xpanel-mcast`. Code: `runtime/sh_gemm.inc.c` (`sh_gemm_xmcast`, `sh_gemm_xmcast_ex`,
`sh_gemm_xmcast_l1_bytes`; `sh_gemm` is unchanged), `softhier.gemm {xmcast}` (`softhier_mlir/backend/emit_c.py`,
FileCheck `tests/filecheck/gemm_xmcast_translate.mlir`), `emit_flow(..., gemm="xmcast")`
(`softhier_mlir/frontend/smolvla_expert.py`, `tests/gvsoc/expert.py flow --gemm xmcast`), `run.py gemm` shapes with
cluster field `xm | xmp | xmw` + `--dump-z` (numpy check) + `--trace-dma` (bytes), cost model
`softhier_mlir/dse/cost.py` (`xmcast_plan`, `gemm_xmcast_est`, `gemm_xmcast_traffic`, `expert_step_traffic(gemm=)`,
`EXPERT_OP_US_XMCAST`, `expert_step_est(gemm=)`; workload kind `xmcast`).

All numbers: gvsoc, 16 clusters (4 x 4), 1 GHz (cycles = ns), ideal HBM, fast ISS libraries, 2026-10-09.

## 1. The problem

`sh_gemm(..., SH_ALL)` deals output tiles round-robin over the clusters, and every output tile streams its own X row
panel from HBM. For the expert's GEMMs (M = 50 N rows, one row tile, 15-25 column tiles) the X panel crosses HBM once
per column tile: 127 MB of the 368 MB a 1-candidate flow step moves, 508 of 840 MB at N = 4
(docs/FLOW_DATAFLOW.md, docs/WORLD_MODEL.md). That term grows with the number of candidates N; the weights do not.

## 2. The dataflow (`sh_gemm_xmcast_ex`)

```
cluster c owns Z[:, c Nc : c Nc + nc]       Nc = ceil(N / 16) rounded up to 4 (720 -> 48 on 15 clusters, 1600 -> 100)
for each row block (tm rows, default all M):
    Y [rows x nc] stays in TCDM over the whole K loop
    whole schedule (X <= 320 KB):  cluster 0 loads X panel-major (KT panels of rows x tk) and multicasts it once
                                   (32 KB collectives, one in flight); meanwhile every cluster loads W panel 0;
                                   global barrier; then a local double-buffered loop over K-panels:
                                   RedMulE Y += X[kk] W[kk]  ||  DMA W[kk+1]  (own columns only)
    panel schedule (larger X):     panel kk+1 is loaded + multicast by cluster (kk+1) % 16 during the RedMulE of
                                   panel kk; double-buffered X and W; one global barrier per panel
    store Y (row by row)
```

HBM reads are X once + W once per row block + nothing else; X also crosses the NoC once as a multicast (line rate,
independent of the number of receivers, collectives_measured.md). The multicast is `sh_dma_bcast_1d` (register-safe,
SIMULATOR_NOTES #15) and every collective is waited before the next one (#4); the RedMulE trigger is one asm statement
(#12). `tk` = largest divisor of K <= 256 whose scratch fits under 0x90000 (below the KV-stationary resident region).
The auto rule (whole up to 320 KB of X, else panels) comes from section 3: a large X spends its load + multicast before
the first RedMulE when it is sent whole.

## 3. Per shape: sh_gemm SH_ALL vs sh_gemm_xmcast (auto)

`tests/gvsoc/run.py gemm --dump-z 256 --trace-dma --shapes ...`. sh_gemm tiles = the frontends' (TILES /
`batch_tiles(N)` for the expert, VLM_TILES for the prefix, which pads 241 rows to 256). Every run: 256 device-sampled
elements of Z vs numpy `X @ W` (integer-valued fp16 data, exact) **0 mismatches, max abs 0**, plus the device's own
256-sample check. Bytes from the gvsoc iDMA trace (`dma_bytes.py`); HBM writes are equal in both (= Z).

| GEMM | M x N x K | sh_gemm tile | sh_gemm us | xmcast us (schedule) | speed-up | HBM read MB sh_gemm -> xmcast | NoC c2c MB (xmcast) |
|---|---|---|---|---|---|---|---|
| qkv (self), N=1 | 50x1600x720 | 50,64,720 | 53.65 | 40.62 (whole) | 1.32x | 4.104 -> 2.376 | 0.072 |
| q (cross), N=1 | 50x960x720 | 50,64,720 | 34.48 | 29.31 (whole) | 1.18x | 2.462 -> 1.454 | 0.072 |
| o, N=1 | 50x720x960 | 50,48,960 | 44.58 | 26.34 (whole) | 1.69x | 2.822 -> 1.478 | 0.096 |
| gate/up, N=1 | 50x4096x720 | 50,256,240 | 91.06 | 78.05 (whole) | 1.17x | 7.050 -> 5.970 | 0.072 |
| down, N=1 | 50x720x2048 | 50,48,512 | 85.41 | 50.86 (whole) | 1.68x | 6.021 -> 3.154 | 0.205 |
| qkv, N=4 | 200x1600x720 | 200,64,720 | 124.64 | 50.21 (whole) | 2.48x | 9.504 -> 2.592 | 0.288 |
| o, N=4 | 200x720x960 | 200,48,960 | 101.12 | 32.99 (panel) | 3.07x | 7.142 -> 1.766 | 0.384 |
| gate/up, N=4 | 200x4096x720 | 200,256,240 | 150.40 | 97.08 (whole) | 1.55x | 10.506 -> 6.186 | 0.288 |
| down, N=4 | 200x720x2048 | 200,48,512 | 196.13 | 83.63 (panel) | 2.35x | 15.237 -> 3.768 | 0.819 |
| qkv, N=8 | 400x1600x720 | 400,64,360 | 217.88 | 77.07 (panel) | 2.83x | 16.704 -> 2.880 | 0.576 |
| o, N=8 | 400x720x960 | 400,48,480 | 171.56 | 59.74 (panel) | 2.87x | 12.902 -> 2.150 | 0.768 |
| gate/up, N=8 | 400x4096x720 | 400,256,240 | 231.40 | 137.66 (panel) | 1.68x | 15.114 -> 6.474 | 0.576 |
| down, N=8 | 400x720x2048 | 400,48,512 | 345.52 | 111.21 (panel) | 3.11x | 27.525 -> 4.588 | 1.638 |
| prefix q/o | 241x960x960 | 256 rows: 128,192,320 / 241 rows: 241,192,320 | 66.49 / 50.88 | 47.81 (panel) | 1.39x / 1.06x | 6.144 / 4.157 -> 2.306 | 0.463 |
| prefix gate or up | 241x2560x960 | 256: 128,256,320 / 241: 241,256,320 | 153.62 / 108.29 | 81.89 (panel) | 1.88x / 1.32x | 14.746 / 9.542 -> 5.378 | 0.463 |

The xmcast HBM read is exactly the floor X + W (e.g. o, N=1: 0.096 + 1.382 MB). Whole vs panel (both measured, same
numbers): 50 rows: whole 26.3-78.1 us vs panel 27.5-79.9 (whole wins by 2-9 %); 200 x 720 x 960 (X 384 KB): whole 38.5
vs panel 33.0; 241 x 960 x 960 (X 463 KB): 52.2 vs 47.8; 241 x 2560 x 960: 95.5 vs 81.9; 200 x 1600 x 720 (288 KB):
50.2 vs 51.8. Hence the 320 KB threshold. The prefix row: against the 241-row single-row-tile sh_gemm (which the prefix
does not use: it pads to 256 rows and tiles 128) the gain is small on q/o, because the weights dominate the bytes.

What is left is the weight stream: gate/up at N = 1 moves 5.9 MB of W in 78 us = 76 B/cycle, at the measured aggregate
HBM rate (85-125 B/cycle, DSE.md section 8). The GEMMs whose X share was large (o, down, all of N >= 4) gain most.

`cost.gemm_xmcast_est` (X load at link rate, W panels at the measured shared HBM rate, 32 KB multicasts at
106 cycles + 64 B/cycle, one barrier per panel) reproduces the 24 measured (shape, schedule) points with mean |error|
10 %, max 29 % (it underestimates the 50-row whole cases by 16-29 %).

## 4. In the expert flow step

`emit_flow(gemm="xmcast")` lowers every step GEMM (embedding a / t / t, per layer qkv or q, o, gate|up, down, the
output projection) to `softhier.gemm {xmcast}` with tile_m = all rows; the once-per-chunk cross-layer KV projections stay
on sh_gemm. Program 60.5 KB (sh_gemm only: 56.9 KB) of the 64 KB instruction memory. Attention: streaming (`sh_x_attention`).

| run | ms / step sh_gemm | ms / step xmcast | change | HBM read MB / step | HBM write | NoC c2c | x_t vs lerobot (max / mean) |
|---|---|---|---|---|---|---|---|
| N=1, 4 layers, 2 steps | 2.884 | 2.590 | -10.2 % | - | - | - | step 2: 0.0044 / 0.0006 both |
| N=1, 16 layers, 10 steps | 9.963 (FLOW_DATAFLOW.md; 1-step rerun 9.902) | **9.338** | **-6.3 %** | 345.7 -> **237.1** (-31 %, trace) | 22.7 both | 0 -> 7.3 | x_10: 0.0093 / 0.0009 both (twin 0.0107) |
| N=4, 4 layers, 2 steps (bytes: 1-step trace) | 7.903 | 6.625 | -16.2 % | 197.2 -> **82.4** (-58 %, trace) | 24.5 both | 0 -> 8.0 | every cand within 0.0044-0.0059 of its twin, both |
| N=4, 16 layers | 28.92 (WORLD_MODEL.md) | **24.90** (3 steps) | **-13.9 %** | read + write: 840 -> **406** (-52 %, program count) | (in the total) | ~29 (4 x the N=1 trace) | cand 0 vs lerobot per step 0.0029 / 0.0032 / 0.0042, as sh_gemm |

Accuracy is unchanged to the bit level that matters: the per-step x_t errors against lerobot and the fp16-floor twin
are identical to the sh_gemm runs (each output element is still one RedMulE accumulation over the same K order).
Per-step HBM read is from the iDMA trace of a 1-step 16-layer run (both programs, excluding the once-per-chunk KV
projection's 15.6 MB); `cost.expert_step_traffic` counts 368 / 259 MB (read + write) for the same programs, within
0.3 % of the trace (368.4 / 259.8 MB); at N = 4 and 4 layers the trace gives 221.7 / 106.9 MB against the count's 221.4 /
106.6 MB, so the 16-layer N = 4 column uses the program count.

Per-op, per layer call, 16 layers (us; the GEMM entries include their rmsnorm / residual add):

| N | qkv | o_proj | gate/up | down | sum | attention | silu*up | rope | emb / step |
|---|---|---|---|---|---|---|---|---|---|
| 1, sh_gemm | 64.0 | 41.4 | 96.1 | 66.2 | 267.7 | 238.9 | 78.4 | 22.2 | 144.6 |
| 1, xmcast | 57.7 | 32.4 | 90.7 | 53.0 | 233.8 (-13 %) | 238.9 | 78.5 | 22.5 | 116.1 |
| 4, sh_gemm | 169.3 | 109.3 | 187.7 | 184.0 | 650.3 | 813.8 | 257.9 | 59.1 | 319.9 |
| 4, xmcast | 115.5 | 53.0 | 155.8 | 80.1 | 404.4 (-38 %) | 813.9 | 257.9 | 59.2 | 223.7 |

## 5. Amdahl: what is left

After the change the GEMM entries (with their rmsnorm / residual adds) are **40 % of the N = 1 step** (3.74 of
9.34 ms) and **26 % of the N = 4 step** (6.47 of 24.90 ms). Attention (the fetch-bound fp16 softmax on 3 Snitch cores
per head) is 41 % / 52 %, silu*up + rope 17 % / 21 %. The GEMMs now read only weights plus X once; the weights
(195 MB / step) at the measured 85-125 B/cycle aggregate HBM rate take 1.6-2.3 ms per step, so even a perfect GEMM
(weight-bound, nothing else) saves at most ~1.4-2.1 ms more at N = 1 (<= 22 % of the step) and ~4 ms at N = 4 (<= 17 %);
with the GEMMs at zero time the step would still be 5.6 ms (N = 1) / 18.4 ms (N = 4). The step stays softmax-bound.

## 6. Conclusion

Multicasting the activation panel instead of re-reading it per output tile removes the term of the GEMM traffic that
grows with the number of candidates: a flow step reads 31 % fewer HBM bytes at N = 1 (346 -> 237 MB, measured) and
52 % less read + write traffic at N = 4 (840 -> 406 MB, program count; the 4-layer trace reads 58 % less), and the step's HBM traffic now grows by ~49 MB per extra candidate instead of
~157 MB. Single GEMMs get 1.2-1.7x faster at 50 rows and 1.6-3.1x at 200-400 rows, at the exact X + W byte floor and
bit-identical numbers. The step itself gains less: -6.3 % at N = 1 (9.96 -> 9.34 ms) and -13.9 % at N = 4
(28.92 -> 24.90 ms), because GEMMs were only 43 % / 36 % of the step and are now 40 % / 26 %, the rest being the fp16
softmax and row ops on the cores, which this change does not touch. N candidates now cost 0.67 N single-chunk latencies
at N = 4 (was 0.73 N). The next lever is the attention softmax; on the GEMM side only the weight stream remains
(per-step fp8 weight bytes in the DMA path, docs/FLOW_DATAFLOW.md section 3).

## 7. Reproduce

```bash
.venv/bin/python tests/gvsoc/run.py gemm --dump-z 256 --trace-dma --shapes 50x720x960:50,48,960,1,0,all 50x720x960:0,0,0,1,0,xm
#   cluster field: xm = auto, xmp = panel, xmw = whole; tm,tn,tk = rows per block, column granule, K-panel (0 = auto)
.venv/bin/python tests/gvsoc/expert.py flow --steps 2 --layers 4 --profile --gemm xmcast [--cands 4]
.venv/bin/python tests/gvsoc/expert.py flow --profile --gemm xmcast                       # 16 layers, 10 steps
.venv/bin/python tests/gvsoc/expert.py flow --steps 1 --profile --trace-dma --gemm xmcast  # HBM / NoC bytes per segment
python -c "from softhier_mlir.dse import cost as C; print(C.expert_step_est(4, gemm='xmcast'))"
```
