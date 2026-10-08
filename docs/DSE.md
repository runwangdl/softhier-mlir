# Design-space exploration: workload IR, cost model, sweep

All numbers: SoftHier flex_cluster GVSoC at `/app/install/softhier` (private copy
`/app/softhier_dse` for architecture changes), ideal HBM (`SOFTHIER_IDEAL_HBM=1`), 1 GHz, so
cycles = ns. Measured 2026-10-08 with the library at this commit.

## 1. The layer

| module | what it does |
|---|---|
| `softhier_mlir/dse/workload.py` | parses a softhier module (xDSL) into `OpRec`s: kind (gemm / summa / layernorm / softmax / gelu / add / add_bias / transpose / barrier), shape, tile, pipeline, cluster, bytes moved, MACs; `unique()` counts shapes (12-layer SigLIP = 12 x 12 unique shapes, 660 ops); `retile()`, `remap_clusters()` |
| `softhier_mlir/dse/cost.py` | analytic model per op as a function of `Arch` + `CostParams`; `estimate()` composes a workload on a timeline where ops pinned to one cluster run as concurrent lanes until the next sync point |
| `softhier_mlir/dse/calibrate.py` + `tests/gvsoc/ubench` | micro-benchmarks timed with the `mcycle` CSR, many per simulation; `fit()` derives `CostParams` |
| `softhier_mlir/dse/validate.py` | model vs simulated whole kernels |
| `softhier_mlir/dse/sweep.py` | knob grid -> analytic ranking -> top-K simulated per unique shape (one gvsoc run per arch point, on the private copy) -> composed end-to-end estimate; JSON cache keyed by (arch, case); markdown + CSV |

```bash
python -m softhier_mlir.dse.workload siglip.mlir --ops          # workload summary
python -m softhier_mlir.dse.cost --layers 1                       # analytic breakdown (SigLIP S=256)
python -m softhier_mlir.dse.calibrate --home /app/softhier_dse --fit docs/dse/params.json
python -m softhier_mlir.dse.validate  --home /app/softhier_dse --params docs/dse/params.json
python -m softhier_mlir.dse.sweep --home /app/softhier_dse --params docs/dse/params.json \
   --arch noc_link_width=256,512,1024 --arch redmule_ce=64x64,128x32 --arch mesh=1x1,4x4 --top 12 \
   --cache docs/dse/cache.json --md docs/dse/sweep_siglip_s256.md --csv docs/dse/sweep_siglip_s256.csv
```

Rules the tools enforce: `apply_arch` refuses the shared install (`SOFTHIER_ALLOW_SHARED_APPLY=1`
overrides); gvsoc runs from the binary's directory (gapy writes `gvsoc_config.json`, which
names the binary, into the cwd: two runs sharing `/app/install/softhier` as cwd swap binaries;
this affected every concurrent simulation on the machine before).

## 2. The model

* **RedMulE tile** `redmule_cycles(tm, tn, tk)` mirrors `light_redmule.cpp`: with
  `bh = ce_height`, `bw = ce_width*(ce_pipe+1)`, `bn = tcdm_bank_width/8*banks/elem` (128, 128,
  256 by default) the FSM visits `ceil(tm/bh) x ceil(tn/bw) x ceil(tk/bn)` buffer tiles; each
  costs `max(TCDM block accesses, n*(ce_pipe+1) rounded up to bw)` cycles, plus preload /
  store / `bw` latency / a fixed cost. Consequences: an output-column tile `tn < 128` or a
  contraction tile `tk < 256` idles part of the array (attention `dh = 64`: 38-48 %
  utilization, not 50 %); tiles of 256 reach 86 %, 512 reach 93 %.
* **DMA**: 2-D load `178 + bytes/(noc_link_width/8)`; per-row store
  `175 + max(22*rows, bytes/link)` (the stores are issued one `bare_dma_start_1d` per row and
  are issue-bound for rows <= 1.4 KB); the ZOMEM zero fill runs at ~750 B/cycle.
* **HBM sharing**: the aggregate HBM->TCDM rate when clusters `0..n-1` stream together is
  measured, not derived: it is **not monotonic in n** (it depends on which mesh rows are busy)
  and is used as an interpolated table, scaled with the link width.
* **sh_gemm**: per output tile `zero + load(k=0) + sum_k max(compute, load(k+1)) + store` when
  pipelined (sum when not), tiles dealt round-robin over the mesh for `cluster = -1`.
* **sh_gemm_mesh (SUMMA)**: per K step `max(compute, diag load + 2 serialized broadcasts)` +
  group barrier, with the collective constants of `collectives_measured.md` (106 cycles +
  bytes/64 per broadcast, independent of the receiver count).
* **Row ops**: sequential stage-in / scalar / stage-out per row block (block size as in
  `sh_rowop`), blocks round-robin over the clusters; cycles per element per op are parameters.
* **Composition**: ops with `cluster = c` accumulate on lane c; a `cluster = -1` op, a SUMMA or
  a `group_barrier` closes the region with `max(lanes)`. Concurrent lanes share HBM through the
  table above (n = number of lanes).

## 3. Calibration (`docs/dse/calibration_default.json`, fitted `docs/dse/params.json`)

Barrier around an empty timed region: 244-266 cycles (subtracted from every case below).

RedMulE, one trigger, cycles after barrier subtraction (model: 1 cycle/block, fixed 150;
max error 0.4 % over the 13 tiles):

| tile tm x tn x tk | sim | ideal (4096 MAC/cyc) | utilization |
|---|---|---|---|
| 256x256x256 | 4763 | 4096 | 0.86 |
| 512x256x256, 256x512x256, 256x256x512 | 8859 | 8192 | 0.93 |
| 256x256x128 | 2715 | 2048 | 0.75 |
| 256x128x256, 128x256x256 | 2715 | 2048 | 0.75 |
| 256x256x64 (attention Q.K^T) | 2139 | 1024 | 0.48 |
| 256x64x256 (attention P.V) | 2715 | 1024 | 0.38 |
| 64x256x256 | 2523 | 1024 | 0.41 |
| 256x256x32 | 1947 | 512 | 0.26 |
| 256x32x256 | 2715 | 512 | 0.19 |
| 128x128x128 | 1179 | 512 | 0.43 |

DMA, one cluster: 2-D loads of 16-128 KB run at 64 B/cycle (512 b link) + ~180 cycles; per-row
stores cost ~22-25 cycles per row whatever the row length (256 rows: ~6.3k cycles for 128 KB,
i.e. 21 B/cycle). Concurrent 2-D loads from clusters 0..n-1 (128 KB each):

| n clusters | aggregate B/cycle |
|---|---|
| 1 | 58 |
| 2 | 119 |
| 3 | 93 |
| 4 | 84 |
| 6 | 125 |
| 8 | 126 |
| 12 | 95 |
| 16 | 85 |

Scalar row ops, cycles per element on the first core (software fp16<->fp32 conversion;
layernorm also reads gamma/beta from HBM per element), fitted on one cluster and checked on
16 clusters (all within 0.2 %):

| op | cycles/element |
|---|---|
| softmax | 333 |
| layernorm | 304 |
| gelu | 148 |
| add, add_bias | 92 |
| transpose (64x64 blocks) | 16 |

So the "~150 cycles/element" working number holds for GELU only; softmax and layernorm are
2.2x worse, the residual adds 1.6x better. `CostParams.elem` is the knob the row-op work
should update.

## 4. Validation on whole kernels (`docs/dse/validation_default.json`)

| kernel | sim cycles | model | error | bound |
|---|---|---|---|---|
| gemm 256^3, 1 cluster | 16,408 | 15,520 | -5.4 % | dma |
| gemm 512x768x768, 1 cluster | 150,842 | 150,579 | -0.2 % | dma |
| gemm 512x768x768, 1 cluster, no pipeline | 199,844 | 205,267 | +2.7 % | dma |
| gemm 1024x768x768, 16 clusters (12 busy) | 120,134 | 122,486 | +2.0 % | dma |
| gemm 1024x3072x768, 16 clusters | 520,829 | 539,087 | +3.5 % | dma |
| gemm 256x768x768, 16 clusters (3 busy) | 37,174 | 37,869 | +1.9 % | dma |
| gemm 256x768x768, tk = 128 | 35,175 | 37,068 | +5.4 % | dma |
| gemm 256x3072x768 | 120,584 | 122,486 | +1.6 % | dma |
| gemm 256x768x3072 | 110,904 | 117,810 | +6.2 % | dma |
| gemm 256x256x64 (Q.K^T head), 1 cluster | 10,716 | 9,802 | -8.5 % | dma |
| gemm 256x64x256 (P.V head), 1 cluster | 12,633 | 11,806 | -6.5 % | dma |
| SUMMA 1024^3, 4x4 mesh | 101,946 | 113,263 | +11.1 % | feed |

About the reference points quoted before (256^3 9.9 us, 512x768x768 146 us, 1024x768x768
109 us, 1024x3072x768 509 us): the gemm test started the timer right after a `sh_printf`, and
the two share the slow virtual interconnect, so the start stamp landed late. With a barrier
in between (fixed in `tests/gvsoc/gemm/main.c`) 256^3 reads 14.7-16.7 us, matching the
micro-benchmark; the other three are 150.8 / 120.1 / 520.8 us. The simulator shows a +-5 %
spread between runs of the same DMA-heavy kernel (the 512x768x768 kernel read 138.9 us in the
gemm test and 150.8 us in ubench), which bounds what calibration can achieve.

What the model gets right: every GEMM that streams more than one tile per cluster is within
+-6 %; the RedMulE share is exact; row ops are exact once their per-element cost is measured.
Where it is off: single-tile kernels (-5 to -9 %: a per-kernel fixed cost the model lacks, the
same size as the barrier it subtracts), SUMMA (+11 %: the four diagonal clusters sit in four
different mesh rows and get more HBM bandwidth than the "clusters 0..3" table entry), and any
case where the HBM-sharing table is interpolated between measured points.

## 5. Simulator findings made along the way

* Two gvsoc runs with the same cwd swap binaries (`gvsoc_config.json`), see section 1.
* Some SEQUENCES of DMA cases segfault gvsoc (e.g. 2-cluster load, 2-cluster store, 4-cluster
  load, ... in one program) although each passes alone; `calibrate.run_cases` bisects the case
  list on a crash. Any crash loses the whole buffered stdout, so keep runs small.
* Timed regions must not touch `.rodata` tables (every L3 read is a slow NoC access: 2-6k
  cycles per case before the fix) and the row-op inputs must be initialised (software fp16
  conversion is data dependent: 1.12 M vs 1.27 M cycles for the same `add`).
* `csrr mcycle` works on the Snitch ISS and equals the global timer at 1 GHz.

## 6. Sweep: SigLIP layer, S = 256, D = 768, FF = 3072, 12 heads

Workload: one encoder layer from `softhier_mlir.frontend.siglip` (55 ops, 1.91 GMAC, 12 unique
kernel shapes; attention heads pinned to clusters `hd % 16`, everything else `cluster = -1`).
Grid: `noc_link_width` {256, 512, 1024} x RedMulE CE {64x64, 128x32} (4096 MAC/cycle both) x
mesh {1x1, 4x4}; all 12 points simulated (one gvsoc run of the 12 kernels per point, ~100 s
each, 866 cache entries in `docs/dse/cache.json`). Full tables: `docs/dse/sweep_siglip_s256.md`
/ `.csv`. "composed" = per-kernel simulations x counts through the lane timeline, **not** an
end-to-end measurement (the 12 heads' HBM contention and inter-op effects are not in it).

| link b | CE | mesh | model us | composed us | err | of which softmax | add_bias | layernorm | gelu | gemm (model / composed) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 64x64 | 4x4 | 49,527 | 52,828 | -6.2 % | 21,838 / 21,828 | 10,198 / 13,536 | 7,477 / 7,466 | 7,270 / 7,266 | 270 / 259 |
| 1024 | 128x32 | 4x4 | 49,527 | 52,828 | -6.2 % | same | same | same | same | 270 / 260 |
| 512 | 128x32 | 4x4 | 49,810 | 53,042 | -6.1 % | 21,854 / 21,829 | 10,241 / 13,564 | 7,486 / 7,475 | 7,288 / 7,285 | 450 / 404 |
| 512 | 64x64 | 4x4 | 49,812 | 53,043 | -6.1 % | same | same | same | same | 451 / 405 |
| 256 | 128x32 | 4x4 | 50,402 | 53,509 | -5.8 % | 21,887 / 21,831 | 10,327 / 13,620 | 7,505 / 7,493 | 7,326 / 7,321 | 830 / 727 |
| 256 | 64x64 | 4x4 | 50,403 | 53,510 | -5.8 % | same | same | same | same | 831 / 728 |
| 1024 | 128x32 | 1x1 | 700,243 | 673,503 | +4.0 % | 261,936 / 245,402 | 162,467 / 162,180 | 119,479 / 112,667 | 116,041 / 115,164 | 991 / 916 |
| 512 | 128x32 | 1x1 | 700,423 | 673,677 | +4.0 % | 261,949 / 245,414 | 162,510 / 162,221 | 119,486 / 112,673 | 116,066 / 115,189 | 1,068 / 990 |
| 256 | 128x32 | 1x1 | 701,142 | 674,361 | +4.0 % | 261,973 / 245,439 | 162,623 / 162,329 | 119,511 / 112,693 | 116,115 / 115,238 | 1,533 / 1,433 |

(The 64x64 rows of the 1x1 mesh are within 0.01 % of the 128x32 rows.)

Per-kernel model vs simulation, 4x4 mesh (the six points agree to the digit shown unless a
range is given):

| kernel | count | 256 b sim | 512 b sim | 1024 b sim | model error |
|---|---|---|---|---|---|
| gemm 256x768x768 all | 4 | 63.5k | 37.1k | 26.1k | +3.8 / +1.8 / -3.1 % |
| gemm 256x3072x768 all | 1 | 234.5k | 120.4k | 63.6k | +1.5 / +1.7 / +1.7 % |
| gemm 256x768x3072 all | 1 | 210.6k | 110.9k | 68.3k | +5.3 / +6.2 / -0.8 % |
| gemm 256x256x64 (head), 1 cluster | 12 | 11.2k | 10.2k | 9.7k | -8 .. -9 % |
| gemm 256x64x256 (head), 1 cluster | 12 | 15.2-17.2k | 12.6-14.6k | 11.4-13.3k | -5 .. -7 % |
| softmax 256x256, 1 cluster | 12 | 21.83M | 21.83M | 21.83M | 0.0 % |
| layernorm 256x768 all | 2 | 3.75M | 3.74M | 3.73M | +0.1 % |
| gelu 256x3072 all | 1 | 7.32M | 7.28M | 7.27M | +0.1 % |
| add 256x768 all | 2 | 1.15M | 1.14M | 1.13M | +0.3 % |
| add_bias 256x768 all | 5 | 1.14M | 1.14M | 1.13M | +0.3 % |
| add_bias 256x3072 all | 1 | 7.90M | 7.89M | 7.88M | **-42 %** |
| transpose 256x768 all | 1 | 213k | 208k | 207k | +2 / +0.2 / -0.5 % |

## 7. What the sweep says, and what the model gets right / wrong

Findings about the design:

* At this state of the library the layer is a **scalar-softmax problem**: one head's softmax
  (65k elements x 333 cycles = 21.8 ms) is the critical lane in every 4x4 point, and the row
  ops together are 98.5 % of the composed time. The whole architecture grid moves the
  end-to-end by 1.3 % (52.8 -> 53.5 ms). No GEMM-side knob matters until softmax / GELU /
  layernorm / bias are vectorized (the `CostParams.elem` numbers are the ones to update).
* **The GEMMs are HBM-bound, not RedMulE-bound**: the 16-cluster GEMMs scale linearly with
  the link width (256x3072x768: 234 -> 120 -> 64 us for 256 -> 512 -> 1024 b) and the CE shape
  changes nothing (64x64 and 128x32 differ by < 0.5 % everywhere, both 4096 MAC/cycle). With
  12-16 clusters streaming, each sees 5-8 B/cycle of the shared HBM node while the RedMulE
  tile needs 54 B/cycle to stay busy; the mesh is 7-10x over-provisioned in compute for this
  memory system. Only the single-cluster attention heads come near balance (tile 2.1k vs
  load 1.4k cycles).
* **16 clusters buy 12.7x on the layer** (673 -> 53 ms composed), all of it from the row ops
  splitting over clusters; the GEMM part alone gains only 3.5x (0.92 -> 0.26 ms at 1024 b),
  the rest is the HBM node.
* Attention tiles with `dh = 64` run the array at 38-48 %; the fix is in the kernel (process
  two heads per trigger, or `tn = 128`), not in the CE shape.

Model accuracy:

* Right: the RedMulE tile time (exact), multi-tile GEMMs on 1 or 16 clusters (within +-6 %
  over three link widths, i.e. the HBM-sharing table and the double-buffer max() are the
  right structure), all row ops whose per-element cost was measured (+-0.3 %), the ranking
  of the points (the model orders all 12 points like the composed sims).
* Off: (1) `add_bias 256x3072` on 16 clusters runs at 160 cycles/element in the simulator
  while the same op at 768 columns, and at 3072 columns on one cluster, runs at 92; the
  per-op constant cannot express it and it is the whole -6 % of the 4x4 rows (cause not
  found; the kernel's TCDM layout is the suspect). (2) Single-tile kernels are 5-14 % faster
  than the model (a per-kernel fixed cost that the model's barrier subtraction does not
  capture). (3) On the 1x1 mesh scalar loops run faster than on 4x4 (transpose 3x, softmax
  6 %, layernorm 6 %): with 45 other cores polling barriers the 64 KB L3 instruction memory
  is contended; the model has no instruction-fetch term. (4) SUMMA is +11 % (section 4).
  (5) The HBM-sharing curve was measured for clusters 0..n-1 of a 4x4 mesh and is scaled
  linearly with the link width; other meshes and placements are extrapolations.
