# softhier-mlir

An [xDSL](https://github.com/xdslproject/xdsl)/MLIR **dialect and lowering flow
for [SoftHier](https://github.com/pulp-platform/softhier-sdk)** — a PULP/Snitch
RISC-V *many-cluster* accelerator (RedMule matmul engines + iDMA NoC + HBM),
simulated in GVSoC.

The bet: most of the SoftHier software stack can be expressed with **mature MLIR
dialects** (`linalg`, `memref`, `scf`, `vector`, and a `mesh`-style device grid),
leaving a **thin `softhier` dialect** for only the irreducible hardware ops.
Backend reuse follows [`snax-mlir`](https://github.com/KULeuven-MICAS/snax-mlir)
(Snitch/RISC-V dialects) rather than a from-scratch codegen.

> Status: **working end-to-end.** A network's GEMMs written in standard
> `linalg.matmul` lower to the `softhier` dialect, generate C against the SoftHier
> `flex_` runtime, and run on GVSoC with numerically-verified results — including
> a 2-layer MLP and multi-tile (512³) GEMMs.

**Docs:** [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — how to generate & run, and how to
add a new op (step by step) · [`docs/DESIGN.md`](docs/DESIGN.md) — the dialect
abstraction and lowering pipeline.

## Working end-to-end pipeline

```
linalg.matmul  --softhier-opt -p linalg-to-softhier-->  softhier dialect
               --softhier-translate-->  main.c (flex_ runtime)
               --SDK build + GVSoC-->  runs, self-checks (MLP_PASS / GEMM_PASS)
```

Verified on GVSoC (`../softhier/gvsoc`, RedMule traces + on-device checks):
- **256³ GEMM** — one RedMule tile.
- **2-layer MLP** `Z = ReLU(X@W1)@W2` — two chained GEMMs, `MLP_PASS` (all outputs 1.0).
- **512³ GEMM** — tiled loop-nest, 8 RedMule tiles with K-accumulation, `GEMM_PASS`.
- **linalg-form** of the MLP and the 512³ GEMM — lower + run, same results.
- **non-constant input** (col-parity X) — `GEMM_PASS` (Z=0.75), correctness beyond constants.

## The `softhier` dialect

| Op | Meaning |
|---|---|
| `softhier.redmule %x, %w into %y {fmt}` | RedMule GEMM tile, in-place `y += x·w` |
| `softhier.gemm %x, %w into %z {fmt}` | full (multi-tile) HBM GEMM; backend tiles it |
| `softhier.dma_2d %src -> %dst` | strided HBM↔TCDM copy |
| `softhier.dma_broadcast` / `dma_reduce {kind}` | mesh row/col multicast / in-network REDADD·REDMAX |
| `softhier.group_barrier {grid_x,grid_y}` | two-phase XY group barrier |
| `softhier.relu` / `vexp` / `transpose` | Spatz ReLU / `vfexp` / transpose engine |
| `softhier.l1_zero` / `l1_fill` / `l1_buffer` | TCDM tile clear / fill / declare |
| `softhier.hbm_buffer` / `hbm_fill` / `hbm_fill_col_parity` | HBM buffer declare / fill |
| `softhier.check_const` / `hbm_check_const` | on-device self-verify (prints PASS/FAIL) |
| `softhier.cluster_pos` | this cluster's (x,y) |

Memref memory spaces select the physical space / HBM edge:
`"tcdm"`, `"remote_tcdm"`, `"hbm_west" | "hbm_south" | "hbm_north" | "hbm_east"`.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# lower a standard-linalg 512^3 GEMM onto SoftHier, then emit runnable C
softhier-opt examples/gemm512_linalg.mlir -p linalg-to-softhier | \
  softhier-translate /dev/stdin      # (or: -o main.c)

# run the FileCheck suite
for f in tests/filecheck/*.mlir; do echo "$f"; done   # see each file's RUN line
```

To run generated C on GVSoC, drop it into `soft_hier_sdk/generated/<name>/`
(with a one-line CMakeLists) and `make sh-old-hs app=... && make sh-old-run`.

## Design-space exploration

`softhier_mlir/dse/` turns a module into a shape-level workload (`workload.py`), estimates it
analytically for any `Arch` (`cost.py`: RedMulE FSM, DMA/HBM rates, double-buffered tiles,
SUMMA collectives, scalar row ops), calibrates the constants on micro-benchmarks
(`calibrate.py` + `tests/gvsoc/ubench`) and sweeps knob grids, simulating the top-K points per
unique kernel shape on a **private** SoftHier copy (`sweep.py`). Results, the fitted
constants and the model-vs-simulation errors are in [`docs/DSE.md`](docs/DSE.md).

```bash
python -m softhier_mlir.frontend.siglip --layers 12 --no-test | python -m softhier_mlir.dse.workload -
python -m softhier_mlir.dse.calibrate --home /app/softhier_dse --fit docs/dse/params.json
python -m softhier_mlir.dse.sweep --home /app/softhier_dse --params docs/dse/params.json \
    --arch noc_link_width=256,512,1024 --arch redmule_ce=64x64,128x32 --arch mesh=1x1,4x4 --top 12
```

## Layout

```
docs/DESIGN.md                     the dialect abstraction + lowering pipeline
docs/DSE.md                        cost model calibration + sweep results
softhier_mlir/dialects/softhier.py the dialect (types + ops)
softhier_mlir/transforms/          lowering passes (WIP)
softhier_mlir/dse/                 workload IR, analytic cost model, calibration, sweep driver
softhier_mlir/tools/softhier_opt.py the opt driver
tests/filecheck/                   FileCheck tests
tests/gvsoc/                       on-simulator tests (run.py) + ubench (cost-model micro-benchmarks)
```

## Why this can work

Every SoftHier kernel (GEMM, FlatAttention, DeepSeek-MLA decode) is the same
shape: **output-stationary SUMMA + in-place-accumulate RedMule + in-network iDMA
broadcast/reduce + a double-buffered pipeline.** That regularity is what makes a
compact dialect + a `mesh`-style distribution the right abstraction. Details and
the staged lowering pipeline are in [`docs/DESIGN.md`](docs/DESIGN.md).

## License

Apache-2.0.
