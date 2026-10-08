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

> Status (2026-10-08): **the compiler emits calls into an operator library** (`runtime/`,
> "softhier-ops") instead of inline runtime code. A SigLIP/ViT encoder layer written by the
> frontend (`softhier_mlir/frontend/siglip.py`) lowers to the `softhier` dialect, translates to
> C, builds and runs on the SoftHier GVSoC with every tensor matching a numpy reference on one
> cluster (S=256, D=768, 12 heads, FFN 3072). GEMMs run on one cluster, round-robin over all 16,
> or as a mesh-wide SUMMA; the 8 original examples still pass end to end.

## Layout (what calls what)

```
softhier_mlir/frontend/siglip.py   model -> softhier dialect (HBM buffers, strided views, per-head attention)
softhier_mlir/dialects/softhier.py the dialect: gemm, layernorm, softmax, gelu, add, add_bias, transpose, view,
                                   redmule/l1 ops, test-data ops (hbm_fill_lcg, dump_samples)
softhier_mlir/transforms/          linalg-to-softhier, pipeline-gemm, distribute-summa (policy = attributes)
softhier_mlir/backend/emit_c.py    one library call per op; generated main.c includes only runtime/sh_ops.h
runtime/sh_ops.h                   the library API (SPMD: every op is called by all cores of all clusters)
runtime/sh_gemm.inc.c              tiled GEMM (any tm/tn/tk, split-K, double-buffered, cluster=0|SH_ALL), SUMMA
runtime/sh_rowops.inc.c            layernorm / softmax / gelu / add / bias / scale / transpose on HBM tensors
runtime/sh_test.inc.c              on-device LCG data + sampled dumps; host twin in softhier_mlir/testing/lcg.py
softhier_mlir/sim/gvsoc.py         build (x86 chroot, private build dir) + run (ideal HBM) + Arch knobs
tests/gvsoc/run.py                 gemm | rowops | siglip | siglip-mlir | mlir <files>: build, simulate, compare
docs/SIMULATOR_NOTES.md            the gvsoc model bugs found on the way and the conventions relied on
```

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" numpy
bash tests/run_filecheck.sh                                   # compiler tests (no simulator)
python tests/gvsoc/run.py gemm                                # library GEMM shapes on gvsoc, self-checked
python tests/gvsoc/run.py mlir examples/*.mlir                # every example end to end
python tests/gvsoc/run.py siglip-mlir --seq 256 --cluster 0   # one encoder layer through the compiler
python -m softhier_mlir.frontend.siglip --layers 12 --no-test # the dialect module of a 12-layer encoder
```
Simulator setup (aarch64 host, x86 toolchain chroot, ideal HBM) is described in `softhier_mlir/sim/gvsoc.py`.

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

## Layout

```
docs/DESIGN.md                     the dialect abstraction + lowering pipeline
softhier_mlir/dialects/softhier.py the dialect (types + ops)
softhier_mlir/transforms/          lowering passes (WIP)
softhier_mlir/tools/softhier_opt.py the opt driver
tests/filecheck/                   FileCheck tests
```

## Why this can work

Every SoftHier kernel (GEMM, FlatAttention, DeepSeek-MLA decode) is the same
shape: **output-stationary SUMMA + in-place-accumulate RedMule + in-network iDMA
broadcast/reduce + a double-buffered pipeline.** That regularity is what makes a
compact dialect + a `mesh`-style distribution the right abstraction. Details and
the staged lowering pipeline are in [`docs/DESIGN.md`](docs/DESIGN.md).

## License

Apache-2.0.
