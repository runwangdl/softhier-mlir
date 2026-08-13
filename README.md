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

> Status: **early scaffold (iteration 1).** The `softhier` dialect parses,
> verifies, and round-trips. Lowering passes are next. See
> [`docs/DESIGN.md`](docs/DESIGN.md).

## The `softhier` dialect

| Op | Meaning |
|---|---|
| `softhier.redmule %x, %w into %y {fmt}` | RedMule GEMM, in-place `y += x·w` |
| `softhier.dma_2d %src -> %dst` | strided HBM↔TCDM copy |
| `softhier.dma_broadcast %src -> %dst` | multicast along a mesh row/col (SUMMA) |
| `softhier.dma_reduce %src -> %dst {kind}` | in-network REDADD/REDMAX |
| `softhier.group_barrier {grid_x,grid_y}` | two-phase XY group barrier |
| `softhier.cluster_pos` | this cluster's (x,y) |
| `softhier.transpose` / `softhier.vexp` | transpose engine / Spatz `vfexp` |

Memref memory spaces select the physical space / HBM edge:
`"tcdm"`, `"remote_tcdm"`, `"hbm_west" | "hbm_south" | "hbm_north" | "hbm_east"`.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# parse + verify + reprint the example SUMMA GEMM tile
softhier-opt tests/filecheck/gemm_tile.mlir
```

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
