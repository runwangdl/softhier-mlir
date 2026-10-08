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
softhier_mlir/sim/testdata.py      test inputs generated on the host into the HBM preload image (softhier-translate --preload-elf)
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
python tests/gvsoc/run.py siglip --cluster all --data device  # test inputs generated on the device instead of
                                                              # preloaded from the host (the default; ~20x less wall time)
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
| `softhier.view` / `layernorm` / `softmax` / `gelu` / `add` / `add_bias` | strided HBM views + row-wise fp16 tensor ops (`sh_*` library calls) |
| `softhier.attention %q, %k, %v -> %o {scale, heads, q_block}` | fused multi-head attention: `softmax(scale q k^T) v` in work items of `q_block` query rows inside one cluster's TCDM, fp16 SIMD softmax, any S the L1 holds (`sh_attention_q`); S=256: 0.55 ms vs 0.83 ms per-head, S=1024: 7.8 vs 11.8 ms. GEMM tiles come from the cost model (`softhier_mlir.dse.tiling`, the GEMMs are HBM-bound); numbers in `docs/DSE.md` section 8 |

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
(with a one-line CMakeLists) and `make sh-old-hs app=... && make sh-old-run`, or use
`tests/gvsoc/run.py` (private build dir, ideal HBM, host-side comparison).

### Real weights: SmolVLA's SigLIP vision tower

Parameters go into HBM through the simulator's preload path (`softhier_mlir/sim/preload.py`,
see `docs/SIMULATOR_NOTES.md`), so a network with real weights needs no on-device data generation:

```bash
# checkpoint -> weights in library layout + im2col'd test image + fp32 HF reference (torch/transformers)
python3 -m softhier_mlir.frontend.smolvla prepare --ckpt /app/models/smolvla_base/model.safetensors \
        --seq 256 --out /app/models/smolvla_base/vision_s256.npz
# npz -> MLIR (+ preload image) -> C -> gvsoc; compares sampled EMB / L<n> / OUT against the reference
.venv/bin/python tests/gvsoc/run.py smolvla --npz /app/models/smolvla_base/vision_s256.npz --layers 1
```

`--seq 256` keeps the top-left 16x16 patches with their own position embeddings (exactly the full
model restricted to those tokens); `--seq 1024 --all-layers` is the full 512x512 encoder. The
program is one `scf.for` over the layers and one over the heads (the cluster instruction memory is
64 KB), starts with `softhier.preload_wait` on the image's sentinel segment (the loader's done flag
fires before the data has crossed the NoC) and prints `[mark]` stamps per layer; `--unroll` gives the
spelled-out form with the layer-1 intermediate dumps, `--from-log` re-evaluates a finished run.

Results (16 clusters, ideal HBM, gvsoc at 1 GHz; 256 sampled elements per tensor against the fp32
HF `SiglipVisionModel`):

| run | per layer (simulated) | total | wall | embeddings | layer 1 | layer 12 | post-LN |
|---|---|---|---|---|---|---|---|
| seq 256, 12 layers | 2.29 ms (attention 1.21, proj+MLP 1.07) | 38.8 ms compute + 2.7 ms preload | 502 s | max abs 0.0067 (max 1.23) | 0.039 (max 3.6) | 0.19 (max 359) | 0.21 (max 24.7, median 0.01) |

Everything is at the fp16 floor (fp16 operands and RedMulE accumulation): the device agrees with
the fp16-rounded numpy model of the same program as closely as with HF.

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
softhier_mlir/frontend/siglip.py   SigLIP/ViT encoder emitter (--fused: softhier.attention per layer)
runtime/                           softhier-ops C library the generated code calls (sh_gemm, sh_attention, ...)
tests/filecheck/                   FileCheck tests
softhier_mlir/sim/trace.py         gvsoc trace (redmule/idma/cluster_registers) -> Perfetto JSON / PNG / utilisation; sim/viewer builds an HTML timeline
tests/gvsoc/run.py                 on-simulator tests: gemm | rowops | fp16cvt | attention [--composed] | mesh | siglip | siglip-mlir [--fused] | mlir
tests/gvsoc/ubench/                cost-model micro-benchmarks (docs/DSE.md)
```

## Why this can work

Every SoftHier kernel (GEMM, FlatAttention, DeepSeek-MLA decode) is the same
shape: **output-stationary SUMMA + in-place-accumulate RedMule + in-network iDMA
broadcast/reduce + a double-buffered pipeline.** That regularity is what makes a
compact dialect + a `mesh`-style distribution the right abstraction. Details and
the staged lowering pipeline are in [`docs/DESIGN.md`](docs/DESIGN.md).

## License

Apache-2.0.

### SmolVLA's VLM text prefix (16 Llama layers, GQA, RoPE, token-class attention mask)

The prefix the action expert cross-attends to (connector + 16-layer SmolVLM2 text tower over image, language and
state tokens) runs through `softhier.rmsnorm / rope / silu_mul / pixel_shuffle` and the masked grouped-query
`softhier.attention` (`runtime/sh_llm.inc.c`); references, tolerances, the attention mask, the KV-cache layout
and the simulated times are in [`docs/SMOLVLA.md`](docs/SMOLVLA.md).

```bash
python3 -m softhier_mlir.frontend.smolvla prepare-vlm --cams 3 --out /app/models/smolvla_base/vlm_c3.npz   # 241 tokens
.venv/bin/python tests/gvsoc/run.py smolvla-vlm --npz /app/models/smolvla_base/vlm_c3.npz --all-layers
.venv/bin/python tests/gvsoc/run.py llmops --cluster all                                                     # the ops alone
```
