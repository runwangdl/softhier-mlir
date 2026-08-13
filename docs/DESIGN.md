# SoftHier-MLIR — dialect design

Status: **draft / scaffolding** (iteration 1). Goal: explore whether the SoftHier
software stack can be expressed and lowered through MLIR, **reusing mature
dialects wherever possible** and adding a thin `softhier` dialect only for the
irreducible hardware specifics.

Toolkit: **[xDSL](https://github.com/xdslproject/xdsl)** (Python MLIR), following
the [`snax-mlir`](https://github.com/KULeuven-MICAS/snax-mlir) approach for a
Snitch/RISC-V + accelerator target. xDSL is chosen over C++/TableGen because the
target host (`skylab`, glibc 2.28) makes an out-of-tree LLVM/MLIR build painful,
and because snax-mlir already provides Snitch/RISC-V dialects we want to reuse.

---

## 1. What the hardware actually is (recap)

SoftHier (`pulp.chips.soft_hier_old.flex_cluster`) is an **N×N mesh of RISC-V
clusters** with HBM attached on the four mesh edges. Per cluster:

- **3 RISC-V cores** with fixed roles: a *compute* core (fires RedMule + Spatz
  vector ops), a *data-mover (DM)* core (fires all iDMA), and a third; intra-cluster
  sync via a hardware barrier.
- **RedMule** — a systolic matmul engine computing `Y = X·W` with **`Y` as an
  in-place accumulator** (`Y ← X·W + Y`). fp16/fp8/int datapaths. One in-flight op.
- **iDMA** — async data mover: 1D / 2D-strided copies, **multicast/broadcast**
  (one→many along a mesh row/col) and **in-network reduction** (REDADD / REDMAX
  performed inside the NoC).
- **Spatz** vector unit (on core 0) for softmax-class math (`vfexp`, `vfredmax`,
  `vfredsum`).
- **TCDM** — 1 MB (soft_hier_old) / 384 KB (MLA arch) per-cluster scratchpad.
- **Group barrier** — a two-phase X-then-Y barrier scoped to a rectangular
  sub-mesh ("Group"), not the whole chip.

Addressing: local TCDM, **remote TCDM** (another cluster's L1 over the NoC), and
HBM addressed per edge/node (`hbm_west/south/east/north(node, offset)`).

### The one recurring programming pattern

Every kernel (GEMM, FlatAttention, MLA-decode) is the *same* shape:

> **output-stationary SUMMA** + **in-place-accumulate RedMule** + **in-network
> iDMA broadcast/reduce** + **double-buffered software pipeline**.

- Cluster `(x,y)` permanently owns output tile `[y,x]`.
- Diagonal clusters load operand panels from the HBM edge and **broadcast** them
  along their mesh row / column.
- Every cluster runs its local tile on RedMule.
- Partial results along the contracted dimension are combined with an **iDMA
  in-network reduction** (`REDADD`/`REDMAX`).
- Loads / matmul / vector-math / stores overlap via ping-pong buffers.

This regularity is exactly why an MLIR abstraction is attractive: the whole
family collapses to *linalg on a device mesh* + a small set of accelerator ops.

---

## 2. Reuse map — mature dialects first

| SoftHier concept | Reused dialect(s) | Notes |
|---|---|---|
| GEMM / matmul (RedMule) | `linalg.matmul`, `linalg.generic` | lowered to `softhier.redmule` at the tile level |
| softmax / elementwise / RoPE | `linalg.generic`, `math`, `vector` | reductions + `exp`; `vector` for Spatz lanes |
| tensors / values | `tensor` | pre-bufferization |
| buffers, TCDM vs HBM | `memref` + memory-space attr | space = `#softhier.as<tcdm\|hbm_west\|…\|remote_tcdm>` |
| tiling / K-loop / pipeline | `scf.forall`, `scf.for`, `affine` | `scf.forall` maps to the cluster grid |
| **cluster mesh + SPMD collectives** | **`mesh`** dialect | `mesh.shard` for tiling, `mesh.all_reduce`/`mesh.broadcast` = SUMMA row/col reduce & multicast |
| async overlap / double-buffer | `async` (or `scf` pipeline) | models DM/compute overlap |
| module / control / scalars | `builtin`, `func`, `arith`, `scf`, `cf` | — |
| RISC-V codegen backend | **`snax`/Snitch/RISC-V dialects** (xDSL) | reuse instead of a new backend |

The **`mesh` dialect is the key reuse win**: SoftHier's SUMMA broadcast + NoC
in-network reduction map almost 1:1 onto `mesh.broadcast` / `mesh.all_reduce`
over a 2-D device mesh, so the distributed-attention/GEMM structure is expressed
in-tree and only *lowered* to `softhier` iDMA ops.

---

## 3. The `softhier` dialect — only the irreducible bits

### Types / attributes
- `#softhier.as<tcdm | hbm_west | hbm_south | hbm_north | hbm_east | remote_tcdm>`
  — `memref` memory-space attribute selecting the physical space / HBM edge.
- (mesh handle reused from the `mesh` dialect; a `softhier.grid` alias may wrap it.)

### Ops (initial set — see `softhier_mlir/dialects/softhier.py`)
| Op | Meaning | Lowers from |
|---|---|---|
| `softhier.redmule %x, %w, %y {fmt}` | RedMule GEMM, in-place `y += x·w` | `linalg.matmul` on TCDM tiles |
| `softhier.dma_2d %dst, %src {sizes,strides,repeat}` | strided HBM↔TCDM copy | `memref.copy` / loads-stores |
| `softhier.dma_broadcast %dst, %src {row_mask,col_mask}` | multicast along row/col | `mesh.broadcast` |
| `softhier.dma_reduce %dst, %src {kind,row_mask,col_mask}` | in-network REDADD/REDMAX | `mesh.all_reduce` |
| `softhier.group_barrier {grid_x,grid_y}` | XY two-phase group barrier | `mesh` sync / `scf.forall` boundary |
| `softhier.cluster_pos -> (index,index)` | this cluster's (x,y) | `mesh.process_multi_index` |
| `softhier.transpose %dst, %src` | on-chip transpose engine (K→Kᵀ) | `linalg.transpose` |
| `softhier.vexp %dst, %src` | Spatz `vfexp` | `math.exp` on `vector` |

Everything else (config/trigger/wait sequencing, mask encodings, HBM node
math) is an *implementation detail of the lowering*, not surfaced in the IR.

---

## 4. Lowering pipeline (the story)

```
 (1) linalg on tensor  +  mesh.shard          ← frontend / ILP picks tiling & sharding
        │  bufferize, mesh-spmdization
        ▼
 (2) linalg on memref<…,#softhier.as<tcdm>>   ← per-cluster tiles; mesh collectives explicit
        │  offload + collective lowering
        ▼
 (3) softhier.redmule / dma_2d / dma_broadcast / dma_reduce / group_barrier
        │  schedule: async / double-buffer
        ▼
 (4) scf + async + softhier ops               ← pipelined loop nest
        │  backend (reuse snax / Snitch / RISC-V dialects)
        ▼
 (5) RISC-V + custom-instruction / flex_* runtime calls  →  softhier.elf
```

- **(1)→(2)**: reuse `mesh` spmdization; `mesh.all_reduce`→`softhier.dma_reduce`,
  `mesh.broadcast`→`softhier.dma_broadcast`.
- **(2)→(3)**: `linalg.matmul` on a TCDM tile → `softhier.redmule`; softmax
  `linalg.generic` → `vector` + `softhier.vexp`.
- **(3)→(4)**: introduce ping-pong buffers + `async` to overlap DMA/compute
  (the `flatasync_run` / two-batch MLA pipeline pattern).
- **(4)→(5)**: emit the `flex_*` runtime (RedMule config/trigger/wait, iDMA
  micro-ops, group barriers) — first as **C source targeting the existing SDK
  runtime** (fastest path to a working demo), later as native RISC-V custom
  instructions via the Snitch/RISC-V dialects.

### Where ILP fits
A **decide/apply-separated** pass (per the project goal): a `decide` pass emits
an ILP over tile sizes / mesh sharding / buffer budget (TCDM capacity, RedMule
tile shape, NoC bandwidth) and writes the solution as attributes; a separate
`apply` pass consumes those attributes to drive tiling + `mesh.shard`. Keeps the
policy out of the mechanical rewrite.

---

## 4b. Scheduling (HLS-style) — and why a cost model is needed

The backend expands `softhier.gemm` into a tile loop-nest. Two scheduling
knobs so far, plus a lesson:

- **`pipeline-gemm` pass** (decide) marks gemms; the backend (apply) then
  software-pipelines the K-loop — double-buffered, prefetch K-tile `k+1` on the
  DM core while RedMule computes K-tile `k` on the first core. This is textbook
  modulo scheduling: per-step wall time → `max(t_dma, t_redmule)` instead of the
  sum. Measured: the intra-tile K-step gap dropped 4145 ns → 21 ns.
- **DMA-zeroing**: RedMule accumulates, so each output tile's accumulator is
  cleared first. Doing it with a scalar loop (65536 fp16 stores on one core) cost
  ~200 µs/tile and *dominated* everything. Zeroing via one iDMA from the ZOMEM
  region instead is ~an iDMA burst.

**The lesson (this is what motivates the ILP / a cost model):** naively turning
on `pipeline-gemm` first *regressed* the 512³ GEMM — because the real bottleneck
was the scalar zeroing, not compute-under-DMA, and the pipeline's prologue added
overhead. Only after fixing the dominant cost did pipelining pay off:

| 512³ GEMM (GEMM-phase span) | serial, scalar-zero | serial, zomem-zero | + pipelined |
|---|---|---|---|
| span | 662 µs | 73 µs (**9×**) | 56.6 µs (**+22%**) |

So *whether* a scheduling transform helps depends on the DMA-vs-compute-vs-fixed
balance and K-depth — which you cannot know without measuring or modeling. That
is exactly the job of a **static cost model** feeding the decide/apply ILP:
estimate `t_dma`, `t_redmule`, fixed costs, TCDM budget, RedMule queue-depth-1 and
iDMA-16-outstanding limits, and choose tiling + which loops to pipeline + buffer
depth to minimize estimated latency.

## 5. Roadmap

- [x] Repo scaffold + `softhier` dialect stub (types + core ops) — **iteration 1**
- [ ] `softhier-opt` round-trips the dialect; FileCheck tests
- [ ] `linalg.matmul` → `softhier.redmule` tile-and-offload pass (GEMM-first)
- [ ] `mesh` → `softhier` iDMA collective lowering (SUMMA broadcast/reduce)
- [ ] Emit `flex_*` C runtime backend; run one GEMM through GVSoC
- [ ] FlatAttention / MLA-decode fronted as `linalg` + `mesh`
- [ ] ILP decide/apply tiling pass

See the verified SoftHier simulator env in `../softhier/gvsoc` (GEMM validated).
