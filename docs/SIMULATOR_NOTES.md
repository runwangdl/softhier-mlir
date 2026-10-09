# SoftHier GVSoC: what the stock model gets wrong, and how this repo works around it

All findings are from the `flex_cluster` target of the SoftHier GVSoC fork at
`/app/install/softhier` (branch `bowwang-dev/softhier-deeploy`, 2026-10). Patches live in
`AI_AGENT/SoftHier/dse/patches/` (not in this repo, because they belong to the simulator).
As of 2026-10-08 none of them is fixed in the public upstream repos (`gvsoc/gvsoc`,
`gvsoc/gvsoc-pulp`, `pulp-platform/softhier-sdk`); the flex_cluster chip models were even
removed from `gvsoc-pulp` in 2026-01 and the public SDK (2026-06) only ships the runtime.

| # | Symptom | Root cause | Fix / workaround |
|---|---|---|---|
| 1 | scalar `flw`/`fsw` to HBM, L3 `.rodata` or remote TCDM read stale values; NoC error `No entry found for burst (base: 0x2d68)` | Snitch FP subsystem (`snitch_fp_ss.cpp`) completes an offloaded instruction after a fixed latency even when the access returned `IO_REQ_PENDING`; the next access reuses the in-flight `IoReq` | patch `gvsoc_snitch_fp_ss_pending.patch`: wait in `Iss::handle_event` while `exec.is_stalled()` |
| 2 | `flh`/`fsh` (Zfh) return garbage even from TCDM | ISA generator tags `flh` as an integer `load`, and `flh_exec` uses the integer load path; the half never reaches the FP register file | handler patched (`gvsoc_zfh_flh_fsh.patch`) but still returns 0: **the library avoids scalar fp16 entirely** (integer loads + software fp16<->fp32 + fp32 FPU) |
| 3 | 768-term fp16 dot products come out ~10 % low | `float_to_fp16` in `light_redmule.cpp` truncates the mantissa; every FMA of the accumulation rounds toward zero | patch `gvsoc_redmule_fp16_rne.patch`: IEEE round-to-nearest-even (real RedMulE rounds) |
| 4 | gvsoc segfaults (no output) with two outstanding collective broadcasts from one DM core | NoC collective model | `sh_gemm_mesh` waits after every broadcast |
| 5 | gvsoc segfaults at elaboration | DRAMSys ships as an x86-64 `.so` on an aarch64 host | `SOFTHIER_IDEAL_HBM=1` swaps HBM for gvsoc's ideal memory (HBM-bound numbers are optimistic) |
| 6 | `fcvt.w.s` followed by `fcvt.s.w` on the same integer register (floor/round via int) returns a stale integer: 25 % wrong results when eight independent pairs are issued back to back (`tests/gvsoc/fp16cvt`) | the integer operand of an offloaded int->FP instruction is forwarded before the preceding FP->int result landed | never round through the integer file: `t + 1.5*2^23 - 1.5*2^23` in FP, exponent bits via `fmv.x.w` + integer ops + `fmv.w.x` (`sh_exp2_clamped`, `sh_v4_exp2`) |
| 7 | three cores in a cluster are not faster than one on straight-line code; a 5-instruction loop runs 5.5x slower depending on its address | one `instr_router` (8 B/cycle) shared by the three cores, 32 B single-line prefetchers, no RVC: ~1 instruction/cycle for the whole cluster unless the loop fits one 32 B line | minimise instructions per element (fp16 SIMD `vfadd.h` ... = 4 lanes per instruction), `#pragma GCC optimize("align-loops=32")` in the library |
| 8 | an `fsd` by the FP subsystem followed by an integer `lw`/`lhu` of the same address (and `sw` followed by `fld`) reads stale data; data stored with `fsd` was not yet in TCDM when the DMA read it after the barrier | the integer core and its FP subsystem are two masters; neither program order nor the barrier CSR orders their memory accesses | `sh_v4_after_fsd` / `sh_v4_after_sw` / `sh_fp_fence` (`runtime/sh_simd.inc.c`): an `fmv.x.w` whose result the loads depend on drains the subsystem; a read-back feeding the `fld` address orders the other direction |
| 9 | core 0 hangs forever in a loop over a TCDM buffer at offset 0 | TCDM address 0 is `NULL` to GCC (and the SDK L1 allocator lives at 0x10) | the library never stages data below TCDM 0x1000 (`SH_ROWOPS_L1_BASE`) |
| 10 | software softmax / LN / GELU results depend on the *binary layout*: `siglip --cluster all` failed (P0 = softmax(-x), negative "probabilities", O off by 0.8) while `--cluster 0`, or the same binary plus an unrelated `printf`, passed; identical numbers whether the heads ran concurrently or serialized | Snitch core <-> FP-subsystem integer-register hazard: `fcvt.w.s`, `flt.s`, `feq.s`, `fmv.x.s`, `fclass.s` (and the fp16/fp8 scalar variants) lack the `nseq` tag that the `.d` versions have, so they are queued in the FPU sequence buffer; the integer core runs ahead, reads the destination register before the subsystem wrote it, and the late write-back clobbers newer values (insn trace: `addi a5,a5,127` / `slli` execute before `fcvt.w.s a5`, then `fmv.w.x fa4,a5` gets the stale `a5`). Only bites when the sequence buffer is non-empty, i.e. depends on cycle timing / code alignment | patch `gvsoc_snitch_fp_int_nseq.patch` (ISA tables: add `nseq` to every scalar FP instruction with an integer operand), regenerate `isa_snitch_rv32imfdva.cpp`, rebuild `gen_isa_snitch_rv32imfdva_cpp_*`; until it is installed, run with `SOFTHIER_MODEL_DIR=<dir with the patched .so>` (see below) |

| 11 | `Executing illegal instruction (pc: 0xc0059000 / 0x0 / 0x10 ...)` on `cluster_0/pe1` at a data address, late in a long program (the SmolVLA expert flow); the layer alone passes | not a model bug: the SDK start code (`flex_start.s`, `sll t0, a0, 0xa`) spaces the three harts' stacks **1 KB** apart in the 128 KB stack memory. The generated kernel function keeps one 64-bit address per HBM buffer (70+ spills) and core 0 then calls `sh_printf` (256 B buffer + vsnprintf) for marks / dumps: > 1 KB of stack, overrunning core 1's stack top and its saved return address (the pc is one of core 0's spilled buffer addresses) | `sh_call_on_core_stack` (`runtime/sh_rt.inc.c`): the generated `main` runs the inputs / kernel / checks phases on private per-core stacks of `SH_CORE_STACK_BYTES` (40 KB) carved from the top of the stack memory |

Findings from the TTT backward work (docs/TTT.md, 2026-10-09), not patched:

| # | Symptom | Root cause | Workaround |
|---|---|---|---|
| 12 | `Executing illegal instruction (pc: <memset / memcpy>, opcode: 0x...433d)` | the toolchain's newlib is built with the C extension; the cores do not execute compressed instructions. GCC calls `memset` / `memcpy` for struct zeroing (`= { 0 }`) and struct / compound-literal copies (always at -Os) | `runtime/sh_train.inc.c` defines plain-RV32 `memcpy` / `memset` (the linker takes them before libc's) |
| 13 | in-network REDADD returns sum + garbage, but only after a few reductions in a run (constant per-cluster data: first rounds exact) | `floonoc_router.cpp collective_generate` / `floonoc.cpp process_collective_operations`: the pulled data are added INTO the root request's data buffer, which for a read is one of the iDMA AXI back-end's static read-burst buffers (`idma_be_axi.cpp`: `ARCH_IDMA_OUTSTAND_BURST` = 256 x 4 KB, FIFO-reused, never cleared); every intermediate router's kid copies that stale content too | `sh_t_red_flush`: read 256 x 4 KB from the cluster's zero memory (same AXI back-end) right before the reduction so its bursts hold zeros |
| 14 | fp16 REDADD of 16 operands: rel-L2 1.0e-3 vs the exact sum (fp16 rounding of the exact sum: 2.1e-4) | `floonoc.cpp float_to_fp16` truncates the mantissa (and `fp16_to_float` reads subnormals as ~0), applied after every pairwise add | use it for gradients (their own error is ~1e-2); `sh_t_allreduce` mode 1 is exact (integer limbs) but core-bound |
| - | a library function built with `__attribute__((optimize("Os")))` fails to link: `undefined reference to bare_dma_wait_all` | the SDK's `bare_dma_*` are C99 `inline` without `static`: -O3 always inlined them, -Os emits calls | `extern inline` declarations in the unity build emit the external definitions |
| - | `--gc-sections` keeps `sh_gemm_mesh` (and what it calls) although nothing references it (4.5 KB of the 64 KB instruction memory) | not investigated | `SH_NO_GEMM_MESH` guard |

Findings from the fused attention kernel (`runtime/sh_attention.inc.c`, 2026-10-08), not patched:

| # | Symptom | Root cause | Workaround |
|---|---|---|---|
| 6 | `fcvt.w.s` followed by `fcvt.s.w` of its result (the usual `(float)(int)x` range reduction) returns garbage; an integer ALU consumer of the same result is correct | `snitch_fp_ss.cpp` snapshots the integer register file when an instruction is offloaded; an FP instruction that reads the integer result of a preceding FP->int move is offloaded before that result is written back | never feed an FP->int result straight back into the FPU: consume it with ALU ops first (`sh_attention_head` builds the `2^k` exponent field on the integer side) |
| 7 | the 3 cores of a cluster together run no faster than one: an ALU-only loop goes from 16 to 47 cycles/iteration when the other two cores run the same loop | no instruction cache: every core refetches each line over the shared `instr_router` (8 B/cycle) from `instr_mem`, ~2 cycles per instruction even for one core | instruction count is the cost metric on the cores (an int op is not free); use RedMulE / iDMA for everything that can be; the softmax in `sh_attention_head` is fetch-bound at ~60 cycles/element |

Measured costs on one Snitch core, data in TCDM (cycles per operation, `tests` micro-benchmarks 2026-10-08): dependent
TCDM load ~13 (non-volatile, pipelined: ~5), `flw` 3, `fsw` ~12, dependent `fadd`/`fmul`/`fmadd` 3-4, every FP->int
move (`fmv.x.w`, `fcvt.w.s`, `flt`+branch) ~15 round trip, `fcvt.s.h`/`fcvt.h.s` (Zfh conversions, unlike `flh`/`fsh`)
work and cost ~4, software `sh_fp16_to_f32` 40, `sh_f32_to_fp16` 72, `sh_expf` ~60. iDMA 2-D transfers with 2-byte
elements (an in-TCDM transpose: 64 descriptors of 256 x 2 B) work and take ~1 cycle per element.
## HBM preload (real weights without a host round trip)

`gvsoc ... run --preload <elf>` feeds an ELF to the chip's `hbm_preloader`
(`utils.loader.loader.ElfLoader`, wired in `flex_cluster.py`). At reset it walks the ELF's
`PT_LOAD` program headers (sections are ignored) and writes each segment to its `p_paddr` through
the data NoC at cluster (0,0), one 64 KB request at a time, then raises `hbm_preload_done` in
`ctrl_registers`; `has_preload_binary=1` makes the control registers hold back every global
barrier until that flag is set.

**The flag does not mean the data is in HBM.** The program's first mark comes ~25-35 us after
reset whether the image is 44 KB or 166 MB, while the data arrives at NoC link bandwidth after
that: on the ideal-HBM model a 15.4 MiB image is complete ~0.21 ms after the program starts
(~75 B/ns) and the 166 MB SmolVLA image after ~2.7 ms; segments land in file order. (The loader
issues 64 KB bursts into the FlooNoC network interface at cluster (0,0), which splits them into
64 B flits and lets the next burst start while flits are still in flight.) A program that reads a
late segment early gets zeros: this showed up as an all-zero `layer_norm1` output (gamma/beta
are among the last segments) that changed with how much the program printed before the first
layer. The fix is in the image + runtime, not the SDK: `make_preload_elf` writes segments in
offset order, the frontend places `softhier_mlir.sim.preload.sentinel_array()` (64 B = one
flit, so it lands atomically) at the highest offset, and `softhier.preload_wait %sentinel`
(`sh_preload_wait`) spins on it from cluster 0, waits 8 us for earlier chunks on longer mesh
paths and ends with a global barrier. It prints `[sh_preload_wait] image visible after <cycles>`.
`tests/gvsoc/run.py preload` covers the race (16 MB filler, last segment dumped first;
`--no-wait` reproduces the zeros).

`softhier_mlir/sim/preload.py` writes that ELF32 directly from `{hbm_offset: ndarray}`
(no toolchain; the SDK's `flex_cluster_utilities/preload.py` spells arrays out as C initialisers,
which does not scale past a few MB). Constraints: offsets >= 4 KB (the SDK's `flex_alloc_init`
keeps the HBM allocator state at `0xC0000000` and the first block header at `+0x400`), 64 B
aligned, non-overlapping. `run_sim(preload=path)` passes the flag.

## Test inputs come from the host, through the same preload path (2026-10-08)

Every gvsoc test starts from LCG matrices (`sh_test_fill_fp16`, host twin `softhier_mlir/testing/lcg.py`).
Generated on the device they are a scalar loop on one core: one SigLIP layer is ~7 M elements, ~200 ms
of simulated time and ~70 s of wall time for a 2.3 ms kernel. `tests/gvsoc/run.py` now generates the
same bytes on the host and puts them in HBM through the preload image (`--data preload`, the default;
`--data device` is the old path). Two mechanisms, one convention:

* Generated programs: `softhier-translate --preload-elf <file>` (`softhier_mlir/sim/testdata.py`) collects
  every top-level `softhier.hbm_fill_lcg` / `hbm_fill` / `hbm_fill_col_parity` into `{offset: array}`,
  writes the image (sentinel at the first 4 KB boundary above every declared buffer) and replaces the
  fills with `softhier.preload_wait` on the sentinel. Both are prologue ops (before the timer), so the
  ROI is untouched. A fill that cannot be preloaded (buffer below 0x1000, unaligned, indexed, through a
  view) leaves the whole module on the device path with a note on stderr: the `examples/*.mlir` all
  start at offset 0 and stay on the device (their constant fills are DMA-based and cheap anyway).
* Hand-written C tests (`gemm`, `gemm_seq`, `rowops`, `attention`, `siglip_layer`): `run.py` mirrors
  each test's HBM layout and fills in Python, writes `preload.elf` and `#define SH_PRELOAD <sentinel
  offset>` into `shape.h`; `main.c` then does `sh_preload_wait(sh_hbm_addr(SH_PRELOAD))` instead of its
  `#else` fill block. All layouts now start at `HBM_START = 0x1000` (the SDK allocator's region).

Wall time per run (`run_sim` wall, i.e. gvsoc only; machine shared with other simulations, load ~6-7),
same PASS/FAIL and the same max errors in every case:

| run | wall, inputs on the device | wall, inputs preloaded | ROI device -> preloaded | image |
|---|---|---|---|---|
| `gemm 1024x3072x768 ... all` | 32.3 s | **4.2 s** | 520521 -> 520539 ns | 12 MiB, 4 segments |
| `rowops 256x768 --cluster all` | 17.2 s | 15.9 s | 508504 -> 508473 ns (sum of 7 ROIs) | 0.8 MiB, 5 |
| `siglip-mlir --seq 256 --cluster all` | 94.4 s | **48.4 s** | 4782253 -> 4782093 ns | 13.9 MiB, 18 |
| `siglip --seq 256 --cluster all` | 90.4 s | **44.5 s** | 2276565 -> 2276815 ns | 13.9 MiB, 18 |
| `smolvla --layers 1` | 48.3 s | unchanged (it already preloads everything; nothing is generated on the device) | 5355905 ns, bit-identical | 15.4 MiB, 23 |

The ROI deltas (<= 250 ns on ms-scale ROIs, <= 0.01 %) are the run-to-run jitter of the timer start,
not the data path: the device path itself moves by ~40 ns between two runs of `rowops` or `attention`,
and `gemm --data device` with the new layout reproduces the old 520521 ns exactly (so the 18 ns in the
preload run is the loader's tail). `rowops` gains little because its inputs are small (0.4 M elements);
its wall time is the row ops and the sample printing. What remains in the SigLIP runs is the kernel
itself (4.8 ms / 2.3 ms simulated on 48 cores) plus gvsoc start-up and the dumps.

The host LCG is a leapfrog (4096 lanes advanced by the composed affine map in wrapping uint32), bit-equal
to the scalar loop (`tests/test_testdata.py`): 7 M elements in ~0.1 s instead of ~2 s. Limits seen so far:
none. Images are one `PT_LOAD` per array (18 segments, 14 MiB for a SigLIP layer; the 12-layer SmolVLA
image is ~200 segments, 166 MB), the loader streams them in ~0.2-2.7 ms of simulated time, and the first global
barrier + sentinel wait absorb that before the timer starts.

## Program size: 64 KB of instruction memory

`ARCH_INSTRUCTION_MEM_SIZE` is 0x10000. The program ELF is loaded from 0x80000000 and anything
past 64 KB lands in the 1-byte `debug_mem` (`Received out-of-bound request (reqAddr: 0x80010000
...)` at ~16 us, then nothing). The runtime is ~21 KB; one unrolled SigLIP layer (36 per-head
attention calls + 16 ops, 64-bit address arithmetic at every call site) is ~33 KB, so 12 unrolled
layers are ~140 KB. `softhier-translate` therefore lowers `scf.for` over `index` values to C loops
and `softhier.hbm_buffer %i {offset, stride}` / `softhier.view %src, %i {stride}` to
`offset + i * stride` addressing; `frontend/smolvla.py` emits one loop over the layers (per-layer
parameters are allocated at a constant stride) and one over the heads, which keeps the 12-layer
program at ~50 KB for any depth or sequence length. `tests/gvsoc/run.py smolvla` prints the
program size; `--unroll` is for 1-2 layers of debugging with the layer-1 intermediate dumps.

Host memory: `gvsoc_launcher` holds ~850 MB RSS for the full SmolVLA run (166 MB image, 16
clusters, seq 1024) and this host has 7 GB for every agent's simulations together; when the
kernel OOM-kills it the streamed log simply stops mid-line, `run_sim` returns `ok=False` with
`roi=None` and no error text (`dmesg | grep oom-kill` shows it). `run.py smolvla --from-log`
still evaluates every tensor the run got to.

Timing probe: `softhier.mark {tag}` prints `[mark] tag <mcycle>` from cluster 0 (clock 1 GHz,
so cycle deltas are ns; the counter is 32-bit and wraps every 4.29 s, `tests/gvsoc/run.py`
unwraps it). The ROI timer (`[Performance Counter]`) starts after `sh_init`, i.e. after the preload.
Other facts the library relies on:
- RedMulE convention: `flex_redmule_config(m, n, k)` computes `Y[m,k] += X[m,n] . W[n,k]`
  (the contraction is `n`). `sh_gemm` therefore calls `config(tm, tk, tn)`.
- The iDMA model does 2-D transfers HBM -> TCDM but not TCDM -> HBM: stores are per-row 1-D.
- Only one `IoReq` is in flight per FP subsystem; the 64 KB instruction memory is enough
  for the whole library (`.text` ~20 KB). Every cluster has its own copy of that memory
  (`.text`/`.rodata`/`.sdata`) and its own stack memory behind the cluster's `narrow_axi` router.
- Every core executes `main()`: never keep mutable program state in `.bss` (it is shared
  L3 memory); allocate HBM offsets on the stack or with constants.

## Row-wise / elementwise fp16 ops: scalar conversions -> fp16 SIMD (2026-10-08)

`sh_layernorm / sh_softmax_rows / sh_gelu / sh_add / sh_add_bias / sh_scale / sh_transpose`
(`runtime/sh_rowops.inc.c`, `runtime/sh_simd.inc.c`) were rewritten; `tests/gvsoc/run.py rowops`
(256 x 768 fp16, numpy reference, `--nsamples 1024`) and `run.py siglip --seq 256 --cluster 0`
pass. Baseline = commit 8597b42 (software fp16<->fp32 on the first core only).

What changed, in the order it mattered:
1. **Hardware conversions.** `lhu` + `fmv.w.x` (NaN-boxed) + `fcvt.s.h`, and `fcvt.h.s` + `fmv.x.w` + `sh`
   (`sh_h2f` / `sh_f2h` in `sh_ops.h`) are bit-exact against the software converters for all 65536
   halves and 200k floats (`run.py fp16cvt`) and ~5x faster per element. Independent chains must be
   interleaved by hand (`sh_h2f4` / `sh_f2h4`): the FPU model overlaps independent instructions
   (~2 cycles each) but serialises dependent ones (~6).
2. **All three cores, double-buffered DMA.** Each block is split over the cores (the DM core also
   computes); block i+1 streams in and block i-1 streams out during compute. On this model the
   three cores do *not* add throughput on straight-line code (#7): with everything on one core the
   ops took the same time. The split still matters for the fetch-free inner loops (ADD/SCALE/BIAS).
3. **fp16 SIMD (Xfvec).** `vfadd.h / vfmul.h / vfmac.h / vfmax.h / vfdiv.h / vfcpka.h.s ...` on four
   fp16 lanes of a 64-bit FP register, loaded with a plain `fld` (the broken `flh` is not involved),
   emitted with `.insn r` because the upstream assembler lacks the mnemonics. Softmax and GELU
   evaluate `2^t` on four lanes (`sh_v4_exp2`: round via `t + 1536`, degree-3 polynomial, `2^k` from
   the bits of the sum with two SWAR integer ops per lane pair); reductions keep fp16 partial sums
   of <= 8 terms per lane and fold into fp32. LayerNorm subtracts the mean as an fp16 high+low pair.
4. **32-byte loop alignment** (#7) and the **FP/integer ordering fences** (#8).
5. **Transpose on the iDMA**: one 2-D gather of 2-byte elements per column inside TCDM instead
   of a scalar loop (`SH_TRANSPOSE_CORES` restores the core loop).

Simulated time of `run.py rowops --rows 256 --cols 768` (196 608 elements per op, gvsoc at 1 GHz):

| op | cluster 0: before | after | speedup | all 16 clusters: before | after | speedup |
|---|---|---|---|---|---|---|
| layernorm | 49.16 ms | 2.02 ms | 24.3x | 3.727 ms | 0.149 ms | 25.1x |
| softmax | 60.37 ms | 2.14 ms | 28.3x | 3.841 ms | 0.145 ms | 26.5x |
| gelu | 28.73 ms | 1.81 ms | 15.9x | 1.816 ms | 0.123 ms | 14.8x |
| add | 17.45 ms | 0.145 ms | 120x | 1.123 ms | 0.022 ms | 50.7x |
| add_bias | 17.29 ms | 0.158 ms | 110x | 1.106 ms | 0.027 ms | 40.8x |
| scale | 14.00 ms | 0.128 ms | 109x | 0.967 ms | 0.017 ms | 57.5x |
| transpose | 3.27 ms | 0.231 ms | 14.2x | 0.208 ms | 0.025 ms | 8.2x |
| **all 7** | **190.3 ms** | **6.63 ms** | **28.7x** | **12.79 ms** | **0.508 ms** | **25.2x** |

Per element on one cluster the ops went from 71-307 cycles to 0.65-10.9 cycles. The SigLIP layer
(`siglip --seq 256 --cluster 0`, GEMMs on RedMulE unchanged) went from 700.5 ms to 22.8 ms (30.7x)
with every tensor within tolerance. Remaining cost is instruction fetch (#7): softmax/GELU are
~8-11 instructions per element (the `2^k` assembly and the FP<->int fences), LayerNorm ~10.

Open point (not needed by the library): why `flh` still reads 0 after the handler patch was not
investigated further; the library avoids scalar fp16 loads entirely (`fld` of four halves).

## Integer-register hazard between the Snitch core and its FP subsystem (#6, 2026-10-08)

How it was found (`tests/gvsoc/mesh_slices`, `run.py siglip --define ...`): every isolated multi-cluster
ingredient passed (12 clusters storing 64-column slices of one 256x768 matrix with GEMM / 1-D DMA /
scalar stores, the full scores-softmax-P.V sequence on 12 clusters, the follow-up 16-cluster GEMM), the
failing run gave bit-identical wrong numbers with a global barrier between the heads, and the P0
checksum did not change after the softmax wrote it. So neither the DMA, the NoC nor the HBM model
corrupts anything concurrently: the softmax itself produced `softmax(-0.125 x)` (row dump matched it to
3e-3, with tiny negative entries where the `exp` polynomial was evaluated far outside its range), and
the instruction trace of `cluster_0/fp_ss0` + `cluster_0/pe0` showed the integer core consuming the
`fcvt.w.s` result before the subsystem produced it. The decisive clue was that two binaries whose only
difference was dead code after the attention loop (`.sdata` shifted by 0x20) passed and failed
deterministically.

Timing effect of the fix: the core now stalls on every FP -> int transfer. `siglip --seq 256`:
`--cluster all` 52.14 -> 52.72 M ns (+1.1 %), `--cluster 0` 700.5 -> 718.2 M ns (+2.5 %, more software
row ops per cluster); RedMulE-only runs are unchanged. All 13 sampled tensors match numpy in both modes
(O/H/G/OUT maxerr 0.003-0.028).

Running against a model build that is not installed: `softhier_mlir.sim.gvsoc.run_sim` calls `gapy`
with `--work-dir=<ELF dir>` (private `gvsoc_config.json`; without it two concurrent runs from any
agents on this machine swap each other's ELF) and prepends every directory in `SOFTHIER_MODEL_DIR`
(colon separated) to the model search path, so a patched `.so` can be tested without touching
`/app/install/softhier/install/models`. The patched integer-core ISA model was built from
`build/core`'s compile/link lines with only the regenerated `isa_snitch_rv32imfdva.cpp` replaced (the
patch header lists the targets); the FP-subsystem decoder does not matter for this bug because the
sequencer decides on the core's instruction descriptor.

## Host speed of the RedMulE functional model (patch `gvsoc_redmule_neon.patch`, 2026-10-08)

`light_redmule.cpp` computes every 128 x 2048 x 128 tile (33.5 MMAC per `process_compute`)
with a scalar per-MAC `fp16 -> float -> fp16` round trip through hand-written bit converters:
~14-25 ns per MAC on the aarch64 host, i.e. 0.5-0.8 s per tile and ~60 s of pure arithmetic for a
1024x3072x768 GEMM. The patch replaces only the arithmetic helpers (address generation, DMA and
timing code are untouched, so every ROI stays bit-identical):

- `matmul_fp16`: AArch64 NEON `vfmaq_f16` (8 output columns per vector, 4x32 register block,
  accumulator kept in fp16 lanes, zero-padded lanes for a column tail). The host needs
  `fphp asimdhp` (checked at run time through `HWCAP`; the old scalar loop stays as fallback,
  `REDMULE_NO_NEON=1` forces it for A/B runs). No CMake change: the functions carry
  `__attribute__((target("arch=armv8.2-a+fp16")))`. The kernel saves FPCR, forces round-to-nearest
  with FZ/FZ16 clear and restores it: the Snitch ISS emulates RISC-V rounding modes with
  `fesetround()` and leaves the engine thread in RTZ/RDN/RUP, which the hand-written scalar
  converter ignored but hardware fp16 FMAs honour (first NEON build: `siglip` P.V came out ~4 % low
  and H/G/OUT failed).
- `matmul_{u,}int{8,16}`: accumulate 64 columns in a local `uint32_t` block so GCC vectorises;
  bit-identical (per-step wrap-around == one wrap at the end).
- `fp8e4m3_fma`: 256-entry table instead of `powf` per operand.

Numerics (fp16): the per-MAC rounding of the accumulator to fp16 is kept, but the FMA is now
*fused* (one RNE rounding of the exact `a*b+c`, as in FPnew) instead of double-rounded through
fp32. Checked against an exact long-double FMA with a single RNE rounding: the NEON result matched
it in 100 % of ~30k outputs; the old path differed in 0.01 % (`gemm --real` data) to 0.13 % (random
normal fp16 bits). On the integer-valued data of `run.py gemm` both are bit-identical. Subnormal,
inf and NaN now follow IEEE (`FZ16 = 0`); the old converter read a subnormal *accumulator* back as
~0 and inf/NaN as the finite `2^16 * 1.m`, so results differ only for data that leaves the fp16
normal range (|x| < 6.1e-5 or overflow).

| run (`tests/gvsoc/run.py`) | ROI (ns) scalar / NEON | wall scalar | wall NEON |
|---|---|---|---|
| kernel alone, 128x2048x128 tile, host | - | 14.35 ns/MAC (0.48 s) | 0.076 ns/MAC (2.6 ms), ~190x |
| `gemm 1024x3072x768:256,256,256,1,0,all` | 509224 / 509224 | 71.6 s | 36.9-40.7 s |
| `gemm 512x768x768:256,256,256` | 146466 / 146466 | 13.2-14.6 s | 10.0-10.4 s |
| `gemm` default set (7 shapes) + `--real` | all identical | - | all PASS |
| `siglip --seq 256 --cluster 0` (13 tensors) | 700522026 / 700514592 | 242 s | 228 s |

The GEMM ROIs are identical because the RedMulE timing code is untouched; the siglip ROI moves by
1e-5 because the software softmax/LN/GELU on the Snitch cores have data-dependent timing and now
see slightly different fp16 GEMM outputs (siglip accuracy: OUT maxerr 0.0283 -> 0.0205).
Host load from other jobs was 7-9 (10 cores) during all measurements. The remaining wall time is
the rest of the platform (Snitch ISS, iDMA/NoC/TCDM models), not RedMulE arithmetic: with
`REDMULE_NO_NEON=1` the 512x768x768 run takes 13.2 s, with NEON 10.4 s.

## Host speed of the Snitch ISS (patch `gvsoc_iss_host_speed.patch`, `install/models_fast/`, 2026-10-08)

With RedMulE vectorised, gdb sampling of `gvsoc_launcher` (60 stacks of the engine thread during
`siglip-mlir --cluster all`) put the host time of the platform in the Snitch core model, on the
path every offloaded FP instruction takes (integer core -> FPU sequencer -> FP subsystem -> response):

| share | where | why |
|---|---|---|
| 37 % | `iss_insn_s::operator=`, `std::string::_M_assign`, `OffloadReq::operator=` | `iss_insn_t` (~900 B, eight `std::string` arg names + a vector) copied by value up to 12x per FP instruction: `Iss::handle_req` 3x, `sequencer::req/gen_entry/write_entry/read_entry/offload_event` 6x, `snitch_fp_ss.cpp` `handle_notif/get_latency/handle_event/acc_rsp` 5x |
| 32 % | `sprintf`, `__vfprintf_internal`, `iss_trace_dump_insn` | `snitch_fp_ss.cpp` formatted the full instruction trace (`iss_trace_save_args` + `iss_trace_dump`) for every instruction and then dropped the string because `insn_trace` is inactive (in the optim build it cannot even be activated) |
| 5 % | `strstr` | up to 26 `strstr(label, ...)` per instruction: `int_offload_exec` on *every* integer instruction, `fp_offload_exec`, `handle_result`, `get_latency` |
| 26 % | `Exec::exec_instr`, `ClockEngine`, sequencer logic, FPU emulation | the model itself |

The patch (header lists every change; all marked `[SoftHier-fast]`) is behaviour-neutral: the arg
name becomes an interned `const char *` (only `csr_decode` sets it, only the trace reads it), the
label predicates are evaluated once per decoded instruction and cached in the instruction
(`iss_insn_label_class`), the instruction is copied once per stage (core -> request, request ->
ring buffer, ring buffer -> sequencer output, -> FP subsystem; the response carries a pointer), and
the FP-subsystem trace work is guarded by `trace.insn_trace.get_active()` exactly like the integer
core's `iss_exec_insn_with_trace`. After the patch the engine thread spends its time in the model
proper (`Exec::exec_instr`/prefetcher 25 %, sequencer 22 %, the four remaining `memcpy`s 21 %,
FP-subsystem offload 17 %, engine/NoC 10 %).

Build / install: the three libraries (`gen_isa_snitch_rv32imfdva_cpp_60335333`, `gen_isa_snitch_fp_ss_rv32imfdva_cpp_61161152`, `pulp/snitch/sequencer`) share the `iss_insn_t` / `OffloadReq` /
`OffloadRsp` layout and must come from one build. `tools/iss_fast/gen_makefile.py` writes a Makefile
that compiles a private copy of `core/models/cpu/iss` + `sequencer.cpp` (`/app/iss_fast`, patched with
the pending / zfh / nseq fixes and this patch, generated ISA tables from `install/models_fix` and
`build/core`) with the exact flags and object lists of the shared build tree, optim and debug, and
installs them under `install/models_fast/` (+ `debug/`). `softhier_mlir/sim/gvsoc.py` searches
`models_fast`, then `models_fix`, then `models`; `SOFTHIER_STOCK_MODELS=1` skips both patched
directories, `SOFTHIER_MODEL_DIR` still goes first.

Verification (`tests/gvsoc/run.py`, same ELFs, before = `models_fix` + `models`, after = `models_fast`;
host shared with other agents' simulations, load average 5-12 during "before" and 18-23 during "after",
so the wall times understate the gain; the CPU time of build+simulation is listed too):

| run (`tests/gvsoc/run.py`) | ROI ns before = after | wall before (load 5-12) | wall before, rerun (load 18-22) | wall after (load 19-23) | CPU s before / after (rerun, build+sim) |
|---|---|---|---|---|---|
| `gemm --shapes 256x256x256` | 14739 | 2.5 | 4.5 | 3.6 | 56.7 / 33.6 (both shapes) |
| `gemm 1024x3072x768:256,256,256,1,0,all` | 520521 | 30.0 | 47.6 | 14.0 | " |
| `rowops --rows 256 --cols 768 --cluster all` | 508504 | 17.4 | 16.8 | 10.2 | 25.4 / 17.4 |
| `siglip-mlir --seq 256 --cluster all` | 4782253 | 170.1 | 106.9 | 28.1 | 112.1 / 42.6 |
| `attention --cluster 0` | 26819562 | 43.5 | 56.7 | 8.1 | 58.0 / 20.2 |
| `mlir examples/*.mlir`: 8 small examples | all identical | 17.3 (sum) | 20.7 | 22.9 | 458.8 / 214.0 (all 9) |
| `mlir examples/siglip_encoder_layer.mlir` | 17851422 | 447.5 | 424.7 | 174.5 | " |

Every ROI (and every sampled tensor / checksum) is identical; the Snitch-bound runs are 2.4-7x
faster in wall time and 1.7-2.9x in CPU time (the CPU time includes the unchanged ELF build in the
x86 chroot, ~10-15 s per case, and the RedMulE/NoC/memory models). The GEMM runs profit from the
integer-core part of the change (`int_offload_exec` ran four `strstr` per integer instruction).

Traced debug run (`gemm 256x256x256` with `--trace=cluster_0/pe0/insn --trace=cluster_0/fp_ss0/insn`,
i.e. the debug libraries, previous set vs `models_fast/debug`): the first 300 MB of simulator output
are byte-identical (1 448 009 lines, 105 516 FP-subsystem instruction dumps with register values,
CSR names such as `misa`/`mhartid` in the integer-core dumps). Do not ask the wrapper for more traces
than that on this host: `fpu_sequencer0/trace` is LEVEL_TRACE spam and a five-trace run of the same
ELF wrote 45 GB before it was stopped.

Two caveats for anyone measuring the host speed again: a `gdb -p` sampling loop stops the simulator
for ~1 s per attach (a sampled `attention` run took 41 s, unsampled 8 s), and the host was at times
out of memory (OOM kills of other agents' `gvsoc_launcher`s in `dmesg`); one sampled `siglip-mlir`
run under those conditions ended without a ROI and passed on every unsampled repeat.
