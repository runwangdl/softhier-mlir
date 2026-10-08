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
| 6 | software softmax / LN / GELU results depend on the *binary layout*: `siglip --cluster all` failed (P0 = softmax(-x), negative "probabilities", O off by 0.8) while `--cluster 0`, or the same binary plus an unrelated `printf`, passed; identical numbers whether the heads ran concurrently or serialized | Snitch core <-> FP-subsystem integer-register hazard: `fcvt.w.s`, `flt.s`, `feq.s`, `fmv.x.s`, `fclass.s` (and the fp16/fp8 scalar variants) lack the `nseq` tag that the `.d` versions have, so they are queued in the FPU sequence buffer; the integer core runs ahead, reads the destination register before the subsystem wrote it, and the late write-back clobbers newer values (insn trace: `addi a5,a5,127` / `slli` execute before `fcvt.w.s a5`, then `fmv.w.x fa4,a5` gets the stale `a5`). Only bites when the sequence buffer is non-empty, i.e. depends on cycle timing / code alignment | patch `gvsoc_snitch_fp_int_nseq.patch` (ISA tables: add `nseq` to every scalar FP instruction with an integer operand), regenerate `isa_snitch_rv32imfdva.cpp`, rebuild `gen_isa_snitch_rv32imfdva_cpp_*`; until it is installed, run with `SOFTHIER_MODEL_DIR=<dir with the patched .so>` (see below) |

Other facts the library relies on:
- RedMulE convention: `flex_redmule_config(m, n, k)` computes `Y[m,k] += X[m,n] . W[n,k]`
  (the contraction is `n`). `sh_gemm` therefore calls `config(tm, tk, tn)`.
- The iDMA model does 2-D transfers HBM -> TCDM but not TCDM -> HBM: stores are per-row 1-D.
- Only one `IoReq` is in flight per FP subsystem; the 64 KB instruction memory is enough
  for the whole library (`.text` ~20 KB). Every cluster has its own copy of that memory
  (`.text`/`.rodata`/`.sdata`) and its own stack memory behind the cluster's `narrow_axi` router.
- Every core executes `main()`: never keep mutable program state in `.bss` (it is shared
  L3 memory); allocate HBM offsets on the stack or with constants.

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
