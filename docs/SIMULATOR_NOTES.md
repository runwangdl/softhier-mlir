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

Other facts the library relies on:
- RedMulE convention: `flex_redmule_config(m, n, k)` computes `Y[m,k] += X[m,n] . W[n,k]`
  (the contraction is `n`). `sh_gemm` therefore calls `config(tm, tk, tn)`.
- The iDMA model does 2-D transfers HBM -> TCDM but not TCDM -> HBM: stores are per-row 1-D.
- Only one `IoReq` is in flight per FP subsystem; the 64 KB instruction memory is enough
  for the whole library (`.text` ~20 KB).
- Every core executes `main()`: never keep mutable program state in `.bss` (it is shared
  L3 memory); allocate HBM offsets on the stack or with constants.

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
