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

Other facts the library relies on:
- RedMulE convention: `flex_redmule_config(m, n, k)` computes `Y[m,k] += X[m,n] . W[n,k]`
  (the contraction is `n`). `sh_gemm` therefore calls `config(tm, tk, tn)`.
- The iDMA model does 2-D transfers HBM -> TCDM but not TCDM -> HBM: stores are per-row 1-D.
- Only one `IoReq` is in flight per FP subsystem; the 64 KB instruction memory is enough
  for the whole library (`.text` ~20 KB).
- Every core executes `main()`: never keep mutable program state in `.bss` (it is shared
  L3 memory); allocate HBM offsets on the stack or with constants.
