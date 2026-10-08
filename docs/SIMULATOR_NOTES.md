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

## HBM preload (real weights without a host round trip)

`gvsoc ... run --preload <elf>` feeds an ELF to the chip's `hbm_preloader`
(`utils.loader.loader.ElfLoader`, wired in `flex_cluster.py`). At reset it walks the ELF's
`PT_LOAD` program headers (sections are ignored) and writes each segment to its `p_paddr` through
the data NoC at cluster (0,0), 64 KB per request, then raises `hbm_preload_done` in
`ctrl_registers`; `has_preload_binary=1` makes the control registers hold back every global
barrier until that flag is set, so the first barrier of `sh_init` is where the program waits for
the data. Verified on the ideal-HBM model (`tests/gvsoc/run.py preload`): fp16 matrices placed at
4 KB, 70 MB and 200 MB (three different HBM nodes) read back bit-exact; a 165 MiB image
(the SmolVLA vision tower) loads in a few simulated ms.

`softhier_mlir/sim/preload.py` writes that ELF32 directly from `{hbm_offset: ndarray}`
(no toolchain; the SDK's `flex_cluster_utilities/preload.py` spells arrays out as C initialisers,
which does not scale past a few MB). Constraints: offsets >= 4 KB (the SDK's `flex_alloc_init`
keeps the HBM allocator state at `0xC0000000` and the first block header at `+0x400`), 64 B
aligned, non-overlapping. `run_sim(preload=path)` passes the flag.

Timing probe: `softhier.mark {tag}` prints `[mark] tag <mcycle>` from cluster 0 (clock 1 GHz,
so cycle deltas are ns; the counter is 32-bit and wraps every 4.29 s, `tests/gvsoc/run.py`
unwraps it). The ROI timer (`[Performance Counter]`) starts after `sh_init`, i.e. after the preload.

Other facts the library relies on:
- RedMulE convention: `flex_redmule_config(m, n, k)` computes `Y[m,k] += X[m,n] . W[n,k]`
  (the contraction is `n`). `sh_gemm` therefore calls `config(tm, tk, tn)`.
- The iDMA model does 2-D transfers HBM -> TCDM but not TCDM -> HBM: stores are per-row 1-D.
- Only one `IoReq` is in flight per FP subsystem; the 64 KB instruction memory is enough
  for the whole library (`.text` ~20 KB).
- Every core executes `main()`: never keep mutable program state in `.bss` (it is shared
  L3 memory); allocate HBM offsets on the stack or with constants.
