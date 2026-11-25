# Docker image: softhier-mlir + SoftHier GVSoC

One image with the compiler, the simulator and the RISC-V toolchain, so the whole flow
(`.mlir` -> C -> ELF -> gvsoc) runs with no other setup. 3.8 GB, builds in about 15 minutes.

```bash
docker compose -f docker/compose.yaml build                     # or: docker build -t softhier-mlir:dev docker/
docker compose -f docker/compose.yaml run --rm softhier         # shell in /app/softhier-mlir (this checkout, mounted)
smoke                                                           # FileCheck + GEMMs + two examples on gvsoc, ~1 min
smoke full                                                      # + every example + one SigLIP layer, ~3 min
```

## What is inside

| path | content |
|---|---|
| `/app/softhier-mlir` | this repo, `pip install -e` into `/opt/venv` (the compose file mounts the checkout over it) |
| `/app/install/softhier` | `gvsoc/gvsoc`, branch `bowwang-dev/softhier-deeploy`, built for `pulp.chips.flex_cluster.flex_cluster` |
| `.../third_party/toolchain/install` | prebuilt `riscv32-unknown-elf-gcc` 14.2 (rv32 + V + Zfh), C only |

Deeploy itself is not installed. The simulator is set up the way `pulp-platform/Deeploy`'s
`make softhier` does it, because `runtime/` is written against that SDK variant: the linker
script `flex_memory_deeploy.ld` and the headers in `deeploy_include/` (`flex_alloc_api.h`,
`flex_alloc_init`, `flex_hbm_malloc`).

Three things differ from the Deeploy container:

- **Model fixes.** `patches/` is applied before the build (root causes in `docs/SIMULATOR_NOTES.md`):
  ideal HBM switch, RedMulE fp16 round-to-nearest-even, RedMulE NEON kernel, Snitch FP-subsystem
  pending fix, Zfh `flh`/`fsh`, `nseq` tags for FP instructions with an integer operand, ISS host speed.
  They are compiled into the stock model directory, so there is no `models_fix` / `models_fast`.
- **Architecture.** The build ends with `apply_arch(Arch())`: the defaults of
  `softhier_mlir.sim.gvsoc.Arch` (4x4 clusters, HBM on the west and south edges, 64 MiB per node).
  `arch_deeploy.py` only has HBM on the west edge and gvsoc segfaults in `hbm_ctrl` on the first
  `hbm_south` access.
- **Toolchain without a chroot.** The toolchain is x86-64 only. On an arm64 host the image carries
  the x86-64 loader and glibc and the compiler runs through Docker's binfmt emulation
  (`SOFTHIER_CHROOT=""`); gvsoc is native. On an amd64 host everything is native.

The HBM is gvsoc's ideal memory (`SOFTHIER_IDEAL_HBM=1`) on every host: DRAMSys ships as an
x86-64 `.so`, and the numbers in this repo were measured this way.

## Not included

- torch / transformers / lerobot and the SmolVLA checkpoint. `frontend.smolvla prepare*` needs them;
  mount the checkpoint at `/app/models` and `pip install torch transformers` in the container.
- The C++ and LTO front ends of the toolchain and gdb (1.9 GB).
