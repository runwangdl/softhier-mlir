"""Build an HBM preload image for the SoftHier GVSoC (`gvsoc ... run --preload <elf>`).

The flex_cluster chip has an `hbm_preloader` (utils.loader.loader.ElfLoader) that, at reset,
writes every PT_LOAD segment of the given ELF through the data NoC (cluster (0,0) input) to the
segment's physical address, 64 KB per request, and then raises `hbm_preload_done` in the control
registers; the first global barrier of the program blocks until that flag is set. That flag
only means the loader has *issued* its last 64 KB request: the data still crosses the NoC at
link bandwidth (~75 B/ns measured on the ideal-HBM model, i.e. ~13 ms per GB) and the last
segments land long after the program has started. A program must therefore wait for a sentinel
written as the LAST segment (`sentinel_array()` placed at the highest offset; the ELF is written
in offset order) with `softhier.preload_wait` / `sh_preload_wait` before touching preloaded data.
The SDK's own helper
(soft_hier/flex_cluster_utilities/preload.py) spells the arrays out as C initialisers and
compiles them, which does not scale to 100+ MB of weights; this module writes the ELF32
directly -- the loader only reads the program headers, so no toolchain is involved.

HBM layout rules (flex_cluster_sdk):
  * HBM base is 0xC0000000 (`hbm_addr(off)` = base + off); west nodes 0..3 are the first 256 MB.
  * `flex_alloc_init` keeps the HBM allocator state in the first 1 KB and the first block header
    at +0x400, so preloaded data must start at >= 4 KB (this module enforces it).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HBM_BASE = 0xC0000000
MIN_OFFSET = 0x1000
_ALIGN = 64

SENTINEL_MAGIC = 0x5EEDC0DE   # must match SH_PRELOAD_MAGIC in runtime/sh_rt.inc.c
SENTINEL_WORDS = 16           # 64 B = one 512-bit NoC flit, so it lands atomically

_EM_RISCV = 243
_PT_LOAD = 1
_PF_R, _PF_W = 4, 2


def make_preload_elf(path: str | Path, arrays: dict[int, np.ndarray], hbm_base: int = HBM_BASE) -> Path:
    """Write an ELF32 (little-endian, RISC-V) whose PT_LOAD segments place each array at
    ``hbm_base + offset``. ``arrays`` maps HBM byte offsets to numpy arrays (any dtype; fp16
    tensors are written as their raw little-endian bits, row-major). Returns the path."""
    items = sorted((int(off), np.ascontiguousarray(a)) for off, a in arrays.items())
    prev_end = 0
    for off, a in items:
        if off < MIN_OFFSET:
            raise ValueError(f"offset 0x{off:x} overlaps the SDK's HBM allocator metadata (< 0x{MIN_OFFSET:x})")
        if off % _ALIGN:
            raise ValueError(f"offset 0x{off:x} is not {_ALIGN}-byte aligned")
        if off < prev_end:
            raise ValueError(f"array at 0x{off:x} overlaps the previous one (ends at 0x{prev_end:x})")
        prev_end = off + a.nbytes
    ehdr_size, phdr_size = 52, 32
    data_start = ehdr_size + phdr_size * len(items)
    data_start = (data_start + _ALIGN - 1) & ~(_ALIGN - 1)
    phdrs, file_off = [], data_start
    for off, a in items:
        phdrs.append((file_off, hbm_base + off, a.nbytes))
        file_off += (a.nbytes + _ALIGN - 1) & ~(_ALIGN - 1)
    path = Path(path)
    with open(path, "wb") as f:
        ident = b"\x7fELF" + bytes([1, 1, 1, 0]) + bytes(8)        # ELFCLASS32, little-endian, version 1
        f.write(ident + struct.pack("<HHIIIIIHHHHHH", 2, _EM_RISCV, 1, hbm_base, ehdr_size, 0, 0,
                                    ehdr_size, phdr_size, len(items), 0, 0, 0))
        for fo, paddr, size in phdrs:
            f.write(struct.pack("<IIIIIIII", _PT_LOAD, fo, paddr, paddr, size, size, _PF_R | _PF_W, _ALIGN))
        for (fo, _, _), (_, a) in zip(phdrs, items):
            f.seek(fo)
            f.write(a.tobytes())
        f.truncate(file_off)
    return path


def sentinel_array() -> np.ndarray:
    """The 64 B end-of-preload marker as a [1, 32] fp16 array (raw bits: 16 x SENTINEL_MAGIC), to be
    placed at the highest preloaded offset; `sh_preload_wait(addr)` spins until it is visible."""
    return np.full(SENTINEL_WORDS, SENTINEL_MAGIC, dtype=np.uint32).view(np.float16).reshape(1, SENTINEL_WORDS * 2)


def read_preload_elf(path: str | Path, hbm_base: int = HBM_BASE) -> dict[int, bytes]:
    """Inverse of make_preload_elf (for tests): {offset: raw bytes} of every PT_LOAD segment."""
    data = Path(path).read_bytes()
    e_phoff, e_phnum = struct.unpack_from("<I", data, 28)[0], struct.unpack_from("<H", data, 44)[0]
    out = {}
    for i in range(e_phnum):
        p_type, p_offset, _, p_paddr, p_filesz = struct.unpack_from("<IIIII", data, e_phoff + 32 * i)
        if p_type == _PT_LOAD:
            out[p_paddr - hbm_base] = data[p_offset:p_offset + p_filesz]
    return out
