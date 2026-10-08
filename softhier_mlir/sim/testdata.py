"""Test inputs from the host instead of the device: turn a module's fill ops into an HBM preload image.

A test module declares its inputs with `softhier.hbm_fill_lcg` / `hbm_fill` / `hbm_fill_col_parity`.
On the device those are scalar loops on one core: one SigLIP layer (~7 M fp16 elements) spends
~200 ms of simulated time (~70 s of host wall time) generating data for a 2.3 ms kernel.
`extract_preload` generates the very same bytes on the host (`softhier_mlir.testing.lcg`, the twin
of runtime/sh_test.inc.c), removes the fill ops from the module and inserts the end-of-image
sentinel buffer + `softhier.preload_wait` in their place, so the program only waits for the loader
(`run_sim(preload=make_preload_elf(...))`). The emitted program is otherwise identical: fills and
the wait are both prologue ops in the backend (before the timer), so ROIs do not move.

Falls back (returns None, module untouched) when a fill cannot be preloaded: a buffer below the
SDK allocator's first 4 KB, not 64-byte aligned, indexed (inside a loop) or filled through a view.
"""
from __future__ import annotations

import sys

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import Float16Type, IntegerAttr, MemRefType, ModuleOp, StringAttr, i32

from softhier_mlir.dialects.softhier import (
    HbmBufferOp,
    HbmFillColParityOp,
    HbmFillLcgOp,
    HbmFillOp,
    PreloadWaitOp,
)
from softhier_mlir.sim.preload import MIN_OFFSET, make_preload_elf, sentinel_array
from softhier_mlir.testing import lcg

FILL_OPS = (HbmFillLcgOp, HbmFillOp, HbmFillColParityOp)
_ALIGN = 64


def _fill_array(op, rows: int, cols: int) -> np.ndarray:
    if isinstance(op, HbmFillLcgOp):
        return lcg.fill_fp16(rows, cols, op.seed.value.data, op.lo.value.data, op.hi.value.data, op.scale.value.data)
    if isinstance(op, HbmFillOp):
        return lcg.const_fp16(rows, cols, op.value_bits.value.data)
    return lcg.col_parity_fp16(rows, cols, op.even_bits.value.data, op.odd_bits.value.data)


def extract_preload(module: ModuleOp, verbose: bool = True) -> dict[int, np.ndarray] | None:
    """Move every top-level fill op of the module's function into a preload dict {hbm_offset: array}
    (sentinel included, at the first 4 KB boundary above every declared buffer) and rewrite the
    module to wait for it instead. Returns {} when the module has no fill ops (nothing changed) and
    None when some fill cannot be preloaded (nothing changed; a note goes to stderr)."""
    fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    top = list(fn.body.block.ops)
    fills = [op for op in top if isinstance(op, FILL_OPS)]
    if not fills:
        return {}
    arrays: dict[int, np.ndarray] = {}
    end = 0
    for op in top:
        if isinstance(op, HbmBufferOp) and op.index is None:
            mt = op.result.type
            n = 1
            for d in mt.get_shape():
                n *= d
            end = max(end, op.offset.value.data + n * 2)
    for op in fills:
        src = op.buf.owner
        why = None
        if not isinstance(src, HbmBufferOp):
            why = "filled through a view"
        elif src.index is not None:
            why = "an indexed (in-loop) buffer"
        else:
            off, mt = src.offset.value.data, src.result.type
            if off < MIN_OFFSET:
                why = f"offset 0x{off:x} is inside the HBM allocator's first 0x{MIN_OFFSET:x} bytes"
            elif off % _ALIGN:
                why = f"offset 0x{off:x} is not {_ALIGN}-byte aligned"
            elif not isinstance(mt.element_type, Float16Type):
                why = f"element type {mt.element_type} is not f16"
            elif off in arrays:
                why = "filled twice"
        if why:
            if verbose:
                print(f"[testdata] {op.name} on %{src.results[0].name_hint or '?'}: {why}; test inputs stay on the device",
                      file=sys.stderr)
            return None
        shape = mt.get_shape()
        rows, cols = (1, shape[0]) if len(shape) == 1 else (shape[0], shape[1])
        arrays[off] = _fill_array(op, rows, cols)
    sent_off = (max(end, max(o + a.nbytes for o, a in arrays.items())) + 0xFFF) & ~0xFFF
    sent = sentinel_array()
    arrays[sent_off] = sent
    # rewrite: sentinel buffer + wait where the first fill was; drop the fills
    sent_t = MemRefType(Float16Type(), list(sent.shape), memory_space=StringAttr("hbm_west"))
    buf = HbmBufferOp(operands=[[]], properties={"offset": IntegerAttr(sent_off, i32)}, result_types=[sent_t])
    wait = PreloadWaitOp(operands=[buf.result])
    fn.body.block.insert_ops_before([buf, wait], fills[0])
    for op in fills:
        op.detach()
        op.erase()
    if verbose:
        tot = sum(a.nbytes for a in arrays.values())
        print(f"[testdata] {len(fills)} fill ops -> preload image: {len(arrays)} segments, {tot / 2**20:.1f} MiB, "
              f"sentinel at 0x{sent_off:x}", file=sys.stderr)
    return arrays


def write_preload(path, arrays: dict[int, np.ndarray]):
    """make_preload_elf with the dict extract_preload returned (kept here so callers import one module)."""
    return make_preload_elf(path, arrays)
