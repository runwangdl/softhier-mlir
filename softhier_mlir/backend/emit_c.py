"""SoftHier dialect -> C backend, targeting the softhier-ops operator library.

The generated ``main.c`` includes only ``sh_ops.h`` (see ``runtime/``) and calls one
library function per ``softhier`` op. Policy (tiling, pipelining, cluster mapping) is
carried as op attributes and forwarded to the library through ``sh_gemm_cfg``; the
mechanics (DMA, RedMulE, barriers) live in the library, where they can be measured and
swapped without touching the compiler.

Every library op is SPMD-safe: generated code runs on all cores of all clusters, flat,
with no cluster guards.
"""

from __future__ import annotations

from xdsl.dialects import func
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, MemRefType, ModuleOp, StridedLayoutAttr

from softhier_mlir.dialects.softhier import (
    AddBiasOp,
    AddOp,
    AttentionOp,
    CheckConstOp,
    DumpSamplesOp,
    GeluOp,
    HbmFillLcgOp,
    LayerNormOp,
    SoftmaxOp,
    TransposeOp,
    ViewOp,
    Dma2DOp,
    GemmOp,
    GroupBarrierOp,
    HbmBufferOp,
    HbmCheckConstOp,
    HbmFillColParityOp,
    HbmFillOp,
    L1AddOp,
    L1BufferOp,
    L1FillOp,
    L1ZeroOp,
    RedmuleOp,
    ReluOp,
)

_ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i16": 2, "i32": 4}
_FMT = {"fp16": "SH_FP16", "fp8": "SH_FP8", "int16": "SH_INT16", "int8": "SH_INT8"}


def _elem_bytes(memref: MemRefType) -> int:
    return _ELEM_BYTES.get(str(memref.element_type), 2)


def _nelem(memref: MemRefType) -> int:
    n = 1
    for d in memref.get_shape():
        n *= d
    return n


def _bytes(memref: MemRefType) -> int:
    return _nelem(memref) * _elem_bytes(memref)


def _space(memref: MemRefType) -> str:
    return getattr(memref.memory_space, "data", "tcdm")


def _int_attr(op, name: str, default: int) -> int:
    a = op.attributes.get(name)
    return a.value.data if isinstance(a, IntegerAttr) else default


def _cluster(op) -> str:
    """The `cluster` attribute: -1 -> SH_ALL (split over all clusters), absent -> 0."""
    c = _int_attr(op, "cluster", 0)
    return "SH_ALL" if c < 0 else str(c)


def _f(x: FloatAttr | float) -> str:
    v = x.value.data if isinstance(x, FloatAttr) else float(x)
    return f"{v!r}f"


class _Buffers:
    """Resolves each SSA buffer value to (space, byte-offset, size, memref, c-name)."""

    def __init__(self) -> None:
        self.info: dict = {}
        self._l1_cursor = 0
        self._n = 0

    def _name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def add_hbm(self, op: HbmBufferOp) -> None:
        mt = op.result.type
        self.info[op.result] = ("hbm", op.offset.value.data, _bytes(mt), mt, self._name("hb"))

    def add_l1(self, op: L1BufferOp) -> None:
        mt = op.result.type
        off = self._l1_cursor
        self._l1_cursor += _bytes(mt)
        self.info[op.result] = ("tcdm", off, _bytes(mt), mt, self._name("l1b"))

    def add_view(self, op: ViewOp) -> None:
        """A view shares its source's name; the strided layout gives offset/ld."""
        mt = op.result.type
        space, off, _, _, name = self.info[op.src]
        self.info[op.result] = (space, off, _bytes(mt), mt, name)

    # ---- HBM tensor geometry (rows, cols, ld, elem offset) -------------------------
    def geom(self, val) -> tuple[int, int, int, int]:
        mt = self.memref(val)
        shape = mt.get_shape()
        rows, cols = (1, shape[0]) if len(shape) == 1 else (shape[0], shape[1])
        if isinstance(mt.layout, StridedLayoutAttr):
            ld = mt.layout.get_strides()[0] if len(shape) > 1 else cols
            eoff = mt.layout.get_offset() or 0
        else:
            ld, eoff = cols, 0
        return rows, cols, ld, eoff

    def haddr(self, val) -> str:
        """64-bit HBM address expression of a (possibly viewed) buffer."""
        _, _, _, eoff = self.geom(val)
        return self.name(val) if eoff == 0 else f"({self.name(val)} + {eoff * 2})"

    def name(self, val) -> str:
        return self.info[val][4]

    def addr(self, val) -> str:
        """A 64-bit address expression usable by sh_dma_copy."""
        space, off, _, _, name = self.info[val]
        return name if space == "hbm" else f"(uint64_t)sh_l1_addr({off})"

    def raw_off(self, val) -> int:
        return self.info[val][1]

    def space(self, val) -> str:
        return self.info[val][0]

    def size(self, val) -> int:
        return self.info[val][2]

    def memref(self, val) -> MemRefType:
        return self.info[val][3]

    def decls(self) -> list[str]:
        out, seen = [], set()
        for space, off, size, mt, name in self.info.values():
            if name in seen:
                continue
            seen.add(name)
            shape = "x".join(str(d) for d in mt.get_shape())
            if space == "hbm":
                out.append(f"    const uint64_t {name} = sh_hbm_addr({off});  // {shape}x{mt.element_type} HBM, {size} B")
            else:
                out.append(f"    const uint32_t {name} = {off};  // {shape}x{mt.element_type} TCDM offset, {size} B")
        return out


def _gemm_cfg(op: GemmOp) -> str:
    fmt = _FMT.get(op.fmt.data, "SH_FP16")
    return (f"{{ .tm = {_int_attr(op, 'tile_m', 0)}, .tn = {_int_attr(op, 'tile_n', 0)}, "
            f".tk = {_int_attr(op, 'tile_k', 0)}, .pipeline = {1 if 'pipeline' in op.attributes else 0}, "
            f".accumulate = {1 if 'accumulate' in op.attributes else 0}, .fmt = {fmt}, .l1_base = 0 }}")


_PROLOGUE_OPS = (HbmFillOp, HbmFillColParityOp, HbmFillLcgOp)   # test inputs: before the timer
_EPILOGUE_OPS = (DumpSamplesOp, HbmCheckConstOp)                  # test outputs: after the timer


def emit_kernel(fn: func.FuncOp, bufs: _Buffers, phase: str = "kernel") -> str:
    """Emit one of three phases of the function: the test-input fills ("prologue", run before the
    timer starts, followed by a global barrier), the computation ("kernel", timed), or the test
    checks/dumps ("epilogue", after the timer). Op order is preserved inside each phase."""
    body: list[str] = bufs.decls()
    b = body.append
    tag = fn.sym_name.data.upper()

    for op in fn.body.block.ops:
        if isinstance(op, (HbmBufferOp, L1BufferOp, ViewOp, func.ReturnOp)):
            continue
        op_phase = "prologue" if isinstance(op, _PROLOGUE_OPS) else "epilogue" if isinstance(op, _EPILOGUE_OPS) else "kernel"
        if op_phase != phase:
            continue

        if isinstance(op, Dma2DOp):
            b(f"    sh_dma_copy({bufs.addr(op.dst)}, {bufs.addr(op.src)}, {bufs.size(op.src)});")

        elif isinstance(op, RedmuleOp):
            m, k = bufs.memref(op.x).get_shape()
            _k2, n = bufs.memref(op.w).get_shape()
            fmt = _FMT.get(op.fmt.data, "SH_FP16")
            b(f"    sh_redmule({bufs.name(op.x)}, {bufs.name(op.w)}, {bufs.name(op.y)}, {m}, {n}, {k}, {fmt});")

        elif isinstance(op, L1ZeroOp):
            b(f"    sh_l1_zero({bufs.name(op.buf)}, {bufs.size(op.buf)});")

        elif isinstance(op, L1FillOp):
            b(f"    sh_l1_fill_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))}, {op.value_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, ReluOp):
            b(f"    sh_l1_relu_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))});")

        elif isinstance(op, L1AddOp):
            b(f"    sh_l1_add_fp16({bufs.name(op.dst)}, {bufs.name(op.src)}, {_nelem(bufs.memref(op.dst))});")

        elif isinstance(op, CheckConstOp):
            b(f"    sh_test_check_const_l1_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))}, "
              f"{op.value_bits.value.data & 0xFFFF}u, {op.tol.value.data}, \"{tag}\");")

        elif isinstance(op, GemmOp):
            m, k, ldx, _ = bufs.geom(op.x)
            _k2, n, ldw, _ = bufs.geom(op.w)
            _m2, _n2, ldz, _ = bufs.geom(op.z)
            cfg = _gemm_cfg(op)
            args = f"{bufs.haddr(op.x)}, {bufs.haddr(op.w)}, {bufs.haddr(op.z)}, {m}, {n}, {k}, {ldx}, {ldw}, {ldz}, &cfg"
            if "summa" in op.attributes:
                b(f"    {{ sh_gemm_cfg cfg = {cfg};  // mesh-wide SUMMA")
                b(f"      sh_gemm_mesh({args}); }}")
            else:
                b(f"    {{ sh_gemm_cfg cfg = {cfg};")
                b(f"      sh_gemm({args}, {_cluster(op)}); }}")

        elif isinstance(op, HbmFillOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"    sh_test_fill_const_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, {op.value_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, HbmFillColParityOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"    sh_test_fill_colparity_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, "
              f"{op.even_bits.value.data & 0xFFFF}u, {op.odd_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, HbmCheckConstOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"    sh_test_check_const_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, "
              f"{op.value_bits.value.data & 0xFFFF}u, {op.tol.value.data}, \"{tag}\");")

        elif isinstance(op, LayerNormOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"    sh_layernorm({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {bufs.haddr(op.gamma)}, {bufs.haddr(op.beta)}, "
              f"{rows}, {cols}, {ld}, {_f(op.eps)}, {_cluster(op)});")

        elif isinstance(op, SoftmaxOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"    sh_softmax_rows({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {rows}, {cols}, {ld}, {_f(op.scale)}, {_cluster(op)});")

        elif isinstance(op, GeluOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"    sh_gelu({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AddOp):
            rows, cols, ld, _ = bufs.geom(op.a)
            b(f"    sh_add({bufs.haddr(op.y)}, {bufs.haddr(op.a)}, {bufs.haddr(op.b)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AddBiasOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"    sh_add_bias({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {bufs.haddr(op.bias)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AttentionOp):
            S, D, ldq, _ = bufs.geom(op.q)
            _, _, ldk, _ = bufs.geom(op.k)
            _, _, ldv, _ = bufs.geom(op.v)
            _, _, ldo, _ = bufs.geom(op.o)
            b(f"    sh_attention({bufs.haddr(op.q)}, {bufs.haddr(op.k)}, {bufs.haddr(op.v)}, {bufs.haddr(op.o)}, "
              f"{S}, {D}, {op.heads.value.data}, {ldq}, {ldk}, {ldv}, {ldo}, {_f(op.scale)}, {_cluster(op)});")

        elif isinstance(op, TransposeOp) and bufs.space(op.src) != "tcdm":
            rows, cols, lds, _ = bufs.geom(op.src)
            _, _, ldd, _ = bufs.geom(op.dst)
            b(f"    sh_transpose({bufs.haddr(op.dst)}, {bufs.haddr(op.src)}, {rows}, {cols}, {lds}, {ldd}, {_cluster(op)});")

        elif isinstance(op, HbmFillLcgOp):
            rows, cols, ld, _ = bufs.geom(op.buf)
            b(f"    if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_fill_fp16({bufs.haddr(op.buf)}, {rows}, {cols}, {ld}, "
              f"{op.seed.value.data}, {op.lo.value.data}, {op.hi.value.data}, {_f(op.scale)});")

        elif isinstance(op, DumpSamplesOp):
            rows, cols, ld, _ = bufs.geom(op.buf)
            b(f"    if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_samples({bufs.haddr(op.buf)}, {rows}, {cols}, {ld}, "
              f"{op.seed.value.data}, {op.n.value.data}, \"{op.tag.data}\");")

        elif isinstance(op, GroupBarrierOp):
            b("    sh_barrier_global();")

        else:
            b(f"    // (unhandled op: {op.name})")

    return "\n".join(body)


_MAIN_TEMPLATE = '''\
// Generated by softhier-mlir (softhier dialect -> softhier-ops calls). Do not edit by hand.
#include "sh_ops.h"

static void {kernel_name}_inputs(void) {{   // test inputs (not timed)
{prologue_body}
}}

static void {kernel_name}(void) {{
{kernel_body}
}}

static void {kernel_name}_checks(void) {{   // test outputs (not timed)
{epilogue_body}
}}

int main(void) {{
    sh_init();
    const int timekeeper = (sh_cluster_id() == 0 && sh_is_first_core());  // the timer is global: one core stamps it
    {kernel_name}_inputs();
    sh_barrier_global();
    if (timekeeper) sh_timer_start();
    {kernel_name}();
    sh_barrier_global();
    if (timekeeper) sh_timer_end();
    sh_barrier_global();
    {kernel_name}_checks();
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}}
'''


def emit_c(module: ModuleOp) -> str:
    """Emit a complete ``main.c`` from a module containing one softhier func."""
    fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    bufs = _Buffers()
    for op in fn.body.block.ops:
        if isinstance(op, HbmBufferOp):
            bufs.add_hbm(op)
        elif isinstance(op, L1BufferOp):
            bufs.add_l1(op)
        elif isinstance(op, ViewOp):
            bufs.add_view(op)
    return _MAIN_TEMPLATE.format(kernel_name=fn.sym_name.data,
                                 prologue_body=emit_kernel(fn, bufs, "prologue"),
                                 kernel_body=emit_kernel(fn, bufs, "kernel"),
                                 epilogue_body=emit_kernel(fn, bufs, "epilogue"))
