"""SoftHier dialect -> C backend, targeting the softhier-ops operator library.

The generated ``main.c`` includes only ``sh_ops.h`` (see ``runtime/``) and calls one
library function per ``softhier`` op. Policy (tiling, pipelining, cluster mapping) is
carried as op attributes and forwarded to the library through ``sh_gemm_cfg``; the
mechanics (DMA, RedMulE, barriers) live in the library, where they can be measured and
swapped without touching the compiler.

Every library op is SPMD-safe: generated code runs on all cores of all clusters, flat,
with no cluster guards.

Control flow: ``scf.for`` over ``index`` values becomes a C ``for`` loop and ``arith``
constants / addi / subi / muli on indices become C expressions. An ``softhier.hbm_buffer``
or ``softhier.view`` with an ``index`` operand is addressed as ``offset + index * stride``
and declared where it appears (inside the loop); ``mark`` / ``dump_samples`` with an index
append it to their tag. The 64 KB instruction memory makes this necessary for anything
deeper than a layer or two (an unrolled 12-layer SigLIP is ~140 KB of code).
"""

from __future__ import annotations

from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, MemRefType, ModuleOp, StridedLayoutAttr
from xdsl.ir import Block, BlockArgument, Operation, SSAValue

from softhier_mlir.dialects.softhier import (
    AttentionOp,
    AddBiasOp,
    AddOp,
    CheckConstOp,
    DumpSamplesOp,
    GeluOp,
    HbmFillLcgOp,
    LayerNormOp,
    MarkOp,
    PreloadWaitOp,
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
    a = op.attributes.get(name) or op.properties.get(name)
    return a.value.data if isinstance(a, IntegerAttr) else default


def _cluster(op) -> str:
    """The `cluster` attribute: -1 -> SH_ALL (split over all clusters), absent -> 0."""
    c = _int_attr(op, "cluster", 0)
    return "SH_ALL" if c < 0 else str(c)


def _f(x: FloatAttr | float) -> str:
    v = x.value.data if isinstance(x, FloatAttr) else float(x)
    return f"{v!r}f"


def _walk(block: Block):
    """Every op of a block, descending into nested regions (scf.for bodies)."""
    for op in block.ops:
        yield op
        for region in op.regions:
            for blk in region.blocks:
                yield from _walk(blk)


class _Index:
    """C expressions for `index`-typed SSA values: arith constants, scf.for induction variables
    (named i1, i2, ... as their loops are emitted) and addi/subi/muli of those."""

    def __init__(self) -> None:
        self.names: dict[SSAValue, str] = {}
        self._n = 0

    def bind_iv(self, iv: BlockArgument) -> str:
        self._n += 1
        self.names[iv] = f"i{self._n}"
        return self.names[iv]

    def expr(self, v: SSAValue) -> str:
        if v in self.names:
            return self.names[v]
        op = v.owner
        if isinstance(op, arith.ConstantOp):
            return str(op.value.value.data)
        if isinstance(op, arith.AddiOp):
            return f"({self.expr(op.lhs)} + {self.expr(op.rhs)})"
        if isinstance(op, arith.SubiOp):
            return f"({self.expr(op.lhs)} - {self.expr(op.rhs)})"
        if isinstance(op, arith.MuliOp):
            return f"({self.expr(op.lhs)} * {self.expr(op.rhs)})"
        raise NotImplementedError(f"index expression from {op.name if isinstance(op, Operation) else op}")

    def is_const(self, v: SSAValue) -> bool:
        return isinstance(v.owner, arith.ConstantOp)


class _Buffers:
    """Resolves each SSA buffer value to (space, byte-offset, size, memref, c-name-or-expression).

    Plain HBM / L1 buffers are `const` names declared at the top of the kernel. An indexed HBM
    buffer (offset + index * stride) is declared where it appears, so its name is valid inside the
    enclosing loop; a view's "name" is an address expression over its source."""

    def __init__(self, idx: _Index) -> None:
        self.info: dict = {}
        self.top_decls: list[str] = []
        self._l1_cursor = 0
        self._n = 0
        self.idx = idx

    def _name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def add_hbm(self, op: HbmBufferOp) -> str | None:
        """Register the buffer; returns the declaration to emit in place for an indexed one."""
        mt = op.result.type
        off, name = op.offset.value.data, self._name("hb")
        shape = "x".join(str(d) for d in mt.get_shape())
        self.info[op.result] = ("hbm", off, _bytes(mt), mt, name)
        if op.index is None:
            self.top_decls.append(f"    const uint64_t {name} = sh_hbm_addr({off});  // {shape}x{mt.element_type} HBM, {_bytes(mt)} B")
            return None
        stride = op.stride.value.data if op.stride is not None else 0
        return (f"const uint64_t {name} = sh_hbm_addr((uint64_t){off} + (uint64_t)({self.idx.expr(op.index)}) * {stride}u);"
                f"  // {shape}x{mt.element_type} HBM, {_bytes(mt)} B, indexed")

    def add_l1(self, op: L1BufferOp) -> None:
        mt = op.result.type
        off = self._l1_cursor
        self._l1_cursor += _bytes(mt)
        name = self._name("l1b")
        shape = "x".join(str(d) for d in mt.get_shape())
        self.info[op.result] = ("tcdm", off, _bytes(mt), mt, name)
        self.top_decls.append(f"    const uint32_t {name} = {off};  // {shape}x{mt.element_type} TCDM offset, {_bytes(mt)} B")

    def add_view(self, op: ViewOp) -> None:
        """A view addresses its source; the strided layout gives ld and the constant element
        offset, an `index` operand adds index * stride elements."""
        mt = op.result.type
        space, off, _, _, name = self.info[op.src]
        if op.index is not None:
            stride = op.stride.value.data if op.stride is not None else 0
            name = f"({name} + (uint64_t)({self.idx.expr(op.index)}) * {stride * _elem_bytes(mt)}u)"
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
        return self.name(val) if eoff == 0 else f"({self.name(val)} + {eoff * _elem_bytes(self.memref(val))})"

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


def _gemm_cfg(op: GemmOp) -> str:
    fmt = _FMT.get(op.fmt.data, "SH_FP16")
    return (f"{{ .tm = {_int_attr(op, 'tile_m', 0)}, .tn = {_int_attr(op, 'tile_n', 0)}, "
            f".tk = {_int_attr(op, 'tile_k', 0)}, .pipeline = {1 if 'pipeline' in op.attributes else 0}, "
            f".accumulate = {1 if 'accumulate' in op.attributes else 0}, .fmt = {fmt}, .l1_base = 0 }}")


def _tagged(idx: _Index, tag: str, index) -> tuple[str, str]:
    """(printf format piece, extra args) for a tag with an optional runtime index suffix."""
    if index is None:
        return f"\"{tag}\"", ""
    return f"\"{tag}\"", f", (uint32_t)({idx.expr(index)})"


def _emit_ops(ops, bufs: _Buffers, idx: _Index, b, tag: str, ind: str) -> None:
    for op in ops:
        if isinstance(op, (L1BufferOp, func.ReturnOp, arith.ConstantOp, arith.AddiOp, arith.SubiOp, arith.MuliOp, scf.YieldOp)):
            continue

        if isinstance(op, HbmBufferOp):
            if op.result in bufs.info:      # top-level buffer, declared with the kernel prologue
                continue
            decl = bufs.add_hbm(op)
            if decl:
                b(f"{ind}{decl}")

        elif isinstance(op, ViewOp):
            bufs.add_view(op)

        elif isinstance(op, scf.ForOp):
            iv = idx.bind_iv(op.body.block.args[0])
            b(f"{ind}for (uint32_t {iv} = {idx.expr(op.lb)}; {iv} < {idx.expr(op.ub)}; {iv} += {idx.expr(op.step)}) {{")
            _emit_ops(op.body.block.ops, bufs, idx, b, tag, ind + "    ")
            b(f"{ind}}}")

        elif isinstance(op, Dma2DOp):
            b(f"{ind}sh_dma_copy({bufs.addr(op.dst)}, {bufs.addr(op.src)}, {bufs.size(op.src)});")

        elif isinstance(op, RedmuleOp):
            m, k = bufs.memref(op.x).get_shape()
            _k2, n = bufs.memref(op.w).get_shape()
            fmt = _FMT.get(op.fmt.data, "SH_FP16")
            b(f"{ind}sh_redmule({bufs.name(op.x)}, {bufs.name(op.w)}, {bufs.name(op.y)}, {m}, {n}, {k}, {fmt});")

        elif isinstance(op, L1ZeroOp):
            b(f"{ind}sh_l1_zero({bufs.name(op.buf)}, {bufs.size(op.buf)});")

        elif isinstance(op, L1FillOp):
            b(f"{ind}sh_l1_fill_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))}, {op.value_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, ReluOp):
            b(f"{ind}sh_l1_relu_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))});")

        elif isinstance(op, L1AddOp):
            b(f"{ind}sh_l1_add_fp16({bufs.name(op.dst)}, {bufs.name(op.src)}, {_nelem(bufs.memref(op.dst))});")

        elif isinstance(op, CheckConstOp):
            b(f"{ind}sh_test_check_const_l1_fp16({bufs.name(op.buf)}, {_nelem(bufs.memref(op.buf))}, "
              f"{op.value_bits.value.data & 0xFFFF}u, {op.tol.value.data}, \"{tag}\");")

        elif isinstance(op, GemmOp):
            m, k, ldx, _ = bufs.geom(op.x)
            _k2, n, ldw, _ = bufs.geom(op.w)
            _m2, _n2, ldz, _ = bufs.geom(op.z)
            cfg = _gemm_cfg(op)
            args = f"{bufs.haddr(op.x)}, {bufs.haddr(op.w)}, {bufs.haddr(op.z)}, {m}, {n}, {k}, {ldx}, {ldw}, {ldz}, &cfg"
            if "summa" in op.attributes:
                b(f"{ind}{{ sh_gemm_cfg cfg = {cfg};  // mesh-wide SUMMA")
                b(f"{ind}  sh_gemm_mesh({args}); }}")
            else:
                b(f"{ind}{{ sh_gemm_cfg cfg = {cfg};")
                b(f"{ind}  sh_gemm({args}, {_cluster(op)}); }}")

        elif isinstance(op, HbmFillOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"{ind}sh_test_fill_const_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, {op.value_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, HbmFillColParityOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"{ind}sh_test_fill_colparity_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, "
              f"{op.even_bits.value.data & 0xFFFF}u, {op.odd_bits.value.data & 0xFFFF}u);")

        elif isinstance(op, HbmCheckConstOp):
            rows, cols = bufs.memref(op.buf).get_shape()
            b(f"{ind}sh_test_check_const_fp16({bufs.name(op.buf)}, {rows}, {cols}, {cols}, "
              f"{op.value_bits.value.data & 0xFFFF}u, {op.tol.value.data}, \"{tag}\");")

        elif isinstance(op, LayerNormOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"{ind}sh_layernorm({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {bufs.haddr(op.gamma)}, {bufs.haddr(op.beta)}, "
              f"{rows}, {cols}, {ld}, {_f(op.eps)}, {_cluster(op)});")

        elif isinstance(op, SoftmaxOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"{ind}sh_softmax_rows({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {rows}, {cols}, {ld}, {_f(op.scale)}, {_cluster(op)});")

        elif isinstance(op, GeluOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"{ind}sh_gelu({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AddOp):
            rows, cols, ld, _ = bufs.geom(op.a)
            b(f"{ind}sh_add({bufs.haddr(op.y)}, {bufs.haddr(op.a)}, {bufs.haddr(op.b)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AddBiasOp):
            rows, cols, ld, _ = bufs.geom(op.x)
            b(f"{ind}sh_add_bias({bufs.haddr(op.y)}, {bufs.haddr(op.x)}, {bufs.haddr(op.bias)}, {rows}, {cols}, {ld}, {_cluster(op)});")

        elif isinstance(op, AttentionOp):
            S, D, ldq, _ = bufs.geom(op.q)
            _, _, ldk, _ = bufs.geom(op.k)
            _, _, ldv, _ = bufs.geom(op.v)
            _, _, ldo, _ = bufs.geom(op.o)
            qb = _int_attr(op, "q_block", 0)      # q-block policy attribute; 0 / absent = the library's rule
            b(f"{ind}sh_attention{'_q' if qb else ''}({bufs.haddr(op.q)}, {bufs.haddr(op.k)}, {bufs.haddr(op.v)}, {bufs.haddr(op.o)}, "
              f"{S}, {D}, {op.heads.value.data}, {ldq}, {ldk}, {ldv}, {ldo}, {_f(op.scale)}, {_cluster(op)}{f', {qb}' if qb else ''});")

        elif isinstance(op, TransposeOp) and bufs.space(op.src) != "tcdm":
            rows, cols, lds, _ = bufs.geom(op.src)
            _, _, ldd, _ = bufs.geom(op.dst)
            b(f"{ind}sh_transpose({bufs.haddr(op.dst)}, {bufs.haddr(op.src)}, {rows}, {cols}, {lds}, {ldd}, {_cluster(op)});")

        elif isinstance(op, HbmFillLcgOp):
            rows, cols, ld, _ = bufs.geom(op.buf)
            b(f"{ind}if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_fill_fp16({bufs.haddr(op.buf)}, {rows}, {cols}, {ld}, "
              f"{op.seed.value.data}, {op.lo.value.data}, {op.hi.value.data}, {_f(op.scale)});")

        elif isinstance(op, DumpSamplesOp):
            rows, cols, ld, _ = bufs.geom(op.buf)
            if op.index is None:
                b(f"{ind}if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_samples({bufs.haddr(op.buf)}, {rows}, {cols}, {ld}, "
                  f"{op.seed.value.data}, {op.n.value.data}, \"{op.tag.data}\");")
            else:
                b(f"{ind}if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_samples_idx({bufs.haddr(op.buf)}, {rows}, {cols}, {ld}, "
                  f"{op.seed.value.data}, {op.n.value.data}, \"{op.tag.data}\", (uint32_t)({idx.expr(op.index)}));")

        elif isinstance(op, GroupBarrierOp):
            b(f"{ind}sh_barrier_global();")

        elif isinstance(op, PreloadWaitOp):
            b(f"{ind}sh_preload_wait({bufs.haddr(op.buf)});")

        elif isinstance(op, MarkOp):
            if op.index is None:
                b(f"{ind}if (sh_cluster_id() == 0 && sh_is_first_core()) sh_printf(\"[mark] %s %u\\n\", \"{op.tag.data}\", sh_cycles());")
            else:
                b(f"{ind}if (sh_cluster_id() == 0 && sh_is_first_core()) sh_printf(\"[mark] %s%u %u\\n\", \"{op.tag.data}\", "
                  f"(uint32_t)({idx.expr(op.index)}), sh_cycles());")

        else:
            b(f"{ind}// (unhandled op: {op.name})")


_PROLOGUE_OPS = (HbmFillOp, HbmFillColParityOp, HbmFillLcgOp, PreloadWaitOp)   # test inputs / preload: before the timer
_EPILOGUE_OPS = (HbmCheckConstOp,)   # whole-buffer checks: after the timer. Sample dumps stay IN PLACE: a
                                     # buffer may be reused later in the program (residual stream across layers)
_STRUCTURAL_OPS = (HbmBufferOp, L1BufferOp, ViewOp, arith.ConstantOp, arith.AddiOp, arith.SubiOp, arith.MuliOp, func.ReturnOp)


def _phase_of(op, later_ops=()) -> str:
    """Prologue: test fills / preload wait. Epilogue: whole-buffer checks, and sample dumps whose
    buffer no later top-level op touches (a dump of a buffer that is reused later, e.g. the residual
    stream across layers, has to stay in place; it then costs printing time inside the ROI)."""
    if isinstance(op, _PROLOGUE_OPS):
        return "prologue"
    if isinstance(op, _EPILOGUE_OPS):
        return "epilogue"
    if isinstance(op, DumpSamplesOp):
        base = _base_value(op.buf)
        for lo in later_ops:
            if isinstance(lo, (DumpSamplesOp, HbmCheckConstOp)):
                continue
            for operand in _all_operands(lo):
                if _base_value(operand) is base:
                    return "kernel"
        return "epilogue"
    return "kernel"


def _base_value(v):
    """Follow softhier.view chains to the underlying buffer SSA value."""
    while isinstance(v.owner, ViewOp):
        v = v.owner.src
    return v


def _all_operands(op):
    """Operands of an op, descending into nested regions (scf.for bodies)."""
    out = list(op.operands)
    for region in op.regions:
        for block in region.blocks:
            for inner in block.ops:
                out.extend(_all_operands(inner))
    return out


def emit_kernel(fn: func.FuncOp, bufs: _Buffers | None = None, phase: str | None = None) -> str:
    """Emit the function body, or one phase of it: test-input fills and the preload wait ("prologue",
    run before the timer, followed by a global barrier), the computation ("kernel", timed) or the
    test dumps/checks ("epilogue", after the timer). Only TOP-LEVEL ops are assigned to phases; a
    loop body stays whole inside the kernel. Buffer/view declarations are repeated in every phase."""
    idx = _Index()
    bufs = bufs or _Buffers(idx)
    bufs.idx = idx
    for op in fn.body.block.ops:      # top-level buffers: declared in the prologue, named in program order
        if isinstance(op, L1BufferOp):
            bufs.add_l1(op)
        elif isinstance(op, HbmBufferOp) and op.index is None:
            bufs.add_hbm(op)
    top = list(fn.body.block.ops)
    ops = [op for i, op in enumerate(top)
           if phase is None or isinstance(op, _STRUCTURAL_OPS) or _phase_of(op, top[i + 1:]) == phase]
    body: list[str] = []
    _emit_ops(ops, bufs, idx, body.append, fn.sym_name.data.upper(), "    ")
    return "\n".join(bufs.top_decls + body)


_MAIN_TEMPLATE = '''\
// Generated by softhier-mlir (softhier dialect -> softhier-ops calls). Do not edit by hand.
#include "sh_ops.h"

static void {kernel_name}_inputs(void) {{   // preload wait + test inputs (not timed)
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
    return _MAIN_TEMPLATE.format(kernel_name=fn.sym_name.data,
                                 prologue_body=emit_kernel(fn, phase="prologue"),
                                 kernel_body=emit_kernel(fn, phase="kernel"),
                                 epilogue_body=emit_kernel(fn, phase="epilogue"))
