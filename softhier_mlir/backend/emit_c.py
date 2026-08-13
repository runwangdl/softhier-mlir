"""SoftHier dialect -> C backend.

Walks a `softhier` function and emits a runnable ``main.c`` against the SoftHier
``flex_*`` runtime (the fastest path to executing generated code on GVSoC).

Supported ops: ``hbm_buffer``, ``l1_buffer``, ``dma_2d`` (as contiguous 1-D
copies), ``redmule``, ``group_barrier``. The emitted kernel runs on cluster 0,
with the DM core issuing DMA and the first core driving RedMule — matching the
SDK's ``example_one_cluster_gemm``.
"""

from __future__ import annotations

from xdsl.dialects import func
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp

from softhier_mlir.dialects.softhier import (
    CheckConstOp,
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

_T = 256  # RedMule tile dimension
_TB = _T * _T * 2  # tile size in bytes (fp16)

_ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i16": 2, "i32": 4}
_FMT = {"fp16": "REDMULE_FP_16", "fp8": "REDMULE_FP_8", "int8": "REDMULE_INT_8",
        "int16": "REDMULE_INT_16"}


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
    ms = memref.memory_space
    return getattr(ms, "data", "tcdm")


class _Buffers:
    """Resolves each SSA buffer value to (space, byte-offset, size)."""

    def __init__(self) -> None:
        self.info: dict = {}
        self._l1_cursor = 0

    def add_hbm(self, op: HbmBufferOp) -> None:
        mt = op.result.type
        self.info[op.result] = ("hbm", op.offset.value.data, _bytes(mt), mt)

    def add_l1(self, op: L1BufferOp) -> None:
        mt = op.result.type
        off = self._l1_cursor
        self._l1_cursor += _bytes(mt)
        self.info[op.result] = ("tcdm", off, _bytes(mt), mt)

    def addr(self, val) -> str:
        space, off, _, _ = self.info[val]
        return f"hbm_addr({off})" if space == "hbm" else f"local({off})"

    def raw_off(self, val) -> int:
        return self.info[val][1]

    def space(self, val) -> str:
        return self.info[val][0]

    def size(self, val) -> int:
        return self.info[val][2]

    def memref(self, val) -> MemRefType:
        return self.info[val][3]


def emit_kernel(fn: func.FuncOp, bufs: _Buffers) -> str:
    """Emit the body of the cluster-0 kernel for one softhier function."""
    body: list[str] = []
    b = body.append
    ew = 0  # unique-id counter for elementwise loops

    for op in fn.body.block.ops:
        if isinstance(op, (HbmBufferOp, L1BufferOp, func.ReturnOp)):
            continue

        if isinstance(op, Dma2DOp):
            src, dst = op.src, op.dst
            size = bufs.size(src)
            b("    if (flex_is_dm_core()) {")
            b(f"        flex_dma_async_1d({bufs.addr(dst)}, {bufs.addr(src)}, {size});")
            b("        flex_dma_async_wait_all();")
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, RedmuleOp):
            xmt, wmt = bufs.memref(op.x), bufs.memref(op.w)
            m, k = xmt.get_shape()
            k2, n = wmt.get_shape()
            fmt = _FMT.get(op.fmt.data, "REDMULE_FP_16")
            b("    if (flex_is_first_core()) {")
            b(f"        flex_redmule_config({m}, {n}, {k});")
            b(f"        flex_redmule_trigger({bufs.raw_off(op.x)}, "
              f"{bufs.raw_off(op.w)}, {bufs.raw_off(op.y)}, {fmt});")
            b("        flex_redmule_wait();")
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, (L1ZeroOp, ReluOp, L1FillOp)):
            off = bufs.raw_off(op.buf)
            n = _nelem(bufs.memref(op.buf))
            p, i = f"p{ew}", f"i{ew}"
            ew += 1
            if isinstance(op, L1ZeroOp):
                stmt, comment = f"{p}[{i}] = 0;", "zero accumulator"
            elif isinstance(op, ReluOp):
                stmt = f"if ({p}[{i}] & 0x8000u) {p}[{i}] = 0;"
                comment = "relu (fp16: clear negatives)"
            else:  # L1FillOp
                v = op.value_bits.value.data & 0xFFFF
                stmt, comment = f"{p}[{i}] = {v}u;", "fill constant fp16"
            b(f"    if (flex_is_first_core()) {{  // {comment}")
            b(f"        volatile uint16_t *{p} = (volatile uint16_t *)local({off});")
            b(f"        for (int {i} = 0; {i} < {n}; ++{i}) {stmt}")
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, CheckConstOp):
            off = bufs.raw_off(op.buf)
            n = _nelem(bufs.memref(op.buf))
            exp = op.value_bits.value.data & 0xFFFF
            tol = op.tol.value.data
            c, i, ok, d = f"c{ew}", f"i{ew}", f"ok{ew}", f"d{ew}"
            ew += 1
            b("    if (flex_is_first_core() && flex_get_cluster_id() == 0) {  // verify")
            b(f"        volatile uint16_t *{c} = (volatile uint16_t *)local({off});")
            b(f"        int {ok} = 0;")
            b(f"        for (int {i} = 0; {i} < {n}; ++{i}) {{")
            b(f"            int {d} = (int){c}[{i}] - {exp}; if ({d} < 0) {d} = -{d};")
            b(f"            if ({d} <= {tol}) {ok}++;")
            b("        }")
            b('        flex_print((char *)"MLP_CHECK ok="); flex_print_int(' + ok + ");")
            b(f'        flex_print((char *)({ok} == {n} ? " MLP_PASS\\n" : " MLP_FAIL\\n"));')
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, GemmOp):
            xb, wb, zb = bufs.raw_off(op.x), bufs.raw_off(op.w), bufs.raw_off(op.z)
            m, k = bufs.memref(op.x).get_shape()
            _k2, n = bufs.memref(op.w).get_shape()
            mt, nt, kt = m // _T, n // _T, k // _T
            fmt = _FMT.get(op.fmt.data, "REDMULE_FP_16")
            r, c, kk = f"r{ew}", f"c{ew}", f"k{ew}"
            ew += 1
            b(f"    // GEMM {m}x{n}x{k}: {mt}x{nt} output tiles, {kt} K-steps")
            b(f"    for (int {r} = 0; {r} < {mt}; ++{r})")
            b(f"    for (int {c} = 0; {c} < {nt}; ++{c}) {{")
            b(f"        if (flex_is_first_core()) {{ volatile uint16_t *zt = "
              f"(volatile uint16_t *)local({2 * _TB}); for (int t = 0; t < {_T * _T}; ++t) zt[t] = 0; }}")
            b("        flex_intra_cluster_sync();")
            b(f"        for (int {kk} = 0; {kk} < {kt}; ++{kk}) {{")
            b("            if (flex_is_dm_core()) {")
            b(f"                flex_dma_async_1d(local(0), hbm_addr({xb} + ({r}*{kt} + {kk})*{_TB}), {_TB});")
            b(f"                flex_dma_async_1d(local({_TB}), hbm_addr({wb} + ({kk}*{nt} + {c})*{_TB}), {_TB});")
            b("                flex_dma_async_wait_all();")
            b("            }")
            b("            flex_intra_cluster_sync();")
            b("            if (flex_is_first_core()) {")
            b(f"                flex_redmule_config({_T}, {_T}, {_T});")
            b(f"                flex_redmule_trigger(0, {_TB}, {2 * _TB}, {fmt});")
            b("                flex_redmule_wait();")
            b("            }")
            b("            flex_intra_cluster_sync();")
            b("        }")
            b("        if (flex_is_dm_core()) {")
            b(f"            flex_dma_async_1d(hbm_addr({zb} + ({r}*{nt} + {c})*{_TB}), local({2 * _TB}), {_TB});")
            b("            flex_dma_async_wait_all();")
            b("        }")
            b("        flex_intra_cluster_sync();")
            b("    }")

        elif isinstance(op, HbmFillOp):
            hb = bufs.raw_off(op.buf)
            rows, cols = bufs.memref(op.buf).get_shape()
            ntiles = (rows // _T) * (cols // _T)
            v = op.value_bits.value.data & 0xFFFF
            t, i = f"t{ew}", f"i{ew}"
            ew += 1
            b(f"    // fill HBM {rows}x{cols} = fp16 {v} ({ntiles} tiles)")
            b(f"    if (flex_is_first_core()) {{ volatile uint16_t *fp = (volatile uint16_t *)local(0);"
              f" for (int {i} = 0; {i} < {_T * _T}; ++{i}) fp[{i}] = {v}u; }}")
            b("    flex_intra_cluster_sync();")
            b(f"    if (flex_is_dm_core()) for (int {t} = 0; {t} < {ntiles}; ++{t}) {{"
              f" flex_dma_async_1d(hbm_addr({hb} + {t}*{_TB}), local(0), {_TB}); flex_dma_async_wait_all(); }}")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, HbmFillColParityOp):
            hb = bufs.raw_off(op.buf)
            rows, cols = bufs.memref(op.buf).get_shape()
            ntiles = (rows // _T) * (cols // _T)
            ev = op.even_bits.value.data & 0xFFFF
            od = op.odd_bits.value.data & 0xFFFF
            t, i = f"t{ew}", f"i{ew}"
            ew += 1
            b(f"    // fill HBM {rows}x{cols} col-parity even={ev}/odd={od} ({ntiles} tiles)")
            b(f"    if (flex_is_first_core()) {{ volatile uint16_t *fp = (volatile uint16_t *)local(0);"
              f" for (int {i} = 0; {i} < {_T * _T}; ++{i}) fp[{i}] = ({i} % 2 == 0) ? {ev}u : {od}u; }}")
            b("    flex_intra_cluster_sync();")
            b(f"    if (flex_is_dm_core()) for (int {t} = 0; {t} < {ntiles}; ++{t}) {{"
              f" flex_dma_async_1d(hbm_addr({hb} + {t}*{_TB}), local(0), {_TB}); flex_dma_async_wait_all(); }}")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, HbmCheckConstOp):
            hb = bufs.raw_off(op.buf)
            rows, cols = bufs.memref(op.buf).get_shape()
            ntiles = (rows // _T) * (cols // _T)
            total = rows * cols
            exp = op.value_bits.value.data & 0xFFFF
            tol = op.tol.value.data
            t, i, ok, d = f"t{ew}", f"i{ew}", f"ok{ew}", f"d{ew}"
            ew += 1
            b(f"    // verify HBM {rows}x{cols} == {exp} ({ntiles} tiles, {total} elems)")
            b(f"    int {ok} = 0;")
            b(f"    for (int {t} = 0; {t} < {ntiles}; ++{t}) {{")
            b(f"        if (flex_is_dm_core()) {{ flex_dma_async_1d(local(0), hbm_addr({hb} + {t}*{_TB}), {_TB}); flex_dma_async_wait_all(); }}")
            b("        flex_intra_cluster_sync();")
            b("        if (flex_is_first_core() && flex_get_cluster_id() == 0) {")
            b("            volatile uint16_t *cp = (volatile uint16_t *)local(0);")
            b(f"            for (int {i} = 0; {i} < {_T * _T}; ++{i}) {{ int {d} = (int)cp[{i}] - {exp};"
              f" if ({d} < 0) {d} = -{d}; if ({d} <= {tol}) {ok}++; }}")
            b("        }")
            b("        flex_intra_cluster_sync();")
            b("    }")
            b("    if (flex_is_first_core() && flex_get_cluster_id() == 0) {")
            b(f'        flex_print((char *)"GEMM_CHECK ok="); flex_print_int({ok});')
            b(f'        flex_print((char *)({ok} == {total} ? " GEMM_PASS\\n" : " GEMM_FAIL\\n"));')
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, L1AddOp):
            src_off = bufs.raw_off(op.src)
            dst_off = bufs.raw_off(op.dst)
            n = _nelem(bufs.memref(op.dst))
            a, d, i = f"a{ew}", f"d{ew}", f"i{ew}"
            ew += 1
            b("    if (flex_is_first_core()) {  // dst += src (fp16)")
            b(f"        volatile _Float16 *{a} = (volatile _Float16 *)local({src_off});")
            b(f"        volatile _Float16 *{d} = (volatile _Float16 *)local({dst_off});")
            b(f"        for (int {i} = 0; {i} < {n}; ++{i}) {d}[{i}] += {a}[{i}];")
            b("    }")
            b("    flex_intra_cluster_sync();")

        elif isinstance(op, GroupBarrierOp):
            b("    flex_global_barrier_xy();")

        else:
            b(f"    // (unhandled op: {op.name})")

    return "\n".join(body)


_MAIN_TEMPLATE = '''\
// Generated by softhier-mlir (softhier -> C backend). Do not edit by hand.
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

static inline void {kernel_name}(void) {{
    flex_global_barrier_xy();
    uint32_t CID = flex_get_cluster_id();
    if (CID == 0) {{
{kernel_body}
    }}
    flex_global_barrier_xy();
}}

int main() {{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();

    {kernel_name}();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
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
    kernel_body = emit_kernel(fn, bufs)
    return _MAIN_TEMPLATE.format(
        kernel_name=fn.sym_name.data, kernel_body=kernel_body
    )
