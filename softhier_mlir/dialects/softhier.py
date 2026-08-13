"""The `softhier` xDSL dialect.

A thin accelerator dialect for the SoftHier RISC-V many-cluster accelerator
(``pulp.chips.soft_hier_old.flex_cluster`` in GVSoC). It surfaces *only* the
irreducible hardware operations; everything else (matmul, softmax, tiling,
control flow) is meant to be expressed with mature dialects (linalg, memref,
scf, arith, vector) and lowered into these ops.

See ``docs/DESIGN.md`` for the full abstraction and lowering pipeline.

Target: xDSL >= 0.69.
"""

from __future__ import annotations

from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    StringAttr,
    i32,
)
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)

# --------------------------------------------------------------------------- #
# Memory-space conventions
#
# A memref's memory space is a StringAttr drawn from this set; it selects the
# physical space and, for HBM, the mesh edge nearest the owning cluster:
#   "tcdm"        local scratchpad (L1)
#   "remote_tcdm" another cluster's L1 over the NoC
#   "hbm_west" | "hbm_south" | "hbm_north" | "hbm_east"
# A dedicated ParametrizedAttribute can replace this later; a string keeps the
# scaffold dependency-free and round-trippable today.
# --------------------------------------------------------------------------- #
SPACES = ("tcdm", "remote_tcdm", "hbm_west", "hbm_south", "hbm_north", "hbm_east")


@irdl_op_definition
class RedmuleOp(IRDLOperation):
    """RedMule GEMM offload: ``y += x @ w`` (in-place accumulate into ``y``).

    Mirrors the hardware config/trigger/wait sequence. ``fmt`` selects the
    datapath: fp16 / fp8 / int16 / int8. Operands live in TCDM.
    """

    name = "softhier.redmule"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    w = operand_def(MemRefType)
    y = operand_def(MemRefType)
    fmt = prop_def(StringAttr)  # "fp16" | "fp8" | "int16" | "int8"
    assembly_format = (
        "$x `,` $w `into` $y attr-dict `:` type($x) `,` type($w) `,` type($y)"
    )


@irdl_op_definition
class Dma2DOp(IRDLOperation):
    """Strided 2-D iDMA copy (HBM <-> TCDM).

    ``repeat`` rows of ``size`` bytes with independent src/dst strides.
    """

    name = "softhier.dma_2d"
    irdl_options = (ParsePropInAttrDict(),)
    dst = operand_def(MemRefType)
    src = operand_def(MemRefType)
    size = prop_def(IntegerAttr)
    dst_stride = prop_def(IntegerAttr)
    src_stride = prop_def(IntegerAttr)
    repeat = prop_def(IntegerAttr)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


@irdl_op_definition
class DmaBroadcastOp(IRDLOperation):
    """Multicast ``src`` to the same TCDM offset across a mesh row/column.

    Lowering target for ``mesh.broadcast``. ``row_mask``/``col_mask`` select the
    participating clusters (SUMMA row/column broadcast).
    """

    name = "softhier.dma_broadcast"
    irdl_options = (ParsePropInAttrDict(),)
    dst = operand_def(MemRefType)
    src = operand_def(MemRefType)
    row_mask = prop_def(IntegerAttr)
    col_mask = prop_def(IntegerAttr)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


@irdl_op_definition
class DmaReduceOp(IRDLOperation):
    """In-network iDMA reduction across a mesh row/column (REDADD / REDMAX).

    Lowering target for ``mesh.all_reduce``. Used for SUMMA partial-sum and for
    the softmax row-max / row-sum over the KV partition.
    """

    name = "softhier.dma_reduce"
    irdl_options = (ParsePropInAttrDict(),)
    dst = operand_def(MemRefType)
    src = operand_def(MemRefType)
    kind = prop_def(StringAttr)  # "add" | "max"
    row_mask = prop_def(IntegerAttr)
    col_mask = prop_def(IntegerAttr)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


@irdl_op_definition
class GroupBarrierOp(IRDLOperation):
    """Two-phase X-then-Y barrier scoped to a ``grid_x`` x ``grid_y`` group."""

    name = "softhier.group_barrier"
    irdl_options = (ParsePropInAttrDict(),)
    grid_x = prop_def(IntegerAttr)
    grid_y = prop_def(IntegerAttr)
    assembly_format = "attr-dict"


@irdl_op_definition
class ClusterPosOp(IRDLOperation):
    """This cluster's (x, y) position in the mesh."""

    name = "softhier.cluster_pos"
    x = result_def(IndexType)
    y = result_def(IndexType)
    assembly_format = "attr-dict `:` type($x) `,` type($y)"


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """On-chip transpose engine (e.g. K -> K^T in TCDM before Q@K^T)."""

    name = "softhier.transpose"
    dst = operand_def(MemRefType)
    src = operand_def(MemRefType)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


@irdl_op_definition
class VExpOp(IRDLOperation):
    """Spatz vector exponential (``vfexp``) over a TCDM tile, in place."""

    name = "softhier.vexp"
    dst = operand_def(MemRefType)
    src = operand_def(MemRefType)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


SoftHier = Dialect(
    "softhier",
    [
        RedmuleOp,
        Dma2DOp,
        DmaBroadcastOp,
        DmaReduceOp,
        GroupBarrierOp,
        ClusterPosOp,
        TransposeOp,
        VExpOp,
    ],
    [],
)
