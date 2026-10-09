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
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    StringAttr,
    i32,
)
from xdsl.ir import Dialect
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
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


@irdl_op_definition
class ReluOp(IRDLOperation):
    """Elementwise ReLU over a TCDM tile, in place (``x = max(0, x)``)."""

    name = "softhier.relu"
    buf = operand_def(MemRefType)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class L1ZeroOp(IRDLOperation):
    """Zero a TCDM tile (RedMule accumulates into ``y``, so clear it first)."""

    name = "softhier.l1_zero"
    buf = operand_def(MemRefType)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class L1AddOp(IRDLOperation):
    """Elementwise add of two TCDM tiles, in place: ``dst += src`` (fp16).

    Useful for residual/bias adds when composing layers.
    """

    name = "softhier.l1_add"
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    assembly_format = "$src `into` $dst attr-dict `:` type($src) `,` type($dst)"


@irdl_op_definition
class L1FillOp(IRDLOperation):
    """Fill a TCDM tile with a constant fp16 bit pattern (for test inputs)."""

    name = "softhier.l1_fill"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    value_bits = prop_def(IntegerAttr)  # raw 16-bit fp16 pattern
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class GemmOp(IRDLOperation):
    """A full (multi-tile) GEMM ``z = x @ w`` on HBM-resident matrices.

    The backend tiles it into 256x256x256 RedMule ops with K-accumulation and
    per-tile HBM<->TCDM DMA (matrices assumed tile-major in HBM). ``x``:MxK,
    ``w``:KxN, ``z``:MxN.
    """

    name = "softhier.gemm"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    w = operand_def(MemRefType)
    z = operand_def(MemRefType)
    step = opt_operand_def(IndexType)       # optional: with ``fmt_steps``, the RedMulE format is fmt_steps[step]
    fmt = prop_def(StringAttr)
    fmt_steps = opt_prop_def(ArrayAttr)     # per-step formats ["fp16", "fp8", ...] indexed by ``step`` (flow-matching
                                            # steps of different precision, docs/SMOLVLA_EXPERT.md); ``fmt`` without it
    assembly_format = (
        "$x `,` $w `into` $z (`step` $step^)? attr-dict `:` type($x) `,` type($w) `,` type($z)"
    )


@irdl_op_definition
class HbmFillOp(IRDLOperation):
    """Fill an HBM matrix with a constant fp16 bit pattern (tile by tile)."""

    name = "softhier.hbm_fill"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    value_bits = prop_def(IntegerAttr)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class HbmFillColParityOp(IRDLOperation):
    """Fill an HBM matrix with a column-parity pattern: even columns get
    ``even_bits``, odd columns get ``odd_bits`` (fp16). A non-uniform input for
    correctness testing beyond constants."""

    name = "softhier.hbm_fill_col_parity"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    even_bits = prop_def(IntegerAttr)
    odd_bits = prop_def(IntegerAttr)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class HbmCheckConstOp(IRDLOperation):
    """Verify an HBM matrix ~= ``value_bits`` (fp16) within ``tol`` ULPs, tile by
    tile; prints ``GEMM_PASS`` / ``GEMM_FAIL``."""

    name = "softhier.hbm_check_const"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    value_bits = prop_def(IntegerAttr)
    tol = prop_def(IntegerAttr)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class CheckConstOp(IRDLOperation):
    """Verify every element of a TCDM tile equals ``value_bits`` (fp16) within
    ``tol`` ULPs; prints ``MLP_PASS`` / ``MLP_FAIL`` via the runtime log."""

    name = "softhier.check_const"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    value_bits = prop_def(IntegerAttr)
    tol = prop_def(IntegerAttr)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class HbmBufferOp(IRDLOperation):
    """Declares an HBM buffer at a fixed byte ``offset`` from the HBM base.

    Codegen resolves uses of the result to ``hbm_addr(offset)``. With an ``index`` operand
    (an scf.for induction variable or arith expression over one) the buffer is the
    ``index``-th of a family laid out ``stride`` bytes apart: ``hbm_addr(offset + index * stride)``,
    e.g. the weights of layer ``index``.
    """

    name = "softhier.hbm_buffer"
    irdl_options = (ParsePropInAttrDict(),)
    index = opt_operand_def(IndexType)
    offset = prop_def(IntegerAttr)
    stride = opt_prop_def(IntegerAttr)
    result = result_def(MemRefType)
    assembly_format = "($index^)? attr-dict `:` type($result)"


@irdl_op_definition
class L1BufferOp(IRDLOperation):
    """Declares a TCDM (L1) tile buffer; codegen bump-allocates its offset.

    Resolves to ``local(offset)`` in the emitted C.
    """

    name = "softhier.l1_buffer"
    result = result_def(MemRefType)
    assembly_format = "attr-dict `:` type($result)"



# --------------------------------------------------------------------------- #
# HBM tensor ops (fp16, row-major, strided views allowed). These lower 1:1 to the
# softhier-ops library (runtime/sh_ops.h). Optional attribute on every one of them:
#   cluster = <i32>   executing cluster, -1 = split over all clusters (SH_ALL); default 0
# --------------------------------------------------------------------------- #
@irdl_op_definition
class ViewOp(IRDLOperation):
    """A strided sub-view of an HBM buffer. The result type carries the layout:
    ``memref<rows x cols x f16, strided<[ld, 1], offset: elems>, "hbm_*">`` where the
    offset is in elements from the source buffer's base. An optional ``index`` operand adds
    ``index * stride`` elements (the head loop of attention: head ``index`` of q/k/v/o)."""
    name = "softhier.view"
    irdl_options = (ParsePropInAttrDict(),)
    src = operand_def(MemRefType)
    index = opt_operand_def(IndexType)      # optional: the view starts ``index * stride`` elements further
    stride = opt_prop_def(IntegerAttr)
    result = result_def(MemRefType)
    assembly_format = "$src (`,` $index^)? attr-dict `:` type($src) `->` type($result)"


@irdl_op_definition
class LayerNormOp(IRDLOperation):
    """``y = layernorm(x) * gamma + beta`` over the last dim (gamma/beta are 1 x cols)."""
    name = "softhier.layernorm"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    gamma = operand_def(MemRefType)
    beta = operand_def(MemRefType)
    y = operand_def(MemRefType)
    eps = prop_def(FloatAttr)
    assembly_format = "$x `,` $gamma `,` $beta `->` $y attr-dict `:` type($x) `,` type($gamma) `,` type($beta) `->` type($y)"


@irdl_op_definition
class SoftmaxOp(IRDLOperation):
    """``y = softmax(scale * x)`` per row (in place allowed). With the optional ``mask`` operand (a
    ``memref<S x i16>`` of token classes, see ``sh_softmax_masked``: query i attends key j iff
    ``mask[j] <= mask[i]`` unsigned, 0xFFFF = padding) the rows are queries and the columns keys of
    the same sequence."""
    name = "softhier.softmax"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    mask = opt_operand_def(MemRefType)
    y = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    assembly_format = "$x (`,` $mask^)? `->` $y attr-dict `:` type($x) (`,` type($mask)^)? `->` type($y)"


@irdl_op_definition
class GeluOp(IRDLOperation):
    """``y = gelu(x)`` (tanh approximation)."""
    name = "softhier.gelu"
    x = operand_def(MemRefType)
    y = operand_def(MemRefType)
    assembly_format = "$x `->` $y attr-dict `:` type($x) `->` type($y)"


@irdl_op_definition
class AddOp(IRDLOperation):
    """``y = a + b`` elementwise (unit attribute ``train``: ``sh_t_add``, the training library's kernel)."""
    name = "softhier.add"
    a = operand_def(MemRefType)
    b = operand_def(MemRefType)
    y = operand_def(MemRefType)
    assembly_format = "$a `,` $b `->` $y attr-dict `:` type($a) `,` type($b) `->` type($y)"


@irdl_op_definition
class AddBiasOp(IRDLOperation):
    """``y = x + bias`` with ``bias`` a 1 x cols row broadcast over rows."""
    name = "softhier.add_bias"
    x = operand_def(MemRefType)
    bias = operand_def(MemRefType)
    y = operand_def(MemRefType)
    assembly_format = "$x `,` $bias `->` $y attr-dict `:` type($x) `,` type($bias) `->` type($y)"


@irdl_op_definition
class AttentionOp(IRDLOperation):
    """Fused multi-head attention on HBM tensors: for every head ``h`` (columns
    ``[h*dh, (h+1)*dh)`` of the ``S x D`` operands, ``dh = D / heads``)
    ``o_h = softmax(scale * q_h k_h^T) v_h``, processed as work items of ``q_block`` query rows
    (the head's K / K^T / V and the ``q_block x S`` scores stay inside one cluster's TCDM).
    ``cluster = -1`` deals the ``heads * S / q_block`` items in contiguous chunks over the
    clusters. ``q_block`` is a policy attribute (``softhier_mlir.frontend.siglip.attention_q_block``
    is the rule); absent, the library applies the same rule (``sh_attention_q_block``)."""
    name = "softhier.attention"
    irdl_options = (ParsePropInAttrDict(),)
    q = operand_def(MemRefType)
    k = operand_def(MemRefType)
    v = operand_def(MemRefType)
    mask = opt_operand_def(MemRefType)      # optional token-class mask (memref<S x i16>, see SoftmaxOp) -> sh_attention_gqa
    o = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    heads = prop_def(IntegerAttr)
    kv_heads = opt_prop_def(IntegerAttr)    # grouped-query attention: k / v are S x (kv_heads * dh); query head h uses kv head h // (heads / kv_heads)
    q_block = opt_prop_def(IntegerAttr)     # q-block work-item policy (0/absent = library rule), see sh_attention_q
    assembly_format = "$q `,` $k `,` $v (`,` $mask^)? `->` $o attr-dict `:` type($q) `,` type($k) `,` type($v) (`,` type($mask)^)? `->` type($o)"


@irdl_op_definition
class RmsNormOp(IRDLOperation):
    """``y = x * rsqrt(mean(x^2) + eps) * gamma`` over the last dim (HF LlamaRMSNorm; gamma is 1 x cols)."""
    name = "softhier.rmsnorm"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    gamma = operand_def(MemRefType)
    y = operand_def(MemRefType)
    eps = prop_def(FloatAttr)
    assembly_format = "$x `,` $gamma `->` $y attr-dict `:` type($x) `,` type($gamma) `->` type($y)"


@irdl_op_definition
class RopeOp(IRDLOperation):
    """Rotary position embedding, HF Llama rotate-half convention, on every head of ``head_dim`` columns of
    each row: ``y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin`` with ``cos_sin`` a ``rows x head_dim`` table
    (cos[head_dim/2] then sin[head_dim/2] per row, precomputed by the host for the row's position id)."""
    name = "softhier.rope"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    cos_sin = operand_def(MemRefType)
    y = operand_def(MemRefType)
    head_dim = prop_def(IntegerAttr)
    assembly_format = "$x `,` $cos_sin `->` $y attr-dict `:` type($x) `,` type($cos_sin) `->` type($y)"


@irdl_op_definition
class SiluMulOp(IRDLOperation):
    """``y = silu(a) * b`` elementwise (the SiLU-gated MLP activation; ``b`` may be omitted: ``y = silu(a)``, the expert's time MLP)."""
    name = "softhier.silu_mul"
    irdl_options = (ParsePropInAttrDict(),)
    a = operand_def(MemRefType)
    b = opt_operand_def(MemRefType)
    y = operand_def(MemRefType)
    assembly_format = "$a (`,` $b^)? `->` $y attr-dict `:` type($a) (`,` type($b)^)? `->` type($y)"


@irdl_op_definition
class ScaleOp(IRDLOperation):
    """``y = scale * x`` elementwise (``sh_scale``)."""
    name = "softhier.scale"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    y = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    assembly_format = "$x `->` $y attr-dict `:` type($x) `->` type($y)"


@irdl_op_definition
class PixelShuffleOp(IRDLOperation):
    """SmolVLM connector pixel shuffle: ``src`` = ``grid*grid x D`` raster patch tokens ->
    ``dst`` = ``(grid/scale)^2 x D*scale^2`` (output token (gr, gb) = its scale x scale patch block,
    row-major). ``sh_pixel_shuffle``; data movement only."""
    name = "softhier.pixel_shuffle"
    irdl_options = (ParsePropInAttrDict(),)
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    scale = prop_def(IntegerAttr)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


@irdl_op_definition
class AxpyOp(IRDLOperation):
    """``y = a + alpha * b`` elementwise (the flow-matching Euler update ``x_t + dt * v_t``)."""
    name = "softhier.axpy"
    irdl_options = (ParsePropInAttrDict(),)
    a = operand_def(MemRefType)
    b = operand_def(MemRefType)
    y = operand_def(MemRefType)
    alpha = prop_def(FloatAttr)
    assembly_format = "$a `,` $b `->` $y attr-dict `:` type($a) `,` type($b) `->` type($y)"


@irdl_op_definition
class CrossAttentionOp(IRDLOperation):
    """GQA attention of ``Sq`` query tokens (``q``: Sq x heads*dh) over a stationary prefix KV (``kp`` /
    ``vp``: Lp x kv_heads*dh, e.g. the VLM prefix of SmolVLA) plus, with ``own %ko, %vo`` (Sq x kv_heads*dh),
    the tokens' own keys/values attended causally (query i sees own key j iff j <= i). ``mask %tok``: the
    prefix's 1 x Lp i16 token-class array (as for ``softhier.attention``'s mask: 0xFFFF = padding key; the
    expert's queries attend every non-padding prefix key). Query head h uses kv head h / (heads / kv_heads).
    Each head runs inside one cluster's TCDM; ``cluster = -1`` deals head h to cluster h % P.
    ``n_batch = N``: the rows of ``q`` / ``ko`` / ``vo`` / ``o`` are N candidate blocks (N x Sq query tokens that
    share the prefix KV); candidate c's queries attend the prefix and, causally, only their own block c
    (``sh_x_attention_n``, docs/WORLD_MODEL.md). With the unit attribute ``train`` (single candidate) it lowers to
    ``sh_t_attention_fwd`` (same result, the backward kernel's staging code: training programs keep one attention
    implementation in the 64 KB instruction memory)."""
    name = "softhier.cross_attention"
    irdl_options = (ParsePropInAttrDict(), AttrSizedOperandSegments(as_property=True))
    q = operand_def(MemRefType)
    kp = operand_def(MemRefType)
    vp = operand_def(MemRefType)
    ko = opt_operand_def(MemRefType)
    vo = opt_operand_def(MemRefType)
    mask = opt_operand_def(MemRefType)
    o = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    heads = prop_def(IntegerAttr)
    kv_heads = prop_def(IntegerAttr)
    n_batch = opt_prop_def(IntegerAttr)
    assembly_format = ("$q `,` $kp `,` $vp (`own` $ko^ `,` $vo)? (`mask` $mask^)? `->` $o attr-dict `:` "
                       "type($q) `,` type($kp) `,` type($vp) (`own` type($ko)^ `,` type($vo))? (`mask` type($mask)^)? `->` type($o)")


@irdl_op_definition
class DumpAllOp(IRDLOperation):
    """Test output: print every element as ``<tag> r c hex`` lines (small tensors, e.g. the 50 x 32 action chunk)."""
    name = "softhier.dump_all"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    index = opt_operand_def(IndexType)      # optional: printed after the tag
    tag = prop_def(StringAttr)
    assembly_format = "$buf (`,` $index^)? attr-dict `:` type($buf)"


@irdl_op_definition
class HbmFillLcgOp(IRDLOperation):
    """Test input: fill with ``scale * randint(lo, hi)`` from the runtime's LCG(seed); the host
    regenerates the same data with softhier_mlir.testing.lcg.fill_fp16."""
    name = "softhier.hbm_fill_lcg"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    seed = prop_def(IntegerAttr)
    lo = prop_def(IntegerAttr)
    hi = prop_def(IntegerAttr)
    scale = prop_def(FloatAttr)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class DumpSamplesOp(IRDLOperation):
    """Test output: print ``n`` fp16 codes at LCG(seed) positions as ``<tag> r c hex`` lines."""
    name = "softhier.dump_samples"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    index = opt_operand_def(IndexType)      # optional: printed after the tag (``L`` + 3 -> ``L3``)
    seed = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    tag = prop_def(StringAttr)
    assembly_format = "$buf (`,` $index^)? attr-dict `:` type($buf)"


@irdl_op_definition
class PreloadWaitOp(IRDLOperation):
    """Wait until the HBM preload image has landed: cluster 0 spins on the 64 B sentinel buffer
    (softhier_mlir.sim.preload.sentinel_array(), the image's last segment), then a global barrier.
    The chip's `hbm_preload_done` only gates on the loader having issued its requests; the data
    arrives at NoC bandwidth afterwards (docs/SIMULATOR_NOTES.md, HBM preload)."""
    name = "softhier.preload_wait"
    irdl_options = (ParsePropInAttrDict(),)
    buf = operand_def(MemRefType)
    assembly_format = "$buf attr-dict `:` type($buf)"


@irdl_op_definition
class MarkOp(IRDLOperation):
    """Timing probe: cluster 0 prints ``[mark] <tag> <mcycle>`` (the global clock is 1 GHz, so the
    difference between two marks is the simulated time in ns; the host unwraps the 32-bit counter).
    Place it after an op that ends with a global barrier."""
    name = "softhier.mark"
    irdl_options = (ParsePropInAttrDict(),)
    index = opt_operand_def(IndexType)      # optional: printed after the tag (``layer`` + 3 -> ``layer3``)
    tag = prop_def(StringAttr)
    assembly_format = "($index^)? attr-dict"


# --------------------------------------------------------------------------- #
# Training ops (test-time adaptation of a LoRA adapter; runtime/sh_train.inc.c, docs/TTT.md). Same conventions as
# the HBM tensor ops; ``cluster = -2`` = the calling cluster (data-parallel programs: every cluster runs its own copy
# on its own buffers, addressed with ``softhier.cluster_id``).
# --------------------------------------------------------------------------- #
@irdl_op_definition
class ClusterIdOp(IRDLOperation):
    """This cluster's id as an ``index`` (per-cluster buffers: ``softhier.hbm_buffer %cid {offset, stride}``)."""
    name = "softhier.cluster_id"
    result = result_def(IndexType)
    assembly_format = "attr-dict `:` type($result)"


@irdl_op_definition
class GemmTransOp(IRDLOperation):
    """``z (+)= op(x) op(w)`` with physically transposed operands: with ``trans_x`` ``x`` is stored as ``K x M``
    (``x^T``), with ``trans_w`` ``w`` is stored as ``N x K``. RedMulE reads row-major operands only, so the library
    first transposes each marked operand into ``scratch`` (HBM, ``(trans_x ? M K : 0) + (trans_w ? K N : 0)``
    elements) and then runs the plain tiled GEMM (``sh_t_gemm_tr``). Tile / pipeline / accumulate / cluster attributes
    as for ``softhier.gemm``."""
    name = "softhier.gemm_t"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    w = operand_def(MemRefType)
    z = operand_def(MemRefType)
    scratch = operand_def(MemRefType)
    assembly_format = "$x `,` $w `into` $z `scratch` $scratch attr-dict `:` type($x) `,` type($w) `,` type($z) `,` type($scratch)"


@irdl_op_definition
class RmsNormBwdOp(IRDLOperation):
    """Backward of ``softhier.rmsnorm`` w.r.t. its input: ``dx = d rmsnorm(x, gamma) / dx ^T dy`` (+ ``res %dres``: the
    residual stream's gradient added in the same pass). fp32 statistics."""
    name = "softhier.rmsnorm_bwd"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    gamma = operand_def(MemRefType)
    dy = operand_def(MemRefType)
    dres = opt_operand_def(MemRefType)
    dx = operand_def(MemRefType)
    eps = prop_def(FloatAttr)
    assembly_format = ("$x `,` $gamma `,` $dy (`res` $dres^)? `->` $dx attr-dict `:` type($x) `,` type($gamma) `,` type($dy) "
                       "(`res` type($dres)^)? `->` type($dx)")


@irdl_op_definition
class SiluMulBwdOp(IRDLOperation):
    """Backward of ``y = silu(a) * b``: ``da = dy b silu'(a)``, ``db = dy silu(a)`` (fp16 SIMD; strided views allowed)."""
    name = "softhier.silu_mul_bwd"
    irdl_options = (ParsePropInAttrDict(),)
    a = operand_def(MemRefType)
    b = operand_def(MemRefType)
    dy = operand_def(MemRefType)
    da = operand_def(MemRefType)
    db = operand_def(MemRefType)
    assembly_format = "$a `,` $b `,` $dy `->` $da `,` $db attr-dict `:` type($a) `,` type($b) `,` type($dy) `->` type($da) `,` type($db)"


@irdl_op_definition
class SoftmaxBwdOp(IRDLOperation):
    """Backward of ``y = softmax(scale x)`` per row: ``dx = scale y (dy - rowdot(y, dy))``."""
    name = "softhier.softmax_bwd"
    irdl_options = (ParsePropInAttrDict(),)
    y = operand_def(MemRefType)
    dy = operand_def(MemRefType)
    dx = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    assembly_format = "$y `,` $dy `->` $dx attr-dict `:` type($y) `,` type($dy) `->` type($dx)"


@irdl_op_definition
class MseGradOp(IRDLOperation):
    """MSE loss gradient: ``dy = gscale (pred - tgt)`` (``gscale`` = 2 / numel x the loss scale); ``loss %l`` (rows x 1
    f32): per-row sums of squares."""
    name = "softhier.mse_grad"
    irdl_options = (ParsePropInAttrDict(),)
    pred = operand_def(MemRefType)
    tgt = operand_def(MemRefType)
    dy = operand_def(MemRefType)
    loss = opt_operand_def(MemRefType)
    gscale = prop_def(FloatAttr)
    assembly_format = "$pred `,` $tgt `->` $dy (`loss` $loss^)? attr-dict `:` type($pred) `,` type($tgt) `->` type($dy) (`loss` type($loss)^)?"


@irdl_op_definition
class OptimStepOp(IRDLOperation):
    """One optimizer step on a flat parameter arena: fp32 master ``w32`` (updated in place) and its fp16 copy ``w16``
    (the forward's operand), gradient ``g`` (f16 or f32) times ``inv_scale``. ``kind = "sgd"``: ``w -= lr g``;
    ``kind = "adam"`` with ``moments %m, %v`` (f32): bias corrections ``bc1 = 1 - b1^t``, ``bc2 = 1 - b2^t``."""
    name = "softhier.optim_step"
    irdl_options = (ParsePropInAttrDict(), AttrSizedOperandSegments(as_property=True))
    g = operand_def(MemRefType)
    w32 = operand_def(MemRefType)
    w16 = operand_def(MemRefType)
    m = opt_operand_def(MemRefType)
    v = opt_operand_def(MemRefType)
    kind = prop_def(StringAttr)
    lr = prop_def(FloatAttr)
    inv_scale = prop_def(FloatAttr)
    assembly_format = ("$g `,` $w32 `->` $w16 (`moments` $m^ `,` $v)? attr-dict `:` type($g) `,` type($w32) `->` type($w16) "
                       "(`moments` type($m)^ `,` type($v))?")


@irdl_op_definition
class AttentionBwdOp(IRDLOperation):
    """Backward of ``softhier.cross_attention`` (the prefix ``kp``/``vp`` are frozen): ``dq`` for every query head from
    the forward's ``q``, ``o`` and the output gradient ``do``; with ``own %ko, %vo`` also ``grads %dko, %dvo`` (summed
    over each kv group; ``scratch %s``: Sq x 2 heads dh per-head partials). Scores are recomputed in TCDM."""
    name = "softhier.attention_bwd"
    irdl_options = (ParsePropInAttrDict(), AttrSizedOperandSegments(as_property=True))
    q = operand_def(MemRefType)
    kp = operand_def(MemRefType)
    vp = operand_def(MemRefType)
    ko = opt_operand_def(MemRefType)
    vo = opt_operand_def(MemRefType)
    mask = opt_operand_def(MemRefType)
    o = operand_def(MemRefType)
    do = operand_def(MemRefType)
    dq = operand_def(MemRefType)
    dko = opt_operand_def(MemRefType)
    dvo = opt_operand_def(MemRefType)
    scratch = opt_operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    heads = prop_def(IntegerAttr)
    kv_heads = prop_def(IntegerAttr)
    assembly_format = ("$q `,` $kp `,` $vp (`own` $ko^ `,` $vo)? (`mask` $mask^)? `,` $o `,` $do `->` $dq "
                       "(`grads` $dko^ `,` $dvo `scratch` $scratch)? attr-dict `:` type($q) `,` type($kp) `,` type($vp) "
                       "(`own` type($ko)^ `,` type($vo))? (`mask` type($mask)^)? `,` type($o) `,` type($do) `->` type($dq) "
                       "(`grads` type($dko)^ `,` type($dvo) `scratch` type($scratch))?")


@irdl_op_definition
class GradAllReduceOp(IRDLOperation):
    """Data-parallel gradient sum over all clusters with the NoC's in-network REDADD: cluster c's gradients are the
    ``src`` buffer shifted by ``c * src_stride`` bytes, ``dst`` receives the sum. ``mode = 0``: fp16 REDADD (dst f16);
    ``mode = 1``: exact two-limb integer REDADD with one global power-of-two scale (dst f32, ``scal`` = 64 B scratch)."""
    name = "softhier.grad_allreduce"
    irdl_options = (ParsePropInAttrDict(),)
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    scal = operand_def(MemRefType)
    src_stride = prop_def(IntegerAttr)
    mode = prop_def(IntegerAttr)
    assembly_format = "$src `->` $dst `,` $scal attr-dict `:` type($src) `->` type($dst) `,` type($scal)"


@irdl_op_definition
class LoraFwdOp(IRDLOperation):
    """LoRA branch of a linear layer, accumulated into its output: ``y += s (x A) B`` with ``t = s x A`` (M x r) kept for
    the backward (``sh_t_lora_fwd``; ``scale`` = s, ``tile_k`` = the K tile of x A)."""
    name = "softhier.lora_fwd"
    irdl_options = (ParsePropInAttrDict(),)
    x = operand_def(MemRefType)
    a = operand_def(MemRefType)
    b = operand_def(MemRefType)
    y = operand_def(MemRefType)
    t = operand_def(MemRefType)
    scale = prop_def(FloatAttr)
    assembly_format = "$x `,` $a `,` $b `->` $y `,` $t attr-dict `:` type($x) `,` type($a) `,` type($b) `->` type($y) `,` type($t)"


@irdl_op_definition
class LinearBwdOp(IRDLOperation):
    """Backward of ``y = x W`` (W: K x N, [in, out]) w.r.t. x, plus with ``lora %a, %b, %t, %x grads %da, %db`` the LoRA
    ``y[:, :nl] += s (x A) B`` (nl = columns of B, ``t`` = the forward's s x A): ``dx = dy W^T + s (dy B^T) A^T``,
    ``dB = t^T dy``, ``dA = s x^T (dy B^T)``. The frozen-weight product is formed transposed (``dx^T = W dy^T``) so W is
    read in its stored layout; ``scratch`` holds the transposed activations (``sh_t_linear_bwd``). ``tile_m`` / ``tile_k``:
    tiles of ``W dy^T`` (K and N)."""
    name = "softhier.linear_bwd"
    irdl_options = (ParsePropInAttrDict(), AttrSizedOperandSegments(as_property=True))
    dy = operand_def(MemRefType)
    w = operand_def(MemRefType)
    a = opt_operand_def(MemRefType)
    b = opt_operand_def(MemRefType)
    t = opt_operand_def(MemRefType)
    x = opt_operand_def(MemRefType)
    da = opt_operand_def(MemRefType)
    db = opt_operand_def(MemRefType)
    dx = operand_def(MemRefType)
    scratch = operand_def(MemRefType)
    assembly_format = ("$dy `,` $w (`lora` $a^ `,` $b `,` $t `,` $x `grads` $da `,` $db)? `->` $dx `scratch` $scratch attr-dict `:` "
                       "type($dy) `,` type($w) (`lora` type($a)^ `,` type($b) `,` type($t) `,` type($x) `grads` type($da) `,` type($db))? "
                       "`->` type($dx) `scratch` type($scratch)")


@irdl_op_definition
class CopyOp(IRDLOperation):
    """``dst = src`` for HBM tensors (strided views allowed): per-row 1-D DMA on the DM cores, rows dealt over the
    clusters (``sh_t_copy``). Training programs seed an accumulating GEMM with it (residual / bias + x W)."""
    name = "softhier.copy"
    irdl_options = (ParsePropInAttrDict(),)
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    assembly_format = "$src `->` $dst attr-dict `:` type($src) `->` type($dst)"


SoftHier = Dialect(
    "softhier",
    [
        MarkOp,
        PreloadWaitOp,
        HbmBufferOp,
        L1BufferOp,
        RedmuleOp,
        Dma2DOp,
        DmaBroadcastOp,
        DmaReduceOp,
        GroupBarrierOp,
        ClusterPosOp,
        TransposeOp,
        VExpOp,
        ReluOp,
        L1ZeroOp,
        L1AddOp,
        L1FillOp,
        CheckConstOp,
        GemmOp,
        HbmFillOp,
        HbmFillColParityOp,
        HbmCheckConstOp,
        ViewOp,
        LayerNormOp,
        SoftmaxOp,
        GeluOp,
        AddOp,
        AddBiasOp,
        AttentionOp,
        AxpyOp,
        CrossAttentionOp,
        DumpAllOp,
        HbmFillLcgOp,
        DumpSamplesOp,
        RmsNormOp,
        RopeOp,
        SiluMulOp,
        ScaleOp,
        PixelShuffleOp,
        ClusterIdOp,
        GemmTransOp,
        RmsNormBwdOp,
        SiluMulBwdOp,
        SoftmaxBwdOp,
        MseGradOp,
        OptimStepOp,
        AttentionBwdOp,
        GradAllReduceOp,
        LoraFwdOp,
        LinearBwdOp,
        CopyOp,
    ],
    [],
)
