"""Lower `linalg` compute ops to the `softhier` accelerator dialect.

Currently: ``linalg.matmul`` (buffer/memref form) -> ``softhier.l1_zero`` +
``softhier.redmule`` (RedMule accumulates, so the output tile is cleared first).

This is the compiler bridge: a network's GEMMs can be written in standard MLIR
``linalg`` and lowered onto SoftHier's RedMule engine, then carried to C by
``softhier-translate``.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import Float16Type, ModuleOp, StringAttr
from xdsl.dialects.linalg.ops import MatmulOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from softhier_mlir.dialects.softhier import GemmOp, L1ZeroOp, RedmuleOp


def _fmt(elem_type) -> str:
    # RedMule datapath by element type; fp16 is the default/verified path.
    if isinstance(elem_type, Float16Type):
        return "fp16"
    return "fp16"


def _space(memref) -> str:
    return getattr(memref.memory_space, "data", "tcdm")


class LowerMatmul(RewritePattern):
    """Lower ``linalg.matmul`` by where its operands live:

    - output in an HBM space  -> ``softhier.gemm`` (backend tiles it, any size)
    - output in TCDM          -> ``softhier.l1_zero`` + ``softhier.redmule``
      (a single 256x256x256 RedMule tile)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        x, w = op.inputs
        y = op.outputs[0]
        fmt = _fmt(y.type.element_type)
        if _space(y.type).startswith("hbm"):
            gemm = GemmOp(operands=[x, w, y], properties={"fmt": StringAttr(fmt)})
            rewriter.replace_matched_op([gemm])
        else:
            zero = L1ZeroOp(operands=[y])
            redmule = RedmuleOp(operands=[x, w, y], properties={"fmt": StringAttr(fmt)})
            rewriter.replace_matched_op([zero, redmule])


@dataclass(frozen=True)
class LinalgToSoftHier(ModulePass):
    name = "linalg-to-softhier"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerMatmul()).rewrite_module(op)
