"""Software-pipeline `softhier.gemm` tile loops (HLS-style modulo scheduling).

This is the *decide* half of a decide/apply split: it marks which GEMMs should
be software-pipelined by attaching a ``pipeline`` unit attribute. The backend
(*apply*) then expands a marked gemm into a double-buffered loop-nest that
prefetches K-tile k+1 (DM core) while RedMule computes K-tile k (first core), so
the per-step wall time is ``max(t_dma, t_redmule)`` instead of their sum.

Today the policy is "pipeline every gemm". A future version can replace it with
an ILP that picks which loops to pipeline and how deep to buffer, subject to the
TCDM budget and the RedMule/iDMA resource limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, UnitAttr
from xdsl.passes import ModulePass

from softhier_mlir.dialects.softhier import GemmOp


@dataclass(frozen=True)
class PipelineGemm(ModulePass):
    name = "pipeline-gemm"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for o in op.walk():
            if isinstance(o, GemmOp):
                o.attributes["pipeline"] = UnitAttr()
