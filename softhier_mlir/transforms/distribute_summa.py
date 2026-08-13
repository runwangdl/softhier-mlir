"""Distribute `softhier.gemm` across the cluster mesh with SUMMA.

Marks each `softhier.gemm` with a ``summa`` attribute; the backend then emits the
output-stationary SUMMA schedule (cluster (px,py) owns output tile Z[py,px];
diagonal clusters load X/W panels from the HBM edges and broadcast along their
mesh row/column; per-cluster RedMule accumulates over K; group barriers).

Assumes M = N = P*256 (one 256-tile per cluster, P = mesh dim). Combine with
`pipeline-gemm` for the K-loop modulo scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, UnitAttr
from xdsl.passes import ModulePass

from softhier_mlir.dialects.softhier import GemmOp


@dataclass(frozen=True)
class DistributeSumma(ModulePass):
    name = "distribute-summa"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for o in op.walk():
            if isinstance(o, GemmOp):
                o.attributes["summa"] = UnitAttr()
