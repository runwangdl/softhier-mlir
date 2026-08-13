"""`softhier-opt` — an xDSL opt driver with the `softhier` dialect registered.

Usage:
    softhier-opt input.mlir            # parse + verify + reprint
    softhier-opt input.mlir -p <pass>  # run a pass pipeline
"""

from __future__ import annotations

from xdsl.xdsl_opt_main import xDSLOptMain

from softhier_mlir.dialects.softhier import SoftHier
from softhier_mlir.transforms.distribute_summa import DistributeSumma
from softhier_mlir.transforms.linalg_to_softhier import LinalgToSoftHier
from softhier_mlir.transforms.pipeline_gemm import PipelineGemm


class SoftHierOptMain(xDSLOptMain):
    def register_all_dialects(self) -> None:
        super().register_all_dialects()
        self.ctx.load_dialect(SoftHier)  # linalg et al. already registered by base

    def register_all_passes(self) -> None:
        super().register_all_passes()
        self.register_pass(LinalgToSoftHier.name, lambda: LinalgToSoftHier)
        self.register_pass(PipelineGemm.name, lambda: PipelineGemm)
        self.register_pass(DistributeSumma.name, lambda: DistributeSumma)


def main() -> None:
    SoftHierOptMain().run()


if __name__ == "__main__":
    main()
