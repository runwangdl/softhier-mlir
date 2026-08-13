"""`softhier-opt` — an xDSL opt driver with the `softhier` dialect registered.

Usage:
    softhier-opt input.mlir            # parse + verify + reprint
    softhier-opt input.mlir -p <pass>  # run a pass pipeline
"""

from __future__ import annotations

from xdsl.xdsl_opt_main import xDSLOptMain

from softhier_mlir.dialects.softhier import SoftHier


class SoftHierOptMain(xDSLOptMain):
    def register_all_dialects(self) -> None:
        super().register_all_dialects()
        self.ctx.load_dialect(SoftHier)


def main() -> None:
    SoftHierOptMain().run()


if __name__ == "__main__":
    main()
