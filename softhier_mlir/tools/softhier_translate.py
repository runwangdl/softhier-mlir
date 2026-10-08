"""`softhier-translate` — lower a `softhier` .mlir module to a runnable C file.

Usage:
    softhier-translate input.mlir [-o main.c] [--preload-elf preload.elf]

--preload-elf: the module's test inputs (`softhier.hbm_fill_lcg` / `hbm_fill` / `hbm_fill_col_parity`)
are generated on the host and written into that HBM preload image (`gvsoc ... --preload`); the
program only waits for it (softhier_mlir/sim/testdata.py). Without it they are generated on the device.
"""

from __future__ import annotations

import argparse
import sys

from xdsl.context import Context
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.parser import Parser

from softhier_mlir.backend.emit_c import emit_c
from softhier_mlir.dialects.softhier import SoftHier


def main() -> None:
    ap = argparse.ArgumentParser(description="softhier dialect -> C (flex_ runtime)")
    ap.add_argument("input", help="input .mlir file")
    ap.add_argument("-o", "--output", help="output .c file (default: stdout)")
    ap.add_argument("--preload-elf", help="write the test inputs into this HBM preload image instead of generating "
                    "them on the device (the file is not written when some fill cannot be preloaded)")
    args = ap.parse_args()

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(SoftHier)

    with open(args.input) as f:
        module = Parser(ctx, f.read(), args.input).parse_module()

    if args.preload_elf:
        from pathlib import Path

        from softhier_mlir.sim.testdata import extract_preload, write_preload
        Path(args.preload_elf).unlink(missing_ok=True)
        arrays = extract_preload(module)
        if arrays:
            write_preload(args.preload_elf, arrays)
    c_src = emit_c(module)
    if args.output:
        with open(args.output, "w") as f:
            f.write(c_src)
    else:
        sys.stdout.write(c_src)


if __name__ == "__main__":
    main()
