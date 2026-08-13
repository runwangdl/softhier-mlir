"""`softhier-translate` — lower a `softhier` .mlir module to a runnable C file.

Usage:
    softhier-translate input.mlir [-o main.c]
"""

from __future__ import annotations

import argparse
import sys

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.parser import Parser

from softhier_mlir.backend.emit_c import emit_c
from softhier_mlir.dialects.softhier import SoftHier


def main() -> None:
    ap = argparse.ArgumentParser(description="softhier dialect -> C (flex_ runtime)")
    ap.add_argument("input", help="input .mlir file")
    ap.add_argument("-o", "--output", help="output .c file (default: stdout)")
    args = ap.parse_args()

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SoftHier)

    with open(args.input) as f:
        module = Parser(ctx, f.read(), args.input).parse_module()

    c_src = emit_c(module)
    if args.output:
        with open(args.output, "w") as f:
            f.write(c_src)
    else:
        sys.stdout.write(c_src)


if __name__ == "__main__":
    main()
