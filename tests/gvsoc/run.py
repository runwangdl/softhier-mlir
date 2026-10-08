#!/usr/bin/env python3
"""Run the on-simulator tests of the softhier-ops library.

    python tests/gvsoc/run.py gemm                 # default shape set
    python tests/gvsoc/run.py gemm --shapes 256x256x256 512x768x768:256,256,256

Each case writes tests/gvsoc/<test>/shape.h, builds the SDK app in the x86 chroot and runs
GVSoC natively (ideal HBM). Prints PASS/FAIL and the ROI in ns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.sim.gvsoc import build_sw, run_sim  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_GEMM = ["256x256x256", "256x768x192:256,192,256", "512x768x768:256,256,256",
                "256x256x256:256,256,256,0", "256x256x512:256,256,256,1,1"]


def parse_shape(s: str) -> dict:
    """MxNxK[:tm,tn,tk[,pipeline[,accumulate]]]"""
    dims, _, rest = s.partition(":")
    m, n, k = (int(v) for v in dims.lower().split("x"))
    opts = [int(v) for v in rest.split(",")] if rest else []
    tm, tn, tk = (opts + [0, 0, 0])[:3]
    pipe = opts[3] if len(opts) > 3 else 1
    acc = opts[4] if len(opts) > 4 else 0
    return dict(M=m, N=n, K=k, tm=tm, tn=tn, tk=tk, pipeline=pipe, accumulate=acc)


def run_gemm(shapes: list[str], nsamples: int = 256) -> bool:
    app = HERE / "gemm"
    all_ok = True
    for s in shapes:
        c = parse_shape(s)
        (app / "shape.h").write_text(
            f"#define GEMM_M {c['M']}\n#define GEMM_N {c['N']}\n#define GEMM_K {c['K']}\n"
            f"#define TILE_M {c['tm']}\n#define TILE_N {c['tn']}\n#define TILE_K {c['tk']}\n"
            f"#define PIPELINE {c['pipeline']}\n#define ACCUMULATE {c['accumulate']}\n"
            f"#define NSAMPLES {nsamples}\n")
        build_sw(app)
        r = run_sim()
        lines = [ln for ln in r["stdout"].splitlines() if ln.startswith("[gemm]") or "mismatch" in ln]
        ok = r["ok"] and any("GEMM_PASS" in ln for ln in lines)
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {s:<28} roi={r['roi_ns']} ns wall={r['wall_s']}s")
        for ln in lines:
            print("     " + ln)
        if not r["ok"]:
            print(r["stdout"][-1500:])
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["gemm"])
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--nsamples", type=int, default=256)
    a = ap.parse_args()
    ok = run_gemm(a.shapes or DEFAULT_GEMM, a.nsamples)
    sys.exit(0 if ok else 1)
