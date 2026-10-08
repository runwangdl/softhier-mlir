#!/usr/bin/env python3
"""Run the on-simulator tests of the softhier-ops library.

    python tests/gvsoc/run.py gemm                 # default shape set
    python tests/gvsoc/run.py gemm --shapes 256x256x256 512x768x768:256,256,256
    python tests/gvsoc/run.py mlir examples/gemm512_linalg.mlir -p linalg-to-softhier
    python tests/gvsoc/run.py mlir examples/*.mlir          # every example, auto passes

Each case writes tests/gvsoc/<test>/shape.h, builds the SDK app in the x86 chroot and runs
GVSoC natively (ideal HBM). Prints PASS/FAIL and the ROI in ns.
"""
from __future__ import annotations

import argparse
import subprocess
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


# passes each example needs (none = already in the softhier dialect)
EXAMPLE_PASSES = {
    "gemm512_linalg.mlir": "linalg-to-softhier",
    "mlp_linalg.mlir": "linalg-to-softhier",
    "gemm1024_summa.mlir": "linalg-to-softhier,distribute-summa,pipeline-gemm",
}


def lower_and_translate(mlir: Path, passes: str | None) -> str:
    """softhier-opt [-p passes] | softhier-translate -> C source."""
    py = sys.executable
    src = mlir.read_text()
    if passes:
        r = subprocess.run([py, "-m", "softhier_mlir.tools.softhier_opt", str(mlir), "-p", passes],
                           capture_output=True, text=True, check=True)
        src = r.stdout
    r = subprocess.run([py, "-m", "softhier_mlir.tools.softhier_translate", "/dev/stdin"],
                       input=src, capture_output=True, text=True, check=True)
    return r.stdout


def run_mlir(files: list[str], passes: str | None) -> bool:
    app = HERE / "mlir_app"
    all_ok = True
    for f in files:
        mlir = Path(f)
        p = passes if passes is not None else EXAMPLE_PASSES.get(mlir.name)
        (app / "main.c").write_text(lower_and_translate(mlir, p))
        build_sw(app)
        r = run_sim()
        lines = [ln for ln in r["stdout"].splitlines() if "_CHECK" in ln or "[sh_" in ln]
        checks = [ln for ln in lines if "_CHECK" in ln]
        # examples without a self-check (pure timing runs) pass when the simulation completes
        ok = r["ok"] and all("_PASS" in ln for ln in checks) and not any("[sh_" in ln for ln in lines)
        all_ok &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {mlir.name:<26} passes={p or '-':<34} roi={r['roi_ns']} ns wall={r['wall_s']}s")
        for ln in lines:
            print("     " + ln)
        if not r["ok"]:
            print(r["stdout"][-1500:])
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["gemm", "mlir"])
    ap.add_argument("files", nargs="*", help="mlir: input .mlir files")
    ap.add_argument("-p", "--passes", help="mlir: pass pipeline for softhier-opt (default: per-example table)")
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--nsamples", type=int, default=256)
    a = ap.parse_args()
    if a.test == "gemm":
        ok = run_gemm(a.shapes or DEFAULT_GEMM, a.nsamples)
    else:
        ok = run_mlir(a.files, a.passes)
    sys.exit(0 if ok else 1)
