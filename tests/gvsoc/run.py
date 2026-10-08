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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.sim.gvsoc import build_sw, run_sim  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_GEMM = ["256x256x256", "256x768x192:256,256,192", "512x768x768:256,256,256",
                "256x256x256:256,256,256,0", "256x256x512:256,256,256,1,1",
                "1024x768x768:256,256,256,1,0,all", "1024x3072x768:256,256,256,1,0,all"]


def parse_shape(s: str) -> dict:
    """MxNxK[:tm,tn,tk[,pipeline[,accumulate[,cluster]]]]   cluster = 0 | all"""
    dims, _, rest = s.partition(":")
    m, n, k = (int(v) for v in dims.lower().split("x"))
    opts = rest.split(",") if rest else []
    tm, tn, tk = (int(v) for v in (opts + ["0", "0", "0"])[:3])
    pipe = int(opts[3]) if len(opts) > 3 else 1
    acc = int(opts[4]) if len(opts) > 4 else 0
    cluster = "SH_ALL" if len(opts) > 5 and opts[5] == "all" else "0"
    return dict(M=m, N=n, K=k, tm=tm, tn=tn, tk=tk, pipeline=pipe, accumulate=acc, cluster=cluster)


def run_gemm(shapes: list[str], nsamples: int = 256, real: bool = False) -> bool:
    app = HERE / "gemm"
    all_ok = True
    for s in shapes:
        c = parse_shape(s)
        (app / "shape.h").write_text(
            f"#define GEMM_M {c['M']}\n#define GEMM_N {c['N']}\n#define GEMM_K {c['K']}\n"
            f"#define TILE_M {c['tm']}\n#define TILE_N {c['tn']}\n#define TILE_K {c['tk']}\n"
            f"#define PIPELINE {c['pipeline']}\n#define ACCUMULATE {c['accumulate']}\n"
            f"#define CLUSTER {c['cluster']}\n#define NSAMPLES {nsamples}\n" + ("#define REAL_DATA 1\n" if real else ""))
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


def run_rowops(rows: int, cols: int, cluster: str, nsamples: int = 64) -> bool:
    from softhier_mlir.testing import lcg
    app = HERE / "rowops"
    (app / "shape.h").write_text(f"#define ROWS {rows}\n#define COLS {cols}\n#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n")
    build_sw(app)
    r = run_sim()
    x = lcg.fill_fp16(rows, cols, 11, -16, 16, 0.125).astype(np.float32)
    b = lcg.fill_fp16(rows, cols, 12, -16, 16, 0.125).astype(np.float32)
    g = lcg.fill_fp16(1, cols, 13, 1, 8, 0.25).astype(np.float32)
    be = lcg.fill_fp16(1, cols, 14, -4, 4, 0.25).astype(np.float32)
    mean = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
    ref = {
        "LN": (x - mean) / np.sqrt(var + 1e-5) * g + be,
        "SM": (lambda e: e / e.sum(1, keepdims=True))(np.exp(0.5 * x - (0.5 * x).max(1, keepdims=True))),
        "GELU": 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
        "ADD": x + b, "BIAS": x + be, "SCALE": x * np.float32(0.3), "T": x.T,
    }
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "ROWOPS_DONE" in r["stdout"]
    print(f"{'PASS' if ok else 'FAIL'} rowops {rows}x{cols} cluster={cluster} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=2e-2, rtol=2e-2)
        print(f"     {tag:<6} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def run_siglip(seq: int, d: int, ff: int, heads: int, cluster: str, nsamples: int = 64) -> bool:
    from softhier_mlir.testing import lcg, siglip_ref
    app = HERE / "siglip_layer"
    (app / "shape.h").write_text(f"#define SEQ {seq}\n#define D_MODEL {d}\n#define D_FF {ff}\n#define N_HEADS {heads}\n"
                                 f"#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n")
    build_sw(app)
    r = run_sim(timeout=7200)
    ref = siglip_ref.layer_reference(seq, d, ff, heads)
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "SIGLIP_LAYER_DONE" in r["stdout"]
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_") or ln.startswith("[head"):
            print("     " + ln)
    (HERE / "siglip_layer" / "last_run.log").write_text(r["stdout"])
    macs = 4 * seq * d * d + 2 * seq * d * ff + 2 * seq * seq * d
    print(f"{'PASS' if ok else 'FAIL'} siglip_layer S={seq} D={d} F={ff} H={heads} cluster={cluster} "
          f"roi={r['roi_ns']} ns ({macs / 1e6:.0f} MMAC, {macs / r['roi_ns'] if r['roi_ns'] else 0:.0f} MAC/ns) wall={r['wall_s']}s")
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<4} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=3)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def run_siglip_mlir(seq: int, d: int, ff: int, heads: int, cluster: str, layers: int = 1, nsamples: int = 64, fused: bool = False) -> bool:
    """Frontend -> softhier-translate -> gvsoc, compared against the same numpy reference as `siglip`."""
    from softhier_mlir.frontend import siglip
    from softhier_mlir.testing import lcg, siglip_ref
    app = HERE / "mlir_app"
    mlir = siglip.emit(seq, d, ff, heads, layers, -1 if cluster == "SH_ALL" else int(cluster), True, nsamples=nsamples, fused_attention=fused)
    (app / "siglip.mlir").write_text(mlir)
    (app / "main.c").write_text(lower_and_translate(app / "siglip.mlir", None))
    build_sw(app)
    r = run_sim(timeout=7200)
    ref = siglip_ref.layer_reference(seq, d, ff, heads)
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} siglip-mlir S={seq} D={d} F={ff} H={heads} L={layers} cluster={cluster} attention={'fused' if fused else 'per-head'} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    for tag, arr in ref.items():
        if tag not in got:
            continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=2)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    (app / "last_run.log").write_text(r["stdout"])
    return ok


def attention_reference(S: int, D: int, H: int, scale: float = 0.125) -> np.ndarray:
    """numpy twin of tests/gvsoc/attention/main.c (fp32 math, fp16 rounding where the device stores fp16)."""
    from softhier_mlir.testing import lcg
    f = lambda *a, **k: lcg.fill_fp16(*a, **k).astype(np.float32)  # noqa: E731
    r16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    q = f(S, D, 21, -8, 8, 0.125); k = f(S, D, 22, -8, 8, 0.125); v = f(S, D, 23, -16, 16, 0.125)
    dh = D // H
    o = np.zeros((S, D), np.float32)
    for hd in range(H):
        sl = slice(hd * dh, (hd + 1) * dh)
        s = r16(q[:, sl] @ k[:, sl].T) * scale
        p = np.exp(s - s.max(1, keepdims=True)); p = r16(p / p.sum(1, keepdims=True))
        o[:, sl] = r16(p @ v[:, sl])
    return o


def run_attention(seq: int, d: int, heads: int, cluster: str, composed: bool, nsamples: int = 128) -> bool:
    """Fused sh_attention (or the composed transpose+gemm+softmax+gemm path) vs numpy; prints the ROI."""
    from softhier_mlir.testing import lcg
    app = HERE / "attention"
    (app / "shape.h").write_text(f"#define SEQ {seq}\n#define D_MODEL {d}\n#define N_HEADS {heads}\n"
                                 f"#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n#define COMPOSED {1 if composed else 0}\n")
    build_sw(app)
    r = run_sim(timeout=7200)
    ref = attention_reference(seq, d, heads)
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "ATTENTION_DONE" in r["stdout"] and "ATTENTION_FAIL" not in r["stdout"]
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_") or ln.startswith("[attention]"):
            print("     " + ln)
    dh = d // heads
    print(f"{'PASS' if ok else 'FAIL'} attention S={seq} D={d} H={heads} dh={dh} cluster={cluster} path={'composed' if composed else 'fused'} "
          f"roi={r['roi_ns']} ns (1 GHz: {r['roi_ns']} cycles) wall={r['wall_s']}s")
    for tag, arr in (("O0", ref[:, :dh]), ("O", ref)):
        if tag not in got:
            print(f"     {tag:<4} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=3)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    (app / "last_run.log").write_text(r["stdout"])
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


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
    ap.add_argument("test", choices=["gemm", "mlir", "rowops", "siglip", "siglip-mlir", "attention"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--ff", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--cluster", default="0", help="rowops: 0 or all")
    ap.add_argument("files", nargs="*", help="mlir: input .mlir files")
    ap.add_argument("-p", "--passes", help="mlir: pass pipeline for softhier-opt (default: per-example table)")
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--real", action="store_true", help="gemm: real-valued data instead of small ints")
    ap.add_argument("--composed", action="store_true", help="attention: the per-head library-call path instead of the fused kernel")
    ap.add_argument("--fused", action="store_true", help="siglip-mlir: use the fused softhier.attention op")
    a = ap.parse_args()
    if a.test == "gemm":
        ok = run_gemm(a.shapes or DEFAULT_GEMM, a.nsamples, a.real)
    elif a.test == "rowops":
        ok = run_rowops(a.rows, a.cols, "SH_ALL" if a.cluster == "all" else "0")
    elif a.test == "siglip":
        ok = run_siglip(a.seq, a.d, a.ff, a.heads, "SH_ALL" if a.cluster == "all" else "0")
    elif a.test == "attention":
        ok = run_attention(a.seq, a.d, a.heads, "SH_ALL" if a.cluster == "all" else a.cluster, a.composed, a.nsamples)
    elif a.test == "siglip-mlir":
        ok = run_siglip_mlir(a.seq, a.d, a.ff, a.heads, "SH_ALL" if a.cluster == "all" else a.cluster, a.layers, fused=a.fused)
    else:
        ok = run_mlir(a.files, a.passes)
    sys.exit(0 if ok else 1)
