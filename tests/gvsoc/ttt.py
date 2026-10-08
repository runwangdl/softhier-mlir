#!/usr/bin/env python3
"""On-simulator tests of test-time adaptation (LoRA on the SmolVLA action expert; softhier_mlir/frontend/smolvla_ttt.py,
docs/TTT.md). Every check is against a float64 torch / numpy reference.

    python tests/gvsoc/ttt.py ops                       # step 1: each backward primitive / optimizer once, timed
    python tests/gvsoc/ttt.py layer [--opt sgd|adam]    # step 2: a self + a cross layer fwd + bwd + update, LCG data
    python tests/gvsoc/ttt.py expert --layers 16        # step 3: the full expert, real weights, MSE on the velocity
    python tests/gvsoc/ttt.py reduce                    # step 3: data-parallel REDADD of 16 per-sample gradient arenas

Runs write tests/gvsoc/ttt_app/{ttt.mlir, main.c, preload.elf, last_run.log}.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.testing import lcg  # noqa: E402
from tests.gvsoc.expert import _build_and_run, marks_seq  # noqa: E402

HERE = Path(__file__).resolve().parent


def _app(app_dir: Path) -> Path:
    """The SDK app dir; the library is built without the mesh SUMMA (unused, 4.5 KB of the 64 KB instruction memory)."""
    app = Path(app_dir)
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c -DSH_NO_GEMM_MESH PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    return app
F32_RE = re.compile(r"^(\S+) (\d+) (\d+) ([0-9a-fA-F]{8})$")


def parse_all(stdout: str) -> dict:
    out = lcg.parse_samples(stdout)
    for ln in stdout.splitlines():
        m = F32_RE.match(ln.strip())
        if m:
            out.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), struct.unpack(">f", bytes.fromhex(m.group(4)))[0]))
    return out


def cmp(tag: str, got: list, ref: np.ndarray, rtol: float) -> tuple[bool, float, float]:
    """max abs error of the samples and the error relative to the reference's max |value|"""
    d = np.array([v - float(ref[r, c]) for r, c, v in got]); w = np.array([float(ref[r, c]) for r, c, _ in got])
    err = float(np.abs(d).max())
    scale = float(np.abs(ref).max()) or 1.0
    l2 = float(np.linalg.norm(d) / (np.linalg.norm(w) or 1.0))
    ok = err <= rtol * scale
    print(f"     {tag:<8} n={len(got):<4} max abs err {err:.3e}  |ref| max {scale:.3e}  rel {err / scale:.2e}  rel-L2 {l2:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok, err, err / scale


def timing(stdout: str) -> list[tuple[str, int]]:
    m = marks_seq(stdout)
    return [(t, b - a) for (_, a), (t, b) in zip(m, m[1:])]


def run_ops(seed: int, nsamples: int) -> bool:
    from softhier_mlir.frontend import smolvla_ttt as TT
    app = _app(HERE / "ttt_app")
    mlir, pre, ref = TT.emit_ops_test(seed, -1, nsamples)
    r = _build_and_run(app, mlir, pre, 3600, None)
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} ttt ops roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for t, dt in timing(r["stdout"]):
        print(f"     time {t:<16} {dt / 1e3:9.1f} us")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    got = parse_all(r["stdout"])
    tol = {"DA": 2e-3, "UB": 2e-3, "WBIGT": 0.0, "O": 2e-2, "DQ": 3e-2, "DQX": 3e-2, "DGU": 1e-2}
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<8} MISSING"); ok = False; continue
        ok &= cmp(tag, got[tag], arr, tol.get(tag, 5e-3))[0]
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


def _grad_report(got: dict, ref: dict, layers_dumped, scale: float, lr: float, rtol: float) -> bool:
    from softhier_mlir.frontend import smolvla_ttt as TT
    ok = True
    rows = []
    for L in layers_dumped:
        for nm, _, _ in TT.LORA:
            gref = TT.lora_tensor(ref["grad"], L, nm) * scale
            wref = TT.lora_tensor(ref["w_new"], L, nm)
            tg, tw = f"G{nm.upper()}{L}", f"W{nm.upper()}{L}"
            if tg not in got or tw not in got:
                print(f"     {tg} / {tw} MISSING"); ok = False; continue
            g_ok, g_err, g_rel = cmp(tg, got[tg], gref, rtol)
            w0 = TT.lora_tensor(ref["w0"], L, nm)
            upd = wref - w0
            w_err = max(abs(v - float(wref[r, c])) for r, c, v in got[tw])
            u_err = max(abs((v - float(w0[r, c])) - float(upd[r, c])) for r, c, v in got[tw])
            print(f"     {tw:<8} updated weight max abs err {w_err:.3e} (|w| max {np.abs(wref).max():.3e}); update lr*g err {u_err:.3e} "
                  f"of |update| max {np.abs(upd).max():.3e}")
            rows.append((L, nm, g_err, g_rel, w_err))
            ok &= g_ok
    return ok


def run_layer(seed: int, nsamples: int, opt: str, lr: float) -> bool:
    from softhier_mlir.frontend import smolvla_ttt as TT
    layers = 2
    W = TT.lcg_weights(layers, seed)
    lora = TT.lora_init(layers, seed)
    h_in = lcg.fill_fp16(TT.S, TT.D, seed * 100 + 90, -16, 16, 0.125)
    dh_in = lcg.fill_fp16(TT.S, TT.D, seed * 100 + 91, -8, 8, 1 / 16)
    app = _app(HERE / "ttt_app")
    mlir, pre, info = TT.emit_ttt(W, layers, False, lora, lr=lr, opt=opt, nsamples=nsamples, profile=True, dump_layers=(0, 1),
                                  h_in=h_in, dh_in=dh_in)
    ref = TT.torch_ttt(W, layers, False, lora, lr=lr, opt=opt, h_in=h_in, dh_in=dh_in)
    ref["w0"] = lora.astype(np.float64)
    r = _build_and_run(app, mlir, pre, 7200, None)
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} ttt layer (self + cross) opt={opt} lr={lr} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    for t, dt in timing(r["stdout"]):
        print(f"     time {t:<10} {dt / 1e3:9.1f} us")
    got = parse_all(r["stdout"])
    ok &= _grad_report(got, ref, (0, 1), 1.0, lr, 3e-2)
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["ops", "layer", "expert", "reduce"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--opt", default="sgd")
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()
    if a.test == "ops":
        ok = run_ops(a.seed, a.nsamples)
    elif a.test == "layer":
        ok = run_layer(a.seed, a.nsamples, a.opt, a.lr)
    else:
        raise SystemExit("not yet")
    sys.exit(0 if ok else 1)
