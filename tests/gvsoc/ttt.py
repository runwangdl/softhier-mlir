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


def _app(app_dir: Path, defines: str = "") -> Path:
    """The SDK app dir; the library is built without the mesh SUMMA (unused, 4.5 KB of the 64 KB instruction memory)."""
    app = Path(app_dir)
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c -DSH_NO_GEMM_MESH{defines} PARENT_SCOPE)\n"
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
            if tg not in got:
                print(f"     {tg} MISSING"); ok = False; continue
            g_ok, g_err, g_rel = cmp(tg, got[tg], gref, rtol)
            ok &= g_ok
            if tw not in got:
                continue
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


def run_expert(npz: str, layers: int, nsamples: int, opt: str, lr: float, profile: bool, seed: int, loss_scale: float | None,
               dump_layers=(0, 7, 15)) -> bool:
    """Step 3: one TTT step of the full expert on the real weights: x_t = noise at t = 1 (flow step 0), loss = MSE between the
    predicted velocity and the target u = noise - a (a = lerobot's action chunk for this observation: self-distillation on
    the policy's own sample), backward through `layers` layers into the LoRA, one optimizer step."""
    from softhier_mlir.frontend import smolvla_ttt as TT
    data = np.load(npz)
    W = {k[2:]: data[k] for k in data.files if k.startswith("p_")}
    x0 = W["x0"].astype(np.float16)
    act = data["ref_actions"].reshape(TT.S, TT.AD)
    tgt = (x0.astype(np.float32) - act).astype(np.float16)
    lora = TT.lora_init(layers, seed)
    ref = TT.torch_ttt(W, layers, True, lora, lr=lr, opt=opt, x_in=x0, target=tgt)
    gmax = float(np.abs(ref["grad"]).max())
    if loss_scale is None:     # largest power of two keeping every activation gradient and LoRA gradient below 2^12
        loss_scale = float(2.0 ** np.floor(np.log2(1024.0 / max(gmax, max(ref["dh_max"])))))
    print(f"[ttt] torch float64: loss {ref['loss']:.6f}; max |dL/dLoRA| {gmax:.3e}; max |dL/dh| per layer "
          f"{' '.join(f'{v:.1e}' for v in ref['dh_max'])}; loss scale {loss_scale:g}")
    ref["w0"] = lora.astype(np.float64)
    app = _app(HERE / "ttt_app")
    dl = tuple(L for L in dump_layers if L < layers)
    mlir, pre, info = TT.emit_ttt(W, layers, True, lora, lr=lr, opt=opt, loss_scale=loss_scale, nsamples=nsamples, profile=profile,
                                  dump_layers=dl, x_in=x0, target=tgt)
    r = _build_and_run(app, mlir, pre, 48 * 3600, app / "run.log")
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} ttt expert layers={layers} opt={opt} lr={lr} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    tm = timing(r["stdout"])
    agg: dict = {}
    for t, dt in tm:
        k = re.sub(r"\d+$", "", t)
        agg.setdefault(k, [0, 0]); agg[k][0] += dt; agg[k][1] += 1
    for k, (t, n) in agg.items():
        print(f"     time {k:<8} {t / 1e3:10.1f} us  ({n} x {t / n / 1e3:.1f})")
    got = parse_all(r["stdout"])
    if "V" in got:
        cmp("V", got["V"], ref["v"], 5e-2)
    if "LOSS" in got:
        lr_ = np.zeros((TT.S, 1))
        dev = sum(v for _, _, v in got["LOSS"]) / len(got["LOSS"]) * TT.S / (TT.S * TT.AD)
        print(f"     loss (device, sampled rows) ~{dev:.6f} vs torch {ref['loss']:.6f}")
        del lr_
    ok &= _grad_report(got, ref, dl, loss_scale, lr, 5e-2)
    return ok


def sample_grads(npz: str, layers: int, nsamp: int, seed: int, loss_scale: float, cache: Path) -> np.ndarray:
    """[nsamp, layers * 98048] fp16 LoRA gradients (times loss_scale) of nsamp TTT samples from torch float64: sample c has its
    own noise x_c ~ N(0, 1) and target x_c - a (cached)."""
    from softhier_mlir.frontend import smolvla_ttt as TT
    if cache.exists():
        return np.load(cache)["g"]
    data = np.load(npz)
    W = {k[2:]: data[k] for k in data.files if k.startswith("p_")}
    act = data["ref_actions"].reshape(TT.S, TT.AD)
    lora = TT.lora_init(layers, seed)
    out = []
    for c in range(nsamp):
        x = np.random.default_rng(1000 + c).standard_normal((TT.S, TT.AD)).astype(np.float16)
        ref = TT.torch_ttt(W, layers, True, lora, lr=0.0, x_in=x, target=(x.astype(np.float32) - act).astype(np.float16))
        out.append((ref["grad"] * loss_scale).astype(np.float16))
        print(f"[ttt] sample {c}: loss {ref['loss']:.4f} max |g| {np.abs(ref['grad']).max():.2e}", flush=True)
    g = np.stack(out)
    np.savez(cache, g=g)
    return g


def run_reduce(npz: str, layers: int, seed: int, loss_scale: float, nsamples: int) -> bool:
    """Step 3: the data-parallel gradient sum alone, on 16 real per-sample LoRA gradient arenas (full size): device REDADD
    (fp16, and the exact two-limb integer scheme) vs the float64 sum of the same fp16 inputs."""
    from softhier_mlir.frontend import smolvla_ttt as TT
    g = sample_grads(npz, layers, 16, seed, loss_scale, HERE / "ttt_app" / f"grads_L{layers}_s{seed}.npz")
    exact = g.astype(np.float64).sum(0)
    rows = exact.size // TT.ARENA_COLS
    ex2 = exact.reshape(rows, TT.ARENA_COLS)
    floor = ex2.astype(np.float16).astype(np.float64)
    tiny = np.abs(g.astype(np.float64)) < 2.0 ** -14
    print(f"[ttt] {g.shape[0]} arenas x {g.shape[1]} fp16 ({g.shape[1] * 2 / 2 ** 20:.2f} MiB each); |sum| max {np.abs(exact).max():.1f}; "
          f"subnormal fp16 inputs {tiny.mean() * 100:.3f} %; fp16(sum) vs sum: rel-L2 {np.linalg.norm(floor - ex2) / np.linalg.norm(ex2):.2e}")
    app = _app(HERE / "ttt_app")
    mlir, pre = TT.emit_reduce_test(g, nsamples)
    r = _build_and_run(app, mlir, pre, 7200, None)
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} ttt reduce roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    for t, dt in timing(r["stdout"]):
        print(f"     time {t:<14} {dt / 1e3:9.1f} us   ({g.shape[1] * 2 / dt:.1f} B/ns of one cluster's arena)")
    got = parse_all(r["stdout"])
    for tag in ("R16", "R32"):
        if tag not in got:
            print(f"     {tag} MISSING"); ok = False; continue
        cmp(tag, got[tag], ex2, 1e-2)
        zero = sum(1 for rr, cc, v in got[tag] if v == 0.0 and ex2[rr, cc] != 0.0)
        print(f"            {tag}: {zero} of {len(got[tag])} samples flushed to 0; floor fp16(sum) on the same samples: "
              f"max abs {max(abs(floor[rr, cc] - ex2[rr, cc]) for rr, cc, _ in got[tag]):.3e}")
    return ok


def run_dp(npz: str, layers: int, nsamples: int, lr: float, seed: int, loss_scale: float, profile: bool) -> bool:
    """Step 3, data-parallel: 16 clusters, one TTT sample each (own noise x_c, target x_c - a), the whole fwd + bwd on the
    cluster's own sample (SH_SELF), LoRA gradients summed with the in-network fp16 REDADD, SGD with their mean."""
    from softhier_mlir.frontend import smolvla_ttt as TT
    data = np.load(npz)
    W = {k[2:]: data[k] for k in data.files if k.startswith("p_")}
    act = data["ref_actions"].reshape(TT.S, TT.AD)
    lora = TT.lora_init(layers, seed)
    xs = [np.random.default_rng(1000 + c).standard_normal((TT.S, TT.AD)).astype(np.float16) for c in range(16)]
    ts = [(x.astype(np.float32) - act).astype(np.float16) for x in xs]
    refs = [TT.torch_ttt(W, layers, True, lora, lr=lr, x_in=x, target=t) for x, t in zip(xs, ts)]
    g = np.mean([r["grad"] for r in refs], axis=0)
    ref = {"grad": g, "w_new": lora.astype(np.float64) - lr * g, "w0": lora.astype(np.float64)}
    print(f"[ttt] dp: 16 samples, torch float64 mean loss {np.mean([r['loss'] for r in refs]):.5f}, max |mean grad| {np.abs(g).max():.2e}")
    app = _app(HERE / "ttt_app", " -DSH_T_NO_RED_EXACT")
    dl = (0, layers - 1)
    mlir, pre, info = TT.emit_ttt(W, layers, True, lora, lr=lr, loss_scale=loss_scale, nsamples=nsamples, profile=profile,
                                  dump_layers=dl, x_in=xs, target=ts, dp=True)
    print(f"[ttt] per-cluster region {info['pcs'] / 2 ** 20:.1f} MiB")
    r = _build_and_run(app, mlir, pre, 48 * 3600, app / "run.log")
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} ttt dp layers={layers} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    agg: dict = {}
    for t, dt in timing(r["stdout"]):
        k = re.sub(r"\d+$", "", t)
        agg.setdefault(k, [0, 0]); agg[k][0] += dt; agg[k][1] += 1
    for k, (t, n) in agg.items():
        print(f"     time {k:<9} {t / 1e3:10.1f} us  ({n} x {t / n / 1e3:.1f})")
    got = parse_all(r["stdout"])
    if "V" in got:
        cmp("V(c0)", got["V"], refs[0]["v"], 5e-2)
    ok &= _grad_report(got, ref, dl, loss_scale * 16, lr, 5e-2)
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["ops", "layer", "expert", "reduce", "dp"])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--opt", default="sgd")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--npz", default="/app/models/smolvla_base/expert.npz")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--loss-scale", type=float)
    a = ap.parse_args()
    if a.test == "ops":
        ok = run_ops(a.seed, a.nsamples)
    elif a.test == "layer":
        ok = run_layer(a.seed, a.nsamples, a.opt, a.lr)
    elif a.test == "reduce":
        ok = run_reduce(a.npz, a.layers, a.seed, a.loss_scale or 65536.0, a.nsamples)
    elif a.test == "dp":
        ok = run_dp(a.npz, a.layers, a.nsamples, a.lr, a.seed, a.loss_scale or 4096.0, a.profile)
    elif a.test == "expert":
        ok = run_expert(a.npz, a.layers, a.nsamples, a.opt, a.lr, a.profile, a.seed, a.loss_scale)
    else:
        raise SystemExit("not yet")
    sys.exit(0 if ok else 1)
