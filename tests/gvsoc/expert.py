#!/usr/bin/env python3
"""On-simulator tests of the SmolVLA action expert (softhier_mlir/frontend/smolvla_expert.py).

    python tests/gvsoc/expert.py layer                      # step 1: self + cross layer on LCG data vs the numpy twin
    python tests/gvsoc/expert.py flow --npz /app/models/smolvla_base/expert.npz [--steps 10] [--layers 16]
                                      [--fmt fp16,...] [--profile] [--dumps X A EMB H O] [--from-log <log>]
                                                            # step 2/3: the flow loop vs lerobot (expert_ref.npz) and
                                                            # the fp16-floor numpy model; per-step / per-op timing

Each run writes tests/gvsoc/expert_app/{expert.mlir, main.c, preload.elf, last_run.log}; the build goes to a
private directory and gvsoc runs with the preload image (see tests/gvsoc/run.py smolvla for the conventions).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.sim.gvsoc import PERF_RE, build_sw, run_sim  # noqa: E402
from softhier_mlir.sim.preload import make_preload_elf  # noqa: E402
from softhier_mlir.testing import lcg  # noqa: E402
from tests.gvsoc.run import _elf_load_segments, lower_and_translate  # noqa: E402

HERE = Path(__file__).resolve().parent


def _app(app_dir: Path | None) -> Path:
    app = Path(app_dir) if app_dir else HERE / "expert_app"
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    return app


def _build_and_run(app: Path, mlir: str, pre: dict, timeout: int, log: Path | None) -> dict:
    (app / "expert.mlir").write_text(mlir)
    elf = make_preload_elf(app / "preload.elf", pre)
    (app / "main.c").write_text(lower_and_translate(app / "expert.mlir", None))
    build_sw(app)
    text = sum(sz for _, sz in _elf_load_segments(build_sw.last_elf) if sz)
    print(f"[expert] preload {elf.stat().st_size / 2 ** 20:.1f} MiB, program {text / 1024:.1f} KB; simulating...", flush=True)
    r = run_sim(preload=elf, timeout=timeout, log=log)
    (app / "last_run.log").write_text(r["stdout"])
    return r


def marks_seq(stdout: str) -> list[tuple[str, int]]:
    """[(tag, ns since the first mark)] in order, unwrapping the 32-bit mcycle."""
    out, prev, acc = [], None, 0
    for ln in stdout.splitlines():
        if ln.startswith("[mark] "):
            _, tag, c = ln.split()
            c = int(c)
            if prev is not None:
                acc += (c - prev) % (1 << 32)
            prev = c
            out.append((tag, acc))
    return out


def op_breakdown(marks: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """{op name (tag without its index): (total ns, count)} of the segments ending at a mark."""
    out: dict[str, list[int]] = {}
    for (t0_tag, t0), (tag, t) in zip(marks, marks[1:]):
        name = re.sub(r"\d+$", "", tag)
        out.setdefault(name, [0, 0])
        out[name][0] += t - t0; out[name][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def compare(tag: str, got: list, ref: np.ndarray, atol: float, rtol: float, floor: np.ndarray | None = None, show: int = 2) -> tuple[bool, float]:
    bad, maxerr = lcg.compare_samples(got, ref, atol=atol, rtol=rtol, show=show)
    msg = f"     {tag:<6} n={len(got):<5} vs ref: max abs {maxerr:.4f} (|ref| max {np.abs(ref).max():.3f}) bad={bad}"
    if floor is not None:
        _, fl = lcg.compare_samples(got, floor, atol=atol, rtol=rtol)
        msg += f"  vs fp16 floor: max abs {fl:.4f}"
    print(msg + f" {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0, maxerr


def run_layer(seed: int, cluster: int, nsamples: int, app_dir=None, timeout: int = 7200) -> bool:
    from softhier_mlir.frontend import smolvla_expert as E
    app = _app(app_dir)
    mlir, pre, ref = E.emit_layer_test(seed, cluster, nsamples)
    r = _build_and_run(app, mlir, pre, timeout, None)
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} expert layer test seed={seed} cluster={cluster} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    marks = marks_seq(r["stdout"])
    for (t0_tag, t0), (tag, t) in zip(marks, marks[1:]):
        print(f"     time {t0_tag:>8} -> {tag:<8} {(t - t0) / 1e6:8.3f} ms")
    got = lcg.parse_samples(r["stdout"])
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        scale = float(np.abs(arr).max())
        good, _ = compare(tag, got[tag], arr, atol=max(0.02, 0.01 * scale), rtol=0.03)
        ok &= good
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


def run_op(which: str, seed: int, cluster: int, nsamples: int, app_dir=None, cols: int = 960) -> bool:
    from softhier_mlir.frontend import smolvla_expert as E
    app = _app(app_dir)
    mlir, pre, ref = E.emit_op_test(which, seed, cluster, nsamples, cols)
    r = _build_and_run(app, mlir, pre, 3600, None)
    ok = r["ok"]
    marks = marks_seq(r["stdout"])
    print(f"{'PASS' if ok else 'FAIL'} expert op {which} cluster={cluster} roi={r['roi_ns']} ns wall={r['wall_s']}s "
          f"marks {[(t, v) for t, v in marks]}")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    got = lcg.parse_samples(r["stdout"])
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        good, _ = compare(tag, got[tag], arr, atol=max(0.02, 0.01 * float(np.abs(arr).max())), rtol=0.03)
        ok &= good
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


def cand_seeds(n_cand: int, seeds=None) -> list:
    """Noise seed per candidate: None = the npz (lerobot) noise. Default: candidate 0 lerobot's, candidate c seed c."""
    if seeds is None:
        return [None] + list(range(1, n_cand))
    out = [None if str(v).lower() in ("none", "ref", "-1") else int(v) for v in seeds]
    assert len(out) == n_cand, (out, n_cand)
    return out


def run_flow(npz: str, steps: int, layers: int, cluster: int, fmt_steps, profile: bool, dumps, nsamples: int = 64,
             app_dir=None, log: Path | None = None, from_log: Path | None = None, timeout: int = 48 * 3600, tiles=None,
             n_cand: int = 1, seeds=None, save_x: Path | None = None) -> bool:
    from softhier_mlir.frontend import smolvla_expert as E
    data = dict(np.load(npz))
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    ref_xt = data["ref_xt"]
    seeds = cand_seeds(n_cand, seeds)
    x0 = E.cand_noise(P["x0"], seeds)
    if n_cand > 1:
        print(f"[expert] {n_cand} candidates, noise seeds {seeds}, tiles {E.batch_tiles(n_cand, tiles or E.TILES)}")
    if from_log is None:
        app = _app(app_dir)
        mlir, pre = E.emit_flow(data, steps, layers, cluster, fmt_steps, profile, tuple(dumps), nsamples, tiles=tiles or E.TILES,
                                n_cand=n_cand, x0=x0)
        r = _build_and_run(app, mlir, pre, timeout, log)
        stdout, ok = r["stdout"], r["ok"]
        print(f"{'PASS' if ok else 'FAIL'} expert flow steps={steps} layers={layers} cand={n_cand} cluster={cluster} fmt={fmt_steps} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    else:
        stdout = Path(from_log).read_text()
        rois = [int(v) for v in PERF_RE.findall(stdout)]
        ok = bool(rois)
        print(f"[expert] re-evaluating {from_log}: steps={steps} layers={layers} roi={rois[0] if rois else None} ns")
    for ln in stdout.splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    # ---- timing
    marks = marks_seq(stdout)
    mt = dict(marks)
    if "kvproj" in mt:
        print(f"     KV projection of the cross layers (once per chunk): {(mt['kvproj'] - mt['start']) / 1e6:.3f} ms")
    step_t, prev = [], mt.get("kvproj", mt.get("start", 0))
    for s in range(steps):
        if f"step{s}" not in mt:
            break
        step_t.append(mt[f"step{s}"] - prev)
        prev = mt.get(f"xdump{s}", mt[f"step{s}"])
    if step_t:
        print(f"     per step: " + " ".join(f"{t / 1e6:.3f}" for t in step_t) + f" ms; mean {np.mean(step_t) / 1e6:.3f} ms, "
              f"chunk ({steps} steps) {sum(step_t) / 1e6:.3f} ms" + (f" + KV projection {(mt['kvproj'] - mt['start']) / 1e6:.3f} ms" if "kvproj" in mt else ""))
    if profile:
        bd = op_breakdown(marks)
        tot = sum(v[0] for k, v in bd.items() if k not in ("xdump", "start", "kvproj", "end"))
        print("     per-op breakdown (sum over steps and layers):")
        for k, (t, n) in sorted(bd.items(), key=lambda kv: -kv[1][0]):
            if k in ("xdump", "start", "kvproj", "end"):
                continue
            print(f"       {k:<8} {t / 1e6:9.3f} ms  {100 * t / tot:5.1f} %   {n:4d} x {t / n / 1e3:8.1f} us")
    # ---- accuracy: device x_t per step vs lerobot and vs the fp16-floor numpy model
    got = lcg.parse_samples(stdout)
    if n_cand > 1:
        return _check_cands(E, P, ref_xt, got, steps, layers, n_cand, seeds, x0, ok, save_x)
    floor_xt, floor_inter = E.np_flow(P, steps, layers, record=("H" in dumps or "O" in dumps))
    if save_x is not None:
        np.savez(save_x, x=np.stack([_full(got.get(f"X{s}", []), 50, 32) for s in range(steps)]), seeds=np.array([-1 if v is None else v for v in seeds]))
    for s in range(steps):
        tag = f"X{s}"
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        want = ref_xt[s + 1] if layers == 16 else floor_xt[s + 1]
        good, err = compare(tag, got[tag], want, atol=0.05, rtol=0.02, floor=floor_xt[s + 1])
        ok &= good
    if "A" in got:
        want = ref_xt[steps] if layers == 16 else floor_xt[steps]      # x_t after `steps` steps (== the actions at 10)
        vals = np.zeros((50, 32), np.float32)
        for r_, c, v in got["A"]:
            vals[r_, c] = v
        err = np.abs(vals - want)
        print(f"     x_{steps} (50 x 32, all elements){' = actions' if steps == 10 else ''} vs lerobot fp32: max abs {err.max():.4f} mean {err.mean():.4f} "
              f"(|actions| max {np.abs(want).max():.3f}); vs fp16 floor: max abs {np.abs(vals - floor_xt[steps]).max():.4f}; "
              f"floor vs lerobot: {np.abs(floor_xt[steps] - want).max():.4f}")
        ok &= err.max() < 0.1
    full = layers == 16          # the lerobot intermediates exist for the full model only; else compare with the twin
    if "EMB0" in got:
        compare("EMB0", got["EMB0"], data["ref_s0_emb"] if full else floor_inter["EMB"], atol=0.05, rtol=0.03, floor=floor_inter.get("EMB"))
    for L in range(layers):
        if f"H{L}" in got:
            ref = data[f"ref_s0_h_{L}"] if full else floor_inter[f"H{L}"]
            compare(f"H{L}", got[f"H{L}"], ref, atol=max(0.05, 0.02 * float(np.abs(ref).max())), rtol=0.05, floor=floor_inter.get(f"H{L}"))
        if f"O{L}" in got:
            compare(f"O{L}", got[f"O{L}"], data[f"ref_s0_att_{L}"] if full else floor_inter[f"O{L}"], atol=0.05, rtol=0.05, floor=floor_inter.get(f"O{L}"))
    if not ok and from_log is None:
        print(stdout[-1500:])
    return ok


def _full(samples, rows: int, cols: int) -> np.ndarray:
    a = np.full((rows, cols), np.nan, np.float32)
    for r_, c, v in samples:
        a[r_, c] = v
    return a


def _check_cands(E, P, ref_xt, got, steps, layers, n_cand, seeds, x0, ok, save_x) -> bool:
    """n_cand candidates: x_t of candidate c = rows [50c, 50c + 50) of the X dumps, compared with the fp16-floor twin
    run on that candidate's noise alone (and with lerobot for the lerobot-noise candidate of the full model)."""
    S = E.S
    xs = np.stack([_full(got.get(f"X{s}", []), S * n_cand, E.AD) for s in range(steps)])
    if save_x is not None:
        np.savez(save_x, x=xs, seeds=np.array([-1 if v is None else v for v in seeds]))
    for c, sd in enumerate(seeds):
        Pc = dict(P); Pc["x0"] = x0[c * S:(c + 1) * S]
        floor_xt, _ = E.np_flow(Pc, steps, layers)
        lero = sd is None and layers == E.LAYERS
        errs, ferrs = [], []
        for s in range(steps):
            dev = xs[s, c * S:(c + 1) * S]
            if np.isnan(dev).any():
                print(f"     cand {c} X{s} MISSING"); ok = False; continue
            want = ref_xt[s + 1] if lero else floor_xt[s + 1]
            errs.append(float(np.abs(dev - want).max())); ferrs.append(float(np.abs(dev - floor_xt[s + 1]).max()))
        good = bool(errs) and max(errs) < 0.05
        ok &= good
        print(f"     cand {c} (noise {'lerobot' if sd is None else f'seed {sd}'}): x_t max abs vs {'lerobot' if lero else 'fp16 twin'} per step "
              + " ".join(f"{e:.4f}" for e in errs) + f"; vs fp16 twin max {max(ferrs) if ferrs else float('nan'):.4f} {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["layer", "flow", "op"])
    ap.add_argument("--which", default="attn", help="op: attn | xattn | rmsnorm | rope | silu | silu_view | axpy")
    ap.add_argument("--cols", type=int, default=960, help="op: columns of the row-op tests")
    ap.add_argument("--npz", default="/app/models/smolvla_base/expert.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--cluster", type=int, default=-1)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--fmt", help="flow: comma-separated RedMulE format per step (fp16|fp8|int16|int8)")
    ap.add_argument("--profile", action="store_true", help="flow: per-op marks")
    ap.add_argument("--dumps", nargs="*", default=["X", "A"], help="flow: X (x_t per step) A (actions) EMB H O (step-0 intermediates)")
    ap.add_argument("--app-dir")
    ap.add_argument("--log")
    ap.add_argument("--from-log")
    ap.add_argument("--tiles", help="flow: override tile shapes, e.g. qkv=50,240,720;o=50,240,960")
    ap.add_argument("--cands", type=int, default=1, help="flow: candidate chunks denoised at once (S_q = 50 N)")
    ap.add_argument("--seeds", nargs="*", help="flow: noise seed per candidate (none = the npz/lerobot noise); default none,1,2,...")
    ap.add_argument("--save-x", help="flow: write the device x_t per step to this .npz")
    a = ap.parse_args()
    if a.test == "layer":
        ok = run_layer(a.seed, a.cluster, a.nsamples, a.app_dir)
    elif a.test == "op":
        ok = run_op(a.which, a.seed, a.cluster, a.nsamples, a.app_dir, a.cols)
    else:
        fmt = a.fmt.split(",") if a.fmt else None
        tiles = None
        if a.tiles:
            from softhier_mlir.frontend import smolvla_expert as E
            tiles = dict(E.TILES)
            for item in a.tiles.split(";"):
                k, v = item.split("=")
                tiles[k] = tuple(int(x) for x in v.split(","))
        ok = run_flow(a.npz, a.steps, a.layers, a.cluster, fmt, a.profile, a.dumps, a.nsamples, a.app_dir,
                      Path(a.log) if a.log else None, Path(a.from_log) if a.from_log else None, tiles=tiles,
                      n_cand=a.cands, seeds=a.seeds, save_x=Path(a.save_x) if a.save_x else None)
    sys.exit(0 if ok else 1)
