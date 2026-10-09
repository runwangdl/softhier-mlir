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


def _build_and_run(app: Path, mlir: str, pre: dict, timeout: int, log: Path | None, traces: tuple = ()) -> dict:
    (app / "expert.mlir").write_text(mlir)
    elf = make_preload_elf(app / "preload.elf", pre)
    (app / "main.c").write_text(lower_and_translate(app / "expert.mlir", None))
    build_sw(app)
    text = sum(sz for _, sz in _elf_load_segments(build_sw.last_elf) if sz)
    print(f"[expert] preload {elf.stat().st_size / 2 ** 20:.1f} MiB, program {text / 1024:.1f} KB; simulating...", flush=True)
    r = run_sim(preload=elf, timeout=timeout, log=log, traces=traces)
    (app / "last_run.log").write_text(r["stdout"])
    return r


def marks_seq(stdout: str) -> list[tuple[str, int]]:
    """[(tag, ns since the first mark)] in order, unwrapping the 32-bit mcycle."""
    out, prev, acc = [], None, 0
    if "[iDMA] Finished" in stdout:      # traced run: program output interleaved with trace lines (tests/gvsoc/dma_bytes.py)
        from tests.gvsoc.dma_bytes import parse
        lines = [f"[mark] {t} {c}" for t, c in parse(stdout)[0]]
    else:
        lines = stdout.splitlines()
    for ln in lines:
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
    if which.startswith("gemm8") and "Z0" in got and "Z2" in got:   # expanded on the cores == host-expanded copy, bit for bit
        same = got["Z0"] == got["Z2"]
        print(f"     Z0 (fp8 bytes, expanded on the cores) vs Z2 (host-expanded fp16 copy): {'bit-identical' if same else 'DIFFERENT'} "
              f"on {len(got['Z0'])} samples")
        ok &= same
        t = dict(marks)
        seq = [k for k, _ in marks]
        for a_, b_ in zip(seq, seq[1:]):
            print(f"     time {b_:<9} {(t[b_] - t[a_]) / 1e3:8.1f} us")
    if not r["ok"]:
        print(r["stdout"][-2000:])
    return ok


def run_flow(npz: str, steps: int, layers: int, cluster: int, fmt_steps, profile: bool, dumps, nsamples: int = 64,
             app_dir=None, log: Path | None = None, from_log: Path | None = None, timeout: int = 48 * 3600, tiles=None,
             attn: str = "stream", fp8_mode: int | None = None, save_x: str | None = None, trace_dma: bool = False) -> bool:
    from softhier_mlir.frontend import smolvla_expert as E
    data = dict(np.load(npz))
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    ref_xt = data["ref_xt"]
    if from_log is None:
        app = _app(app_dir)
        mlir, pre = E.emit_flow(data, steps, layers, cluster, fmt_steps, profile, tuple(dumps), nsamples, tiles=tiles or E.TILES,
                                attn=attn, fp8_mode=fp8_mode)
        r = _build_and_run(app, mlir, pre, timeout, log, ("idma",) if trace_dma else ())
        stdout, ok = r["stdout"], r["ok"]
        print(f"{'PASS' if ok else 'FAIL'} expert flow steps={steps} layers={layers} cluster={cluster} fmt={fmt_steps} attn={attn} fp8_mode={fp8_mode} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    else:
        stdout = Path(from_log).read_text()
        rois = [int(v) for v in PERF_RE.findall(stdout)]
        ok = bool(rois)
        print(f"[expert] re-evaluating {from_log}: steps={steps} layers={layers} roi={rois[0] if rois else None} ns")
    for ln in stdout.splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    if trace_dma:   # bytes moved by every iDMA, by kind, per mark segment (tests/gvsoc/dma_bytes.py)
        from tests.gvsoc.dma_bytes import report
        report(stdout)
    # ---- timing
    marks = marks_seq(stdout)
    mt = dict(marks)
    if "kvproj" in mt:
        print(f"     KV projection of the cross layers (once per chunk): {(mt['kvproj'] - mt['start']) / 1e6:.3f} ms")
    if "kvdeal" in mt:
        print(f"     KV deal into TCDM (once per chunk): {(mt['kvdeal'] - mt['kvproj']) / 1e6:.3f} ms")
    step_t, prev = [], mt.get("kvdeal", mt.get("kvproj", mt.get("start", 0)))
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
        tot = sum(v[0] for k, v in bd.items() if k not in ("xdump", "start", "kvproj", "kvdeal", "end"))
        print("     per-op breakdown (sum over steps and layers):")
        for k, (t, n) in sorted(bd.items(), key=lambda kv: -kv[1][0]):
            if k in ("xdump", "start", "kvproj", "kvdeal", "end"):
                continue
            print(f"       {k:<8} {t / 1e6:9.3f} ms  {100 * t / tot:5.1f} %   {n:4d} x {t / n / 1e3:8.1f} us")
    # ---- accuracy: device x_t per step vs lerobot and vs the fp16-floor numpy model
    got = lcg.parse_samples(stdout)
    fp8 = fp8_mode is not None and fmt_steps is not None and "fp8" in fmt_steps
    if fp8 and fp8_mode in (1, 3):
        print(f"     fp8 mode {fp8_mode} (timing model, W' tiles zero): the numbers are invalid by construction, not compared")
        return ok
    floor_xt, floor_inter = E.np_flow(P, steps, layers, record=("H" in dumps or "O" in dumps), fmt_steps=fmt_steps if fp8 else None)
    if fp8:     # the reference of an fp8 schedule is its own fp16-floor twin; lerobot / all-fp16 are reported as distances
        f16_xt, _ = E.np_flow(P, steps, layers)
    xs = {}
    for s in range(steps):
        tag = f"X{s}"
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        xs[s] = np.zeros((50, 32), np.float32)
        for r_, c, v in got[tag]:
            xs[s][r_, c] = v
        want = ref_xt[s + 1] if (layers == 16 and not fp8) else floor_xt[s + 1]
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
        ok &= err.max() < 0.1 or fp8
        if fp8:
            print(f"     fp8 schedule {fmt_steps}: x_{steps} vs its fp8 twin max {np.abs(vals - floor_xt[steps]).max():.4f}; "
                  f"vs all-fp16 twin max {np.abs(vals - f16_xt[steps]).max():.4f} mean {np.abs(vals - f16_xt[steps]).mean():.5f}; "
                  f"vs lerobot max {err.max():.4f} mean {err.mean():.5f}")
            ok &= np.abs(vals - floor_xt[steps]).max() < 0.1
    if save_x and xs:
        np.savez(save_x, x=np.stack([xs[s] for s in sorted(xs)]), steps=np.array(sorted(xs)))
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
    ap.add_argument("--attn", default="stream", choices=["stream", "kvs"], help="flow: attention dataflow (kvs = KV-stationary)")
    ap.add_argument("--fp8-mode", type=int, choices=[0, 1, 2, 3], help="flow: real fp8 steps (--fmt fp8 entries) of the layer GEMMs: "
                    "0 expand on the cores, 1 no expansion (timing of a DMA-path cast), 2 host-expanded copy (numbers), "
                    "3 = 1 without the activation cast pass (bound of a cast fused into the producer)")
    ap.add_argument("--save-x", help="flow: save the device x_t per step (npz)")
    ap.add_argument("--trace-dma", action="store_true", help="flow: gvsoc iDMA trace -> HBM / cluster-to-cluster bytes per segment")
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
                      attn=a.attn, fp8_mode=a.fp8_mode, save_x=a.save_x, trace_dma=a.trace_dma)
    sys.exit(0 if ok else 1)
