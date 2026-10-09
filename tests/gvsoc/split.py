#!/usr/bin/env python3
"""H3 on gvsoc: SmolVLA prefill (vision + connector + prefix) and decode (the expert's flow) on disjoint cluster sets at the
same time, pipelined over chunks, against time-sharing all 16 clusters (softhier_mlir.frontend.smolvla_split,
docs/SPATIAL_SPLIT.md).

    python tests/gvsoc/split.py run --name ts --mode time                       # time-shared baseline (A then B, 16 clusters)
    python tests/gvsoc/split.py run --name s8_8 --a rows:0-1 --b rows:2-3       # A || B on 8 + 8 clusters
    python tests/gvsoc/split.py run --name s8_8_A --a rows:0-1 --b rows:2-3 --stages A --periods 1   # A alone on its set
    python tests/gvsoc/split.py run ... --trace                                  # + RedMulE / iDMA / barrier traces
    python tests/gvsoc/split.py png --name s8_8 --period 1                       # timeline of one period (traced run)

Each run: tests/gvsoc/split_app/<name>/ (mlir, main.c, preload, log, outputs.npz); the marks, stage times and checks
go to docs/dse/split/<name>.json. Every run is checked bit for bit against the time-shared run `ts` (the last action
chunk and the last KV block); `ts` itself is checked against the fp16 numpy twin of the expert on its own KV.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.frontend.clusters import parse as parse_set, set_mask  # noqa: E402
from softhier_mlir.sim.gvsoc import build_sw, run_sim  # noqa: E402
from softhier_mlir.sim.preload import make_preload_elf  # noqa: E402
from tests.gvsoc.run import _elf_load_segments, lower_and_translate  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "dse" / "split"
NPZ = "/app/models/smolvla_base/e2e_c1_t256.npz"
FLAGS = "-DSH_NO_GEMM_MESH -DSH_NOCLONE_ROWOP -DSH_FAR_CODE -DSH_TINY_PRINTF -fdata-sections"
TRACES = ("redmule", "idma", "cluster_registers")
_MARK = re.compile(r"^\[mark\] ([A-Za-z_]+)(\d*) (\d+)$")


def marks_of(stdout: str) -> tuple[list[tuple[str, int]], int]:
    """[(tag, ns)] in print order (32-bit mcycle unwrapped), number of garbled mark lines (two set leaders printing
    through the one UART at the same time)."""
    out, prev, acc, bad = [], None, 0, 0
    for ln in stdout.splitlines():
        if "[mark]" not in ln:
            continue
        m = _MARK.match(ln.strip())
        if not m:
            bad += 1
            continue
        c = int(m[3])
        if prev is not None:
            acc += (c - prev) % (1 << 32)
        prev = c
        out.append((m[1] + m[2], acc))
    return out, bad


def stage_times(marks: list[tuple[str, int]], periods: int) -> dict:
    """period p = start of p (start / period<p-1>) -> period<p>; A<p> / B<p> measured from the start of p."""
    t = dict(marks)
    t["start"] = next(v for k, v in marks if k == "start")
    res = {"periods": []}
    for p in range(periods):
        t0 = t["start"] if p == 0 else t[f"period{p - 1}"]
        row = {"p": p, "period_ns": t[f"period{p}"] - t0}
        for s in ("A", "B"):
            if f"{s}{p}" in t:
                row[f"{s}_ns"] = t[f"{s}{p}"] - t0
        res["periods"].append(row)
    return res


def dumps_of(stdout: str) -> dict[str, dict]:
    """dump lines '<tag> r c hex' -> {tag: {(r, c): fp16 bits}} (dump_all: every element; dump_samples: a sample)"""
    vals: dict[str, dict] = {}
    for ln in stdout.splitlines():
        p = ln.split()
        if len(p) == 4 and (p[0] == "ACT" or re.match(r"^[KV]C\d+$", p[0])):
            try:
                vals.setdefault(p[0], {})[(int(p[1]), int(p[2]))] = int(p[3], 16)
            except ValueError:
                continue
    return vals


def dense(d: dict) -> np.ndarray:
    R = 1 + max(r for r, _ in d); C = 1 + max(c for _, c in d)
    a = np.zeros((R, C), np.uint16)
    for (r, c), v in d.items():
        a[r, c] = v
    return a.view(np.float16)


def run(name: str, mode: str, a: str, b: str, stages: list[str], periods: int, trace: bool, build_only: bool,
        vlayers: int, players: int, xlayers: int, timeout: int, steps: int = 10, kv_dump: str = "samples",
        far: bool = True) -> bool:
    from softhier_mlir.frontend import smolvla_split as SP
    ma, mb = set_mask(parse_set(a)) or 0xFFFF, set_mask(parse_set(b)) or 0xFFFF
    app = HERE / "split_app" / name
    app.mkdir(parents=True, exist_ok=True)
    rt = (ROOT / "runtime").resolve()
    flags = FLAGS if far else FLAGS.replace(" -DSH_FAR_CODE", "")
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c {flags} PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    e2e = np.load(NPZ)
    mlir, pre, info = SP.emit_pipeline(e2e, mode=mode, mask_a=ma, mask_b=mb, stages=tuple(stages), periods=periods,
                                       vlayers=vlayers, players=players, xlayers=xlayers, steps=steps, dump_kv=kv_dump, far=far)
    (app / "prog.mlir").write_text(mlir)
    elf_pre = make_preload_elf(app / "preload.elf", pre)
    image = sum(x.nbytes for x in pre.values())
    del pre
    gc.collect()
    (app / "main.c").write_text(lower_and_translate(app / "prog.mlir", None))
    build_sw(app, build_dir=app / "build")
    segs = {hex(s): sz for s, sz in _elf_load_segments(build_sw.last_elf)}
    imem = segs.get("0x80000000", 0)
    print(f"[split] {name}: mode={mode} A=0x{ma:04x} B=0x{mb:04x} stages={stages} periods={periods} depth V{vlayers}/P{players}/X{xlayers}; "
          f"preload {image / 2 ** 20:.1f} MiB, instruction memory {imem / 1024:.1f} KB, far code {segs.get('0xc0000000', 0) / 1024:.1f} KB", flush=True)
    if imem > 0x10000:
        print(f"FAIL {name}: program {imem} B > 64 KB instruction memory")
        return False
    if build_only:
        return True
    log = app / "run.log"
    r = run_sim(preload=elf_pre, timeout=timeout, log=log, traces=TRACES if trace else (), program_output_only=True)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    out = r["stdout"]
    (app / "program.log").write_text(out)
    marks, garbled = marks_of(out)
    print(f"{'PASS' if r['ok'] else 'FAIL'} {name}: roi={r['roi_ns']} ns wall={r['wall_s']} s (max child RSS {rss:.0f} MB), "
          f"{len(marks)} marks, {garbled} garbled", flush=True)
    if not r["ok"]:
        print(out[-3000:])
        return False
    res = {"name": name, "mode": mode, "mask_a": ma, "mask_b": mb, "stages": stages, "info": {k: v for k, v in info.items()},
           "roi_ns": r["roi_ns"], "wall_s": r["wall_s"], "program_bytes": imem, "image_bytes": image, "trace": trace,
           "marks": marks, "garbled_marks": garbled}
    res.update(stage_times(marks, periods))
    dd = dumps_of(out)
    d = {tag: dense(v) for tag, v in dd.items()}
    np.savez(app / "outputs.npz", **d)
    ok = True
    ref_app = HERE / "split_app" / "ts"
    same_cfg = name != "ts" and (OUT / "ts.json").exists() and all(
        json.loads((OUT / "ts.json").read_text())["info"][k] == info[k] for k in ("vlayers", "players", "xlayers", "steps"))
    if same_cfg and (ref_app / "outputs.npz").exists():
        ref = {k: v.view(np.uint16) for k, v in np.load(ref_app / "outputs.npz").items()}
        cmp = {}
        for tag, vals in dd.items():
            # ACT of a B-only run is computed on the preloaded (lerobot) KV block, not on A's: nothing to compare
            if tag in ref and (tag != "ACT" or ("A" in stages and periods >= 2)):
                same = all(r < ref[tag].shape[0] and c < ref[tag].shape[1] and int(ref[tag][r, c]) == v for (r, c), v in vals.items())
                cmp[tag] = bool(same)
                ok &= same
        res["bit_identical_to_ts"] = cmp
        print(f"     vs ts: {sum(cmp.values())}/{len(cmp)} outputs bit-identical {sorted(k for k, v in cmp.items() if not v)}")
    if kv_dump == "all" and "B" in stages and "A" in stages:
        res["twin"] = twin_check(e2e, d, info)
        print(f"     expert fp16 twin on the device KV: actions max abs {res['twin']['max_abs']:.4f}")
    for p in res["periods"]:
        print("     period {p}: {period_ns} ns  ".format(**p) + "  ".join(f"{k}={v}" for k, v in p.items() if k.endswith("_ns") and k != "period_ns"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    if trace:
        res_t = trace_stats(log, marks, periods, ma, mb, mode)
        (OUT / f"{name}.json").write_text(json.dumps({**res, "trace_stats": res_t}, indent=1))
    return ok


def twin_check(e2e, d: dict, info: dict) -> dict:
    """The expert's fp16 numpy twin (frontend.smolvla_expert.np_flow) on the KV the device left in the last block: the
    action chunk the device's B of the last period computed from A's KV of the period before (same KV: every chunk of A
    computes the same numbers)."""
    from softhier_mlir.frontend import smolvla_e2e as E2E
    from softhier_mlir.frontend import smolvla_expert as X
    L = info["xlayers"]
    if "ACT" not in d or f"KC{L - 1}" not in d:
        return {"max_abs": float("nan"), "note": "outputs missing"}
    kv = {l: (d[f"KC{l}"].astype(np.float32), d[f"VC{l}"].astype(np.float32)) for l in range(L)}
    data = E2E.expert_data(e2e, kv)
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    xs, _ = X.np_flow(P, steps=info["steps"], layers=L, num_steps=info["steps"])
    x = np.asarray(xs[-1], np.float32)
    err = np.abs(d["ACT"].astype(np.float32) - x)
    return {"max_abs": float(err.max()), "mean_abs": float(err.mean()), "max_ref": float(np.abs(x).max())}


def _group_of(c: int, ma: int, mb: int) -> str:
    return "A" if (ma >> c) & 1 else "B" if (mb >> c) & 1 else "-"


def trace_stats(log: Path, marks, periods: int, ma: int, mb: int, mode: str) -> dict:
    """Per period: RedMulE / iDMA busy % of each set (mean over its clusters), HBM bytes (softhier_mlir.sim.trace)."""
    from softhier_mlir.sim.trace import segment_stats
    segs = segment_stats(log)
    t = dict(marks)
    t["start"] = next(v for k, v in marks if k == "start")
    out = []
    for p in range(periods):
        t0 = t["start"] if p == 0 else t[f"period{p - 1}"]
        t1 = t[f"period{p}"]
        sel = [s for s in segs if t0 < s["t_ns"] <= t1]
        row = {"p": p, "period_ns": t1 - t0, "hbm_MB": sum(s["hbm_rd"] + s["hbm_wr"] for s in sel) / 1e6}
        for g, m in (("A", ma), ("B", mb), ("all", 0xFFFF)):
            cl = [c for c in range(16) if (m >> c) & 1]
            for u in ("redmule", "idma", "sync"):
                row[f"{g}_{u}_pct"] = 100.0 * sum(s[u][c] for s in sel for c in cl) / max(1, (t1 - t0) * len(cl))
        out.append(row)
    return {"periods": out}


def png(name: str, period: int, out: Path | None) -> None:
    """Timeline of one period of a traced run: RedMulE / iDMA / barrier per cluster, the sets labelled."""
    from softhier_mlir.sim.trace import _ANSI, _DMA_ANY, _RED, _SYNC, to_png
    res = json.loads((OUT / f"{name}.json").read_text())
    t = dict(res["marks"])
    t["start"] = next(v for k, v in res["marks"] if k == "start")
    t0 = t["start"] if period == 0 else t[f"period{period - 1}"]
    t1 = t[f"period{period}"]
    ev = []
    with open(HERE / "split_app" / name / "run.log", errors="replace") as f:
        for ln in f:
            if "Finished" not in ln and "Cluster Sync" not in ln:
                continue
            ln = _ANSI.sub("", ln)
            for rx, unit in ((_RED, "redmule"), (_DMA_ANY, "idma"), (_SYNC, "sync")):
                m = rx.search(ln)
                if m:
                    a, b = int(m[2]), int(m[3])
                    if b > t0 and a < t1:
                        ev.append(dict(cluster=int(m[1]), unit=unit, t0=max(a, t0) - t0, t1=min(b, t1) - t0))
                    break
    ma, mb = res["mask_a"], res["mask_b"]
    title = (f"{name}: period {period} = {(t1 - t0) / 1e6:.2f} ms; " +
             ("time-shared: A then B on all 16 clusters" if res["mode"] == "time" else
              f"A (vision+prefix) on {bin(ma).count('1')} clusters 0x{ma:04x} || B (expert) on {bin(mb).count('1')} clusters 0x{mb:04x}") +
             "\nRedMulE (orange, upper lane) / iDMA (blue, lower lane) / barrier wait (grey)")
    path = out or (OUT / f"{name}_period{period}.png")
    to_png(ev, t1 - t0, path, title)
    print(f"[split] {path}: {len(ev)} events")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--name", required=True)
    pr.add_argument("--mode", default="split", choices=("split", "time"))
    pr.add_argument("--a", default="rows:0-1", help="set of stage A (vision + prefix): set:<mask> | rows:<y0>-<y1> | all")
    pr.add_argument("--b", default="rows:2-3", help="set of stage B (expert)")
    pr.add_argument("--stages", nargs="*", default=["A", "B"])
    pr.add_argument("--periods", type=int, default=2)
    pr.add_argument("--vlayers", type=int, default=3)
    pr.add_argument("--players", type=int, default=4)
    pr.add_argument("--xlayers", type=int, default=4)
    pr.add_argument("--trace", action="store_true")
    pr.add_argument("--build-only", action="store_true")
    pr.add_argument("--steps", type=int, default=10)
    pr.add_argument("--kv-dump", default="samples", choices=("all", "samples", "none"),
                    help="all: every KV element (the fp16 twin check of the expert), samples: 256 per tensor (bitwise vs ts)")
    pr.add_argument("--no-far", action="store_true", help="control code in instruction memory (fits only for one stage)")
    pr.add_argument("--timeout", type=int, default=12 * 3600)
    pp = sub.add_parser("png")
    pp.add_argument("--name", required=True)
    pp.add_argument("--period", type=int, default=1)
    pp.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        ok = run(a.name, a.mode, a.a, a.b, a.stages, a.periods, a.trace, a.build_only, a.vlayers, a.players, a.xlayers, a.timeout,
                 a.steps, a.kv_dump, not a.no_far)
        sys.exit(0 if ok else 1)
    png(a.name, a.period, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
