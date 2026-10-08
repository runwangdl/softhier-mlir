"""DSE sweep driver: analytic ranking of a knob grid, simulation of the top-K points per unique
kernel shape on a PRIVATE SoftHier copy, composition into end-to-end estimates.

    python -m softhier_mlir.dse.sweep --home /app/softhier_dse \
        --arch noc_link_width=256,512,1024 --arch redmule_ce=64x64,128x32 --arch mesh=1x1,4x4 \
        --top 12 --cache docs/dse/cache.json --md docs/dse/sweep.md --csv docs/dse/sweep.csv

Arch knobs are any field of softhier_mlir.sim.gvsoc.Arch (plus the shorthands ``mesh=XxY`` and
``redmule_ce=HxW``); kernel knobs are ``tile=TMxTNxTK`` and ``pipeline=0,1`` (applied to every
gemm the tile divides, see Workload.retile).

Numbers labelled "composed" are sums / maxima of per-shape kernel simulations weighted by the
workload's counts: they are not end-to-end measurements (no inter-op cache effects, no HBM
contention between concurrently running per-cluster ops).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from softhier_mlir.dse import calibrate as cb
from softhier_mlir.dse import cost
from softhier_mlir.dse import workload as wlm
from softhier_mlir.sim import gvsoc
from softhier_mlir.sim.gvsoc import Arch

KERNEL_KNOBS = ("tile", "pipeline")


# ----------------------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    arch: Arch
    tile: tuple | None = None
    pipeline: int | None = None

    def label(self, knobs: tuple[str, ...]) -> dict:
        out = {}
        for k in knobs:
            if k == "mesh":
                out[k] = f"{self.arch.num_cluster_x}x{self.arch.num_cluster_y}"
            elif k == "redmule_ce":
                out[k] = f"{self.arch.redmule_ce_height}x{self.arch.redmule_ce_width}"
            elif k == "tile":
                out[k] = "x".join(map(str, self.tile)) if self.tile else "default"
            elif k == "pipeline":
                out[k] = self.pipeline if self.pipeline is not None else "default"
            else:
                out[k] = getattr(self.arch, k)
        return out

    def workload(self, wl: wlm.Workload) -> wlm.Workload:
        w = wl.retile(self.tile, self.pipeline) if (self.tile or self.pipeline is not None) else wl
        return w.remap_clusters(self.arch.n_clusters)


def parse_knob(spec: str) -> tuple[str, list]:
    """'noc_link_width=256,512' -> ('noc_link_width', [256, 512]); mesh / redmule_ce / tile keep strings."""
    name, _, vals = spec.partition("=")
    name = name.strip()
    items = [v.strip() for v in vals.split(",") if v.strip()]
    if name in ("mesh", "redmule_ce", "tile"):
        return name, items
    return name, [int(v, 0) for v in items]


def grid_points(knobs: dict[str, list], base: Arch | None = None) -> list[Point]:
    base = base or Arch()
    names = list(knobs)
    pts = []
    for combo in itertools.product(*(knobs[n] for n in names)):
        arch_kw, tile, pipe = {}, None, None
        for n, v in zip(names, combo):
            if n == "mesh":
                x, y = (int(s) for s in v.lower().split("x"))
                arch_kw["num_cluster_x"], arch_kw["num_cluster_y"] = x, y
            elif n == "redmule_ce":
                h, w = (int(s) for s in v.lower().split("x"))
                arch_kw["redmule_ce_height"], arch_kw["redmule_ce_width"] = h, w
            elif n == "tile":
                tile = tuple(int(s) for s in v.lower().split("x"))
            elif n == "pipeline":
                pipe = int(v)
            else:
                arch_kw[n] = v
        pts.append(Point(replace(base, **arch_kw), tile, pipe))
    return pts


# ----------------------------------------------------------------------------------------
# cache + simulation
# ----------------------------------------------------------------------------------------
def arch_key(arch: Arch) -> dict:
    d = asdict(arch)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def cache_key(arch: Arch, case: cb.Case) -> str:
    return json.dumps({"arch": arch_key(arch), "case": [case.kind, list(case.p)]}, sort_keys=True, separators=(",", ":"))


class Cache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text())

    def get(self, arch: Arch, case: cb.Case):
        e = self.data.get(cache_key(arch, case))
        return e["cycles"] if e else None

    def put(self, arch: Arch, case: cb.Case, cycles, extra: dict | None = None) -> None:
        self.data[cache_key(arch, case)] = {"cycles": cycles, "when": time.strftime("%Y-%m-%d %H:%M"), **(extra or {})}

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))


def case_of(op: wlm.OpRec) -> cb.Case:
    mode = "all" if op.cluster < 0 else "one"
    if op.kind == "gemm":
        return cb.Case.gemm(*op.shape, *op.tiles, op.pipeline, mode)
    if op.kind == "summa":
        return cb.Case.summa(*op.shape, op.tiles[0], op.tiles[2], op.pipeline)
    return cb.Case.rowop(op.kind, op.shape[0], op.shape[1], mode)


def simulate_point(pt: Point, wl: wlm.Workload, cache: Cache, build_dir: str | Path | None = None,
                   feasible=None, verbose: bool = True) -> dict[tuple, float | None]:
    """Simulate every unique kernel of `wl` on arch `pt.arch` (one gvsoc run for all of them),
    through the cache. Returns sig -> cycles (None = failed / infeasible)."""
    wl = pt.workload(wl)
    todo, out = [], {}
    for sig, (rec, _) in wl.unique().items():
        case = case_of(rec)
        if feasible and not feasible(rec):
            out[sig] = None
            continue
        c = cache.get(pt.arch, case)
        if c is None:
            todo.append((sig, case))
        else:
            out[sig] = c
    if todo:
        gvsoc.apply_arch(pt.arch)
        cases = [cb.Case.barrier()] + [c for _, c in todo]
        if verbose:
            print(f"  simulating {len(todo)} kernels on {gvsoc.SH} ...", end="", flush=True)
        t0 = time.time()
        rows = cb.run_cases(cases, pt.arch, build_dir=build_dir)
        base = rows[0]["cycles"] or 0
        for (sig, case), row in zip(todo, rows[1:]):
            cyc = (row["cycles"] - base) if row["cycles"] is not None else None
            cache.put(pt.arch, case, cyc, {"raw": row["cycles"], "barrier": base})
            out[sig] = cyc
        cache.save()
        if verbose:
            print(f" {time.time() - t0:.0f} s, {sum(r['ok'] for r in rows[1:])}/{len(todo)} ok")
    return out


def compose_sim(wl: wlm.Workload, sims: dict[tuple, float | None]) -> dict | None:
    if any(v is None for v in sims.values()):
        return None
    return cost.compose(wl, lambda op, n: 0.0 if op.kind == "barrier" else sims[op.sig])


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------
def evaluate(wl: wlm.Workload, pts: list[Point], prm: cost.CostParams) -> list[dict]:
    rows = []
    for pt in pts:
        w = pt.workload(wl)
        r = cost.estimate(w, pt.arch, prm)
        rows.append({"point": pt, "model": r["cycles"], "model_by_kind": r["by_kind"], "est": r})
    rows.sort(key=lambda r: r["model"])
    return rows


def run(wl: wlm.Workload, knobs: dict[str, list], top: int, home: str | None, cache_path: str | None,
        prm: cost.CostParams | None = None, build_dir: str | None = None, base: Arch | None = None,
        verbose: bool = True) -> list[dict]:
    prm = prm or cost.CostParams()
    pts = grid_points(knobs, base)
    rows = evaluate(wl, pts, prm)
    if top > 0:
        if home:
            gvsoc.set_home(home)
        if gvsoc.SH.resolve() == gvsoc.SHARED_HOME.resolve():
            raise SystemExit("refusing to simulate an arch sweep on the shared install; pass --home <private copy>")
        cache = Cache(cache_path)
        for i, r in enumerate(rows[:top]):
            pt = r["point"]
            if verbose:
                print(f"[{i + 1}/{min(top, len(rows))}] {pt.label(tuple(knobs))} model={r['model']:,.0f}")
            if r["model"] == float("inf"):
                r["sim"] = None
                continue
            feasible = lambda rec, a=pt.arch: cost.op_est(a, prm, rec).cycles != float("inf")  # noqa: E731
            sims = simulate_point(pt, wl, cache, build_dir, feasible, verbose)
            comp = compose_sim(pt.workload(wl), sims)
            r["sims"] = sims
            r["sim"] = comp["cycles"] if comp else None
            r["sim_by_kind"] = comp["by_kind"] if comp else None
    return rows


def table_md(rows: list[dict], knobs: tuple[str, ...], wl: wlm.Workload) -> str:
    kinds = sorted({k for r in rows for k in r["model_by_kind"]}, key=lambda k: -max(r["model_by_kind"].get(k, 0) for r in rows))
    head = list(knobs) + ["model us", "composed us", "err %"] + [f"{k} (model/composed us)" for k in kinds]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lab = r["point"].label(knobs)
        sim = r.get("sim")
        cells = [str(lab[k]) for k in knobs]
        cells.append("inf" if r["model"] == float("inf") else f"{r['model'] / 1e3:.1f}")
        cells.append(f"{sim / 1e3:.1f}" if sim else "-")
        cells.append(f"{100 * (r['model'] - sim) / sim:+.1f}" if sim else "-")
        for k in kinds:
            m = r["model_by_kind"].get(k, 0) / 1e3
            s = (r.get("sim_by_kind") or {}).get(k)
            cells.append(f"{m:.1f} / {s / 1e3:.1f}" if s is not None else f"{m:.1f} / -")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shapes_md(rows: list[dict], knobs: tuple[str, ...], wl: wlm.Workload, prm: cost.CostParams) -> str:
    """Per unique kernel: model vs simulated cycles for every simulated point."""
    out = []
    for r in rows:
        if not r.get("sims"):
            continue
        pt = r["point"]
        w = pt.workload(wl)
        out.append(f"\n**{pt.label(knobs)}**\n")
        out.append("| kernel | count | model cyc | sim cyc | err % | model note |\n|---|---|---|---|---|---|")
        for sig, (rec, n) in w.unique().items():
            s = r["sims"].get(sig)
            e = cost.op_est(pt.arch, prm, rec, 1)
            err = f"{100 * (e.cycles - s) / s:+.1f}" if s else "-"
            out.append(f"| {rec} | {n} | {e.cycles:,.0f} | {s if s is not None else '-'} | {err} | {e.bound}: {e.note} |")
    return "\n".join(out)


def write_csv(rows: list[dict], knobs: tuple[str, ...], path: str) -> None:
    kinds = sorted({k for r in rows for k in r["model_by_kind"]})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(knobs) + ["model_cycles", "composed_cycles", "err_pct"] + [f"model_{k}" for k in kinds] + [f"composed_{k}" for k in kinds])
        for r in rows:
            lab = r["point"].label(knobs)
            sim = r.get("sim")
            w.writerow([lab[k] for k in knobs] + [r["model"], sim if sim else "", (100 * (r["model"] - sim) / sim) if sim else ""]
                       + [r["model_by_kind"].get(k, 0) for k in kinds]
                       + [(r.get("sim_by_kind") or {}).get(k, "") for k in kinds])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help=".mlir workload (default: SigLIP layer from the frontend)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--arch", action="append", default=[], help="knob=v1,v2,...  (Arch field, mesh=XxY, redmule_ce=HxW)")
    ap.add_argument("--kernel", action="append", default=[], help="tile=TMxTNxTK,... | pipeline=0,1")
    ap.add_argument("--top", type=int, default=0, help="simulate the K best model points (0 = model only)")
    ap.add_argument("--home", default=os.environ.get("SOFTHIER_HOME"), help="PRIVATE SoftHier copy for apply_arch")
    ap.add_argument("--cache", default="docs/dse/cache.json")
    ap.add_argument("--params", help="JSON with CostParams overrides (from calibrate --fit)")
    ap.add_argument("--build-dir", default=os.environ.get("SOFTHIER_BUILD_DIR"))
    ap.add_argument("--md", help="write the markdown table here")
    ap.add_argument("--csv", help="write the CSV here")
    a = ap.parse_args()
    wl = wlm.from_file(a.input) if a.input else wlm.siglip(seq=a.seq, layers=a.layers)
    knobs = dict(parse_knob(s) for s in a.arch + a.kernel)
    prm = cost.CostParams()
    if a.params:
        prm = cost.params_from_json(a.params)
    rows = run(wl, knobs, a.top, a.home, a.cache, prm, a.build_dir)
    kn = tuple(knobs)
    md = (f"Workload `{wl.name}`: {len(wl)} ops, {wl.total_macs / 1e9:.2f} GMAC. Model = analytic estimate; "
          f"composed = per-shape kernel simulations x counts through the same timeline (not an end-to-end measurement).\n\n"
          + table_md(rows, kn, wl) + "\n" + shapes_md(rows, kn, wl, prm))
    print(md)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md + "\n")
    if a.csv:
        write_csv(rows, kn, a.csv)


if __name__ == "__main__":
    main()
