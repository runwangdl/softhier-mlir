"""Turn a gvsoc run log with `--trace=redmule --trace=idma --trace=cluster_registers` into an activity
timeline: a Chrome/Perfetto trace-event JSON (open at https://ui.perfetto.dev), a PNG, and a text
utilisation summary. Unlike the SDK's trace_perfetto/parse.py this strips the ANSI colour codes gvsoc
prints and clips to the kernel window (first to last RedMulE/iDMA event).

    python -m softhier_mlir.sim.trace run.log --perfetto out.json --png out.png
    run_sim(traces=("redmule", "idma", "cluster_registers")) produces such a log in r["stdout"].
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RED = re.compile(r"/chip/cluster_(\d+)/redmule/trace\s*\] \[LightRedmule\] Finished : (\d+) ns ---> (\d+) ns .*?uti = ([0-9.]+)")
_DMA = re.compile(r"/chip/cluster_(\d+)/idma/fe/trace\s*\] \[iDMA\] Finished : (\d+) ns ---> (\d+) ns .*?Txn 0 = \{(.*?)\}")
_SYNC = re.compile(r"/chip/cluster_(\d+)/cluster_registers/trace\s*\] Cluster Sync: (\d+) ns -> (\d+) ns .*?Type = (\d)")


def parse(log_text: str) -> list[dict]:
    """-> [{cluster, unit, t0, t1, info}] with ns timestamps (unit: redmule | idma | sync)."""
    ev = []
    for ln in log_text.splitlines():
        ln = _ANSI.sub("", ln)
        m = _RED.search(ln)
        if m:
            ev.append(dict(cluster=int(m[1]), unit="redmule", t0=int(m[2]), t1=int(m[3]), info=f"util {m[4]}"))
            continue
        m = _DMA.search(ln)
        if m:
            ev.append(dict(cluster=int(m[1]), unit="idma", t0=int(m[2]), t1=int(m[3]), info=m[4][:80]))
            continue
        m = _SYNC.search(ln)
        if m:
            ev.append(dict(cluster=int(m[1]), unit="sync", t0=int(m[2]), t1=int(m[3]), info=f"barrier type {m[4]}"))
    return ev


def kernel_window(ev: list[dict]) -> tuple[int, int]:
    k = [e for e in ev if e["unit"] in ("redmule", "idma")]
    return (min(e["t0"] for e in k), max(e["t1"] for e in k)) if k else (0, 0)


def clip(ev: list[dict], w: tuple[int, int]) -> list[dict]:
    t0, t1 = w
    out = []
    for e in ev:
        if e["t1"] <= t0 or e["t0"] >= t1:
            continue
        out.append({**e, "t0": max(e["t0"], t0) - t0, "t1": min(e["t1"], t1) - t0})
    return out


def summary(ev: list[dict], span: int) -> str:
    busy = defaultdict(int)
    n = defaultdict(int)
    for e in ev:
        busy[(e["cluster"], e["unit"])] += e["t1"] - e["t0"]
        n[(e["cluster"], e["unit"])] += 1
    clusters = sorted({e["cluster"] for e in ev})
    lines = [f"kernel window {span / 1e3:.1f} us; busy % of the window per cluster", "cluster  redmule%  idma%   sync%   (#redmule ops)"]
    for c in clusters:
        lines.append(f"{c:>7}  {100 * busy[(c, 'redmule')] / span:7.1f}  {100 * busy[(c, 'idma')] / span:6.1f}  {100 * busy[(c, 'sync')] / span:6.1f}   ({n[(c, 'redmule')]})")
    tot_r = sum(busy[(c, 'redmule')] for c in clusters)
    lines.append(f"mean RedMulE utilisation over {len(clusters)} clusters: {100 * tot_r / (span * max(1, len(clusters))):.1f} %")
    return "\n".join(lines)


def to_perfetto(ev: list[dict]) -> dict:
    events = [{"name": "kernel start", "ph": "I", "ts": 0, "s": "g"}]
    tids = {"redmule": 1, "idma": 2, "sync": 3}
    for c in sorted({e["cluster"] for e in ev}):
        events.append({"name": "process_name", "ph": "M", "pid": c, "args": {"name": f"cluster {c}"}})
        for u, t in tids.items():
            events.append({"name": "thread_name", "ph": "M", "pid": c, "tid": t, "args": {"name": u}})
    for e in ev:
        events.append({"name": e["unit"] if e["unit"] != "redmule" else f"redmule {e['info']}", "ph": "X",
                       "ts": e["t0"] / 1e3, "dur": (e["t1"] - e["t0"]) / 1e3, "pid": e["cluster"],
                       "tid": tids[e["unit"]], "args": {"info": e["info"]}})
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def to_png(ev: list[dict], span: int, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clusters = sorted({e["cluster"] for e in ev})
    color = {"redmule": "#c2410c", "idma": "#1d4ed8", "sync": "#d4d4d8"}
    fig, ax = plt.subplots(figsize=(14, 0.42 * len(clusters) + 1.2))
    for i, c in enumerate(clusters):
        for e in ev:
            if e["cluster"] != c:
                continue
            y = i + (0.3 if e["unit"] == "idma" else 0.0)
            h = 0.28 if e["unit"] != "sync" else 0.6
            ax.broken_barh([(e["t0"] / 1e3, max(0.05, (e["t1"] - e["t0"]) / 1e3))], (y, h), color=color[e["unit"]],
                           alpha=0.35 if e["unit"] == "sync" else 1.0, linewidth=0)
    ax.set_yticks([i + 0.3 for i in range(len(clusters))]); ax.set_yticklabels([f"cluster {c}" for c in clusters], fontsize=8)
    ax.set_xlim(0, span / 1e3); ax.set_xlabel("time in kernel (us)"); ax.invert_yaxis()
    ax.set_title(title or "SoftHier activity: RedMulE (orange, upper lane) / iDMA (blue, lower lane) / barrier wait (grey)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--perfetto")
    ap.add_argument("--png")
    ap.add_argument("--json", help="clipped events as JSON (for custom viewers)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    ev = parse(Path(a.log).read_text())
    w = kernel_window(ev)
    ce = clip(ev, w)
    span = w[1] - w[0]
    print(summary(ce, span))
    if a.perfetto:
        Path(a.perfetto).write_text(json.dumps(to_perfetto(ce)))
    if a.json:
        Path(a.json).write_text(json.dumps({"span_ns": span, "events": ce}))
    if a.png:
        to_png(ce, span, Path(a.png), a.title)


# ----------------------------------------------------------------------------- per-segment statistics (W1 e2e)
_TXN = re.compile(r"Txn \d+ = \{type: (1D|2D), src: 0x([0-9a-f]+), (?:src_stride: 0x[0-9a-f]+, )?dst: 0x([0-9a-f]+), "
                  r"(?:dst_stride: 0x[0-9a-f]+, repeats: 0x([0-9a-f]+), )?size: 0x([0-9a-f]+)")
_MARK = re.compile(r"\[mark\] (\w+) (\d+)")
# any transfer id (the first Txn of a line is not always Txn 0, which _DMA requires)
_DMA_ANY = re.compile(r"/chip/cluster_(\d+)/idma/fe/trace\s*\] \[iDMA\] Finished : (\d+) ns ---> (\d+) ns")
HBM_LO = 0xC0000000


_TRACE_HEAD = re.compile(r": (-?\d+): \[")


def _trace_start(ln: str) -> int:
    """Index where a gvsoc trace line (`<ps>: <ns>: [<path>] ...`, ps = ns * 1000, `0: -1:` before reset) starts inside
    a physical log line, -1 if none. The simulated program prints character by character, so in a traced run a whole
    trace line can be inserted in the middle of one of the program's lines (`[mark] start 30` + `86454000: 86454: [...`)."""
    for m in _TRACE_HEAD.finditer(ln):
        ns = int(m[1])
        ps = "0" if ns < 0 else str(ns * 1000)
        if ln[m.start() - len(ps):m.start()] == ps:
            return m.start() - len(ps)
    return -1


def program_lines(log: str | Path):
    """The simulated program's own output lines of a traced gvsoc log (trace lines cut out, split lines re-joined)."""
    buf = ""
    with open(log, errors="replace") as f:
        for ln in f:
            if "/chip/" in ln or "/sys/" in ln:
                k = _trace_start(ln)
                if k >= 0:
                    buf += ln[:k]
                    continue
            buf += ln
            if buf.endswith("\n"):
                yield buf
                buf = ""
    if buf:
        yield buf


def txn_bytes(info: str) -> tuple[int, int]:
    """(bytes read from HBM, bytes written to HBM) of the Txn list of one iDMA 'Finished' line (every transfer appears in
    exactly one such line: the front end prints and clears its list when the in-flight count drops to 0). 2-D: size is
    the row length, repeats the row count."""
    rd = wr = 0
    for m in _TXN.finditer(info):
        n = int(m[5], 16) * (int(m[4], 16) if m[4] else 1)
        if int(m[2], 16) >= HBM_LO:
            rd += n
        if int(m[3], 16) >= HBM_LO:
            wr += n
    return rd, wr


def segment_stats(log: str | Path, n_clusters: int = 16) -> list[dict]:
    """Per-segment activity of a gvsoc log run with --trace=redmule --trace=idma --trace=cluster_registers. Segments are
    delimited by the program's `[mark] <tag> <cycles>` lines; the mark's mcycle value is the simulated time in ns (1 GHz
    cluster clock, mcycle counts from reset: checked against the trace timestamps, the printf line itself comes out
    ~2 us later). -> [{tag, t_ns, dur_ns, hbm_rd, hbm_wr (bytes), redmule[c], idma[c], sync[c] (busy ns of cluster c
    inside the segment, events clipped to it), n_redmule, n_idma}] for the segment ENDING at each mark. HBM bytes of
    an iDMA busy period are split over the segments in proportion to its overlap."""
    marks: list[tuple[str, int]] = []
    prev_c, acc = None, 0
    for ln in program_lines(log):
        if ln.startswith("[mark] "):
            m = _MARK.match(ln)
            if m:
                c = int(m[2])
                if prev_c is None:
                    acc = c
                else:
                    acc += (c - prev_c) % (1 << 32)
                prev_c = c
                marks.append((m[1], acc))
    bounds = [t for _, t in marks]
    segs = [{"tag": tag, "t_ns": t, "dur_ns": t - (bounds[i - 1] if i else t), "hbm_rd": 0.0, "hbm_wr": 0.0,
             "redmule": [0] * n_clusters, "idma": [0] * n_clusters, "sync": [0] * n_clusters, "n_redmule": 0, "n_idma": 0}
            for i, (tag, t) in enumerate(marks)]
    import bisect

    def spread(t0: int, t1: int):
        """[(segment index, overlap ns)] of the interval [t0, t1) over the segments (i: (bounds[i-1], bounds[i]])"""
        out = []
        i = bisect.bisect_left(bounds, t0 + 1)
        while i < len(bounds):
            lo = bounds[i - 1] if i else bounds[0]
            ov = min(t1, bounds[i]) - max(t0, lo)
            if ov > 0:
                out.append((i, ov))
            if bounds[i] >= t1:
                break
            i += 1
        return out
    with open(log, errors="replace") as f:
        for ln in f:
            if "Finished" not in ln and "Cluster Sync" not in ln:
                continue
            ln = _ANSI.sub("", ln)
            m = _RED.search(ln)
            unit = "redmule"
            if not m:
                m = _DMA_ANY.search(ln); unit = "idma"
            if not m:
                m = _SYNC.search(ln); unit = "sync"
            if not m:
                continue
            c, t0, t1 = int(m[1]), int(m[2]), int(m[3])
            parts = spread(t0, t1)
            if unit == "idma":
                rd, wr = txn_bytes(ln[m.end():])
                tot = max(1, t1 - t0)
            for i, ov in parts:
                segs[i][unit][c] += ov
                if unit == "redmule":
                    segs[i]["n_redmule"] += 1
                elif unit == "idma":
                    segs[i]["n_idma"] += 1
                    segs[i]["hbm_rd"] += rd * ov / tot
                    segs[i]["hbm_wr"] += wr * ov / tot
            if unit == "idma" and not parts and t1 - t0 == 0:      # zero-length burst: by end time
                i = min(bisect.bisect_left(bounds, t1), len(bounds) - 1)
                segs[i]["hbm_rd"] += rd; segs[i]["hbm_wr"] += wr
    return segs


if __name__ == "__main__":
    main()
