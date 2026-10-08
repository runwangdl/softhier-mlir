"""Cost table of one SmolVLA inference on SoftHier (16 clusters, gvsoc at 1 GHz, ideal HBM) for the algorithm side.

    python -m softhier_mlir.dse.e2e_cost --collect      # gvsoc run summaries -> docs/dse/e2e/measured.json
    python -m softhier_mlir.dse.e2e_cost                # measured.json -> docs/dse/e2e_cost.csv + docs/dse/e2e_cost.md

Grid: cameras {1, 3} x SigLIP tokens per camera {256, 1024} x flow steps {1, 2, 5, 10} x action chunk {25, 50}.
Per configuration: simulated ms of vision / connector / prefix / expert (KV projection once per chunk, per step,
total), HBM bytes per segment, RedMulE busy % and per-cluster utilisation. Every number carries its source:

  measured  the chained gvsoc program (tests/gvsoc/run.py smolvla-e2e) at that configuration; fewer flow steps =
            the first k per-step times of the 10-step run (a step's cost does not depend on t)
  composed  gvsoc measurements of the same segment code in separate programs, multiplied out: 1024-token vision =
            embedding + 12 x one measured layer + post-LN (run.py smolvla-e2e --vision-only 1024), full-resolution
            connector + prefix = the 113 / 241-token prefix program (--prefix-only, 8 layers, x 2), expert at a prefix
            length / chunk the chain does not have (--expert-only)
  model     the calibrated segment model below, for cells no simulation covers

Segment model (fitted on the measured / composed points, residuals in the markdown): expert step(Lp, C) =
a + b C + c C Lp (weight streaming of the 16 layers is chunk-independent, row ops scale with C, attention with C x
keys); KV projection(Lp) = a + b Lp; bytes alike; RedMulE busy = k x MACs of the segment (k fitted).
All of it under ideal HBM (DRAMSys is x86-only): the HBM-bound segments are optimistic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests" / "gvsoc" / "smolvla_e2e_app"
MEASURED = ROOT / "docs" / "dse" / "e2e" / "measured.json"
NCL = 16
GROUP_RE = [("vision", re.compile(r"(vemb|attn|layer|vis)\d+$"), ("vision",)), ("connector", re.compile(r"conn$"), ("vision",)),
            ("connector", re.compile(r"emb$"), ("prefix_a",)), ("prefix", re.compile(r"(attn|layer)\d+$"), ("prefix_a", "prefix_b")),
            ("expert_kvproj", re.compile(r"kvproj$"), ("expert",)), ("expert_step", re.compile(r"step\d+$"), ("expert",))]
TD, TFF, TKVD = 960, 2560, 320
XD, XDQ, XFF = 720, 960, 2048


# ----------------------------------------------------------------------------- collection
def groups(phases: dict) -> dict:
    """{group: {dur_ns, hbm_rd, hbm_wr, redmule[c], idma[c], sync[c], n}} from the trace segments of a run's phases."""
    out: dict = {}
    for ph, rec in phases.items():
        for s in rec.get("segments", []):
            for g, rx, phs in GROUP_RE:
                if ph in phs and rx.fullmatch(s["tag"]):
                    if ph == "vision" and g == "vision" and s["tag"].startswith("vemb") and False:
                        pass
                    a = out.setdefault(g, {"dur_ns": 0, "hbm_rd": 0.0, "hbm_wr": 0.0, "redmule": [0] * NCL, "idma": [0] * NCL, "sync": [0] * NCL, "n": 0})
                    a["dur_ns"] += s["dur_ns"]; a["hbm_rd"] += s["hbm_rd"]; a["hbm_wr"] += s["hbm_wr"]; a["n"] += 1
                    for k in ("redmule", "idma", "sync"):
                        a[k] = [x + y for x, y in zip(a[k], s[k])]
                    break
    return out


def collect(runs: Path = RUNS, out: Path = MEASURED) -> dict:
    """Compact records of every finished run summary (chain / vision-only / prefix-only / expert-only)."""
    recs = []
    for f in sorted(runs.glob("*/summary.json")):
        s = json.loads(f.read_text())
        r = {"run": f.parent.name, "trace": "phases" in s and any("segments" in p for p in s["phases"].values())}
        if "tokens_per_camera" in s:
            r.update(kind="chain", cams=s["cams"], T=s["tokens_per_camera"], n=s["prefix_tokens"], chunk=s["chunk"], steps=s["steps"],
                     seg=s["segments_ms"], acc={k: v for k, v in s["acc"].items() if k in ("VIS", "IMG", "actions")},
                     kv_max={"K": max(v["max_abs"] for k, v in s["acc"].items() if re.fullmatch(r"K\d+", k)),
                             "V": max(v["max_abs"] for k, v in s["acc"].items() if re.fullmatch(r"V\d+", k))},
                     xt=[s["acc"][f"X{i}"]["max_abs"] for i in range(s["steps"])])
        elif "kv_src" in s:
            r.update(kind="expert", n=s["prefix_tokens"], chunk=s["chunk"], steps=s["steps"], seg=s["segments_ms"], acc=s["acc"])
        elif "T" in s and "layers" in s:
            r.update(kind="vision", T=s["T"], layers=s["layers"], seg=s["segments_ms"], acc=s["acc"])
        elif "prefix_tokens" in s and "layers" in s:
            r.update(kind="prefix", cams=s["cams"], n=s["prefix_tokens"], layers=s["layers"], seg=s["segments_ms"])
        else:
            continue
        if r["trace"]:
            r["groups"] = groups(s["phases"])
            if r["kind"] == "vision":      # split the vision phase: embedding / layers / post-LN for the composition
                r["vgroups"] = vision_split(s["phases"]["vision"]["segments"])
        recs.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, indent=1))
    print(f"[e2e_cost] {len(recs)} run records -> {out}")
    return recs


def vision_split(segs: list[dict]) -> dict:
    """embed (-> vemb0), layer (attn<l> + layer<l>, summed over the layers run), post (-> vis0), conn."""
    out: dict = {}
    for s in segs:
        g = "embed" if s["tag"] == "vemb0" else "layer" if re.fullmatch(r"(attn|layer)\d+", s["tag"]) else "post" if s["tag"] == "vis0" else \
            "conn" if s["tag"] == "conn" else None
        if g is None:
            continue
        a = out.setdefault(g, {"dur_ns": 0, "hbm_rd": 0.0, "hbm_wr": 0.0, "redmule": [0] * NCL, "idma": [0] * NCL, "sync": [0] * NCL, "n": 0})
        a["dur_ns"] += s["dur_ns"]; a["hbm_rd"] += s["hbm_rd"]; a["hbm_wr"] += s["hbm_wr"]; a["n"] += 1
        for k in ("redmule", "idma", "sync"):
            a[k] = [x + y for x, y in zip(a[k], s[k])]
    return out


# ----------------------------------------------------------------------------- analytic MACs (what RedMulE has to do)
def macs(seg: str, T: int = 256, cams: int = 1, n: int = 65, Lp: int = 65, C: int = 50) -> float:
    S = ((n + 127) // 128) * 128
    if seg == "vision":         # per inference: patch embedding + 12 layers (projections, FFN, per-head QK^T / PV) per camera
        return cams * (T * 768 * 768 + 12 * (4 * T * 768 ** 2 + 2 * T * 768 * 3072 + 2 * 12 * T * T * 64))
    if seg == "connector":
        return cams * (T // 16) * 12288 * TD
    if seg == "prefix":         # 16 layers on S_pad rows
        return 16 * (S * TD * (TD + 2 * TKVD + TD) + 3 * S * TD * TFF + 2 * 15 * S * S * 64)
    if seg == "expert_kvproj":
        return 8 * 2 * Lp * TKVD * TKVD
    if seg == "expert_step":    # suffix embedding + 8 self + 8 cross layers + output projection
        lay = 0
        for self_ in (True, False):
            keys = Lp + (C if self_ else 0)
            lay += 8 * (C * XD * (XDQ + (2 * TKVD if self_ else 0)) + C * XDQ * XD + C * XD * 2 * XFF + C * XFF * XD + 2 * 15 * C * keys * 64)
        return C * 32 * XD + 2 * C * XD * XD + lay + C * XD * 32
    raise KeyError(seg)


# ----------------------------------------------------------------------------- table
STEPS, CHUNKS, CAMS, TOKENS = (1, 2, 5, 10), (25, 50), (1, 3), (256, 1024)


def _scale(g: dict, f: float) -> dict:
    return {"dur_ns": g["dur_ns"] * f, "hbm_rd": g["hbm_rd"] * f, "hbm_wr": g["hbm_wr"] * f,
            "redmule": [x * f for x in g["redmule"]], "idma": [x * f for x in g["idma"]], "sync": [x * f for x in g["sync"]]}


def _add(*gs: dict) -> dict:
    gs = [g for g in gs if g]
    return {"dur_ns": sum(g["dur_ns"] for g in gs), "hbm_rd": sum(g["hbm_rd"] for g in gs), "hbm_wr": sum(g["hbm_wr"] for g in gs),
            "redmule": [sum(x) for x in zip(*(g["redmule"] for g in gs))], "idma": [sum(x) for x in zip(*(g["idma"] for g in gs))],
            "sync": [sum(x) for x in zip(*(g["sync"] for g in gs))]}


class Model:
    """Least-squares segment models of the expert (the only segment whose (prefix length, chunk) points are not all simulated)."""

    def __init__(self, recs: list[dict]):
        pts = {}
        for r in recs:
            if r["kind"] in ("chain", "expert") and "groups" in r:
                pts[(r["n"], r["chunk"])] = r
        self.pts = pts
        X, y_t, y_b, y_k = [], [], [], []
        Xk, yk_t, yk_b = [], [], []
        for (Lp, C), r in pts.items():
            st = r["seg"]["expert_per_step"]
            g = r["groups"]["expert_step"]
            X.append([1.0, C, C * Lp]); y_t.append(np.mean(st))
            y_b.append((g["hbm_rd"] + g["hbm_wr"]) / g["n"])
            y_k.append(sum(g["redmule"]) / g["n"] / macs("expert_step", Lp=Lp, C=C))
            gk = r["groups"]["expert_kvproj"]
            Xk.append([1.0, Lp]); yk_t.append(r["seg"]["expert_kv_projection"]); yk_b.append(gk["hbm_rd"] + gk["hbm_wr"])
        self.X, self.y_t = np.array(X), np.array(y_t)
        self.c_t = np.linalg.lstsq(self.X, self.y_t, rcond=None)[0] if len(X) >= 3 else None
        self.c_b = np.linalg.lstsq(self.X, np.array(y_b), rcond=None)[0] if len(X) >= 3 else None
        self.k_busy = float(np.mean(y_k)) if y_k else None
        self.ck_t = np.linalg.lstsq(np.array(Xk), np.array(yk_t), rcond=None)[0] if len(Xk) >= 2 else None
        self.ck_b = np.linalg.lstsq(np.array(Xk), np.array(yk_b), rcond=None)[0] if len(Xk) >= 2 else None

    def step_ms(self, Lp, C):
        return float(self.c_t @ [1.0, C, C * Lp])

    def residuals(self) -> list[tuple]:
        """leave-one-out error of the step-time model on every simulated (Lp, C) point"""
        out = []
        keys = list(self.pts)
        for i, (Lp, C) in enumerate(keys):
            m = np.ones(len(keys), bool); m[i] = False
            if m.sum() < 3:
                continue
            c = np.linalg.lstsq(self.X[m], self.y_t[m], rcond=None)[0]
            pred = float(c @ [1.0, C, C * Lp])
            out.append((Lp, C, float(self.y_t[i]), pred, 100 * (pred - self.y_t[i]) / self.y_t[i]))
        return out

    def expert(self, Lp, C) -> tuple[dict, dict, list]:
        """(kvproj group, per-step group, per-step ms list of 10) from the model"""
        st = self.step_ms(Lp, C)
        busy = self.k_busy * macs("expert_step", Lp=Lp, C=C)
        g_step = {"dur_ns": st * 1e6, "hbm_rd": float(self.c_b @ [1.0, C, C * Lp]), "hbm_wr": 0.0, "redmule": [busy / NCL] * NCL,
                  "idma": [float("nan")] * NCL, "sync": [float("nan")] * NCL, "n": 1}
        kv = float(self.ck_t @ [1.0, Lp])
        g_kv = {"dur_ns": kv * 1e6, "hbm_rd": float(self.ck_b @ [1.0, Lp]), "hbm_wr": 0.0,
                "redmule": [self.k_busy * macs("expert_kvproj", Lp=Lp) / NCL] * NCL, "idma": [float("nan")] * NCL, "sync": [float("nan")] * NCL, "n": 1}
        return g_kv, g_step, [st] * 10


def build_table(recs: list[dict]) -> list[dict]:
    chains = {(r["cams"], r["T"]): r for r in recs if r["kind"] == "chain" and "groups" in r}
    vis = {r["T"]: r for r in recs if r["kind"] == "vision" and "vgroups" in r}
    pre = {r["cams"]: r for r in recs if r["kind"] == "prefix" and "groups" in r}
    exp = {(r["n"], r["chunk"]): r for r in recs if r["kind"] in ("expert", "chain") and "groups" in r}
    model = Model(recs)
    rows = []
    for cams in CAMS:
        for T in TOKENS:
            n = cams * T // 16 + 49
            ch = chains.get((cams, T))
            seg: dict = {}
            src: dict = {}
            if ch:
                g = ch["groups"]
                seg["vision"], seg["connector"], seg["prefix"] = g["vision"], g["connector"], g["prefix"]
                src.update(vision="measured", connector="measured", prefix="measured")
            else:
                v = vis[T]["vgroups"]
                L = vis[T]["layers"]
                seg["vision"] = _scale(_add(v["embed"], _scale(v["layer"], 12 / L), v["post"]), cams)
                p = pre[cams]
                seg["connector"] = p["groups"]["connector"]
                seg["prefix"] = _scale(p["groups"]["prefix"], 16 / p["layers"])
                src.update(vision="composed", connector="composed", prefix="composed")
            for C in CHUNKS:
                e = exp.get((n, C))
                if e:
                    g_kv, g_step = e["groups"]["expert_kvproj"], e["groups"]["expert_step"]
                    per_step = e["seg"]["expert_per_step"]
                    xsrc = "measured" if e["kind"] == "chain" and ch else "composed"
                else:
                    g_kv, g_step, per_step = model.expert(n, C)
                    xsrc = "model"
                for steps in STEPS:
                    # the first `steps` per-step times; bytes / busy of a step: the group mean
                    f = steps / g_step.get("n", 1)
                    g_steps = _scale(g_step, f)
                    g_steps["dur_ns"] = sum(per_step[:steps]) * 1e6
                    segs = {"vision": seg["vision"], "connector": seg["connector"], "prefix": seg["prefix"], "expert_kvproj": g_kv, "expert_steps": g_steps}
                    tot = _add(*segs.values())
                    row = {"cams": cams, "tokens_per_camera": T, "prefix_tokens": n, "flow_steps": steps, "chunk": C}
                    for k, g in segs.items():
                        row[f"{k}_ms"] = g["dur_ns"] / 1e6
                    row["expert_per_step_ms"] = float(np.mean(per_step[:steps]))
                    row["expert_total_ms"] = row["expert_kvproj_ms"] + row["expert_steps_ms"]
                    row["total_ms"] = tot["dur_ns"] / 1e6
                    for k, g in segs.items():
                        row[f"{k}_hbm_MB"] = (g["hbm_rd"] + g["hbm_wr"]) / 1e6
                    row["total_hbm_MB"] = (tot["hbm_rd"] + tot["hbm_wr"]) / 1e6
                    row["GMAC"] = (macs("vision", T, cams) + macs("connector", T, cams) + macs("prefix", n=n)
                                   + macs("expert_kvproj", Lp=n) + steps * macs("expert_step", Lp=n, C=C)) / 1e9
                    for k, g in segs.items():
                        row[f"{k}_redmule_pct"] = 100 * sum(g["redmule"]) / (NCL * g["dur_ns"]) if g["dur_ns"] else float("nan")
                    per_cl = [100 * b / tot["dur_ns"] for b in tot["redmule"]]
                    row["redmule_pct"] = float(np.mean(per_cl))
                    measured_cl = not any(math.isnan(x) for x in tot["idma"])
                    row["redmule_pct_cluster_min"] = min(per_cl) if measured_cl else float("nan")
                    row["redmule_pct_cluster_max"] = max(per_cl) if measured_cl else float("nan")
                    row["idma_pct"] = float(np.mean([100 * b / tot["dur_ns"] for b in tot["idma"]])) if measured_cl else float("nan")
                    row["redmule_pct_per_cluster"] = ";".join(f"{x:.2f}" for x in per_cl) if measured_cl else ""
                    row["idma_pct_per_cluster"] = ";".join(f"{100 * b / tot['dur_ns']:.2f}" for b in tot["idma"]) if measured_cl else ""
                    s = {**src, "expert": xsrc}
                    row["source"] = "measured" if all(v == "measured" for v in s.values()) else \
                        "model" if "model" in s.values() else "composed"
                    row["source_detail"] = ";".join(f"{k}={v}" for k, v in s.items())
                    rows.append(row)
    build_table.model = model
    return rows


def write(rows: list[dict], recs: list[dict], csv_path: Path, md_path: Path) -> None:
    keys = list(rows[0])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    m = build_table.model
    L = ["# SmolVLA end-to-end cost table (SoftHier 4x4, gvsoc 1 GHz, ideal HBM)", "",
         "Generated by `python -m softhier_mlir.dse.e2e_cost` from `docs/dse/e2e/measured.json`; full columns (HBM MB and RedMulE % per "
         "segment, per-cluster RedMulE / iDMA busy %) in `docs/dse/e2e_cost.csv`. Source: measured = the chained gvsoc program at that "
         "configuration; composed = gvsoc runs of the same segment programs multiplied out (1024-token vision = embedding + 12 x one "
         "layer + post-LN; full-resolution prefix = 2 x 8 layers); model = the fitted expert model (below). Context: docs/SMOLVLA_E2E.md.", "",
         "| cams | tok/cam | prefix | steps | chunk | vision ms | conn ms | prefix ms | expert ms/step | expert total ms | **total ms** | HBM MB | GMAC | RedMulE % (cluster min-max) | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cl = f"{r['redmule_pct']:.1f} ({r['redmule_pct_cluster_min']:.1f}-{r['redmule_pct_cluster_max']:.1f})" if not math.isnan(r["redmule_pct_cluster_min"]) \
            else f"{r['redmule_pct']:.1f} (model)"
        L.append(f"| {r['cams']} | {r['tokens_per_camera']} | {r['prefix_tokens']} | {r['flow_steps']} | {r['chunk']} | {r['vision_ms']:.2f} | "
                 f"{r['connector_ms']:.3f} | {r['prefix_ms']:.2f} | {r['expert_per_step_ms']:.3f} | {r['expert_total_ms']:.2f} | **{r['total_ms']:.2f}** | "
                 f"{r['total_hbm_MB']:.0f} | {r['GMAC']:.1f} | {cl} | {r['source']} ({r['source_detail'].replace(';', ', ')}) |")
    L += ["", "## Per-segment HBM traffic and RedMulE busy (10 steps, chunk 50)", "",
          "| cams | tok/cam | vision MB / RedMulE % | connector MB / % | prefix MB / % | expert KV proj MB / % | expert 10 steps MB / % |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["flow_steps"] == 10 and r["chunk"] == 50:
            L.append(f"| {r['cams']} | {r['tokens_per_camera']} | " + " | ".join(
                f"{r[f'{k}_hbm_MB']:.0f} / {r[f'{k}_redmule_pct']:.1f}" for k in ("vision", "connector", "prefix", "expert_kvproj", "expert_steps")) + " |")
    L += ["", "## Expert model", "",
          f"step ms = {m.c_t[0]:.4f} + {m.c_t[1]:.5f} C + {m.c_t[2]:.3e} C Lp (C = action rows, Lp = prefix tokens), fitted on "
          f"{len(m.pts)} simulated (Lp, C) points; KV projection ms = {m.ck_t[0]:.4f} + {m.ck_t[1]:.3e} Lp; RedMulE busy = "
          f"{m.k_busy:.3f} ns per MAC summed over clusters (from the traces).", "",
          "| Lp | C | simulated ms/step | leave-one-out model | error |", "|---|---|---|---|---|"]
    for Lp, C, y, p, e in m.residuals():
        L.append(f"| {Lp} | {C} | {y:.3f} | {p:.3f} | {e:+.1f} % |")
    L += ["", "## Runs", "", "| run | kind | key numbers |", "|---|---|---|"]
    for r in recs:
        k = {kk: r[kk] for kk in ("cams", "T", "n", "chunk", "layers") if kk in r}
        acc = r.get("acc", {})
        L.append(f"| {r['run']} | {r['kind']} {k} | " + ", ".join(f"{a}: {json.dumps(v)}" for a, v in acc.items())[:300] + " |")
    md_path.write_text("\n".join(L) + "\n")
    print(f"[e2e_cost] {len(rows)} configurations -> {csv_path}, {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true", help="re-read the run summaries under --runs into --measured first")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--measured", default=str(MEASURED))
    ap.add_argument("--csv", default=str(ROOT / "docs" / "dse" / "e2e_cost.csv"))
    ap.add_argument("--md", default=str(ROOT / "docs" / "dse" / "e2e_cost.md"))
    a = ap.parse_args()
    recs = collect(Path(a.runs), Path(a.measured)) if a.collect else json.loads(Path(a.measured).read_text())
    rows = build_table(recs)
    write(rows, recs, Path(a.csv), Path(a.md))


if __name__ == "__main__":
    main()
