#!/usr/bin/env python3
"""Figures of docs/WORLD_MODEL.md: python tools/world_model_plots.py  ->  docs/world_model/*.png

Expert curve from softhier_mlir.dse.cost.expert_step_est (the measured per-op table); RSSM points measured with
tests/gvsoc/wm.py --sweep 32,64,128 --cluster 0,all (2026-10-09; RSSM_MEASURED below)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from softhier_mlir.dse.cost import expert_step_est  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "world_model"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (model, K, clusters): ROI ns, per-step compute cycles of cluster 0, phases gemm / ln / gru / softmax (+head) over H=10
RSSM_MEASURED = {
    ("full", 32, 1): (6138764, 606914, 111575, 3087760, 1110610, 1738570 + 20630),
    ("full", 64, 1): (12135086, 1202950, 119885, 6195760, 2208500, 3468770 + 36580),
    ("full", 128, 1): (23987867, 2380956, 136555, 12268560, 4411730, 6925130 + 67590),
    ("full", 32, 16): (643170, 52080, 103365, 214580, 79010, 117770 + 6070),
    ("full", 64, 16): (1068980, 94406, 104175, 460180, 145300, 227170 + 7230),
    ("full", 128, 16): (1763105, 163276, 105205, 791410, 285330, 441930 + 8890),
    ("nano", 32, 1): (3088977, 302630, 56755, 1652640, 558800, 737470 + 20630),
    ("nano", 64, 1): (6077123, 597846, 64325, 3301470, 1112490, 1463590 + 36580),
    ("nano", 128, 1): (11980369, 1180972, 79135, 6532480, 2211920, 2918590 + 67590),
    ("nano", 32, 16): (325030, 27746, 49425, 123900, 42200, 55870 + 6070),
    ("nano", 64, 16): (541029, 49106, 50375, 253780, 79290, 100390 + 7230),
    ("nano", 128, 16): (884480, 82950, 51165, 431730, 145520, 192190 + 8890),
}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stacked(ax, xs, parts, names, fmt):
    bottom = [0.0] * len(xs)
    for i, (nm, vals) in enumerate(zip(names, parts)):
        ax.bar(xs, vals, 0.6, bottom=bottom, color=C[i], label=nm, edgecolor=SURF, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, b in zip(xs, bottom):
        ax.text(x, b, fmt(b), ha="center", va="bottom", fontsize=9, color=INK)


def expert():
    Ns = [1, 2, 4, 8]
    est = [expert_step_est(n) for n in Ns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    xs = list(range(len(Ns)))
    parts = [[e["split_us"][k] / 1e3 for e in est] for k in ("attention", "gemm", "rowops")]
    stacked(a1, xs, parts, ["attention", "weight GEMMs (+norm, residual)", "row ops (SiLU*up, RoPE, emb, tail)"], lambda v: f"{v:.1f}")
    a1.set_xticks(xs, [f"N={n}" for n in Ns]); a1.set_ylabel("ms per flow step (16 layers)", color=INK2)
    a1.set_title("Expert step with N candidate chunks, 16 clusters", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    style(a1)
    pc = [e["per_cand_cycles"] / est[0]["per_cand_cycles"] for e in est]
    pb = [e["per_cand_bytes"] / est[0]["per_cand_bytes"] for e in est]
    a2.plot(Ns, [1 / n for n in Ns], color=INK2, linewidth=1, linestyle="--", label="ideal 1/N (weight-bound)")
    a2.plot(Ns, pc, color=C[0], linewidth=2, marker="o", markersize=7, label="time per candidate")
    a2.plot(Ns, pb, color=C[1], linewidth=2, marker="s", markersize=7, label="HBM bytes per candidate")
    for n, v, e in zip(Ns, pc, est):
        a2.annotate(f"{e['per_cand_cycles'] / 1e6:.2f} ms", (n, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
    for n, v, e in zip(Ns, pb, est):
        a2.annotate(f"{e['per_cand_bytes'] / 1e6:.0f} MB", (n, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK2)
    a2.set_xscale("log", base=2); a2.set_xticks(Ns, [str(n) for n in Ns]); a2.set_xlabel("N", color=INK2)
    a2.set_ylim(0, 1.15); a2.set_ylabel("per candidate-step, relative to N = 1", color=INK2)
    a2.set_title("Per candidate", fontsize=10, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / "expert_candidates.png", dpi=130, facecolor=SURF)


def rssm():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    Ks = [32, 64, 128]
    for i, (model, cl) in enumerate((("full", 1), ("full", 16), ("nano", 1), ("nano", 16))):
        ys = [RSSM_MEASURED[(model, K, cl)][0] / 1e6 for K in Ks]
        a1.plot(Ks, ys, color=C[i], linewidth=2, marker="o", markersize=7, label=f"{model}, {cl} cluster{'s' if cl > 1 else ''}")
        a1.annotate(f"{ys[-1]:.2f} ms", (Ks[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=INK)
    a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xticks(Ks, [str(k) for k in Ks]); a1.set_xlim(28, 190)
    a1.set_xlabel("K trajectories", color=INK2); a1.set_ylabel("ms for H = 10 steps (log)", color=INK2)
    a1.set_title("RSSM imagination, weights resident in TCDM", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    style(a1)
    labels, parts = [], [[], [], [], []]
    for cl in (1, 16):
        for K in Ks:
            _, _, g, ln, gru, sm = RSSM_MEASURED[("full", K, cl)]
            tot = g + ln + gru + sm
            labels.append(f"K={K}\n{cl} cl")
            for j, v in enumerate((g, ln, gru, sm)):
                parts[j].append(100 * v / tot)
    xs = list(range(len(labels)))
    stacked(a2, xs, parts, ["GEMM (RedMulE)", "LayerNorm", "GRU gates", "softmax+unimix, tanh"], lambda v: "")
    a2.set_xticks(xs, labels, fontsize=8); a2.set_ylim(0, 100); a2.set_ylabel("% of step time (cluster 0)", color=INK2)
    a2.set_title("full RSSM: where a step goes", fontsize=10, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / "rssm_imagination.png", dpi=130, facecolor=SURF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    expert(); rssm()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
