"""Regenerate the README diagrams (docs/img/{flow,chip}-{light,dark}.svg):  python docs/img/make_figs.py"""
from pathlib import Path

OUT = Path(__file__).parent
THEMES = {
    "light": dict(fg="#1f2328", mute="#59636e", fill="#f6f8fa", line="#d0d7de",
                  blue="#0969da", bluef="#ddf4ff", orange="#bc4c00", orangef="#fff1e5",
                  green="#1a7f37", greenf="#dafbe1", purple="#8250df", purplef="#fbefff"),
    "dark": dict(fg="#e6edf3", mute="#9198a1", fill="#161b22", line="#3d444d",
                 blue="#58a6ff", bluef="#121d2f", orange="#f0883e", orangef="#2a1a0c",
                 green="#3fb950", greenf="#12261e", purple="#bc8cff", purplef="#231a33"),
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def svg(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}" font-size="13">\n<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{t["mute"]}"/>'
            f'</marker></defs>\n{body}</svg>\n')


def box(x, y, w, h, t, color=None, r=8):
    stroke = t[color] if color else t["line"]
    fill = t[color + "f"] if color else t["fill"]
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>\n'


def text(x, y, s, t, color="fg", size=13, weight=400, anchor="middle", mono=False):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" fill="{t[color]}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{fam}>{s}</text>\n')


def arrow(x1, y1, x2, y2, t, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{t["mute"]}" stroke-width="1.5" fill="none" marker-end="url(#a)"{d}/>\n'


def flow(t):
    W, H = 980, 300
    b = ""
    stages = [  # title, lines, colour, tool on the arrow leaving the box
        ("Model", ["linalg / frontends", "SigLIP, SmolVLA,", "RSSM, LoRA step"], "purple", "softhier-opt"),
        ("softhier dialect", ["xDSL IR", "gemm, attention,", "dma, barrier, ..."], "blue", "softhier-translate"),
        ("C program", ["one sh_* call", "per op", "(SPMD main.c)"], "green", "riscv32 gcc"),
        ("ELF", ["runtime/ softhier-ops", "+ SoftHier SDK", "(+ HBM preload image)"], "orange", "gapy run"),
        ("GVSoC", ["flex_cluster model", "simulated ns, traces,", "on-device self-checks"], None, None),
    ]
    bw, bh, gap, x0, y0 = 152, 112, 52, 14, 36
    for i, (title, lines, col, tool) in enumerate(stages):
        x = x0 + i * (bw + gap)
        b += box(x, y0, bw, bh, t, col)
        b += text(x + bw / 2, y0 + 28, title, t, col or "fg", 15, 600)
        for j, ln in enumerate(lines):
            b += text(x + bw / 2, y0 + 54 + j * 18, ln, t, "mute", 12)
        if tool:
            b += arrow(x + bw + 4, y0 + bh / 2, x + bw + gap - 4, y0 + bh / 2, t)
            b += text(x + bw + gap / 2, y0 - 12, tool, t, "fg", 11.5, 500, mono=True)
            b += f'<path d="M{x + bw + gap / 2} {y0 - 6}L{x + bw + gap / 2} {y0 + bh / 2 - 6}" stroke="{t["line"]}" stroke-width="1"/>\n'
    # passes under the first arrow
    px = x0 + bw + gap / 2
    b += text(px, y0 + bh + 22, "passes", t, "mute", 11.5, 600)
    for j, p in enumerate(["linalg-to-softhier", "pipeline-gemm", "distribute-summa"]):
        b += text(px, y0 + bh + 40 + j * 16, p, t, "mute", 11.5, mono=True)
    # DSE band
    dy = 236
    dx, dw = x0 + bw + gap, 3 * bw + 2 * gap
    b += box(dx, dy, dw, 46, t, None)
    b += text(dx + 16, dy + 20, "Design-space exploration", t, "fg", 13, 600, "start")
    b += text(dx + 16, dy + 37, "workload IR  →  analytic cost model (calibrated on micro-benchmarks)  →  arch / tiling sweep", t, "mute", 11.5, 400, "start")
    b += arrow(dx + 60, dy - 2, dx + 60, y0 + bh + 6, t, dash=True)
    b += text(dx + 68, dy - 14, "tiles, schedule", t, "mute", 11, 400, "start")
    gx = x0 + 4 * (bw + gap) + bw / 2
    b += f'<path d="M{gx} {y0 + bh + 6}L{gx} {dy + 23}L{dx + dw + 6} {dy + 23}" stroke="{t["mute"]}" stroke-width="1.5" fill="none" stroke-dasharray="5 4" marker-end="url(#a)"/>\n'
    b += text(gx - 8, y0 + bh + 40, "measured ns", t, "mute", 11, 400, "end")
    return svg(W, H, b, t)


def chip(t):
    W, H = 980, 400
    b = ""
    # mesh
    gx, gy, cs, gp = 96, 44, 58, 12
    n = 4
    b += text(gx + (n * cs + (n - 1) * gp) / 2, 24, "SoftHier: 4 x 4 cluster mesh", t, "fg", 14, 600)
    for r in range(n):
        for c in range(n):
            x, y = gx + c * (cs + gp), gy + r * (cs + gp)
            b += box(x, y, cs, cs, t, "blue", 6)
            cid = (n - 1 - r) * n + c
            b += text(x + cs / 2, y + cs / 2 + 4, f"C{cid}", t, "blue", 12, 600)
            if c < n - 1:
                b += f'<path d="M{x + cs} {y + cs / 2}h{gp}" stroke="{t["mute"]}" stroke-width="2"/>\n'
            if r < n - 1:
                b += f'<path d="M{x + cs / 2} {y + cs}v{gp}" stroke="{t["mute"]}" stroke-width="2"/>\n'
    mesh = n * cs + (n - 1) * gp
    for r in range(n):  # west HBM
        y = gy + r * (cs + gp)
        b += box(gx - 64, y + 8, 46, cs - 16, t, "orange", 5)
        b += text(gx - 41, y + cs / 2 + 4, "HBM", t, "orange", 11, 600)
        b += f'<path d="M{gx - 18} {y + cs / 2}h18" stroke="{t["mute"]}" stroke-width="2"/>\n'
    for c in range(n):  # south HBM
        x = gx + c * (cs + gp)
        b += box(x + 6, gy + mesh + 18, cs - 12, 30, t, "orange", 5)
        b += text(x + cs / 2, gy + mesh + 38, "HBM", t, "orange", 11, 600)
        b += f'<path d="M{x + cs / 2} {gy + mesh}v18" stroke="{t["mute"]}" stroke-width="2"/>\n'
    b += text(gx - 41, gy + mesh + 38, "west", t, "mute", 11)
    b += text(gx + mesh + 30, gy + mesh + 38, "south", t, "mute", 11)
    b += text(gx + mesh / 2, gy + mesh + 72, "FlooNoC mesh, 512-bit links · 8 HBM nodes x 64 MiB", t, "mute", 11.5)

    # zoom lines
    zx, zy, zw, zh = 470, 44, 250, 300
    cx, cy = gx + 3 * (cs + gp) + cs, gy
    b += f'<path d="M{cx} {cy}L{zx} {zy}M{cx} {cy + cs}L{zx} {zy + zh}" stroke="{t["line"]}" stroke-width="1" stroke-dasharray="3 3"/>\n'
    b += box(zx, zy, zw, zh, t, None, 10)
    b += text(zx + zw / 2, 24, "one cluster", t, "fg", 14, 600)
    parts = [("3 x Snitch RV32 cores", "compute · data mover · spare", "blue"),
             ("RedMulE", "128 x 32 fp16 FMAs,  Y += X·W", "green"),
             ("iDMA", "2D copies, multicast, in-network reduce", "purple"),
             ("TCDM (L1)", "1 MiB scratchpad, 64 banks", "orange")]
    for i, (a, s, col) in enumerate(parts):
        y = zy + 14 + i * 70
        b += box(zx + 14, y, zw - 28, 58, t, col, 6)
        b += text(zx + zw / 2, y + 25, a, t, col, 13, 600)
        b += text(zx + zw / 2, y + 44, s, t, "mute", 10.5)

    # dialect mapping
    mx = 752
    b += text(mx, 24, "what the dialect exposes", t, "fg", 14, 600, "start")
    rows = [("softhier.gemm / redmule", "tiled GEMM, SUMMA"),
            ("softhier.attention", "fused multi-head attention"),
            ("softhier.dma_2d", "HBM ↔ TCDM copies"),
            ("softhier.dma_broadcast", "row / column multicast"),
            ("softhier.dma_reduce", "REDADD / REDMAX in the NoC"),
            ("softhier.group_barrier", "XY barrier on a sub-mesh"),
            ('memref&lt;..., "hbm_west"&gt;', "placement = memory space")]
    for i, (op, d) in enumerate(rows):
        y = 58 + i * 42
        b += text(mx, y, op, t, "fg", 12, 500, "start", mono=True)
        b += text(mx, y + 17, d, t, "mute", 11.5, 400, "start")
    return svg(W, H, b, t)


for name, t in THEMES.items():
    (OUT / f"flow-{name}.svg").write_text(flow(t))
    (OUT / f"chip-{name}.svg").write_text(chip(t))
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
