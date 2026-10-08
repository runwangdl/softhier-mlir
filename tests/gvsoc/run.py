#!/usr/bin/env python3
"""Run the on-simulator tests of the softhier-ops library.

    python tests/gvsoc/run.py gemm                 # default shape set
    python tests/gvsoc/run.py gemm --shapes 256x256x256 512x768x768:256,256,256
    python tests/gvsoc/run.py mlir examples/gemm512_linalg.mlir -p linalg-to-softhier
    python tests/gvsoc/run.py mlir examples/*.mlir          # every example, auto passes
    python tests/gvsoc/run.py siglip --seq 256 --cluster all [--define ATTN_SERIAL ATTN_CANARY ...]
    python tests/gvsoc/run.py gemm-seq                      # mixed tile shapes back to back (gemm_seq)
    python tests/gvsoc/run.py rowops --data device          # inputs generated on the device instead of preloaded
    python tests/gvsoc/run.py mesh --modes 0 1 2 5 6        # multi-cluster slice-store repro (mesh_slices)

Environment: SOFTHIER_MODEL_DIR=<dir>[:<dir>] puts extra gvsoc model directories in front of
install/models (pin or test a model build); see docs/SIMULATOR_NOTES.md.

Test inputs (`--data`, default `preload`): the LCG matrices every test starts from are generated on the
host (softhier_mlir.testing.lcg, the twin of sh_test_fill_fp16) and put into HBM through the simulator's
preload image before the program starts; `--data device` generates them on the device as before (one
core, ~70 s of wall time for a SigLIP layer). Same bytes either way, so results and ROIs are identical.

Each case writes tests/gvsoc/<test>/shape.h, builds the SDK app in the x86 chroot and runs
GVSoC natively (ideal HBM). Prints PASS/FAIL and the ROI in ns.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.sim.gvsoc import PERF_RE, build_sw, run_sim  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = "preload"        # --data: "preload" (host-generated inputs in the HBM preload image) | "device" (on-device LCG)
HBM_START = 0x1000      # first HBM offset the C tests use: the SDK allocator owns the first 4 KB (preload.MIN_OFFSET)


def preload_image(app: Path, arrays: dict[int, np.ndarray], sentinel_off: int | None = None) -> tuple[Path | None, str]:
    """`--data preload`: write the test's input matrices ({hbm offset: fp16 array}) + the end-of-image sentinel
    (default: the first 4 KB boundary above the arrays) into <app>/preload.elf. Returns (elf, shape.h line
    defining SH_PRELOAD = the sentinel's offset); (None, "") with `--data device`, where main.c keeps its fills."""
    if DATA != "preload":
        return None, ""
    from softhier_mlir.sim.preload import make_preload_elf, sentinel_array
    end = max(off + a.nbytes for off, a in arrays.items())
    sent = (max(end, sentinel_off or 0) + 0xFFF) & ~0xFFF
    elf = make_preload_elf(app / "preload.elf", {**arrays, sent: sentinel_array()})
    return elf, f"#define SH_PRELOAD 0x{sent:x}\n"

DEFAULT_GEMM = ["256x256x256", "256x768x192:256,256,192", "512x768x768:256,256,256",
                "256x256x256:256,256,256,0", "256x256x512:256,256,256,1,1",
                "1024x768x768:256,256,256,1,0,all", "1024x3072x768:256,256,256,1,0,all"]


def parse_shape(s: str) -> dict:
    """MxNxK[:tm,tn,tk[,pipeline[,accumulate[,cluster]]]]   cluster = 0 | all"""
    dims, _, rest = s.partition(":")
    m, n, k = (int(v) for v in dims.lower().split("x"))
    opts = rest.split(",") if rest else []
    tm, tn, tk = (int(v) for v in (opts + ["0", "0", "0"])[:3])
    pipe = int(opts[3]) if len(opts) > 3 else 1
    acc = int(opts[4]) if len(opts) > 4 else 0
    cluster = "SH_ALL" if len(opts) > 5 and opts[5] == "all" else "0"
    return dict(M=m, N=n, K=k, tm=tm, tn=tn, tk=tk, pipeline=pipe, accumulate=acc, cluster=cluster)


def run_gemm(shapes: list[str], nsamples: int = 256, real: bool = False, offsets: tuple | None = None) -> bool:
    """offsets: HBM byte offsets of X, W, Z (default X at HBM_START, W at 16 MB, Z at 32 MB: all in HBM node 0;
    a node is 64 MB and has its own NoC edge port, docs/DSE.md section 8)."""
    from softhier_mlir.testing import lcg
    app = HERE / "gemm"
    all_ok = True
    for s in shapes:
        c = parse_shape(s)
        M, N, K, acc = c["M"], c["N"], c["K"], c["accumulate"]
        # == the fills in gemm/main.c (Z0 = 3.0 when accumulating)
        off = {"x": HBM_START, "w": 0x01000000, "z": 0x02000000}
        if offsets:
            off = {"x": max(offsets[0], HBM_START), "w": offsets[1], "z": offsets[2]}
        x = lcg.fill_fp16(M, K, 1, 0, 64, 1 / 4096) if real else lcg.fill_fp16(M, K, 1, -1, 1)
        w = lcg.fill_fp16(K, N, 2, -16, 16, 0.125) if real else lcg.fill_fp16(K, N, 2, -2, 2)
        z = lcg.fill_fp16(M, N, 3, 3 if acc else 0, 3 if acc else 0)
        pre, pre_h = preload_image(app, {off["x"]: x, off["w"]: w, off["z"]: z})
        (app / "shape.h").write_text(
            f"#define GEMM_M {M}\n#define GEMM_N {N}\n#define GEMM_K {K}\n"
            f"#define TILE_M {c['tm']}\n#define TILE_N {c['tn']}\n#define TILE_K {c['tk']}\n"
            f"#define PIPELINE {c['pipeline']}\n#define ACCUMULATE {acc}\n"
            f"#define CLUSTER {c['cluster']}\n#define NSAMPLES {nsamples}\n" + ("#define REAL_DATA 1\n" if real else "")
            + f"#define OFF_X 0x{off['x']:x}\n#define OFF_W 0x{off['w']:x}\n#define OFF_Z 0x{off['z']:x}\n" + pre_h)
        build_sw(app)
        r = run_sim(preload=pre)
        lines = [ln for ln in r["stdout"].splitlines() if ln.startswith("[gemm]") or "mismatch" in ln]
        ok = r["ok"] and any("GEMM_PASS" in ln for ln in lines)
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {s:<28} roi={r['roi_ns']} ns wall={r['wall_s']}s")
        for ln in lines:
            print("     " + ln)
        if not r["ok"]:
            print(r["stdout"][-1500:])
    return all_ok


def run_rowops(rows: int, cols: int, cluster: str, nsamples: int = 64) -> bool:
    from softhier_mlir.testing import lcg
    app = HERE / "rowops"
    x16 = lcg.fill_fp16(rows, cols, 11, -16, 16, 0.125)
    b16 = lcg.fill_fp16(rows, cols, 12, -16, 16, 0.125)
    g16 = lcg.fill_fp16(1, cols, 13, 1, 8, 0.25)
    be16 = lcg.fill_fp16(1, cols, 14, -4, 4, 0.25)
    mb = rows * cols * 2                        # == the layout in rowops/main.c: x, b, g (+be at +4 KB), then 7 outputs
    assert mb % 64 == 0, "rows*cols must be a multiple of 32 (64 B aligned matrices)"
    pre, pre_h = preload_image(app, {HBM_START: x16, HBM_START + mb: b16, HBM_START + 2 * mb: g16, HBM_START + 2 * mb + 4096: be16},
                               sentinel_off=HBM_START + 10 * mb)
    (app / "shape.h").write_text(f"#define ROWS {rows}\n#define COLS {cols}\n#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n"
                                 f"#define HBM_START 0x{HBM_START:x}\n" + pre_h)
    build_sw(app)
    r = run_sim(preload=pre)
    x, b, g, be = (a.astype(np.float32) for a in (x16, b16, g16, be16))
    mean = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
    ref = {
        "LN": (x - mean) / np.sqrt(var + 1e-5) * g + be,
        "SM": (lambda e: e / e.sum(1, keepdims=True))(np.exp(0.5 * x - (0.5 * x).max(1, keepdims=True))),
        "GELU": 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
        "ADD": x + b, "BIAS": x + be, "SCALE": x * np.float32(0.3), "T": x.T,
    }
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "ROWOPS_DONE" in r["stdout"]
    per_op = dict(zip(["LN", "SM", "GELU", "ADD", "BIAS", "SCALE", "T"], r["rois"]))   # one timer_end per op
    total = sum(r["rois"])
    print(f"{'PASS' if ok else 'FAIL'} rowops {rows}x{cols} cluster={cluster} roi={total} ns wall={r['wall_s']}s")
    print("     per-op ns: " + "  ".join(f"{k}={v}" for k, v in per_op.items()))
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<6} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=2e-2, rtol=2e-2)
        print(f"     {tag:<6} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def run_fp16cvt() -> bool:
    """Hardware (Zfh register) fp16<->fp32 conversion vs the software converters, plus a timing of both."""
    app = HERE / "fp16cvt"
    build_sw(app)
    r = run_sim()
    ok = r["ok"] and "FP16CVT_PASS" in r["stdout"]
    print(f"{'PASS' if ok else 'FAIL'} fp16cvt wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[fp16cvt]") or ln.startswith("h2f") or ln.startswith("f2h"):
            print("     " + ln)
    if len(r["rois"]) >= 2:
        print(f"     timed loop: software {r['rois'][0]} ns, hardware {r['rois'][1]} ns ({r['rois'][0] / max(r['rois'][1], 1):.1f}x); more ROIs: {r['rois'][2:]}")
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def siglip_layer_inputs(S: int, D: int, F: int, H: int) -> tuple[dict[int, np.ndarray], int]:
    """== the ALLOC order and fills of siglip_layer/main.c -> ({hbm offset: array}, end of the layout)."""
    from softhier_mlir.testing import lcg
    nxt = HBM_START

    def alloc(nbytes):
        nonlocal nxt
        a = nxt; nxt += (nbytes + 4095) & ~4095; return a
    off = {}
    for nm, nb in [("x", S * D), ("ln1", S * D), ("q", S * D), ("k", S * D), ("v", S * D), ("kT", D * S), ("sc", H * S * S),
                   ("o", S * D), ("ao", S * D), ("h", S * D), ("ln2", S * D), ("f1", S * F), ("g", S * F), ("f2", S * D), ("out", S * D),
                   ("wq", D * D), ("wk", D * D), ("wv", D * D), ("wo", D * D), ("w1", D * F), ("w2", F * D),
                   ("bq", D), ("bk", D), ("bv", D), ("bo", D), ("b1", F), ("b2", D), ("g1", D), ("be1", D), ("g2", D), ("be2", D)]:
        off[nm] = alloc(nb * 2)
    fills = [("x", S, D, 1, -16, 16, 0.125)] + [(nm, D, D, sd, -8, 8, 1 / 128) for nm, sd in (("wq", 2), ("wk", 3), ("wv", 4), ("wo", 5))] + \
            [("w1", D, F, 6, -8, 8, 1 / 128), ("w2", F, D, 7, -8, 8, 1 / 256)] + \
            [(nm, 1, D, sd, -4, 4, 0.0625) for nm, sd in (("bq", 8), ("bk", 9), ("bv", 10), ("bo", 11))] + \
            [("b1", 1, F, 12, -4, 4, 0.0625), ("b2", 1, D, 13, -4, 4, 0.0625), ("g1", 1, D, 14, 2, 6, 0.25),
             ("be1", 1, D, 15, -4, 4, 0.125), ("g2", 1, D, 16, 2, 6, 0.25), ("be2", 1, D, 17, -4, 4, 0.125)]
    return {off[nm]: lcg.fill_fp16(r, c, sd, lo, hi, sc) for nm, r, c, sd, lo, hi, sc in fills}, nxt


def run_siglip(seq: int, d: int, ff: int, heads: int, cluster: str, nsamples: int = 64, extra: str = "") -> bool:
    from softhier_mlir.testing import lcg, siglip_ref
    app = HERE / "siglip_layer"
    pre, pre_h = preload_image(app, *siglip_layer_inputs(seq, d, ff, heads))
    (app / "shape.h").write_text(f"#define SEQ {seq}\n#define D_MODEL {d}\n#define D_FF {ff}\n#define N_HEADS {heads}\n"
                                 f"#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n#define HBM_START 0x{HBM_START:x}\n" + pre_h + extra)
    build_sw(app)
    r = run_sim(timeout=7200, preload=pre)
    ref = siglip_ref.layer_reference(seq, d, ff, heads)
    got = lcg.parse_samples(r["stdout"])
    if "O0a" in got:
        ref = {**ref, "O0a": ref["O0"]}
    ok = r["ok"] and "SIGLIP_LAYER_DONE" in r["stdout"]
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_") or ln.startswith("[head"):
            print("     " + ln)
    (HERE / "siglip_layer" / "last_run.log").write_text(r["stdout"])
    macs = 4 * seq * d * d + 2 * seq * d * ff + 2 * seq * seq * d
    print(f"{'PASS' if ok else 'FAIL'} siglip_layer S={seq} D={d} F={ff} H={heads} cluster={cluster} "
          f"roi={r['roi_ns']} ns ({macs / 1e6:.0f} MMAC, {macs / r['roi_ns'] if r['roi_ns'] else 0:.0f} MAC/ns) wall={r['wall_s']}s")
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<4} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=3)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def run_siglip_mlir(seq: int, d: int, ff: int, heads: int, cluster: str, layers: int = 1, nsamples: int = 64, fused: bool = False,
                    trace: Path | None = None, tiles: str = "model", hbm_split: bool = False) -> bool:
    """Frontend -> softhier-translate -> gvsoc, compared against the same numpy reference as `siglip`.
    trace: also record the RedMulE / iDMA / barrier activity (gvsoc --trace) into this log, for
    `python -m softhier_mlir.sim.trace <log> --png ...`. tiles: the frontend's GEMM tile policy.
    hbm_split: parameters in HBM node 1, activations in node 0."""
    from softhier_mlir.frontend import siglip
    from softhier_mlir.testing import lcg, siglip_ref
    app = HERE / "mlir_app"
    mlir = siglip.emit(seq, d, ff, heads, layers, -1 if cluster == "SH_ALL" else int(cluster), True, nsamples=nsamples,
                       fused_attention=fused, tiles=tiles, hbm_split=hbm_split)
    (app / "siglip.mlir").write_text(mlir)
    (app / "main.c").write_text(lower_and_translate(app / "siglip.mlir", None, pre := translate_preload(app)))
    build_sw(app)
    r = run_sim(timeout=7200, preload=pre if pre and pre.exists() else None,
                traces=("redmule", "idma", "cluster_registers") if trace else (), log=trace)
    ref = siglip_ref.layer_reference(seq, d, ff, heads)
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} siglip-mlir S={seq} D={d} F={ff} H={heads} L={layers} cluster={cluster} attention={'fused' if fused else 'per-head'} "
          f"tiles={tiles} hbm_split={hbm_split} roi={r['roi_ns']} ns (the ROI includes the sample dumps; see the marks) wall={r['wall_s']}s")
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_") or ln.startswith("[mark]"):
            print("     " + ln)
    marks = parse_marks(r["stdout"])
    if marks:
        prev = None
        for tag, t in marks.items():
            if prev is not None:
                print(f"     time {prev[0]:>8} -> {tag:<8} {(t - prev[1]) / 1e6:9.3f} ms")
            prev = (tag, t)
    for tag, arr in ref.items():
        if tag not in got:
            continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=2)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if trace is None:
        (app / "last_run.log").write_text(r["stdout"])
    return ok


def run_preload(offsets=(0x1000, 70 << 20, 200 << 20), rows: int = 64, cols: int = 96, nsamples: int = 64,
                filler_mb: int = 16, wait: bool = True) -> bool:
    """HBM preload: fp16 matrices at the given HBM byte offsets (spanning several HBM nodes) go in
    through `--preload`; the program only dumps samples of them, the host compares. A `filler_mb`
    block at 1 MB makes the image big enough that its last segments land well after the program
    starts (the loader's done flag only means "issued"); the samples are taken right after
    `softhier.preload_wait` on the sentinel, highest offset first. wait=False shows the race."""
    from softhier_mlir.sim.preload import make_preload_elf, sentinel_array
    from softhier_mlir.testing import lcg
    app = HERE / "mlir_app"
    rng = np.random.default_rng(7)
    arrays = {off: (rng.standard_normal((rows, cols)) * 2).astype(np.float16) for off in offsets}
    if filler_mb:
        arrays[1 << 20] = (rng.standard_normal(((filler_mb << 20) // 8192, 4096)) * 2).astype(np.float16)
    sent_off = max(off + a.nbytes for off, a in arrays.items()); sent_off = (sent_off + 0xFFFF) & ~0xFFFF
    arrays[sent_off] = sentinel_array()
    elf = make_preload_elf(app / "preload.elf", arrays)
    lines, dumps = [], []
    for i, off in enumerate(sorted(arrays)):
        a = arrays[off]
        t = f'memref<{a.shape[0]}x{a.shape[1]}xf16, "hbm_west">'
        lines.append(f"    %b{i} = softhier.hbm_buffer {{offset = {off} : i32}} : {t}")
        if off == sent_off:
            waitop = f"    softhier.preload_wait %b{i} : {t}"
        else:
            dumps.append(f'    softhier.dump_samples %b{i} {{seed = {300 + i} : i32, n = {nsamples} : i32, tag = "P{i}"}} : {t}')
    body = "\n".join(lines + ['    softhier.mark {tag = "t0"}'] + ([waitop] if wait else []) + ['    softhier.mark {tag = "t1"}']
                      + dumps[::-1] + ['    softhier.mark {tag = "t2"}'])
    mlir = f"builtin.module {{\n  func.func @preload_test() {{\n{body}\n    func.return\n  }}\n}}\n"
    (app / "preload.mlir").write_text(mlir)
    (app / "main.c").write_text(lower_and_translate(app / "preload.mlir", None))
    build_sw(app)
    r = run_sim(preload=elf)
    got = lcg.parse_samples(r["stdout"])
    info = [ln for ln in r["stdout"].splitlines() if ln.startswith("[mark]") or ln.startswith("[sh_preload_wait]")]
    ok = r["ok"]
    print(f"{'PASS' if ok else 'FAIL'} preload {len(arrays)} segments ({elf.stat().st_size / 2 ** 20:.1f} MiB elf, wait={wait}) roi={r['roi_ns']} ns wall={r['wall_s']}s {info}")
    for i, off in enumerate(sorted(arrays)):
        if off == sent_off:
            continue
        tag = f"P{i}"
        if tag not in got:
            print(f"     {tag} @0x{off:x} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arrays[off].astype(np.float32), atol=0, rtol=0, show=3)
        print(f"     {tag} @0x{off:08x} {arrays[off].shape} samples={len(got[tag])} bad={bad} maxerr={maxerr} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    if not r["ok"]:
        print(r["stdout"][-1500:])
    (app / "last_run.log").write_text(r["stdout"])
    return ok


def parse_marks(stdout: str) -> dict[str, int]:
    """[mark] tag cycles -> {tag: ns since the first mark}, unwrapping the 32-bit mcycle (1 GHz)."""
    out, prev, acc = {}, None, 0
    for ln in stdout.splitlines():
        if ln.startswith("[mark] "):
            m = re.match(r"\[mark\] (\w+) (\d+)$", ln)   # a gvsoc trace line can be glued to the mark (--trace runs)
            if not m:
                continue
            tag, c = m.group(1), int(m.group(2))
            if prev is not None:
                acc += (c - prev) % (1 << 32)
            prev = c
            out[tag] = acc
    return out


def layer_times(marks: dict[str, int]) -> dict[int, tuple[int, int]]:
    """{layer: (attention ns, mlp+proj ns)} from the mark sequence ... attn<n>, layer<n>, [ldump<n>] ...;
    the attention segment starts at the previous mark, so the dump marks keep the sample printing out."""
    out, prev_t = {}, None
    for tag, t in marks.items():
        m = re.fullmatch(r"(attn|layer)(\d+)", tag)
        if prev_t is not None and m:
            n = int(m.group(2))
            if m.group(1) == "attn":
                out[n] = [t - prev_t, 0]
            else:
                out.setdefault(n, [0, 0])[1] = t - prev_t
        prev_t = t
    return {k: tuple(v) for k, v in out.items()}


def dump_times(marks: dict[str, int]) -> list[int]:
    """ns spent in sample dumps (segments ending at a *dump mark)."""
    out, prev_t = [], None
    for tag, t in marks.items():
        if prev_t is not None and tag.endswith("dump"):
            out.append(t - prev_t)
        prev_t = t
    return out


def _elf_load_segments(elf: Path) -> list[tuple[int, int]]:
    """[(paddr, filesz)] of the PT_LOAD segments of an ELF32 (the program's footprint)."""
    import struct
    raw = Path(elf).read_bytes()
    phoff = struct.unpack_from("<I", raw, 28)[0]; phentsize, phnum = struct.unpack_from("<HH", raw, 42)
    out = []
    for i in range(phnum):
        t, _off, _va, pa, fsz, *_ = struct.unpack_from("<8I", raw, phoff + i * phentsize)
        if t == 1:
            out.append((pa, fsz))
    return out


def run_smolvla(npz: str, layers: int | None, attn: int, cluster: int, nsamples: int = 64, dumps=None,
                log: Path | None = None, timeout: int = 48 * 3600, app_dir: Path | None = None, unroll: bool = False,
                from_log: Path | None = None) -> bool:
    """SmolVLA vision tower (real weights via HBM preload) -> compare sampled device tensors against
    the fp32 HF reference and the fp16-program floor stored in the npz by `smolvla.py prepare`.
    from_log: skip build + simulation and re-evaluate an existing simulator log."""
    from softhier_mlir.frontend import smolvla
    from softhier_mlir.sim.preload import make_preload_elf
    data = dict(np.load(npz))
    npz_layers = int(data["meta"][1])
    layers = npz_layers if layers is None else layers
    seq = int(data["xp"].shape[0])
    dumps = tuple(dumps) if dumps else ("EMB",) + tuple(f"L{n}" for n in range(1, layers + 1)) + ("OUT",)
    if layers != npz_layers:      # the npz's OUT is the post-LN after all its layers: redo it after layer `layers` on the host
        def ln(a, g, b):
            a = a.astype(np.float32); m = a.mean(1, keepdims=True); v = a.var(1, keepdims=True)
            return (a - m) / np.sqrt(v + 1e-6) * g.astype(np.float32) + b.astype(np.float32)
        data["ref_OUT"] = ln(data[f"ref_L{layers}"], data["p_gpost"], data["p_bepost"])
        data["np_OUT"] = ln(data[f"np_L{layers}"], data["p_gpost"], data["p_bepost"]).astype(np.float16).astype(np.float32)
    if from_log is not None:
        stdout = Path(from_log).read_text()
        rois = [int(v) for v in PERF_RE.findall(stdout)]
        print(f"[smolvla] re-evaluating {from_log}: seq={seq} layers={layers} roi={rois[0] if rois else None} ns")
        return report_smolvla(data, stdout, layers, dumps, bool(rois))
    app = Path(app_dir) if app_dir else HERE / "smolvla_app"     # own dir so runs can go concurrently
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    mlir, pre = smolvla.emit(data, layers, attn, cluster, dumps=dumps, nsamples=nsamples, unroll=unroll)
    (app / "smolvla.mlir").write_text(mlir)
    elf = make_preload_elf(app / "smolvla_preload.elf", pre)
    (app / "main.c").write_text(lower_and_translate(app / "smolvla.mlir", None))
    build_sw(app)
    text = sum(sz for _, sz in _elf_load_segments(build_sw.last_elf) if sz)   # the cluster instruction memory is 64 KB
    print(f"[smolvla] seq={seq} layers={layers} attn={attn} cluster={cluster} {'unrolled' if unroll else 'looped'} "
          f"preload {elf.stat().st_size / 2 ** 20:.1f} MiB, program {text / 1024:.1f} KB; simulating...", flush=True)
    r = run_sim(preload=elf, timeout=timeout, log=log)
    (app / "last_run.log").write_text(r["stdout"])
    print(f"{'PASS' if r['ok'] else 'FAIL'} smolvla seq={seq} layers={layers} attn={attn} cluster={cluster} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    ok = report_smolvla(data, r["stdout"], layers, dumps, r["ok"])
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def report_smolvla(data: dict, stdout: str, layers: int, dumps: tuple, ok: bool) -> bool:
    """Timing (marks) + accuracy of the sampled tensors. A tensor passes when every sample is within
    atol = 3% of the tensor's max |ref| (+ 5% relative) of the HF fp32 reference: the program is fp16 end
    to end (fp16 operands and RedMulE fp16 accumulation), so ~1e-2 of the tensor scale is the floor, and
    the post layernorm amplifies the last layer's error by gamma / row-std (~6x for this checkpoint)."""
    from softhier_mlir.testing import lcg
    got = lcg.parse_samples(stdout)
    marks = parse_marks(stdout)
    for ln in stdout.splitlines():
        if ln.startswith("[sh_"):
            print("     " + ln)
    prev_t, prev_tag = None, None
    for tag, t in marks.items():
        if prev_t is not None:
            print(f"     time {prev_tag:>8} -> {tag:<8} {(t - prev_t) / 1e6:9.3f} ms")
        prev_t, prev_tag = t, tag
    per_layer = layer_times(marks)
    for n, (ta, tm) in sorted(per_layer.items()):
        print(f"     layer {n:2d}: attention {ta / 1e6:8.3f} ms  mlp+proj {tm / 1e6:8.3f} ms  total {(ta + tm) / 1e6:8.3f} ms")
    if per_layer:
        tot = [ta + tm for ta, tm in per_layer.values()]
        print(f"     per-layer simulated time: mean {np.mean(tot) / 1e6:.3f} ms over {len(tot)} layers (dump printing excluded); "
              f"total marked compute {(marks.get('end', 0) - marks.get('start', 0) - sum(d for d in dump_times(marks))) / 1e6:.3f} ms")
    for tag in dumps:
        if tag not in got:
            print(f"     {tag:<4} MISSING"); ok = False; continue
        floor = data[tag if tag.startswith("p_") else f"np_{tag}"].astype(np.float32)   # p_<name>: the preloaded parameter itself
        has_hf = f"ref_{tag}" in data
        ref = data[f"ref_{tag}"].astype(np.float32) if has_hf else floor   # HF fp32 when it exists, else the fp16 floor
        vals = np.array([v for _, _, v in got[tag]]); want = np.array([ref[r_, c] for r_, c, _ in got[tag]])
        err = np.abs(vals - want)
        scale = np.abs(ref).max() + 1e-12
        atol = max(0.05, 0.03 * scale) if not tag.startswith("p_") else 0.0
        bad_hf, _ = lcg.compare_samples(got[tag], ref, atol=atol, rtol=0.05 if atol else 0.0, show=2)
        bad_fl, maxerr_fl = lcg.compare_samples(got[tag], floor, atol=atol, rtol=0.05 if atol else 0.0)
        print(f"     {tag:<4} samples={len(got[tag])} vs {'HF fp32' if has_hf else 'fp16 floor'}: max abs {err.max():.4f} median {np.median(err):.4f} "
              f"(|ref| max {scale:.2f}, rel-to-max {err.max() / scale:.2e}) bad={bad_hf}; vs fp16 floor: max abs {maxerr_fl:.4f} bad={bad_fl} "
              f"{'PASS' if bad_hf == 0 else 'FAIL'} (atol {atol:.3f}, rtol 0.05)")
        ok &= bad_hf == 0
    return ok


def attention_reference(S: int, D: int, H: int, scale: float = 0.125) -> np.ndarray:
    """numpy twin of tests/gvsoc/attention/main.c (fp32 math, fp16 rounding where the device stores fp16)."""
    from softhier_mlir.testing import lcg
    f = lambda *a, **k: lcg.fill_fp16(*a, **k).astype(np.float32)  # noqa: E731
    r16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    q = f(S, D, 21, -8, 8, 0.125); k = f(S, D, 22, -8, 8, 0.125); v = f(S, D, 23, -16, 16, 0.125)
    dh = D // H
    o = np.zeros((S, D), np.float32)
    for hd in range(H):
        sl = slice(hd * dh, (hd + 1) * dh)
        s = r16(q[:, sl] @ k[:, sl].T) * scale
        p = np.exp(s - s.max(1, keepdims=True)); p = r16(p / p.sum(1, keepdims=True))
        o[:, sl] = r16(p @ v[:, sl])
    return o


def run_attention(seq: int, d: int, heads: int, cluster: str, composed: bool, nsamples: int = 128, extra: str = "") -> bool:
    """Fused sh_attention (or the composed transpose+gemm+softmax+gemm path) vs numpy; prints the ROI.
    extra: more shape.h lines (Q_BLOCK=<rows>, SH_ATTN_KT_DMA=0, ... from --define)."""
    from softhier_mlir.testing import lcg
    app = HERE / "attention"
    mb = (seq * d * 2 + 4095) & ~4095           # == attention/main.c: q, k, v, o, then kT and the H score matrices
    pre, pre_h = preload_image(app, {HBM_START: lcg.fill_fp16(seq, d, 21, -8, 8, 0.125), HBM_START + mb: lcg.fill_fp16(seq, d, 22, -8, 8, 0.125),
                                     HBM_START + 2 * mb: lcg.fill_fp16(seq, d, 23, -16, 16, 0.125), HBM_START + 3 * mb: lcg.fill_fp16(seq, d, 24, 7, 7)},
                               sentinel_off=HBM_START + 5 * mb + heads * seq * seq * 2)
    (app / "shape.h").write_text(f"#define SEQ {seq}\n#define D_MODEL {d}\n#define N_HEADS {heads}\n"
                                 f"#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n#define COMPOSED {1 if composed else 0}\n"
                                 f"#define HBM_START 0x{HBM_START:x}\n" + pre_h + extra)
    build_sw(app)
    r = run_sim(timeout=7200, preload=pre)
    ref = attention_reference(seq, d, heads)
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "ATTENTION_DONE" in r["stdout"] and "ATTENTION_FAIL" not in r["stdout"]
    for ln in r["stdout"].splitlines():
        if ln.startswith("[sh_") or ln.startswith("[attention]"):
            print("     " + ln)
    dh = d // heads
    print(f"{'PASS' if ok else 'FAIL'} attention S={seq} D={d} H={heads} dh={dh} cluster={cluster} path={'composed' if composed else 'fused'} "
          f"roi={r['roi_ns']} ns (1 GHz: {r['roi_ns']} cycles) wall={r['wall_s']}s")
    for tag, arr in (("O0", ref[:, :dh]), ("O", ref)):
        if tag not in got:
            print(f"     {tag:<4} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=0.05, rtol=0.05, show=3)
        print(f"     {tag:<4} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    (app / "last_run.log").write_text(r["stdout"])
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def run_gemm_seq() -> bool:
    """tests/gvsoc/gemm_seq: five GEMMs of different tile shapes back to back on cluster 0, self-checked on the device."""
    from softhier_mlir.testing import lcg
    app = HERE / "gemm_seq"
    f = lcg.fill_fp16
    arrays = {HBM_START: f(256, 64, 1, -1, 1), 0x100000: f(64, 256, 2, -2, 2),           # == the fills in gemm_seq/main.c
              0x300000: f(256, 256, 3, -1, 1), 0x400000: f(256, 64, 4, -2, 2),
              0x600000: f(256, 256, 5, -1, 1), 0x700000: f(256, 256, 6, -2, 2),
              0x900000: f(256, 256, 7, -1, 1), 0xA00000: f(256, 768, 8, -2, 2), 0xB00000: f(256, 768, 9, 7, 7),
              0xC00000: f(256, 256, 10, -32, 32, 0.125), 0xD00000: f(256, 768, 11, -16, 16, 0.125)}
    pre, pre_h = preload_image(app, arrays, sentinel_off=0xF00000)
    (app / "shape.h").write_text(f"#define HBM_START 0x{HBM_START:x}\n" + pre_h)
    build_sw(app)
    r = run_sim(preload=pre)
    lines = [ln for ln in r["stdout"].splitlines() if ln.startswith("[") and ("PASS" in ln or "FAIL" in ln or "mismatch" in ln)]
    ok = r["ok"] and "GEMM_PASS" in r["stdout"] and "FAIL" not in r["stdout"]
    print(f"{'PASS' if ok else 'FAIL'} gemm_seq wall={r['wall_s']}s")
    for ln in lines:
        print("     " + ln)
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


def translate_preload(app: Path) -> Path | None:
    """The preload image softhier-translate writes for a generated program (`--data preload`), else None."""
    return app / "preload.elf" if DATA == "preload" else None


def run_mesh(modes: list[str], heads: int = 12) -> bool:
    """tests/gvsoc/mesh_slices: `heads` clusters each write one 64-column slice of a 256x768 output.
    MODE 0 gemm, 1 dma stores, 2 scalar stores, 3 gemm serialized, 4 gemm private buffers,
    5 scores+softmax+P.V (the attention sequence), 6/7 as 5 + a follow-up 16-cluster GEMM reading it."""
    app = HERE / "mesh_slices"
    all_ok = True
    for m in modes:
        mode, _, defs = m.partition(":")
        (app / "shape.h").write_text(f"#define MODE {mode}\n#define NH {heads}\n" + "".join(f"#define {d}\n" for d in defs.split(",") if d))
        build_sw(app)
        r = run_sim()
        ok = r["ok"] and "MESH_PASS" in r["stdout"]
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} mesh_slices mode={m} heads={heads} roi={r['roi_ns']} ns wall={r['wall_s']}s")
        for ln in r["stdout"].splitlines():
            if ln.startswith("[") and "FAIL" in ln and "mesh_slices" not in ln:
                print("     " + ln)
        if not r["ok"]:
            print(r["stdout"][-1500:])
    return all_ok


# passes each example needs (none = already in the softhier dialect)
EXAMPLE_PASSES = {
    "gemm512_linalg.mlir": "linalg-to-softhier",
    "mlp_linalg.mlir": "linalg-to-softhier",
    "gemm1024_summa.mlir": "linalg-to-softhier,distribute-summa,pipeline-gemm",
}


# passes each example needs (none = already in the softhier dialect)
EXAMPLE_PASSES = {
    "gemm512_linalg.mlir": "linalg-to-softhier",
    "mlp_linalg.mlir": "linalg-to-softhier",
    "gemm1024_summa.mlir": "linalg-to-softhier,distribute-summa,pipeline-gemm",
}


def lower_and_translate(mlir: Path, passes: str | None, preload_elf: Path | None = None) -> str:
    """softhier-opt [-p passes] | softhier-translate -> C source. With preload_elf the test inputs go into that
    HBM preload image (host-generated) and the program only waits for it; the file is absent afterwards when
    some input could not be preloaded (translate says why on stderr) and the program fills on the device."""
    py = sys.executable
    src = mlir.read_text()
    if passes:
        r = subprocess.run([py, "-m", "softhier_mlir.tools.softhier_opt", str(mlir), "-p", passes],
                           capture_output=True, text=True, check=True)
        src = r.stdout
    r = subprocess.run([py, "-m", "softhier_mlir.tools.softhier_translate", "/dev/stdin"] +
                       (["--preload-elf", str(preload_elf)] if preload_elf else []),
                       input=src, capture_output=True, text=True, check=True)
    if r.stderr.strip():
        print("     " + r.stderr.strip().replace("\n", "\n     "))
    return r.stdout


def run_mlir(files: list[str], passes: str | None) -> bool:
    app = HERE / "mlir_app"
    all_ok = True
    for f in files:
        mlir = Path(f)
        p = passes if passes is not None else EXAMPLE_PASSES.get(mlir.name)
        (app / "main.c").write_text(lower_and_translate(mlir, p, pre := translate_preload(app)))
        build_sw(app)
        r = run_sim(preload=pre if pre and pre.exists() else None)
        lines = [ln for ln in r["stdout"].splitlines() if "_CHECK" in ln or "[sh_" in ln]
        checks = [ln for ln in lines if "_CHECK" in ln]
        # examples without a self-check (pure timing runs) pass when the simulation completes
        ok = r["ok"] and all("_PASS" in ln for ln in checks) and not any("[sh_" in ln for ln in lines)
        all_ok &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {mlir.name:<26} passes={p or '-':<34} roi={r['roi_ns']} ns wall={r['wall_s']}s")
        for ln in lines:
            print("     " + ln)
        if not r["ok"]:
            print(r["stdout"][-1500:])
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["gemm", "gemm-seq", "mlir", "rowops", "fp16cvt", "siglip", "siglip-mlir", "mesh", "attention", "preload", "smolvla"])
    ap.add_argument("--data", choices=["preload", "device"], default="preload",
                    help="test inputs: generated on the host into the HBM preload image (default) or on the device")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--npz", default="/app/models/smolvla_base/vision_s256.npz", help="smolvla: output of `smolvla.py prepare`")
    ap.add_argument("--attn", type=int, default=-1, help="smolvla: cluster of the per-head attention ops (-1 = SH_ALL per op)")
    ap.add_argument("--dumps", nargs="*", help="smolvla: tensors to compare (default EMB, L1..Ln, OUT)")
    ap.add_argument("--log", help="smolvla: stream the simulator output to this file")
    ap.add_argument("--all-layers", action="store_true", help="smolvla: run every layer in the npz (overrides --layers)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--ff", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--cluster", default=None, help="executing cluster: 0 or all (default 0; smolvla: all)")
    ap.add_argument("--app-dir", help="smolvla: app/build dir (default tests/gvsoc/smolvla_app)")
    ap.add_argument("--no-wait", action="store_true", help="preload: skip softhier.preload_wait (demonstrates the race)")
    ap.add_argument("--unroll", action="store_true", help="smolvla: unrolled program (1-2 layers; enables the layer-1 intermediate dumps)")
    ap.add_argument("--from-log", help="smolvla: re-evaluate an existing simulator log instead of building and simulating")
    ap.add_argument("files", nargs="*", help="mlir: input .mlir files")
    ap.add_argument("-p", "--passes", help="mlir: pass pipeline for softhier-opt (default: per-example table)")
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--modes", nargs="*", default=["0", "1", "2", "5", "6"], help="mesh: MODE[:DEF,...]")
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--real", action="store_true", help="gemm: real-valued data instead of small ints")
    ap.add_argument("--offsets", help="gemm: HBM byte offsets X,W,Z (hex ok; 64 MB per HBM node), default all in node 0")
    ap.add_argument("--define", nargs="*", default=[], help="siglip / attention: extra NAME[=VALUE] macros for shape.h")
    ap.add_argument("--composed", action="store_true", help="attention: the per-head library-call path instead of the fused kernel")
    ap.add_argument("--fused", action="store_true", help="siglip-mlir: use the fused softhier.attention op")
    ap.add_argument("--trace", help="siglip-mlir: record the RedMulE/iDMA/barrier activity into this log (softhier_mlir.sim.trace)")
    ap.add_argument("--tiles", default="model", help="siglip-mlir: GEMM tile policy, 'model' (softhier_mlir.dse.tiling) or 'tm,tn,tk'")
    ap.add_argument("--hbm-split", action="store_true", help="siglip-mlir: parameters in HBM node 1, activations in node 0")
    a = ap.parse_args()
    DATA = a.data
    if a.cluster is None:
        a.cluster = "all" if a.test == "smolvla" else "0"
    if a.test == "gemm":
        ok = run_gemm(a.shapes or DEFAULT_GEMM, a.nsamples, a.real,
                      tuple(int(v, 0) for v in a.offsets.split(",")) if a.offsets else None)
    elif a.test == "gemm-seq":
        ok = run_gemm_seq()
    elif a.test == "rowops":
        ok = run_rowops(a.rows, a.cols, "SH_ALL" if a.cluster == "all" else "0", a.nsamples)
    elif a.test == "fp16cvt":
        ok = run_fp16cvt()
    elif a.test == "mesh":
        ok = run_mesh(a.modes, a.heads)
    elif a.test == "siglip":
        ok = run_siglip(a.seq, a.d, a.ff, a.heads, "SH_ALL" if a.cluster == "all" else "0",
                        extra="".join(f"#define {m.replace('=', ' ', 1)}\n" for m in a.define))
    elif a.test == "attention":
        ok = run_attention(a.seq, a.d, a.heads, "SH_ALL" if a.cluster == "all" else a.cluster, a.composed, a.nsamples,
                           extra="".join(f"#define {m.replace('=', ' ', 1)}\n" for m in a.define))
    elif a.test == "siglip-mlir":
        ok = run_siglip_mlir(a.seq, a.d, a.ff, a.heads, "SH_ALL" if a.cluster == "all" else a.cluster, a.layers, fused=a.fused,
                             trace=Path(a.trace) if a.trace else None, tiles=a.tiles, hbm_split=a.hbm_split)
    elif a.test == "preload":
        ok = run_preload(wait=not a.no_wait)
    elif a.test == "smolvla":
        ok = run_smolvla(a.npz, None if a.all_layers else a.layers, a.attn, -1 if a.cluster == "all" else int(a.cluster),
                         a.nsamples, a.dumps, Path(a.log) if a.log else None, app_dir=a.app_dir, unroll=a.unroll,
                         from_log=Path(a.from_log) if a.from_log else None)
    else:
        ok = run_mlir(a.files, a.passes)
    sys.exit(0 if ok else 1)
