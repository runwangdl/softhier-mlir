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
        # == the fills in gemm/main.c (X at HBM_START, W at 16 MB, Z at 32 MB or behind a larger W; Z0 = 3.0 when accumulating)
        off = {"x": HBM_START, "w": 0x01000000, "z": max(0x02000000, 0x01000000 + ((K * N * 2 + 0xFFFFF) & ~0xFFFFF))}
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


def llm_tokens(S: int, kind: str) -> np.ndarray:
    """Token mask classes (uint16, 0xFFFF = padding) == tok_prefix / causal in tests/gvsoc/llmops/main.c."""
    PAD = 0xFFFF
    if kind == "causal":
        return np.arange(S, dtype=np.uint32)
    img = S - 64
    tok = np.full(S, PAD, dtype=np.uint32)
    tok[:img] = 0; tok[img:img + 5] = 0; tok[img + 48] = 1
    return tok


def masked_softmax_ref(x: np.ndarray, scale: float, tok_q: np.ndarray, tok_k: np.ndarray) -> np.ndarray:
    """softmax(scale x) with key j allowed for query i iff tok[j] <= tok[i] (unsigned); a padding query row -> uniform
    (torch's softmax of an all -inf row, which is what lerobot's eager attention produces for padded queries)."""
    allowed = (tok_k[None, :] <= tok_q[:, None]) & (tok_q[:, None] != 0xFFFF)
    z = np.where(allowed, scale * x, -np.inf)
    m = np.where(allowed.any(1, keepdims=True), z.max(1, keepdims=True), 0.0)
    e = np.exp(z - m); e = np.where(allowed, e, 0.0)
    s = e.sum(1, keepdims=True)
    out = np.where(s > 0, e / np.where(s > 0, s, 1.0), 1.0 / x.shape[1])
    return out.astype(np.float32)


def gqa_reference(S: int, D: int, H: int, Hkv: int, tok: np.ndarray | None, scale: float = 0.125) -> np.ndarray:
    """numpy twin of the sh_attention_gqa call in tests/gvsoc/llmops/main.c (fp16 roundings where the device stores fp16)."""
    from softhier_mlir.testing import lcg
    f = lambda *a, **k: lcg.fill_fp16(*a, **k).astype(np.float32)  # noqa: E731
    r16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    dh = D // H; dkv = Hkv * dh
    q = f(S, D, 21, -8, 8, 0.125); k = f(S, dkv, 22, -8, 8, 0.125); v = f(S, dkv, 23, -16, 16, 0.125)
    o = np.zeros((S, D), np.float32)
    grp = H // Hkv
    for hd in range(H):
        sl = slice(hd * dh, (hd + 1) * dh); kv = slice((hd // grp) * dh, (hd // grp + 1) * dh)
        s = r16(q[:, sl] @ k[:, kv].T)
        if tok is None:
            p = np.exp(scale * s - (scale * s).max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
        else:
            p = masked_softmax_ref(s, scale, tok, tok)
        o[:, sl] = r16(r16(p) @ v[:, kv])
    return o


def run_llmops(rows: int, cols: int, heads: int, kv_heads: int, cluster: str, nsamples: int = 256) -> bool:
    """RMSNorm / RoPE / SiLU-mul / masked softmax / GQA attention of runtime/sh_llm.inc.c vs numpy."""
    from softhier_mlir.testing import lcg
    app = HERE / "llmops"
    S, D, dh = rows, cols, 64
    (app / "shape.h").write_text(f"#define SEQ {S}\n#define D_MODEL {D}\n#define N_HEADS {heads}\n#define N_KV_HEADS {kv_heads}\n"
                                 f"#define CLUSTER {cluster}\n#define NSAMPLES {nsamples}\n")
    build_sw(app)
    r = run_sim(timeout=7200)
    f = lambda *a, **k: lcg.fill_fp16(*a, **k).astype(np.float32)  # noqa: E731
    x = f(S, D, 11, -16, 16, 0.125); b = f(S, D, 12, -16, 16, 0.125); g = f(1, D, 13, 1, 8, 0.25)
    tab = f(S, dh, 15, -16, 16, 0.0625); sc = f(S, S, 16, -32, 32, 0.25)
    rms = x / np.sqrt((x * x).mean(1, keepdims=True) + 1e-5) * g
    xh = x.reshape(S, D // dh, dh); x1, x2 = xh[..., :dh // 2], xh[..., dh // 2:]
    c = tab[:, None, :dh // 2]; s = tab[:, None, dh // 2:]
    rope = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(S, D)
    silu = x / (1 + np.exp(-x)) * b
    ta, tc = llm_tokens(S, "prefix"), llm_tokens(S, "causal")
    ref = {"RMS": rms, "ROPE": rope, "SILU": silu, "SMA": masked_softmax_ref(sc, 0.5, ta, ta), "SMC": masked_softmax_ref(sc, 0.5, tc, tc),
           "O": gqa_reference(S, D, heads, kv_heads, ta), "OU": gqa_reference(S, D, heads, kv_heads, None)}
    got = lcg.parse_samples(r["stdout"])
    ok = r["ok"] and "LLMOPS_DONE" in r["stdout"] and "LLMOPS_FAIL" not in r["stdout"]
    per_op = dict(zip(list(ref), r["rois"]))
    for ln in r["stdout"].splitlines():
        if ln.startswith("[llmops]") or ln.startswith("[sh_"):
            print("     " + ln)
    print(f"{'PASS' if ok else 'FAIL'} llmops S={S} D={D} H={heads} Hkv={kv_heads} cluster={cluster} roi={sum(r['rois'])} ns wall={r['wall_s']}s")
    print("     per-op ns: " + "  ".join(f"{k}={v}" for k, v in per_op.items()))
    for tag, arr in ref.items():
        if tag not in got:
            print(f"     {tag:<5} MISSING"); ok = False; continue
        bad, maxerr = lcg.compare_samples(got[tag], arr, atol=2e-2, rtol=2e-2, show=3)
        print(f"     {tag:<5} samples={len(got[tag])} bad={bad} maxerr={maxerr:.4f} {'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
    (app / "last_run.log").write_text(r["stdout"])
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
          f"tiles={tiles} hbm_split={hbm_split} roi={r['roi_ns']} ns wall={r['wall_s']}s (layer segments: the marks below)")
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


def run_smolvla_vlm(npz: str, layers: int | None, attn: int, cluster: int, nsamples: int = 64, dumps=None,
                    log: Path | None = None, timeout: int = 48 * 3600, app_dir: Path | None = None, from_log: Path | None = None,
                    layer0: int = 0) -> bool:
    """SmolVLA VLM text prefix (connector + 16 Llama layers, real weights via HBM preload) -> sampled device tensors
    (EMB, L<n>, K<n>, V<n>, OUT over the valid tokens) against the fp32 lerobot-semantics reference and the fp16 floor
    stored by `smolvla.py prepare-vlm`."""
    from softhier_mlir.frontend import smolvla
    from softhier_mlir.sim.preload import make_preload_elf
    data = dict(np.load(npz))
    npz_layers = int(data["meta"][1])
    layers = (npz_layers - layer0) if layers is None else layers
    last = layer0 + layers
    n = int(data["meta"][0])
    if not dumps:
        dumps = (("EMB",) if layer0 == 0 else ()) + tuple(f"{t}{i}" for i in range(layer0 + 1, last + 1) for t in ("L", "K", "V")) \
            + (("OUT",) if last == npz_layers else ())
    dumps = tuple(dumps)
    # padded language rows are computed by both sides but never read (masked as keys everywhere); lerobot gives such a
    # query uniform attention over its 241 keys, the device over S_pad = 256 (15 zero rows), so they are not compared
    valid = smolvla.prefix_layout(int(data["meta"][2]), data["lang_mask"])["pad"]
    if from_log is not None:
        stdout = Path(from_log).read_text()
        rois = [int(v) for v in PERF_RE.findall(stdout)]
        print(f"[smolvla-vlm] re-evaluating {from_log}: tokens={n} layers {layer0 + 1}..{last} roi={rois[0] if rois else None} ns")
        return report_smolvla(data, stdout, layers, dumps, bool(rois), valid)
    app = Path(app_dir) if app_dir else HERE / "smolvla_vlm_app"
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    mlir, pre = smolvla.emit_vlm(data, layers, cluster, attn, dumps=dumps, nsamples=nsamples, layer0=layer0)
    info = smolvla.emit_vlm.last_info
    (app / "smolvla_vlm.mlir").write_text(mlir)
    elf = make_preload_elf(app / "smolvla_vlm_preload.elf", pre)
    (app / "main.c").write_text(lower_and_translate(app / "smolvla_vlm.mlir", None))
    build_sw(app)
    text = sum(sz for _, sz in _elf_load_segments(build_sw.last_elf) if sz)
    print(f"[smolvla-vlm] tokens={n} S_pad={info['S_pad']} layers {layer0 + 1}..{last} attn={attn} cluster={cluster} preload {elf.stat().st_size / 2 ** 20:.1f} MiB "
          f"({info['n_west']} layers in the west region, {layers - info['n_west']} in the south), program {text / 1024:.1f} KB; "
          f"KV cache at 0x{info['kv_base']:x} stride 0x{info['kv_stride']:x}; simulating...", flush=True)
    r = run_sim(preload=elf, timeout=timeout, log=log)
    (app / "last_run.log").write_text(r["stdout"])
    print(f"{'PASS' if r['ok'] else 'FAIL'} smolvla-vlm tokens={n} layers {layer0 + 1}..{last} attn={attn} cluster={cluster} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    ok = report_smolvla(data, r["stdout"], layers, dumps, r["ok"], valid)
    if not r["ok"]:
        print(r["stdout"][-1500:])
    return ok


E2E_TRACES = ("redmule", "idma", "cluster_registers")


def _e2e_phase(app: Path, name: str, emit, timeout: int, trace: bool, reuse: bool) -> str:
    """One program of the chain: emit -> preload ELF (arrays freed before the simulator starts: host memory) -> C ->
    build -> gvsoc with the log streamed to <app>/<name>.log. reuse: a finished log of the same phase is re-read."""
    import gc
    import json
    import resource
    from softhier_mlir.sim.preload import make_preload_elf
    log = app / f"{name}.log"
    meta = app / f"{name}.json"
    if reuse and log.exists() and meta.exists() and json.loads(meta.read_text()).get("ok"):
        print(f"[e2e] {name}: re-using {log}")
        from softhier_mlir.sim.trace import program_lines
        return "".join(program_lines(log))
    app.mkdir(parents=True, exist_ok=True)
    rt = (HERE / "../../runtime").resolve()
    (app / "CMakeLists.txt").write_text(f"set(SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/main.c {rt}/sh_ops.c PARENT_SCOPE)\n"
                                        f"set(INCLUDE_DIRS {rt} PARENT_SCOPE)\n")
    mlir, pre = emit()
    (app / f"{name}.mlir").write_text(mlir)
    elf = make_preload_elf(app / f"{name}_preload.elf", pre)
    image = sum(a.nbytes for a in pre.values())
    del pre
    gc.collect()
    (app / "main.c").write_text(lower_and_translate(app / f"{name}.mlir", None))
    build_sw(app, build_dir=app / f"build_{name}")
    text = sum(sz for _, sz in _elf_load_segments(build_sw.last_elf) if sz)
    print(f"[e2e] {name}: preload {image / 2 ** 20:.1f} MiB, program {text / 1024:.1f} KB; simulating{' with traces' if trace else ''}...", flush=True)
    r = run_sim(preload=elf, timeout=timeout, log=log, traces=E2E_TRACES if trace else (), program_output_only=trace)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"{'PASS' if r['ok'] else 'FAIL'} e2e {name}: roi={r['roi_ns']} ns wall={r['wall_s']}s (max child RSS so far {rss:.0f} MB)", flush=True)
    meta.write_text(json.dumps({"ok": r["ok"], "roi_ns": r["roi_ns"], "wall_s": r["wall_s"], "image_bytes": image, "program_bytes": text,
                                "max_child_rss_mb": rss, "trace": trace}))
    if not r["ok"]:
        print(r["stdout"][-3000:])
        raise RuntimeError(f"e2e phase {name} failed (returncode {r['returncode']}; a death without output is an OOM kill)")
    return r["stdout"]


def run_smolvla_e2e(npz: str, steps: int = 10, app_dir: Path | None = None, trace: bool = False, reuse: bool = True,
                    nsamples: int = 128, timeout: int = 48 * 3600, split: int = 8) -> bool:
    """One SmolVLA inference chained over four programs (frontend.smolvla_e2e: vision + connector, prefix layers
    1..split, prefix layers split+1..16, expert flow); each phase's HBM hand-over is dumped in full by the device and
    preloaded into the next one. Validates every hand-over and the final action chunk against lerobot and writes
    <app>/summary.json (per-segment simulated time; with trace=True also HBM bytes and RedMulE / iDMA busy time per
    segment and cluster, softhier_mlir.sim.trace.segment_stats)."""
    import json
    from softhier_mlir.frontend import smolvla, smolvla_e2e as E, smolvla_expert as XE
    from softhier_mlir.testing import lcg
    e2e = dict(np.load(npz))
    it = E.info(e2e)
    n, cams, T, n_img, S_pad = it["n"], it["cams"], it["T"], it["n_img"], it["S_pad"]
    app = Path(app_dir) if app_dir else HERE / "smolvla_e2e_app" / f"c{cams}_t{T}{'_trace' if trace else ''}"
    app.mkdir(parents=True, exist_ok=True)
    valid = smolvla.prefix_layout(cams, e2e["lang_mask"], it["img_tok"])["pad"]
    summ: dict = {"npz": str(npz), "cams": cams, "tokens_per_camera": T, "prefix_tokens": n, "valid_tokens": it["n_valid"], "S_pad": S_pad,
                  "steps": steps, "chunk": int(e2e["x_x0"].shape[0]), "phases": {}, "acc": {}}
    ok = True

    def err(name, got, ref, rows=None):
        got, ref = np.asarray(got, np.float32), np.asarray(ref, np.float32)
        if rows is not None:
            got, ref = got[rows], ref[rows]
        d = np.abs(got - ref)
        summ["acc"][name] = {"max_abs": float(d.max()), "median": float(np.median(d)), "ref_max": float(np.abs(ref).max())}
        print(f"     {name:<10} max abs {d.max():.4f} median {np.median(d):.5f} (|ref| max {np.abs(ref).max():.3f})")
        return float(d.max())

    def phase_record(name, stdout, seg_tags):
        """per-phase marks + (trace) segment statistics"""
        rec = {"marks": parse_marks(stdout), **json.loads((app / f"{name}.json").read_text())}
        if trace:
            from softhier_mlir.sim.trace import segment_stats
            rec["segments"] = segment_stats(app / f"{name}.log")
        summ["phases"][name] = rec
        return rec

    # ---- V: vision tower per camera + connector
    out = _e2e_phase(app, "vision", lambda: E.emit_vision(e2e, nsamples=nsamples), timeout, trace, reuse)
    phase_record("vision", out, None)
    got = lcg.parse_samples(out)
    print(f"[e2e] vision + connector ({cams} camera(s) x {T} tokens):")
    vis_ref = np.concatenate([e2e[f"ref_vis_out_{c}"] for c in range(cams)])
    vals = np.array([v for _, _, v in got["VIS"]]); want = np.array([vis_ref[r_, c] for r_, c, _ in got["VIS"]])
    d = np.abs(vals - want)
    summ["acc"]["VIS"] = {"max_abs": float(d.max()), "median": float(np.median(d)), "ref_max": float(np.abs(vis_ref).max()), "samples": len(vals)}
    print(f"     VIS        {len(vals)} samples: max abs {d.max():.4f} median {np.median(d):.5f} (|ref| max {np.abs(vis_ref).max():.2f})")
    x_img = E.full_dump(got["IMG"], n_img, smolvla.TD)
    err("IMG", x_img, np.concatenate([e2e[f"ref_img_emb_{c}"] for c in range(cams)]))
    del got, out

    # ---- Pa / Pb: the 16-layer prefix writing the KV cache
    kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    x_hand = None
    for name, l0, nl in (("prefix_a", 0, split), ("prefix_b", split, smolvla.TLAYERS - split)):
        def emit(l0=l0, nl=nl, x_hand=x_hand):
            d = E.vlm_data(e2e, x_img=x_img if l0 == 0 else None, x_init=x_hand, layers_from=l0, layers=nl)
            d["meta"][1] = smolvla.TLAYERS                     # no final norm after a partial tower
            mlir, pre = smolvla.emit_vlm(d, nl, -1, -1, dumps=(), layer0=l0, dump_state=True, pad_rows_zero=True)
            print(f"[e2e] {name}: KV cache at 0x{smolvla.emit_vlm.last_info['kv_base']:x} stride 0x{smolvla.emit_vlm.last_info['kv_stride']:x}, "
                  f"S_pad {smolvla.emit_vlm.last_info['S_pad']}")
            return mlir, pre
        out = _e2e_phase(app, name, emit, timeout, trace, reuse)
        phase_record(name, out, None)
        got = lcg.parse_samples(out)
        print(f"[e2e] {name}: layers {l0 + 1}..{l0 + nl} (rows compared: the {int(valid.sum())} valid tokens)")
        x_hand = E.full_dump(got["XS"], n, smolvla.TD)
        for L in range(l0 + 1, l0 + nl + 1):
            k, v = E.full_dump(got[f"KC{L}"], n, smolvla.TKVD), E.full_dump(got[f"VC{L}"], n, smolvla.TKVD)
            kv[L - 1] = (k, v)
            err(f"K{L}", k, e2e[f"ref_kv_k_{L - 1}"], valid); err(f"V{L}", v, e2e[f"ref_kv_v_{L - 1}"], valid)
        del got, out

    # ---- X: the expert's flow loop on the device's KV cache
    xd = E.expert_data(e2e, kv)
    def emit_x():
        return XE.emit_flow(xd, steps, XE.LAYERS, -1, None, False, ("X", "A"), nsamples, kv_base=None,
                            kv_stride=2 * S_pad * XE.DKV * 2, s_pad=S_pad, num_steps=10)
    out = _e2e_phase(app, "expert", emit_x, timeout, trace, reuse)
    phase_record("expert", out, None)
    got = lcg.parse_samples(out)
    P = {k[2:]: xd[k] for k in xd if k.startswith("p_")}
    floor_xt, _ = XE.np_flow(P, steps, XE.LAYERS)
    print(f"[e2e] expert: {steps} flow steps on the device's KV ({n} prefix tokens)")
    ref_xt = e2e["ref_xt"]
    xs = []
    for s in range(steps):
        x = E.full_dump(got[f"X{s}"], *ref_xt.shape[1:]).astype(np.float32)
        xs.append(x)
        d_ref, d_fl = np.abs(x - ref_xt[s + 1]).max(), np.abs(x - floor_xt[s + 1]).max()
        summ["acc"][f"X{s}"] = {"max_abs": float(d_ref), "vs_floor": float(d_fl), "ref_max": float(np.abs(ref_xt[s + 1]).max())}
    print("     x_t per step vs lerobot (max abs): " + " ".join(f"{summ['acc'][f'X{s}']['max_abs']:.4f}" for s in range(steps)))
    a = E.full_dump(got["A"], *ref_xt.shape[1:]).astype(np.float32)
    want = ref_xt[steps]
    da = np.abs(a - want)
    summ["acc"]["actions"] = {"max_abs": float(da.max()), "mean_abs": float(da.mean()), "ref_max": float(np.abs(want).max()),
                              "vs_floor": float(np.abs(a - floor_xt[steps]).max()), "floor_vs_lerobot": float(np.abs(floor_xt[steps] - want).max())}
    print(f"     action chunk x_{steps} (all {a.size} elements) vs lerobot fp32: max abs {da.max():.4f} mean {da.mean():.5f} (|a| max {np.abs(want).max():.3f}); "
          f"vs the expert's fp16 floor on the same KV: {summ['acc']['actions']['vs_floor']:.4f}; that floor vs lerobot: {summ['acc']['actions']['floor_vs_lerobot']:.4f}")
    np.savez(app / "chain_outputs.npz", actions=a, xt=np.stack(xs), x_img=x_img, x_hand=x_hand,
             **{f"kv_k_{L}": k for L, (k, v) in kv.items()}, **{f"kv_v_{L}": v for L, (k, v) in kv.items()})
    ok &= bool(da.max() < 0.1) if steps == 10 else True

    # ---- per-segment simulated time
    seg = e2e_segments(summ, steps)
    summ["segments_ms"] = seg
    print("[e2e] simulated time (ms, 1 GHz, ideal HBM; preload waits and hand-over dumps excluded):")
    for k, v in seg.items():
        print(f"     {k:<22} {v:9.3f}" if isinstance(v, float) else f"     {k:<22} {v}")
    (app / "summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{'PASS' if ok else 'FAIL'} smolvla-e2e cams={cams} tokens/camera={T} prefix={n} steps={steps}: actions max abs {da.max():.4f} "
          f"vs lerobot; total {seg['total']:.3f} ms simulated; summary {app / 'summary.json'}")
    return ok


def run_expert_cost(kv_src: str, chunk: int = 50, steps: int = 10, trace: bool = True, app_dir: Path | None = None,
                    reuse: bool = True, timeout: int = 48 * 3600) -> bool:
    """The expert phase alone for the cost table: a prefix length / chunk point that the chained runs do not cover.
    kv_src: an e2e npz whose chain already ran (its device KV from <app>/chain_outputs.npz; validated against lerobot,
    first `chunk` action rows), `expert` (expert.npz: lerobot's 241-token KV), `vlm_c1` (vlm_c1.npz: the fp32 113-token
    prefix reference's KV). Writes <app>/summary.json (per-step times, KV projection, trace segments)."""
    import json
    from softhier_mlir.frontend import smolvla, smolvla_e2e as E, smolvla_expert as XE
    from softhier_mlir.testing import lcg
    ref_xt = None
    if kv_src.endswith(".npz") and "e2e_" in kv_src:
        e2e = dict(np.load(kv_src))
        it = E.info(e2e)
        chain = HERE / "smolvla_e2e_app" / f"c{it['cams']}_t{it['T']}_trace" / "chain_outputs.npz"
        co = np.load(chain)
        d = E.expert_data(e2e, {L: (co[f"kv_k_{L}"], co[f"kv_v_{L}"]) for L in range(16)})
        ref_xt, lp, tag = e2e["ref_xt"], it["n"], f"c{it['cams']}_t{it['T']}"
    elif kv_src == "expert":
        d = dict(np.load(E.EXPERT_NPZ))
        ref_xt, lp, tag = d["ref_xt"], int(d["p_tok"].shape[1]), "expert241"
    elif kv_src == "vlm_c1":
        v = np.load(E.VLM_NPZ)
        lp = int(v["meta"][0])
        lay = smolvla.prefix_layout(int(v["meta"][2]), v["lang_mask"])
        n_valid = int(lay["pad"].sum())
        w = np.load(E.EXPERT_NPZ)
        d = {k: w[k] for k in w.files if k.startswith("p_") and not k.startswith(("p_kp", "p_vp"))}
        d["p_tok"] = XE.prefix_tok(lay["pad"].astype(np.float32))
        d["p_rq_self"] = XE.rope_table(n_valid + np.arange(XE.S))
        for L in range(16):
            d[f"p_kp{L}"], d[f"p_vp{L}"] = v[f"ref_K{L + 1}"].astype(np.float16), v[f"ref_V{L + 1}"].astype(np.float16)
        tag = "vlm113"
    else:
        raise ValueError(kv_src)
    S_pad = ((lp + 127) // 128) * 128
    app = Path(app_dir) if app_dir else HERE / "smolvla_e2e_app" / f"expert_{tag}_chunk{chunk}{'_trace' if trace else ''}"
    out = _e2e_phase(app, "expert", lambda: XE.emit_flow(d, steps, XE.LAYERS, -1, None, False, ("A",), 64, kv_base=None,
                                                          kv_stride=2 * S_pad * XE.DKV * 2, s_pad=S_pad, chunk=chunk), timeout, trace, reuse)
    marks = parse_marks(out)
    kvp = (marks["kvproj"] - marks["start"]) / 1e6
    st, prev = [], marks["kvproj"]
    for s in range(steps):
        st.append((marks[f"step{s}"] - prev) / 1e6)
        prev = marks[f"step{s}"]
    summ = {"kv_src": kv_src, "prefix_tokens": lp, "S_pad": S_pad, "chunk": chunk, "steps": steps, "marks": marks,
            "segments_ms": {"expert_kv_projection": kvp, "expert_per_step": st, "expert_step_mean": float(np.mean(st)), "expert_total": kvp + sum(st)},
            **json.loads((app / "expert.json").read_text())}
    if trace:
        from softhier_mlir.sim.trace import segment_stats
        summ["phases"] = {"expert": {"marks": marks, "segments": segment_stats(app / "expert.log")}}
    got = lcg.parse_samples(out)
    a = E.full_dump(got["A"], chunk, XE.AD).astype(np.float32)
    P = {k[2:]: d[k] for k in d if k.startswith("p_")}
    P["x0"], P["rq_self"], P["rq_cross"] = P["x0"][:chunk], P["rq_self"][:chunk], P["rq_cross"][:chunk]
    floor_xt, _ = XE.np_flow(P, steps, XE.LAYERS)
    summ["acc"] = {"actions_vs_floor": float(np.abs(a - floor_xt[steps]).max())}
    msg = f"vs the fp16-floor twin {summ['acc']['actions_vs_floor']:.4f}"
    if ref_xt is not None:
        summ["acc"]["actions_vs_lerobot"] = float(np.abs(a - ref_xt[steps][:chunk]).max())
        msg += f", vs lerobot (first {chunk} rows of the 50-row chunk: causal own keys) {summ['acc']['actions_vs_lerobot']:.4f}"
    (app / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"PASS expert-cost {tag} Lp={lp} chunk={chunk}: {np.mean(st):.3f} ms/step, KV projection {kvp:.3f} ms; actions {msg}")
    return True


def run_vision_cost(T: int = 1024, layers: int = 1, trace: bool = True, app_dir: Path | None = None, reuse: bool = True,
                    timeout: int = 48 * 3600) -> bool:
    """The chain's vision phase (frontend.smolvla_e2e.emit_vision: SigLIP layers + post-LN + pixel shuffle + connector)
    for one camera at T SigLIP tokens with `layers` layers, on vision_s<T>.npz's test image (T = 1024: the full
    512 x 512 image, position rows 0..1023): the per-layer cost the composed full-resolution numbers multiply by 12."""
    import json
    from softhier_mlir.frontend import smolvla, smolvla_e2e as E
    from softhier_mlir.testing import lcg
    src = np.load(f"/app/models/smolvla_base/vision_s{T}.npz")
    img_tok = T // E.PS
    n = img_tok + smolvla.LANG_LEN + 1
    e2e = {"xp": src["xp"][None], "vis_pos_ids": smolvla.token_ids_for(T), "meta": np.array([n, 16, 1, T, n], dtype=np.int64)}
    app = Path(app_dir) if app_dir else HERE / "smolvla_e2e_app" / f"vision_t{T}_L{layers}{'_trace' if trace else ''}"
    out = _e2e_phase(app, "vision", lambda: E.emit_vision(e2e, layers=layers, nsamples=64), timeout, trace, reuse)
    marks = parse_marks(out)
    got = lcg.parse_samples(out)
    ref = src[f"np_L{layers}"] if layers < smolvla.LAYERS else src["np_OUT"]
    if layers < smolvla.LAYERS:   # the program ends with the post-LN: compare against the fp16 floor's post-LN of layer `layers`
        x = src[f"np_L{layers}"].astype(np.float32)
        mu, var = x.mean(1, keepdims=True), x.var(1, keepdims=True)
        ref = ((x - mu) / np.sqrt(var + smolvla.LN_EPS) * src["p_gpost"].astype(np.float32) + src["p_bepost"].astype(np.float32))
    vals = np.array([v for _, _, v in got["VIS"]]); want = np.array([ref[r_, c] for r_, c, _ in got["VIS"]])
    summ = {"T": T, "layers": layers, "marks": marks, **json.loads((app / "vision.json").read_text()),
            "acc": {"VIS_vs_floor_max_abs": float(np.abs(vals - want).max()), "ref_max": float(np.abs(ref).max())}}
    lt = layer_times(marks)
    summ["segments_ms"] = {"embed": (marks["vemb0"] - marks["start"]) / 1e6, "attention_per_layer": [v[0] / 1e6 for v in lt.values()],
                           "mlp_per_layer": [v[1] / 1e6 for v in lt.values()], "post_ln": (marks["vis0"] - marks[f"layer{layers}"]) / 1e6,
                           "connector": (marks["conn"] - marks["vis0"]) / 1e6}
    if trace:
        from softhier_mlir.sim.trace import segment_stats
        summ["phases"] = {"vision": {"marks": marks, "segments": segment_stats(app / "vision.log")}}
    (app / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"PASS vision-cost T={T} layers={layers}: {summ['segments_ms']}; VIS vs fp16 floor max abs {summ['acc']['VIS_vs_floor_max_abs']:.4f}")
    return True


def run_prefix_cost(cams: int, layers: int = 8, trace: bool = True, app_dir: Path | None = None, reuse: bool = True,
                    timeout: int = 48 * 3600) -> bool:
    """The full-resolution prefix for the cost table: vlm_c<cams>.npz (1024 SigLIP tokens per camera -> 113 / 241
    prefix tokens) through emit_vlm: pixel shuffle + connector + scaling + the first `layers` decoder layers (the
    per-layer time of layers 9-16 equals 1-8 within 2 %, docs/SMOLVLA.md), traced for bytes / utilisation."""
    import json
    from softhier_mlir.frontend import smolvla
    data = dict(np.load(f"/app/models/smolvla_base/vlm_c{cams}.npz"))
    app = Path(app_dir) if app_dir else HERE / "smolvla_e2e_app" / f"prefix_c{cams}_t1024_L{layers}{'_trace' if trace else ''}"
    out = _e2e_phase(app, "prefix_a", lambda: smolvla.emit_vlm(data, layers, -1, -1, dumps=(), layer0=0), timeout, trace, reuse)
    marks = parse_marks(out)
    lt = layer_times(marks)
    summ = {"cams": cams, "prefix_tokens": int(data["meta"][0]), "layers": layers, "marks": marks, **json.loads((app / "prefix_a.json").read_text()),
            "segments_ms": {"connector": (marks["emb"] - marks["start"]) / 1e6, "prefix_per_layer": [sum(v) / 1e6 for v in lt.values()],
                            "attention_per_layer": [v[0] / 1e6 for v in lt.values()]}}
    if trace:
        from softhier_mlir.sim.trace import segment_stats
        summ["phases"] = {"prefix_a": {"marks": marks, "segments": segment_stats(app / "prefix_a.log")}}
    (app / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"PASS prefix-cost cams={cams} n={summ['prefix_tokens']}: connector {summ['segments_ms']['connector']:.3f} ms, "
          f"{np.mean(summ['segments_ms']['prefix_per_layer']):.3f} ms/layer")
    return True


def e2e_segments(summ: dict, steps: int) -> dict:
    """Segment times (ms) from the marks of the four phases."""
    ph = summ["phases"]
    mv, ma, mb, mx = (ph[k]["marks"] for k in ("vision", "prefix_a", "prefix_b", "expert"))
    cams = summ["cams"]
    vis, prev = [], mv["start"]
    for c in range(cams):
        vis.append((mv[f"vis{c}"] - prev) / 1e6)
        prev = mv[f"vis{c}"]
    conn = (mv["conn"] - mv[f"vis{cams - 1}"]) / 1e6 + (ma["emb"] - ma["start"]) / 1e6       # shuffle + GEMM, then the scaling
    pre_l = [sum(v) / 1e6 for v in layer_times(ma).values()] + [sum(v) / 1e6 for v in layer_times(mb).values()]
    kvp = (mx["kvproj"] - mx["start"]) / 1e6
    st, prev = [], mx["kvproj"]
    for s in range(steps):
        st.append((mx[f"step{s}"] - prev) / 1e6)
        prev = mx.get(f"xdump{s}", mx[f"step{s}"])
    out = {"vision_per_camera": vis, "vision": sum(vis), "connector": conn, "prefix_per_layer": pre_l, "prefix": sum(pre_l),
           "expert_kv_projection": kvp, "expert_per_step": st, "expert_step_mean": float(np.mean(st)), "expert_total": kvp + sum(st)}
    out["total"] = out["vision"] + out["connector"] + out["prefix"] + out["expert_total"]
    return out


def report_smolvla(data: dict, stdout: str, layers: int, dumps: tuple, ok: bool, valid_rows=None) -> bool:
    """Timing (marks) + accuracy of the sampled tensors. A tensor passes when every sample is within
    atol = 3% of the tensor's max |ref| (+ 5% relative) of the HF fp32 reference: the program is fp16 end
    to end (fp16 operands and RedMulE fp16 accumulation), so ~1e-2 of the tensor scale is the floor, and
    the post layernorm amplifies the last layer's error by gamma / row-std (~6x for this checkpoint)."""
    from softhier_mlir.testing import lcg
    got = lcg.parse_samples(stdout)
    if valid_rows is not None:      # rows nobody reads (padded language tokens): not compared
        got = {t: [(r_, c, v) for r_, c, v in smp if valid_rows[r_]] for t, smp in got.items()}
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
    ap.add_argument("test", choices=["gemm", "gemm-seq", "mlir", "rowops", "fp16cvt", "siglip", "siglip-mlir", "mesh", "attention", "preload", "smolvla",
                                     "llmops", "smolvla-vlm", "smolvla-e2e"])
    ap.add_argument("--data", choices=["preload", "device"], default="preload",
                    help="test inputs: generated on the host into the HBM preload image (default) or on the device")
    ap.add_argument("--kv-heads", type=int, default=5, help="llmops: key/value heads (GQA)")
    ap.add_argument("--layer0", type=int, default=0, help="smolvla-vlm: first layer to run (the program starts from the reference's L<layer0>)")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--npz", default="/app/models/smolvla_base/vision_s256.npz", help="smolvla: output of `smolvla.py prepare`")
    ap.add_argument("--attn", type=int, default=-1, help="smolvla: cluster of the per-head attention ops (-1 = SH_ALL per op)")
    ap.add_argument("--dumps", nargs="*", help="smolvla: tensors to compare (default EMB, L1..Ln, OUT)")
    ap.add_argument("--log", help="smolvla: stream the simulator output to this file")
    ap.add_argument("--all-layers", action="store_true", help="smolvla: run every layer in the npz (overrides --layers)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--ff", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=None, help="attention heads (default 12; llmops 15)")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=None)
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
    ap.add_argument("--steps", type=int, default=10, help="smolvla-e2e: flow steps simulated (of the 10-step schedule)")
    ap.add_argument("--e2e-trace", action="store_true", help="smolvla-e2e: record RedMulE / iDMA / barrier traces (HBM bytes, utilisation per segment)")
    ap.add_argument("--fresh", action="store_true", help="smolvla-e2e: re-simulate phases that already have a finished log")
    ap.add_argument("--split", type=int, default=8, help="smolvla-e2e: prefix layers in the first prefix program")
    ap.add_argument("--expert-only", help="smolvla-e2e: only the expert phase for the cost table; KV from an e2e npz whose chain ran, "
                                          "'expert' (241-token lerobot KV) or 'vlm_c1' (113-token prefix reference KV)")
    ap.add_argument("--prefix-only", type=int, help="smolvla-e2e: only the full-resolution prefix (vlm_c<N>.npz), --layers layers")
    ap.add_argument("--vision-only", type=int,help="smolvla-e2e: only the vision phase, 1 camera at this many SigLIP tokens, --layers layers")
    ap.add_argument("--chunk", type=int, default=50,help="smolvla-e2e --expert-only: action rows (25 = the first 25 of the 50-row chunk)")
    a = ap.parse_args()
    DATA = a.data
    if a.cluster is None:
        a.cluster = "all" if a.test in ("smolvla", "smolvla-vlm") else "0"
    if a.test == "gemm":
        ok = run_gemm(a.shapes or DEFAULT_GEMM, a.nsamples, a.real,
                      tuple(int(v, 0) for v in a.offsets.split(",")) if a.offsets else None)
    elif a.test == "gemm-seq":
        ok = run_gemm_seq()
    elif a.test == "rowops":
        ok = run_rowops(a.rows, (a.cols or 768), "SH_ALL" if a.cluster == "all" else "0", a.nsamples)
    elif a.test == "fp16cvt":
        ok = run_fp16cvt()
    elif a.test == "llmops":
        ok = run_llmops(a.rows, a.cols or 960, a.heads or 15, a.kv_heads, "SH_ALL" if a.cluster == "all" else "0", a.nsamples)
    elif a.test == "smolvla-e2e" and a.prefix_only:
        ok = run_prefix_cost(a.prefix_only, a.layers, a.e2e_trace, Path(a.app_dir) if a.app_dir else None, not a.fresh)
    elif a.test == "smolvla-e2e" and a.vision_only:
        ok = run_vision_cost(a.vision_only, a.layers, a.e2e_trace, Path(a.app_dir) if a.app_dir else None, not a.fresh)
    elif a.test == "smolvla-e2e" and a.expert_only:
        ok = run_expert_cost(a.expert_only, a.chunk, a.steps, a.e2e_trace, Path(a.app_dir) if a.app_dir else None, not a.fresh)
    elif a.test == "smolvla-e2e":
        ok = run_smolvla_e2e(a.npz, a.steps, Path(a.app_dir) if a.app_dir else None, a.e2e_trace, not a.fresh, min(a.nsamples, 128),
                             split=a.split)
    elif a.test == "smolvla-vlm":
        ok = run_smolvla_vlm(a.npz, None if a.all_layers else a.layers, a.attn, -1 if a.cluster == "all" else int(a.cluster),
                             a.nsamples, a.dumps, Path(a.log) if a.log else None, app_dir=a.app_dir,
                             from_log=Path(a.from_log) if a.from_log else None, layer0=a.layer0)
    elif a.test == "mesh":
        ok = run_mesh(a.modes, (a.heads or 12))
    elif a.test == "siglip":
        ok = run_siglip(a.seq, a.d, a.ff, (a.heads or 12), "SH_ALL" if a.cluster == "all" else "0",
                        extra="".join(f"#define {m.replace('=', ' ', 1)}\n" for m in a.define))
    elif a.test == "attention":
        ok = run_attention(a.seq, a.d, (a.heads or 12), "SH_ALL" if a.cluster == "all" else a.cluster, a.composed, a.nsamples,
                           extra="".join(f"#define {m.replace('=', ' ', 1)}\n" for m in a.define))
    elif a.test == "siglip-mlir":
        ok = run_siglip_mlir(a.seq, a.d, a.ff, (a.heads or 12), "SH_ALL" if a.cluster == "all" else a.cluster, a.layers, fused=a.fused,
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
