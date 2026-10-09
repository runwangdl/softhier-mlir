"""One SmolVLA inference on SoftHier: vision tower per camera -> connector -> 16-layer VLM prefix writing the KV
cache -> action expert's 10-step flow reading it. The ledger is docs/SMOLVLA_E2E.md.

  ref      (lerobot venv)  python -m softhier_mlir.frontend.smolvla_expert_ref --vlm-dtype fp32 --cams 1 --img-size 256 \\
                                   --out /app/models/smolvla_base/e2e_ref_c1_t256.npz
           lerobot's SmolVLAPolicy on 1 or 3 seeded camera images: SigLIP output + connector output per camera, the
           prefix embeddings, the 16 layers' KV cache, x_t per flow step and the action chunk (fp32).
  prepare  (system python: torch + safetensors)
           python3 -m softhier_mlir.frontend.smolvla_e2e prepare --ref .../e2e_ref_c1_t256.npz --out .../e2e_c1_t256.npz
           the chain's inputs and host tables (im2col'd camera images, the vision position rows, language rows, state
           embedding, token classes, RoPE tables, noise) + the reference arrays. The weights are read at emit time from
           the per-model npz files (`smolvla.py prepare --seq 1024`, `prepare-vlm --cams 1`, `smolvla_expert.py prepare`).
  run      tests/gvsoc/run.py smolvla-e2e --npz .../e2e_c1_t256.npz

Phases (the HBM state each one hands to the next is dumped in full by the device and preloaded bit-exactly into
the next program; why there are phases: the weights of the four parts are 162 + 22.5 + 300 + 189.5 MiB fp16 =
674 MiB > the 512 MiB of populated HBM (west + south edges), and gvsoc's host memory is ~2x its preload image):
  V   vision tower (12 SigLIP layers, one scf.for over the cameras around the layer loop) + pixel shuffle +
      connector GEMM per camera                                         -> IMG  [n_img, 960] (connector output)
  Pa  prefix: sqrt(960) scaling, layers 1..8 (frontend.smolvla.emit_vlm, x_img = IMG)
                                                                        -> XS [n, 960], KC/VC 1..8 [n, 320]
  Pb  prefix layers 9..16 (emit_vlm layer0 = 8, x_init = XS)            -> KC/VC 9..16
  X   expert: cross-layer KV projection once per chunk + num_steps Euler steps (frontend.smolvla_expert.emit_flow,
      KV region preloaded with KC/VC 1..16 in the VLM layout)           -> x_t per step, the action chunk

Tokens per camera = SigLIP tokens: 256 (256 x 256 image: 16 x 16 patches with SmolVLM's bucketed position rows,
16 image tokens after the 4 x 4 pixel shuffle) or 1024 (lerobot's 512 x 512, 64 image tokens). The prefix is
n_img + 48 language + 1 state tokens: 65 / 97 at 256 tokens for 1 / 3 cameras, 113 / 241 at 1024.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from softhier_mlir.frontend import smolvla as V
from softhier_mlir.frontend import smolvla_expert as X
from softhier_mlir.frontend.clusters import barrier_op, cl_attr, set_attr
from softhier_mlir.frontend.siglip import _Emitter, gemm_tile_attrs
from softhier_mlir.sim.preload import sentinel_array

MODELS = Path("/app/models/smolvla_base")
VISION_NPZ = MODELS / "vision_s1024.npz"      # SigLIP weights in library layout + the full 1024-row position table
VLM_NPZ = MODELS / "vlm_c1.npz"               # text tower + connector weights (fp16 library layout)
EXPERT_NPZ = MODELS / "expert.npz"            # expert weights + time table (fp16 library layout)
D, FF, HEADS, VLAYERS, PATCH = V.D, V.FF, V.HEADS, V.LAYERS, V.PATCH
PS = V.PIX_SCALE * V.PIX_SCALE                # 16 SigLIP tokens per image token
HBM_DATA_START = V.HBM_DATA_START


# ----------------------------------------------------------------------------- host side
def im2col(pixels: np.ndarray) -> np.ndarray:
    """[3, s, s] -> [(s/16)^2, 3*16*16] patch rows, (c, kh, kw) order, row-major patch grid (any image side)."""
    c, h, w = pixels.shape
    g = h // PATCH
    return np.ascontiguousarray(pixels.reshape(c, g, PATCH, g, PATCH).transpose(1, 3, 0, 2, 4).reshape(g * g, -1))


def prepare(ref: str | Path, out: str | Path, ckpt: str | Path = MODELS / "model.safetensors") -> Path:
    """Chain inputs + host tables + lerobot reference -> npz (small: no weights)."""
    r = dict(np.load(ref))
    cams = int(r["images"].shape[0])
    side = int(r["images"].shape[-1])
    T = (side // PATCH) ** 2
    img_tok = T // PS
    lang_mask = r["lang_mask"].astype(np.int64)
    lay = V.prefix_layout(cams, lang_mask, img_tok)
    assert lay["n"] == int(r["prefix_valid"].shape[0]), (lay["n"], r["prefix_valid"].shape)
    assert np.array_equal(lay["pad"], r["prefix_valid"] != 0), "prefix padding differs from lerobot's"
    tw = V.TextTower(ckpt)
    a = {
        "xp": np.stack([im2col(im.astype(np.float32)) for im in r["images"]]).astype(np.float16),     # [cams, T, 768]
        "vis_pos_ids": r["vis_pos_ids"].astype(np.int64),
        "p_lang": tw.embed_rows(r["lang_tokens"]).astype(np.float16),                               # raw rows (device scales)
        "p_state_emb": (r["state"] @ tw.state_w.T + tw.state_b).reshape(1, -1).astype(np.float16),
        "p_rope": V.rope_table(lay["pos"]).astype(np.float16),
        "tok": lay["tok"], "lang_mask": lang_mask, "img_tokens": np.array(img_tok),
        "meta": np.array([lay["n"], V.TLAYERS, cams, T, int(r["n_valid"])], dtype=np.int64),
        # expert tables: classes 0 / 1 (state) / PAD over the prefix, positions n_valid + i on the self layers
        "x_tok": X.prefix_tok(r["prefix_valid"]),
        "x_rq_self": X.rope_table(int(r["n_valid"]) + np.arange(X.S)),
        "x_rq_cross": X.rope_table(np.arange(X.S)),
        "x_x0": r["noise"].astype(np.float16),
    }
    assert np.array_equal(a["x_tok"][0], lay["tok"]), "expert token classes != prefix token classes"
    for k, v in r.items():
        if k not in ("images",):
            a[f"ref_{k}"] = v
    np.savez(out, **a)
    print(f"[e2e prepare] {out}: {cams} camera(s) x {T} SigLIP tokens -> {img_tok} image tokens each; prefix {lay['n']} tokens "
          f"({int(r['n_valid'])} valid), expert chunk {r['noise'].shape}")
    return Path(out)


def info(e2e) -> dict:
    n, _, cams, T, n_valid = (int(v) for v in e2e["meta"])
    return {"n": n, "cams": cams, "T": T, "img_tok": T // PS, "n_img": cams * T // PS, "n_valid": n_valid,
            "S_pad": ((n + 127) // 128) * 128}


# ----------------------------------------------------------------------------- phase V: vision + connector
def emit_vision(e2e, cluster: int = -1, nsamples: int = 128, tiles: str = "model", vision_npz=VISION_NPZ,
                vlm_npz=VLM_NPZ, layers: int = VLAYERS, hbm_base: int | None = None, marks: bool = True,
                connector: bool = True) -> tuple[str, dict[int, np.ndarray]]:
    """SigLIP (12 layers, the layer loop and head loop of frontend.smolvla.emit) inside one scf.for over the
    cameras, then the pixel shuffle of every camera's output and one connector GEMM over all image tokens.
    Marks: start, vemb<c>, attn<c*12+l>, layer<c*12+l>, vis<c> (post-LN), conn, end. Dumps: VIS (samples of the
    SigLIP outputs, row = c*T + token), IMG (every element of the connector output: the hand-over).
    Spatial split (frontend.smolvla_split): hbm_base = first HBM offset of this program's buffers (default
    HBM_DATA_START), marks = False drops every mark, connector = False stops after the SigLIP output (`vis`, whose
    offset is emit_vision.last_info["vis_off"]; the prefix program then runs pixel shuffle + connector)."""
    it = info(e2e)
    cams, T, n_img = it["cams"], it["T"], it["n_img"]
    vw = np.load(vision_npz)
    PV = {k[2:]: vw[k] for k in vw.files if k.startswith("p_") and k != "p_pos"}
    pos = np.ascontiguousarray(vw["p_pos"][e2e["vis_pos_ids"]])
    wc = np.load(vlm_npz)["p_wc"]
    dh = D // HEADS
    e = _Emitter()
    e.next_off = HBM_DATA_START if hbm_base is None else hbm_base
    Tt: dict[str, str] = {}
    pre: dict[int, np.ndarray] = {}
    cl = cl_attr(cluster)       # an id, -1 = SH_ALL, or a cluster set (frontend.clusters)
    hc = cl
    sp = e.space
    mt = lambda r, c: f'memref<{r}x{c}xf16, "{sp}">'  # noqa: E731
    hv = lambda rows, cols, ld, eoff: f'memref<{rows}x{cols}xf16, strided<[{ld}, 1], offset: {eoff}>, "{sp}">'  # noqa: E731

    def alloc(rows, cols):
        off = e.next_off
        e.next_off += (rows * cols * 2 + 4095) & ~4095
        return off

    def B(name, rows, cols, arr=None):
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            pre[e.next_off] = arr
        Tt[name] = e.buf(name, rows, cols)

    def mark(tag, idx=None):
        if marks:
            e.op(f'softhier.mark {idx + " " if idx else ""}{{tag = "{tag}"{set_attr(cluster)}}}')

    # activations of one camera (reused), the camera families (inputs, SigLIP outputs), connector buffers
    for nm, r, c in [("x", T, D), ("ln1", T, D), ("q", T, D), ("k", T, D), ("v", T, D), ("kT", D, T),
                     ("sc", HEADS * T, T), ("o", T, D), ("ao", T, D), ("h", T, D), ("ln2", T, D),
                     ("f1", T, FF), ("g", T, FF), ("f2", T, D)]:
        B(nm, r, c)
    cam_stride = (T * D * 2 + 4095) & ~4095
    xp_off = e.next_off
    for c in range(cams):
        pre[alloc(T, D)] = np.ascontiguousarray(e2e["xp"][c])
    vis_off = e.next_off
    e.next_off += cams * cam_stride
    assert cam_stride == T * D * 2, "camera blocks must be contiguous (pixel shuffle / sample views)"
    Tt["vis"] = mt(cams * T, D)
    e.op(f"%vis = softhier.hbm_buffer {{offset = {vis_off} : i32}} : {Tt['vis']}")
    if connector:
        B("ps", n_img, D * PS); B("img", n_img, V.TD)
    B("pos", T, D, pos)
    B("wpe", D, D, PV["wpe"]); B("bpe", 1, D, PV["bpe"]); B("gpost", 1, D, PV["gpost"]); B("bepost", 1, D, PV["bepost"])
    if connector:
        B("wc", D * PS, V.TD, wc)
    layer0_off, stride = {}, 0
    for L in range(layers):
        begin = e.next_off
        for nm, r, c in V.PARAMS:
            off = alloc(r, c)
            assert PV[f"{nm}{L}"].shape == (r, c)
            pre[off] = PV[f"{nm}{L}"]
            if L == 0:
                layer0_off[nm] = off
                Tt[nm] = mt(r, c)
        if L == 0:
            stride = e.next_off - begin
    sent = sentinel_array()
    sent_off = (max(off + a.nbytes for off, a in pre.items()) + 0xFFFF) & ~0xFFFF
    pre[sent_off] = sent
    Tt["sentinel"] = mt(*sent.shape)
    e.op(f"%sentinel = softhier.hbm_buffer {{offset = {sent_off} : i32}} : {Tt['sentinel']}")
    mark("preload")
    e.op(f"softhier.preload_wait %sentinel : {Tt['sentinel']}")

    gem = lambda tm, tn, tk: f"tile_m = {tm} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, pipeline"  # noqa: E731
    qk, pv = gem(min(256, T), min(256, T), dh), gem(min(256, T), dh, min(256, T))
    big = gemm_tile_attrs(T, D, D, cluster, tiles)
    fc1, fc2 = gemm_tile_attrs(T, FF, D, cluster, tiles), gemm_tile_attrs(T, D, FF, cluster, tiles)
    for nm, v in (("c0", 0), ("c1", 1), ("cH", HEADS), ("cL", layers), ("cC", cams), ("cLL", layers)):
        e.op(f"%{nm} = arith.constant {v} : index")
    mark("start")
    e.op("scf.for %cam = %c0 to %cC step %c1 {")
    e.op(f"%xpc = softhier.hbm_buffer %cam {{offset = {xp_off} : i32, stride = {cam_stride} : i32}} : {mt(T, D)}")
    e.op(f"%fin = softhier.hbm_buffer %cam {{offset = {vis_off} : i32, stride = {cam_stride} : i32}} : {mt(T, D)}")
    e.op("%camL = arith.muli %cam, %cLL : index")
    e.op(f"softhier.gemm %xpc, %wpe into %x {{fmt = \"fp16\", {big}, {cl}}} : {mt(T, D)}, {Tt['wpe']}, {Tt['x']}")
    e.op(f"softhier.add_bias %x, %bpe -> %x {{{cl}}} : {Tt['x']}, {Tt['bpe']} -> {Tt['x']}")
    e.op(f"softhier.add %x, %pos -> %x {{{cl}}} : {Tt['x']}, {Tt['pos']} -> {Tt['x']}")
    mark("vemb", "%cam")
    e.op("scf.for %L = %c0 to %cL step %c1 {")
    e.op("%Lc = arith.addi %camL, %L : index")
    e.op("%L1 = arith.addi %Lc, %c1 : index")
    for nm, r, c in V.PARAMS:
        e.op(f"%{nm} = softhier.hbm_buffer %L {{offset = {layer0_off[nm]} : i32, stride = {stride} : i32}} : {Tt[nm]}")
    e.op(f"softhier.layernorm %x, %g1, %be1 -> %ln1 {{eps = {V.LN_EPS:.1e} : f32, {cl}}} : {Tt['x']}, {Tt['g1']}, {Tt['be1']} -> {Tt['ln1']}")
    for dst, w, bias in (("q", "wq", "bq"), ("k", "wk", "bk"), ("v", "wv", "bv")):
        e.op(f"softhier.gemm %ln1, %{w} into %{dst} {{fmt = \"fp16\", {big}, {cl}}} : {Tt['ln1']}, {Tt[w]}, {Tt[dst]}")
        e.op(f"softhier.add_bias %{dst}, %{bias} -> %{dst} {{{cl}}} : {Tt[dst]}, {Tt[bias]} -> {Tt[dst]}")
    e.op(f"softhier.transpose %k -> %kT {{{cl}}} : {Tt['k']} -> {Tt['kT']}")
    e.op("scf.for %hd = %c0 to %cH step %c1 {")
    views = {}
    for name, src, rows, cols, ld, st in (("qh", "q", T, dh, D, dh), ("kTh", "kT", dh, T, T, dh * T), ("sh", "sc", T, T, T, T * T),
                                          ("vh", "v", T, dh, D, dh), ("oh", "o", T, dh, D, dh)):
        views[name] = hv(rows, cols, ld, 0)
        e.op(f"%{name} = softhier.view %{src}, %hd {{stride = {st} : i32}} : {Tt[src]} -> {views[name]}")
    e.op(f"softhier.gemm %qh, %kTh into %sh {{fmt = \"fp16\", {qk}, {hc}}} : {views['qh']}, {views['kTh']}, {views['sh']}")
    e.op(f"softhier.softmax %sh -> %sh {{scale = {1 / math.sqrt(dh)!r} : f32, {hc}}} : {views['sh']} -> {views['sh']}")
    e.op(f"softhier.gemm %sh, %vh into %oh {{fmt = \"fp16\", {pv}, {hc}}} : {views['sh']}, {views['vh']}, {views['oh']}")
    e.op("}")
    e.op(barrier_op(cluster))
    mark("attn", "%L1")
    e.op(f"softhier.gemm %o, %wo into %ao {{fmt = \"fp16\", {big}, {cl}}} : {Tt['o']}, {Tt['wo']}, {Tt['ao']}")
    e.op(f"softhier.add_bias %ao, %bo -> %ao {{{cl}}} : {Tt['ao']}, {Tt['bo']} -> {Tt['ao']}")
    e.op(f"softhier.add %x, %ao -> %h {{{cl}}} : {Tt['x']}, {Tt['ao']} -> {Tt['h']}")
    e.op(f"softhier.layernorm %h, %g2, %be2 -> %ln2 {{eps = {V.LN_EPS:.1e} : f32, {cl}}} : {Tt['h']}, {Tt['g2']}, {Tt['be2']} -> {Tt['ln2']}")
    e.op(f"softhier.gemm %ln2, %w1 into %f1 {{fmt = \"fp16\", {fc1}, {cl}}} : {Tt['ln2']}, {Tt['w1']}, {Tt['f1']}")
    e.op(f"softhier.add_bias %f1, %b1 -> %f1 {{{cl}}} : {Tt['f1']}, {Tt['b1']} -> {Tt['f1']}")
    e.op(f"softhier.gelu %f1 -> %g {{{cl}}} : {Tt['f1']} -> {Tt['g']}")
    e.op(f"softhier.gemm %g, %w2 into %f2 {{fmt = \"fp16\", {fc2}, {cl}}} : {Tt['g']}, {Tt['w2']}, {Tt['f2']}")
    e.op(f"softhier.add_bias %f2, %b2 -> %f2 {{{cl}}} : {Tt['f2']}, {Tt['b2']} -> {Tt['f2']}")
    e.op(f"softhier.add %h, %f2 -> %x {{{cl}}} : {Tt['h']}, {Tt['f2']} -> {Tt['x']}")
    mark("layer", "%L1")
    e.op("}")
    e.op(f"softhier.layernorm %x, %gpost, %bepost -> %fin {{eps = {V.LN_EPS:.1e} : f32, {cl}}} : {Tt['x']}, {Tt['gpost']}, {Tt['bepost']} -> {mt(T, D)}")
    mark("vis", "%cam")
    e.op("}")
    if not connector:
        body = "\n".join(e.lines)
        mlir = f"builtin.module {{\n  func.func @smolvla_e2e_vision() {{\n{body}\n    func.return\n  }}\n}}\n"
        emit_vision.last_info = {"image_bytes": sum(a.nbytes for a in pre.values()), "layer_stride": stride, "vis_off": vis_off,
                                 "end": e.next_off}
        return mlir, pre
    # connector: pixel shuffle per camera into rows of ps, one GEMM over every camera's image tokens
    it_c = it["img_tok"]
    for c in range(cams):
        src = hv(T, D, D, c * T * D)
        dst = hv(it_c, D * PS, D * PS, c * it_c * D * PS)
        e.op(f"%visc{c} = softhier.view %vis : {Tt['vis']} -> {src}")
        e.op(f"%ps{c} = softhier.view %ps : {Tt['ps']} -> {dst}")
        e.op(f"softhier.pixel_shuffle %visc{c} -> %ps{c} {{scale = {V.PIX_SCALE} : i32, {cl}}} : {src} -> {dst}")
    tm, tn, tk = V.VLM_TILES["wc"]
    e.op(f"softhier.gemm %ps, %wc into %img {{fmt = \"fp16\", tile_m = {min(tm, n_img)} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, "
         f"pipeline, {cl}}} : {Tt['ps']}, {Tt['wc']}, {Tt['img']}")
    mark("conn")
    mark("end")
    e.op(f'softhier.dump_samples %vis {{seed = 250 : i32, n = {nsamples * cams} : i32, tag = "VIS"}} : {Tt["vis"]}')
    e.op(f'softhier.dump_all %img {{tag = "IMG"}} : {Tt["img"]}')
    body = "\n".join(e.lines)
    mlir = f"builtin.module {{\n  func.func @smolvla_e2e_vision() {{\n{body}\n    func.return\n  }}\n}}\n"
    emit_vision.last_info = {"image_bytes": sum(a.nbytes for a in pre.values()), "layer_stride": stride}
    return mlir, pre


emit_vision.last_info = {}


# ----------------------------------------------------------------------------- phases Pa / Pb / X: data for the existing emitters
class _Merged(dict):
    """dict view the existing emitters accept (`for k in data`, `data[k]`, `in`)."""


def vlm_data(e2e, x_img: np.ndarray | None = None, x_init: np.ndarray | None = None, layers_from: int = 0,
             layers: int = 8, vlm_npz=VLM_NPZ) -> dict:
    """npz-like dict for frontend.smolvla.emit_vlm: text weights of layers [layers_from, layers_from + layers) (fp16),
    the chain's host tables, the hand-over (x_img for the first prefix program, x_init for a later one)."""
    it = info(e2e)
    w = np.load(vlm_npz)
    d = _Merged()
    for k in ("p_gfin", "p_wc"):
        d[k] = w[k]
    for L in range(layers_from, layers_from + layers):
        for nm, _, _ in V.VLM_PARAMS:
            d[f"p_{nm}{L}"] = w[f"p_{nm}{L}"]
    for k in ("p_lang", "p_state_emb", "p_rope", "tok", "lang_mask", "img_tokens"):
        d[k] = e2e[k]
    d["vis_out"] = np.zeros((it["cams"], it["T"], D), np.float16)    # not read when x_img is given
    d["meta"] = np.array([it["n"], layers_from + layers, it["cams"], 0, 0], dtype=np.int64)
    if x_img is not None:
        d["x_img"] = x_img.astype(np.float16)
    if x_init is not None:
        d["x_init"] = x_init.astype(np.float16)
    return d


def expert_data(e2e, kv: dict[int, tuple[np.ndarray, np.ndarray]], expert_npz=EXPERT_NPZ) -> dict:
    """npz-like dict for frontend.smolvla_expert.emit_flow / np_flow: expert weights + time table from expert.npz, the
    chain's prefix tables (token classes, RoPE of the self layers at n_valid + i, noise) and the device's KV cache."""
    w = np.load(expert_npz)
    d = _Merged()
    for k in w.files:
        if k.startswith("p_") and not k.startswith(("p_kp", "p_vp")) and k not in ("p_tok", "p_rq_self", "p_rq_cross", "p_x0"):
            d[k] = w[k]
    d["p_tok"] = e2e["x_tok"]; d["p_rq_self"] = e2e["x_rq_self"]; d["p_rq_cross"] = e2e["x_rq_cross"]; d["p_x0"] = e2e["x_x0"]
    for L, (k, v) in kv.items():
        d[f"p_kp{L}"], d[f"p_vp{L}"] = k.astype(np.float16), v.astype(np.float16)
    d["ref_xt"] = e2e["ref_xt"]
    return d


def full_dump(samples: list[tuple[int, int, float]], rows: int, cols: int) -> np.ndarray:
    """dump_all lines (every element, exact fp16 bits) -> fp16 array; asserts completeness."""
    a = np.full((rows, cols), np.nan, np.float32)
    for r, c, v in samples:
        a[r, c] = v
    assert not np.isnan(a).any(), f"incomplete dump: {int(np.isnan(a).sum())} of {rows * cols} elements missing"
    return a.astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--ref", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--ckpt", default=str(MODELS / "model.safetensors"))
    pe = sub.add_parser("emit-vision")
    pe.add_argument("npz")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.ref, a.out, a.ckpt)
    else:
        import sys
        sys.stdout.write(emit_vision(np.load(a.npz))[0])


if __name__ == "__main__":
    main()
