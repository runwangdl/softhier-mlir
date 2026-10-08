"""SmolVLA's vision tower (the frozen 12-layer SigLIP of `lerobot/smolvla_base`) as a softhier program.

Two halves, so the compiler side needs only numpy:

  prepare  (system python: torch + transformers + safetensors)
      python3 -m softhier_mlir.frontend.smolvla prepare --ckpt /app/models/smolvla_base/model.safetensors \
              --seq 256 --out /app/models/smolvla_base/vision_s256.npz
      Loads `model.vlm_with_expert.vlm.model.vision_model.*` (197 bf16 tensors), converts them to the
      library layout, builds the im2col'd patch matrix of a fixed test image and the fp32 HF reference
      (SiglipVisionModel, eager attention) of the embeddings, every layer output and the final output.

  emit  (numpy only; used by tests/gvsoc/run.py smolvla)
      program = emit(npz, layers=12, attn=-1) -> (mlir text, {hbm_offset: fp16 array}) for
      softhier-translate and softhier_mlir.sim.preload.make_preload_elf.

Library layout: fp16 row-major; GEMM weights [in, out] (= HF Linear weight transposed); biases, LN
gamma/beta and the position embeddings as rows. The patch embedding (16x16 stride-16 conv) is the
GEMM  X[tokens, 3*16*16] . Wpe[768, 768] + bpe  with X the im2col'd pixels (channel-major
(c, kh, kw) per patch, matching conv weight.reshape(768, -1)). Token t of the 32x32 patch grid is
(t // 32, t % 32); a reduced `seq` = g*g keeps the top-left g x g patches *with their original
position embeddings*, so the reduced program is exactly the full model restricted to those tokens
(not a resized image).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from softhier_mlir.frontend.siglip import _Emitter

PREFIX = "model.vlm_with_expert.vlm.model.vision_model."
IMG, PATCH, D, FF, HEADS, LAYERS = 512, 16, 768, 3072, 12, 12
GRID = IMG // PATCH          # 32 x 32 patches = 1024 tokens
LN_EPS = 1e-6
HBM_DATA_START = 0x10000     # first 64 KB left to the SDK allocator metadata


# ----------------------------------------------------------------------------- weights
def load_vision_tower(ckpt: str | Path) -> dict[str, np.ndarray]:
    """{hf_name_without_prefix: fp32 ndarray} of the 197 vision-tower tensors (bf16 -> fp32 via torch)."""
    import torch
    from safetensors.torch import load_file
    sd = load_file(str(ckpt))
    vis = {k[len(PREFIX):]: v.to(torch.float32).numpy() for k, v in sd.items() if k.startswith(PREFIX)}
    assert len(vis) == 197, f"expected 197 vision tensors, got {len(vis)}"
    return vis


def to_library_layout(vis: dict[str, np.ndarray], token_ids: np.ndarray) -> dict[str, np.ndarray]:
    """HF vision-tower tensors -> {library name: fp16 array}. Names match frontend/siglip.py
    (wq0, bq0, g10, be10, ... per layer) plus wpe/bpe (patch embedding), pos (selected rows of the
    position table) and gpost/bepost (post layernorm)."""
    row = lambda v: v.reshape(1, -1)  # noqa: E731
    p: dict[str, np.ndarray] = {
        "wpe": vis["embeddings.patch_embedding.weight"].reshape(D, -1).T,       # [768 (c,kh,kw), 768 out]
        "bpe": row(vis["embeddings.patch_embedding.bias"]),
        "pos": vis["embeddings.position_embedding.weight"][token_ids],           # [seq, 768]
        "gpost": row(vis["post_layernorm.weight"]), "bepost": row(vis["post_layernorm.bias"]),
    }
    for L in range(LAYERS):
        g = f"encoder.layers.{L}."
        for nm, hf in (("wq", "self_attn.q_proj"), ("wk", "self_attn.k_proj"), ("wv", "self_attn.v_proj"),
                       ("wo", "self_attn.out_proj"), ("w1", "mlp.fc1"), ("w2", "mlp.fc2")):
            p[f"{nm}{L}"] = vis[g + hf + ".weight"].T                           # Linear [out,in] -> [in,out]
            p[("b" + nm[1:]) + f"{L}"] = row(vis[g + hf + ".bias"])             # bq/bk/bv/bo/b1/b2
        p[f"g1{L}"], p[f"be1{L}"] = row(vis[g + "layer_norm1.weight"]), row(vis[g + "layer_norm1.bias"])
        p[f"g2{L}"], p[f"be2{L}"] = row(vis[g + "layer_norm2.weight"]), row(vis[g + "layer_norm2.bias"])
    return {k: np.ascontiguousarray(v, dtype=np.float16) for k, v in p.items()}


def token_ids_for(seq: int) -> np.ndarray:
    """Tokens kept for a reduced run: the top-left g x g patches of the 32 x 32 grid (seq = g*g)."""
    g = int(math.isqrt(seq))
    assert g * g == seq and g <= GRID, f"seq must be a square <= {GRID * GRID}, got {seq}"
    py, px = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    return (py * GRID + px).reshape(-1)


def im2col(pixels: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """pixels [3, 512, 512] -> [len(token_ids), 3*16*16] patch rows in (c, kh, kw) order."""
    c, h, w = pixels.shape
    patches = pixels.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).transpose(1, 3, 0, 2, 4).reshape(GRID * GRID, -1)
    return np.ascontiguousarray(patches[token_ids])


def test_image(seed: int = 0) -> np.ndarray:
    """A fixed synthetic input in SigLIP's normalised pixel range [-1, 1] (fp16-representable)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (3, IMG, IMG)).astype(np.float16).astype(np.float32)


# ----------------------------------------------------------------------------- references
def hf_reference(vis: dict[str, np.ndarray], pixels: np.ndarray, token_ids: np.ndarray, layers: int = LAYERS) -> dict[str, np.ndarray]:
    """fp32 HF SiglipVisionModel: EMB (embeddings), L<n> (output of encoder layer n, 1-based), OUT
    (post layernorm of layer `layers`). For a reduced token set the embeddings are built by hand
    (patch conv + the kept tokens' own position rows) and fed to the HF encoder."""
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(hidden_size=D, intermediate_size=FF, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                             image_size=IMG, patch_size=PATCH, num_channels=3, hidden_act="gelu_pytorch_tanh",
                             vision_use_head=False, attn_implementation="eager", layer_norm_eps=LN_EPS)
    model = SiglipVisionModel(cfg)
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(v) for k, v in vis.items()}, strict=False)
    missing = [m for m in missing if "head" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    vm = getattr(model, "vision_model", model)   # transformers >= 5 flattens the nesting
    with torch.no_grad():
        px = torch.from_numpy(pixels)[None]
        emb = vm.embeddings.patch_embedding(px).flatten(2).transpose(1, 2)[:, token_ids]      # [1, seq, 768]
        emb = emb + vm.embeddings.position_embedding.weight[torch.as_tensor(token_ids)][None]
        out = {"EMB": emb[0].numpy()}
        h = emb
        for n, layer in enumerate(vm.encoder.layers[:layers], 1):
            r = layer(h, None)
            h = r[0] if isinstance(r, tuple) else r
            out[f"L{n}"] = h[0].numpy()
        out["OUT"] = vm.post_layernorm(h)[0].numpy()
        if layers == LAYERS and len(token_ids) == GRID * GRID:   # cross-check the manual path against the stock forward
            full = model(pixel_values=px).last_hidden_state[0].numpy()
            assert np.abs(full - out["OUT"]).max() < 1e-3, "manual embedding path differs from SiglipVisionModel.forward"
    return out


def numpy_reference(p: dict[str, np.ndarray], xp: np.ndarray, layers: int) -> dict[str, np.ndarray]:
    """What the device computes: fp32 math with every stored intermediate rounded to fp16 (the
    error floor of the fp16 program; same tags as hf_reference)."""
    f = lambda k: p[k].astype(np.float32)  # noqa: E731
    r16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    seq, dh = xp.shape[0], D // HEADS

    def ln(a, g, b):
        m = a.mean(1, keepdims=True); v = a.var(1, keepdims=True)
        return (a - m) / np.sqrt(v + LN_EPS) * g + b

    x = r16(r16(r16(xp.astype(np.float32) @ f("wpe")) + f("bpe")) + f("pos"))
    out = {"EMB": x}
    for L in range(layers):
        ln1 = r16(ln(x, f(f"g1{L}"), f(f"be1{L}")))
        q = r16(r16(ln1 @ f(f"wq{L}")) + f(f"bq{L}")); k = r16(r16(ln1 @ f(f"wk{L}")) + f(f"bk{L}")); v = r16(r16(ln1 @ f(f"wv{L}")) + f(f"bv{L}"))
        o = np.zeros((seq, D), np.float32)
        for hd in range(HEADS):
            sl = slice(hd * dh, (hd + 1) * dh)
            s = r16(q[:, sl] @ k[:, sl].T) * (1 / math.sqrt(dh))
            pr = np.exp(s - s.max(1, keepdims=True)); pr = r16(pr / pr.sum(1, keepdims=True))
            o[:, sl] = r16(pr @ v[:, sl])
            if L == 0 and hd == 0:
                out["P0"] = pr
        h = r16(x + r16(r16(o @ f(f"wo{L}")) + f(f"bo{L}")))
        ln2 = r16(ln(h, f(f"g2{L}"), f(f"be2{L}")))
        f1 = r16(r16(ln2 @ f(f"w1{L}")) + f(f"b1{L}"))
        g = r16(0.5 * f1 * (1 + np.tanh(0.7978845608 * (f1 + 0.044715 * f1 ** 3))))
        x = r16(h + r16(r16(g @ f(f"w2{L}")) + f(f"b2{L}")))
        out[f"L{L + 1}"] = x
        if L == 0:   # layer-1 intermediates (same tags as the program's optional dumps)
            out.update({"LN1": ln1, "Q": q, "K": k, "O": o, "H": h, "G": g})
    out["OUT"] = r16(ln(x, f("gpost"), f("bepost")))
    return out


def prepare(ckpt: str | Path, seq: int, out: str | Path, layers: int = LAYERS, image_seed: int = 0) -> Path:
    """Weights in library layout + patch matrix + references -> one .npz (numpy only downstream)."""
    vis = load_vision_tower(ckpt)
    ids = token_ids_for(seq)
    pixels = test_image(image_seed)
    p = to_library_layout(vis, ids)
    xp = im2col(pixels, ids).astype(np.float16)
    ref = hf_reference(vis, pixels, ids, layers)
    npref = numpy_reference(p, xp, layers)
    arrays = {f"p_{k}": v for k, v in p.items()}
    arrays["xp"] = xp
    arrays.update({f"ref_{k}": v for k, v in ref.items()})
    arrays.update({f"np_{k}": v for k, v in npref.items()})
    arrays["meta"] = np.array([seq, layers, image_seed], dtype=np.int64)
    out = Path(out)
    np.savez(out, **arrays)
    for tag in ("EMB", "L1", "OUT"):
        if tag in ref:
            print(f"[prepare] fp16-program floor vs HF fp32  {tag}: max abs {np.abs(npref[tag] - ref[tag]).max():.4f}  "
                  f"(|ref| max {np.abs(ref[tag]).max():.2f})")
    print(f"[prepare] {out}  seq={seq} layers={layers}  params {sum(v.nbytes for v in p.values()) / 2 ** 20:.1f} MiB fp16")
    return out


# ----------------------------------------------------------------------------- program
def emit(npz: str | Path | dict, layers: int | None = None, attn: int = -1, cluster: int = -1,
         dumps: tuple[str, ...] = ("EMB", "L1", "OUT"), nsamples: int = 64, marks: bool = True) -> tuple[str, dict[int, np.ndarray]]:
    """-> (mlir, {hbm_offset: fp16 array to preload}).

    cluster: executing cluster of GEMMs / row ops (-1 = SH_ALL: output tiles / row blocks dealt
             round-robin over all clusters, global barrier after each op).
    attn:    cluster of the per-head QK^T / softmax / PV ops (-1 = SH_ALL per op, heads sequential;
             0 = everything on cluster 0). Heads are never run concurrently on different clusters.
    dumps:   EMB (embeddings), L<n> (output of layer n), OUT (post layernorm) and, for layer 1,
             LN1 / Q / K / P0 / O / H / G (same meaning as frontend/siglip.py)."""
    data = np.load(npz) if isinstance(npz, (str, Path)) else npz
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    xp = data["xp"]
    seq = xp.shape[0]
    layers = int(data["meta"][1]) if layers is None else layers
    dh = D // HEADS
    e = _Emitter()
    e.next_off = HBM_DATA_START
    T: dict[str, str] = {}
    pre: dict[int, np.ndarray] = {}
    cl = f"cluster = {cluster} : i32"
    hc = f"cluster = {attn} : i32"

    def B(name, rows, cols, arr=None):
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            pre[e.next_off] = arr
        T[name] = e.buf(name, rows, cols)
        return name

    def mark(tag):
        if marks:
            e.op(f'softhier.mark {{tag = "{tag}"}}')

    def dump(tag, name, seed, view=None):
        if tag in dumps:
            e.op(f'softhier.dump_samples %{name} {{seed = {seed} : i32, n = {nsamples} : i32, tag = "{tag}"}} : {view or T[name]}')

    # inputs + activations
    B("xp", seq, D, xp); B("pos", seq, D, P["pos"])
    for nm, r, c in [("x", seq, D), ("ln1", seq, D), ("q", seq, D), ("k", seq, D), ("v", seq, D), ("kT", D, seq),
                     ("sc", HEADS * seq, seq), ("o", seq, D), ("ao", seq, D), ("h", seq, D), ("ln2", seq, D),
                     ("f1", seq, FF), ("g", seq, FF), ("f2", seq, D), ("out", seq, D), ("fin", seq, D)]:
        B(nm, r, c)
    # parameters (preloaded)
    B("wpe", D, D, P["wpe"]); B("bpe", 1, D, P["bpe"]); B("gpost", 1, D, P["gpost"]); B("bepost", 1, D, P["bepost"])
    for L in range(layers):
        for nm, r, c in [("wq", D, D), ("wk", D, D), ("wv", D, D), ("wo", D, D), ("w1", D, FF), ("w2", FF, D),
                         ("bq", 1, D), ("bk", 1, D), ("bv", 1, D), ("bo", 1, D), ("b1", 1, FF), ("b2", 1, D),
                         ("g1", 1, D), ("be1", 1, D), ("g2", 1, D), ("be2", 1, D)]:
            B(f"{nm}{L}", r, c, P[f"{nm}{L}"])

    gem = lambda tm, tn, tk: f"tile_m = {tm} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, pipeline"  # noqa: E731
    big, qk, pv = gem(256, 256, 256), gem(256, 256, dh), gem(256, dh, 256)
    seeds = {"EMB": 200, "LN1": 206, "Q": 201, "K": 207, "P0": 209, "O": 202, "H": 203, "G": 204, "OUT": 205}

    # patch embedding + position embedding
    mark("start")
    e.op(f"softhier.gemm %xp, %wpe into %x {{fmt = \"fp16\", {big}, {cl}}} : {T['xp']}, {T['wpe']}, {T['x']}")
    e.op(f"softhier.add_bias %x, %bpe -> %x {{{cl}}} : {T['x']}, {T['bpe']} -> {T['x']}")
    e.op(f"softhier.add %x, %pos -> %x {{{cl}}} : {T['x']}, {T['pos']} -> {T['x']}")
    mark("emb")
    dump("EMB", "x", seeds["EMB"])

    xin = "x"
    for L in range(layers):
        W = lambda nm: f"{nm}{L}"  # noqa: E731
        e.op(f"softhier.layernorm %{xin}, %{W('g1')}, %{W('be1')} -> %ln1 {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T[xin]}, {T[W('g1')]}, {T[W('be1')]} -> {T['ln1']}")
        for dst, w, bias in (("q", "wq", "bq"), ("k", "wk", "bk"), ("v", "wv", "bv")):
            e.op(f"softhier.gemm %ln1, %{W(w)} into %{dst} {{fmt = \"fp16\", {big}, {cl}}} : {T['ln1']}, {T[W(w)]}, {T[dst]}")
            e.op(f"softhier.add_bias %{dst}, %{W(bias)} -> %{dst} {{{cl}}} : {T[dst]}, {T[W(bias)]} -> {T[dst]}")
        e.op(f"softhier.transpose %k -> %kT {{{cl}}} : {T['k']} -> {T['kT']}")
        if L == 0:
            dump("LN1", "ln1", seeds["LN1"]); dump("Q", "q", seeds["Q"]); dump("K", "k", seeds["K"])
        for hd in range(HEADS):
            qh = e.view(f"q{L}_{hd}", "q", T["q"], seq, dh, D, hd * dh)
            kh = e.view(f"kT{L}_{hd}", "kT", T["kT"], dh, seq, seq, hd * dh * seq)
            sh = e.view(f"s{L}_{hd}", "sc", T["sc"], seq, seq, seq, hd * seq * seq)
            vh = e.view(f"v{L}_{hd}", "v", T["v"], seq, dh, D, hd * dh)
            oh = e.view(f"o{L}_{hd}", "o", T["o"], seq, dh, D, hd * dh)
            e.op(f"softhier.gemm %q{L}_{hd}, %kT{L}_{hd} into %s{L}_{hd} {{fmt = \"fp16\", {qk}, {hc}}} : {qh}, {kh}, {sh}")
            e.op(f"softhier.softmax %s{L}_{hd} -> %s{L}_{hd} {{scale = {1 / math.sqrt(dh)!r} : f32, {hc}}} : {sh} -> {sh}")
            e.op(f"softhier.gemm %s{L}_{hd}, %v{L}_{hd} into %o{L}_{hd} {{fmt = \"fp16\", {pv}, {hc}}} : {sh}, {vh}, {oh}")
            if L == 0 and hd == 0:
                dump("P0", f"s{L}_{hd}", seeds["P0"], sh)
        e.op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
        mark(f"L{L + 1}attn")
        if L == 0:
            dump("O", "o", seeds["O"])
        e.op(f"softhier.gemm %o, %{W('wo')} into %ao {{fmt = \"fp16\", {big}, {cl}}} : {T['o']}, {T[W('wo')]}, {T['ao']}")
        e.op(f"softhier.add_bias %ao, %{W('bo')} -> %ao {{{cl}}} : {T['ao']}, {T[W('bo')]} -> {T['ao']}")
        e.op(f"softhier.add %{xin}, %ao -> %h {{{cl}}} : {T[xin]}, {T['ao']} -> {T['h']}")
        e.op(f"softhier.layernorm %h, %{W('g2')}, %{W('be2')} -> %ln2 {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T['h']}, {T[W('g2')]}, {T[W('be2')]} -> {T['ln2']}")
        e.op(f"softhier.gemm %ln2, %{W('w1')} into %f1 {{fmt = \"fp16\", {big}, {cl}}} : {T['ln2']}, {T[W('w1')]}, {T['f1']}")
        e.op(f"softhier.add_bias %f1, %{W('b1')} -> %f1 {{{cl}}} : {T['f1']}, {T[W('b1')]} -> {T['f1']}")
        e.op(f"softhier.gelu %f1 -> %g {{{cl}}} : {T['f1']} -> {T['g']}")
        e.op(f"softhier.gemm %g, %{W('w2')} into %f2 {{fmt = \"fp16\", {big}, {cl}}} : {T['g']}, {T[W('w2')]}, {T['f2']}")
        e.op(f"softhier.add_bias %f2, %{W('b2')} -> %f2 {{{cl}}} : {T['f2']}, {T[W('b2')]} -> {T['f2']}")
        e.op(f"softhier.add %h, %f2 -> %out {{{cl}}} : {T['h']}, {T['f2']} -> {T['out']}")
        mark(f"L{L + 1}")
        if L == 0:
            dump("H", "h", seeds["H"]); dump("G", "g", seeds["G"])
        dump(f"L{L + 1}", "out", 220 + L)
        xin = "out"
    e.op(f"softhier.layernorm %{xin}, %gpost, %bepost -> %fin {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T[xin]}, {T['gpost']}, {T['bepost']} -> {T['fin']}")
    mark("end")
    dump("OUT", "fin", seeds["OUT"])
    body = "\n".join(e.lines)
    mlir = f"builtin.module {{\n  func.func @smolvla_vision() {{\n{body}\n    func.return\n  }}\n}}\n"
    return mlir, pre


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="checkpoint -> npz (weights in library layout, patch matrix, HF reference)")
    pp.add_argument("--ckpt", default="/app/models/smolvla_base/model.safetensors")
    pp.add_argument("--seq", type=int, default=256, help="tokens kept (a square: 256 = top-left 16x16 patches, 1024 = all)")
    pp.add_argument("--layers", type=int, default=LAYERS)
    pp.add_argument("--image-seed", type=int, default=0)
    pp.add_argument("--out", required=True)
    pe = sub.add_parser("emit", help="npz -> MLIR on stdout")
    pe.add_argument("npz")
    pe.add_argument("--layers", type=int)
    pe.add_argument("--attn", type=int, default=-1)
    pe.add_argument("--cluster", type=int, default=-1)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.ckpt, a.seq, a.out, a.layers, a.image_seed)
    else:
        import sys
        sys.stdout.write(emit(a.npz, a.layers, a.attn, a.cluster)[0])


if __name__ == "__main__":
    main()
