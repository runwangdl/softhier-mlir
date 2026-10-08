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
from softhier_mlir.sim.preload import sentinel_array

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
PARAMS = [("wq", D, D), ("wk", D, D), ("wv", D, D), ("wo", D, D), ("w1", D, FF), ("w2", FF, D),
          ("bq", 1, D), ("bk", 1, D), ("bv", 1, D), ("bo", 1, D), ("b1", 1, FF), ("b2", 1, D),
          ("g1", 1, D), ("be1", 1, D), ("g2", 1, D), ("be2", 1, D)]


def emit(npz: str | Path | dict, layers: int | None = None, attn: int = -1, cluster: int = -1,
         dumps: tuple[str, ...] = ("EMB", "L1", "OUT"), nsamples: int = 64, marks: bool = True,
         unroll: bool = False) -> tuple[str, dict[int, np.ndarray]]:
    """-> (mlir, {hbm_offset: fp16 array to preload}).

    cluster: executing cluster of GEMMs / row ops (-1 = SH_ALL: output tiles / row blocks dealt
             round-robin over all clusters, global barrier after each op).
    attn:    cluster of the per-head QK^T / softmax / PV ops (-1 = SH_ALL per op, heads sequential;
             0 = everything on cluster 0). Heads are never run concurrently on different clusters.
    dumps:   EMB (embeddings), L<n> (output of layer n), OUT (post layernorm), p_<name> (a preloaded
             parameter read back: checks the preload path itself) and, with unroll=True only, the
             layer-1 intermediates LN1 / Q / K / P0 / O / H / G (same meaning as frontend/siglip.py).
    unroll:  False (default): one scf.for over the layers (per-layer parameters are a constant HBM
             stride apart) and one over the heads, so the code size does not grow with the depth
             (the cluster instruction memory is 64 KB; 12 unrolled layers are ~140 KB). True: every
             op spelled out, for 1-2 layers of debugging with the intermediate dumps.
    Timing marks: start / emb / attn<n> / layer<n> / end (+ embdump / ldump<n> after a dump, so the
    sample printing can be excluded)."""
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

    def alloc(rows, cols):
        """an HBM offset without a declaration (same rounding as _Emitter.buf)"""
        off = e.next_off
        e.next_off += (rows * cols * 2 + 4095) & ~4095
        return off

    def B(name, rows, cols, arr=None):
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            pre[e.next_off] = arr
        T[name] = e.buf(name, rows, cols)
        return name

    def mark(tag, idx=None):
        if marks:
            e.op(f'softhier.mark {idx + " " if idx else ""}{{tag = "{tag}"}}')

    def dump(tag, name, seed, view=None, idx=None, out_tag=None):
        if tag in dumps:
            e.op(f'softhier.dump_samples %{name}{", " + idx if idx else ""} {{seed = {seed} : i32, n = {nsamples} : i32, '
                 f'tag = "{out_tag or tag}"}} : {view or T[name]}')
            return True
        return False

    def dump_mark(tag, dumped, idx=None):
        """dumps print from cluster 0 only; a barrier + mark keeps that time out of the next segment"""
        if dumped and marks:
            e.op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
            mark(tag, idx)

    # inputs + activations (x is the residual stream, updated in place by every layer)
    B("xp", seq, D, xp); B("pos", seq, D, P["pos"])
    for nm, r, c in [("x", seq, D), ("ln1", seq, D), ("q", seq, D), ("k", seq, D), ("v", seq, D), ("kT", D, seq),
                     ("sc", HEADS * seq, seq), ("o", seq, D), ("ao", seq, D), ("h", seq, D), ("ln2", seq, D),
                     ("f1", seq, FF), ("g", seq, FF), ("f2", seq, D), ("fin", seq, D)]:
        B(nm, r, c)
    # parameters (preloaded): per-layer families at a constant stride
    B("wpe", D, D, P["wpe"]); B("bpe", 1, D, P["bpe"]); B("gpost", 1, D, P["gpost"]); B("bepost", 1, D, P["bepost"])
    layer0_off, stride = {}, 0
    for L in range(layers):
        begin = e.next_off
        for nm, r, c in PARAMS:
            if unroll:
                B(f"{nm}{L}", r, c, P[f"{nm}{L}"])
            else:
                off = alloc(r, c)
                assert P[f"{nm}{L}"].shape == (r, c)
                pre[off] = P[f"{nm}{L}"]
                if L == 0:
                    layer0_off[nm] = off
                    T[nm] = f'memref<{r}x{c}xf16, "{e.space}">'
        if L == 0:
            stride = e.next_off - begin
    sent = sentinel_array()
    B("sentinel", *sent.shape, sent)    # highest offset = last segment of the image: lands last
    mark("preload")
    e.op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
    for tag in dumps:                   # p_<name>: read a preloaded parameter back (preload-path check)
        nm = tag[2:]
        if tag.startswith("p_") and (nm in T if not nm[-1].isdigit() else nm in P):
            if nm in T and nm[-1].isdigit() is False:
                dump(tag, nm, 100 + sum(map(ord, tag)) % 50)
            elif unroll:
                dump(tag, nm, 100 + sum(map(ord, tag)) % 50)

    gem = lambda tm, tn, tk: f"tile_m = {tm} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, pipeline"  # noqa: E731
    big, qk, pv = gem(256, 256, 256), gem(256, 256, dh), gem(256, dh, 256)
    seeds = {"EMB": 200, "LN1": 206, "Q": 201, "K": 207, "P0": 209, "O": 202, "H": 203, "G": 204, "OUT": 205}
    hv = lambda rows, cols, ld, eoff: f'memref<{rows}x{cols}xf16, strided<[{ld}, 1], offset: {eoff}>, "{e.space}">'  # noqa: E731

    # patch embedding + position embedding
    mark("start")
    e.op(f"softhier.gemm %xp, %wpe into %x {{fmt = \"fp16\", {big}, {cl}}} : {T['xp']}, {T['wpe']}, {T['x']}")
    e.op(f"softhier.add_bias %x, %bpe -> %x {{{cl}}} : {T['x']}, {T['bpe']} -> {T['x']}")
    e.op(f"softhier.add %x, %pos -> %x {{{cl}}} : {T['x']}, {T['pos']} -> {T['x']}")
    mark("emb")
    dump_mark("embdump", dump("EMB", "x", seeds["EMB"]))

    def head(L, hd):
        """attention of one head; L / hd are ints (unrolled) or (SSA index name, None) in a loop"""
        if hd is None:    # loop mode: views at index %h
            sfx, idx = "", "%hd"
            def view(name, src, rows, cols, ld, stride):
                t = hv(rows, cols, ld, 0)
                e.op(f"%{name} = softhier.view %{src}, {idx} {{stride = {stride} : i32}} : {T[src]} -> {t}")
                return t
            qh = view("qh", "q", seq, dh, D, dh); kh = view("kTh", "kT", dh, seq, seq, dh * seq)
            sh = view("sh", "sc", seq, seq, seq, seq * seq); vh = view("vh", "v", seq, dh, D, dh); oh = view("oh", "o", seq, dh, D, dh)
            names = ("qh", "kTh", "sh", "vh", "oh")
        else:
            sfx = f"{L}_{hd}"
            qh = e.view(f"q{sfx}", "q", T["q"], seq, dh, D, hd * dh)
            kh = e.view(f"kT{sfx}", "kT", T["kT"], dh, seq, seq, hd * dh * seq)
            sh = e.view(f"s{sfx}", "sc", T["sc"], seq, seq, seq, hd * seq * seq)
            vh = e.view(f"v{sfx}", "v", T["v"], seq, dh, D, hd * dh)
            oh = e.view(f"o{sfx}", "o", T["o"], seq, dh, D, hd * dh)
            names = (f"q{sfx}", f"kT{sfx}", f"s{sfx}", f"v{sfx}", f"o{sfx}")
        nq, nk, ns, nv, no = names
        e.op(f"softhier.gemm %{nq}, %{nk} into %{ns} {{fmt = \"fp16\", {qk}, {hc}}} : {qh}, {kh}, {sh}")
        e.op(f"softhier.softmax %{ns} -> %{ns} {{scale = {1 / math.sqrt(dh)!r} : f32, {hc}}} : {sh} -> {sh}")
        e.op(f"softhier.gemm %{ns}, %{nv} into %{no} {{fmt = \"fp16\", {pv}, {hc}}} : {sh}, {vh}, {oh}")
        if L == 0 and hd == 0:
            dump("P0", ns, seeds["P0"], sh)

    def layer(L):
        """one encoder layer; L: int (unrolled: parameters %wq3 ...) or None (loop body: %wq at index %L)"""
        W = (lambda nm: f"{nm}{L}") if L is not None else (lambda nm: nm)  # noqa: E731
        n1 = f"{L + 1}" if L is not None else None          # 1-based layer number for tags
        idx = None if L is not None else "%L1"
        e.op(f"softhier.layernorm %x, %{W('g1')}, %{W('be1')} -> %ln1 {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T['x']}, {T[W('g1')]}, {T[W('be1')]} -> {T['ln1']}")
        for dst, w, bias in (("q", "wq", "bq"), ("k", "wk", "bk"), ("v", "wv", "bv")):
            e.op(f"softhier.gemm %ln1, %{W(w)} into %{dst} {{fmt = \"fp16\", {big}, {cl}}} : {T['ln1']}, {T[W(w)]}, {T[dst]}")
            e.op(f"softhier.add_bias %{dst}, %{W(bias)} -> %{dst} {{{cl}}} : {T[dst]}, {T[W(bias)]} -> {T[dst]}")
        e.op(f"softhier.transpose %k -> %kT {{{cl}}} : {T['k']} -> {T['kT']}")
        if L == 0:
            dump_mark("qkdump", dump("LN1", "ln1", seeds["LN1"]) | dump("Q", "q", seeds["Q"]) | dump("K", "k", seeds["K"]))
        if L is None:
            e.op("scf.for %hd = %c0 to %cH step %c1 {")
            head(L, None)
            e.op("}")
        else:
            for hd in range(HEADS):
                head(L, hd)
        e.op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
        mark(f"attn{n1}" if n1 else "attn", idx)
        if L == 0:
            dump_mark("odump", dump("O", "o", seeds["O"]))
        e.op(f"softhier.gemm %o, %{W('wo')} into %ao {{fmt = \"fp16\", {big}, {cl}}} : {T['o']}, {T[W('wo')]}, {T['ao']}")
        e.op(f"softhier.add_bias %ao, %{W('bo')} -> %ao {{{cl}}} : {T['ao']}, {T[W('bo')]} -> {T['ao']}")
        e.op(f"softhier.add %x, %ao -> %h {{{cl}}} : {T['x']}, {T['ao']} -> {T['h']}")
        e.op(f"softhier.layernorm %h, %{W('g2')}, %{W('be2')} -> %ln2 {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T['h']}, {T[W('g2')]}, {T[W('be2')]} -> {T['ln2']}")
        e.op(f"softhier.gemm %ln2, %{W('w1')} into %f1 {{fmt = \"fp16\", {big}, {cl}}} : {T['ln2']}, {T[W('w1')]}, {T['f1']}")
        e.op(f"softhier.add_bias %f1, %{W('b1')} -> %f1 {{{cl}}} : {T['f1']}, {T[W('b1')]} -> {T['f1']}")
        e.op(f"softhier.gelu %f1 -> %g {{{cl}}} : {T['f1']} -> {T['g']}")
        e.op(f"softhier.gemm %g, %{W('w2')} into %f2 {{fmt = \"fp16\", {big}, {cl}}} : {T['g']}, {T[W('w2')]}, {T['f2']}")
        e.op(f"softhier.add_bias %f2, %{W('b2')} -> %f2 {{{cl}}} : {T['f2']}, {T[W('b2')]} -> {T['f2']}")
        e.op(f"softhier.add %h, %f2 -> %x {{{cl}}} : {T['h']}, {T['f2']} -> {T['x']}")
        mark(f"layer{n1}" if n1 else "layer", idx)
        if L is not None:
            d = dump(f"L{n1}", "x", 220 + L)
            if L == 0:
                d |= dump("H", "h", seeds["H"]) | dump("G", "g", seeds["G"])
        else:   # loop body: every layer's output when any L<n> is requested
            d = any(t.startswith("L") and t[1:].isdigit() for t in dumps)
            if d:
                e.op(f'softhier.dump_samples %x, {idx} {{seed = 220 : i32, n = {nsamples} : i32, tag = "L"}} : {T["x"]}')
        dump_mark(f"ldump{n1}" if n1 else "ldump", d, idx)

    if unroll:
        for L in range(layers):
            layer(L)
    else:
        e.op("%c0 = arith.constant 0 : index")
        e.op("%c1 = arith.constant 1 : index")
        e.op(f"%cH = arith.constant {HEADS} : index")
        e.op(f"%cL = arith.constant {layers} : index")
        e.op("scf.for %L = %c0 to %cL step %c1 {")
        e.op("%L1 = arith.addi %L, %c1 : index")
        for nm, r, c in PARAMS:
            e.op(f"%{nm} = softhier.hbm_buffer %L {{offset = {layer0_off[nm]} : i32, stride = {stride} : i32}} : {T[nm]}")
        layer(None)
        e.op("}")
    e.op(f"softhier.layernorm %x, %gpost, %bepost -> %fin {{eps = {LN_EPS:.1e} : f32, {cl}}} : {T['x']}, {T['gpost']}, {T['bepost']} -> {T['fin']}")
    mark("end")
    dump("OUT", "fin", seeds["OUT"])
    body = "\n".join(e.lines)
    mlir = f"builtin.module {{\n  func.func @smolvla_vision() {{\n{body}\n    func.return\n  }}\n}}\n"
    return mlir, pre


# =============================================================================== VLM text prefix
# SmolVLA's VLM = SmolVLM2-500M-Video-Instruct's Llama text tower cut to 16 layers (lerobot
# SmolVLMWithExpertModel, num_vlm_layers=16): hidden 960, 15 query heads / 5 key-value heads of 64, SiLU-gated
# MLP 2560, RMSNorm eps 1e-5, no biases. The prefix the expert's cross-attention consumes is
# [connector(vision tower(image)) x cameras, 48 language tokens, 1 state token] (241 tokens for 3 cameras):
#   image tokens  = pixel_shuffle(scale 4) of the 1024 SigLIP tokens -> 64 x 12288, projected to 960
#                   (modality_projection, no bias), then scaled by sqrt(960)
#   language      = embed_tokens[ids] * sqrt(960), padded to 48 with the pad token (lang_mask 0 on padding)
#   state         = state_proj(pad(state, 32)) (no scaling)
# RoPE is lerobot's own apply_rope with max_wavelength 10_000 (NOT the checkpoint's rope_theta 100000),
# rotate-half convention, positions = cumsum(pad_mask) - 1 (padding does not advance the position).
# Attention mask (make_att_2d_masks): att_masks = 0 for image + language, 1 for the state token, so image and
# language tokens attend all valid image + language tokens (bidirectional, padding excluded) and NOT the
# state; the state token attends everything valid. KV written by the prefix = k / v after RoPE (= lerobot's
# past_key_values[layer]["key_states"/"value_states"] with heads flattened: [S, 5 * 64]).
TXT = "model.vlm_with_expert.vlm.model.text_model."
CONNECTOR_W = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
STATE_W, STATE_B = "model.state_proj.weight", "model.state_proj.bias"
TD, TFF, TH, TKV, TDH, TLAYERS = 960, 2560, 15, 5, 64, 16
TKVD = TKV * TDH                 # 320
RMS_EPS = 1e-5
ROPE_BASE = 10000.0              # lerobot apply_rope(max_wavelength=10_000)
PIX_SCALE = 4
IMG_TOKENS = (GRID // PIX_SCALE) ** 2    # 64 per camera
LANG_LEN, STATE_DIM = 48, 32
PAD_TOKEN_ID = 2                 # SmolVLM2 tokenizer pad token <|im_end|>
DEFAULT_TASK_IDS = [18188, 614, 260, 20636, 198]   # "pick up the cube\n" (SmolVLM2 tokenizer; lerobot appends the newline)
TOK_PAD = 0xFFFF                 # SH_LLM_PAD
HBM_SOUTH = 0x30000000           # offset of the south HBM edge (west 0-256 MiB, north / east unpopulated, south at 768 MiB)
HBM_WEST_END = 0x10000000


class TextTower:
    """Lazy fp32 access to the text tower / connector / state projection of the checkpoint (the host has ~1 GB
    free: the 16 fp32 layers (630 MB) never live in memory at once; each layer is read when needed)."""

    LAYER = {"g1": "input_layernorm.weight", "g2": "post_attention_layernorm.weight", "wq": "self_attn.q_proj.weight",
             "wk": "self_attn.k_proj.weight", "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
             "wg": "mlp.gate_proj.weight", "wu": "mlp.up_proj.weight", "wd": "mlp.down_proj.weight"}

    def __init__(self, ckpt: str | Path) -> None:
        from safetensors import safe_open
        self.f = safe_open(str(ckpt), "pt")
        n_layers = len({k.split(".")[len(TXT.split(".")) - 1 + 1] for k in self.f.keys() if k.startswith(TXT + "layers.")})
        assert n_layers == TLAYERS, n_layers

    def t(self, name: str) -> np.ndarray:
        import torch
        return self.f.get_tensor(name).to(torch.float32).numpy()

    def layer(self, L: int) -> dict[str, np.ndarray]:
        """HF orientation (Linear weights [out, in])."""
        return {k: self.t(f"{TXT}layers.{L}.{hf}") for k, hf in self.LAYER.items()}

    def embed_rows(self, ids) -> np.ndarray:
        import torch
        sl = self.f.get_slice(TXT + "embed_tokens.weight")
        return torch.cat([sl[int(i):int(i) + 1] for i in ids], 0).to(torch.float32).numpy()

    @property
    def connector(self) -> np.ndarray: return self.t(CONNECTOR_W)          # [960, 12288]
    @property
    def norm(self) -> np.ndarray: return self.t(TXT + "norm.weight")
    @property
    def state_w(self) -> np.ndarray: return self.t(STATE_W)
    @property
    def state_b(self) -> np.ndarray: return self.t(STATE_B)
    @property
    def vocab(self) -> int: return self.f.get_slice(TXT + "embed_tokens.weight").get_shape()[0]


def pixel_shuffle(x: np.ndarray, scale: int = PIX_SCALE) -> np.ndarray:
    """== SmolVLMConnector.pixel_shuffle on one image: [seq, D] -> [seq / scale^2, D scale^2]."""
    seq, d = x.shape
    h = w = int(math.isqrt(seq))
    y = x.reshape(h, w // scale, d * scale).transpose(1, 0, 2).reshape(w // scale, h // scale, d * scale * scale).transpose(1, 0, 2)
    return np.ascontiguousarray(y.reshape(seq // (scale * scale), d * scale * scale))


def pixel_shuffle_gather(x: np.ndarray, scale: int = PIX_SCALE) -> np.ndarray:
    """What sh_pixel_shuffle moves: output token (gr, gb) = the scale x scale patch block, row-major (checked == pixel_shuffle)."""
    seq, d = x.shape
    g = int(math.isqrt(seq)) // scale
    blk = x.reshape(g, scale, g, scale, d).transpose(0, 2, 1, 3, 4)      # (gr, gb, i, j, d)
    return np.ascontiguousarray(blk.reshape(g * g, scale * scale * d))


def rope_table(positions: np.ndarray, dh: int = TDH, base: float = ROPE_BASE) -> np.ndarray:
    """[len(positions), dh] fp32: cos[dh/2] | sin[dh/2] of position / base^(2i/dh) (lerobot apply_rope)."""
    inv = base ** (-(2.0 / dh) * np.arange(dh // 2, dtype=np.float64))
    rad = positions.astype(np.float64)[:, None] * inv[None, :]
    return np.concatenate([np.cos(rad), np.sin(rad)], 1).astype(np.float32)


def apply_rope(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """x [S, H, dh], table [S, dh] -> rotate-half RoPE."""
    half = x.shape[-1] // 2
    c, s = table[:, None, :half], table[:, None, half:]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def prefix_layout(n_cams: int, lang_mask: np.ndarray) -> dict:
    """Token bookkeeping of the prefix: counts, pad mask, 2-D attention mask, token classes (device mask), position ids."""
    n_img = n_cams * IMG_TOKENS
    n = n_img + LANG_LEN + 1
    pad = np.concatenate([np.ones(n_img, bool), lang_mask.astype(bool), np.ones(1, bool)])
    att = np.zeros(n, np.int64); att[-1] = 1
    cum = np.cumsum(att)
    att2d = (cum[None, :] <= cum[:, None]) & pad[None, :] & pad[:, None]
    tok = np.where(pad, cum, TOK_PAD).astype(np.uint16)
    pos = np.cumsum(pad) - 1
    return {"n_img": n_img, "n": n, "pad": pad, "att2d": att2d, "tok": tok, "pos": pos}


def vision_outputs(ckpt: str | Path, seeds: list[int]) -> np.ndarray:
    """fp32 SigLIP outputs (post-LN, 1024 tokens) of test_image(seed) per camera through the HF SiglipVisionModel; the
    first camera of seed 0 is taken from /app/models/smolvla_base/vision_s1024.npz when present (same image, same
    model). The vision model is freed afterwards (memory)."""
    import gc
    outs, todo = {}, []
    cache = Path("/app/models/smolvla_base/vision_s1024.npz")
    per_seed = lambda s: Path(f"/app/models/smolvla_base/vision_out_seed{s}.npz")  # noqa: E731
    for s in seeds:
        if s == 0 and cache.exists():
            d = np.load(cache)
            if int(d["meta"][2]) == 0 and int(d["meta"][1]) == LAYERS:
                outs[s] = d["ref_OUT"].astype(np.float32); continue
        if per_seed(s).exists():
            outs[s] = np.load(per_seed(s))["out"].astype(np.float32); continue
        todo.append(s)
    if todo:
        import torch
        from safetensors import safe_open
        from transformers import SiglipVisionConfig, SiglipVisionModel
        cfg = SiglipVisionConfig(hidden_size=D, intermediate_size=FF, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                                 image_size=IMG, patch_size=PATCH, num_channels=3, hidden_act="gelu_pytorch_tanh",
                                 vision_use_head=False, attn_implementation="eager", layer_norm_eps=LN_EPS)
        model = SiglipVisionModel(cfg).eval()
        f = safe_open(str(ckpt), "pt")          # parameter by parameter (the host has ~1 GB free): no second fp32 copy
        params = dict(model.named_parameters())
        params.update(dict(model.named_buffers()))
        loaded = 0
        with torch.no_grad():
            for k in f.keys():
                if k.startswith(PREFIX):
                    nm = k[len(PREFIX):]
                    nm = nm if nm in params else "vision_model." + nm
                    params[nm].copy_(f.get_tensor(k).to(torch.float32)); loaded += 1
            assert loaded == 197, loaded
            for s in todo:
                outs[s] = model(pixel_values=torch.from_numpy(test_image(s))[None]).last_hidden_state[0].numpy()
                np.savez(per_seed(s), out=outs[s].astype(np.float32))   # cached: a later run (or a crash) does not redo it
        del model; gc.collect()
    return np.stack([outs[s] for s in seeds])


def vlm_inputs(ckpt: str | Path, n_cams: int, task_ids: list[int], image_seed: int = 0, state_seed: int = 0) -> dict:
    """The host-side inputs of the prefix: fp32 SigLIP outputs per camera (camera c = test_image(image_seed + c)),
    language ids (padded to 48 with the pad token, lang_mask 0 there), the padded state vector."""
    ids = list(task_ids)[:LANG_LEN]
    lang_mask = np.array([1] * len(ids) + [0] * (LANG_LEN - len(ids)), np.int64)
    ids = ids + [PAD_TOKEN_ID] * (LANG_LEN - len(ids))
    state = np.zeros(STATE_DIM, np.float32)
    state[:6] = np.random.default_rng(state_seed).uniform(-1, 1, 6).astype(np.float32)
    return {"vis_out": vision_outputs(ckpt, [image_seed + c for c in range(n_cams)]), "ids": np.array(ids, np.int64),
            "lang_mask": lang_mask, "state": state}


def embed_prefix(tw: TextTower, inp: dict) -> np.ndarray:
    """fp32 prefix embeddings [S, 960] as lerobot's embed_prefix builds them (images * sqrt(960), language * sqrt(960), state)."""
    sq = math.sqrt(TD)
    wc = tw.connector
    img = [pixel_shuffle(v) @ wc.T * sq for v in inp["vis_out"]]
    lang = tw.embed_rows(inp["ids"]) * sq
    state = inp["state"] @ tw.state_w.T + tw.state_b
    return np.concatenate(img + [lang, state[None]], 0).astype(np.float32)


def prefix_reference(tw: TextTower, inp: dict, layers: int = TLAYERS) -> dict[str, np.ndarray]:
    """fp32 torch reference of lerobot's SmolVLMWithExpertModel.forward over the prefix (fill_kv_cache=True): EMB
    (the embedded prefix), L<n> (residual stream after layer n), K<n> / V<n> (the layer's RoPE'd keys / values,
    [S, 320] = heads flattened), OUT (final norm). Cross-checked against transformers' LlamaModel with the same
    weights, rope_theta 10000 and the 2-D mask as a 4-D additive mask (hf_llama_crosscheck)."""
    import torch
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    lay = prefix_layout(inp["vis_out"].shape[0], inp["lang_mask"])
    h = T(embed_prefix(tw, inp))
    S = h.shape[0]
    mask = torch.from_numpy(lay["att2d"])
    table = T(rope_table(lay["pos"]))
    big_neg = torch.finfo(torch.float32).min
    out = {"EMB": h.numpy().copy()}

    def rms(x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + RMS_EPS) * w

    def rope(x):   # [S, H, dh]
        half = TDH // 2; c, s = table[:, None, :half], table[:, None, half:]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    with torch.no_grad():
        for L in range(layers):
            w = {k: T(v) for k, v in tw.layer(L).items()}
            ln = rms(h, w["g1"])
            q = (ln @ w["wq"].T).view(S, TH, TDH)
            k = (ln @ w["wk"].T).view(S, TKV, TDH)
            v = (ln @ w["wv"].T).view(S, TKV, TDH)
            q, k = rope(q), rope(k)
            out[f"K{L + 1}"], out[f"V{L + 1}"] = k.reshape(S, TKVD).numpy().copy(), v.reshape(S, TKVD).numpy().copy()
            grp = TH // TKV
            kk = k[:, :, None, :].expand(S, TKV, grp, TDH).reshape(S, TH, TDH)
            vv = v[:, :, None, :].expand(S, TKV, grp, TDH).reshape(S, TH, TDH)
            att = torch.einsum("qhd,khd->hqk", q, kk) * TDH ** -0.5
            att = torch.where(mask[None], att, big_neg).softmax(-1)
            o = torch.einsum("hqk,khd->qhd", att, vv).reshape(S, TD)
            h = h + o @ w["wo"].T
            ln2 = rms(h, w["g2"])
            h = h + (torch.nn.functional.silu(ln2 @ w["wg"].T) * (ln2 @ w["wu"].T)) @ w["wd"].T
            out[f"L{L + 1}"] = h.numpy().copy()
        out["OUT"] = rms(h, T(tw.norm)).numpy()
    return out


def hf_llama_crosscheck(tw: TextTower, inp: dict, ref: dict[str, np.ndarray], layers: int = 2) -> float:
    """Run transformers' LlamaModel (eager) with the first `layers` layers on the same embedded prefix and return the
    max |diff| of the layer outputs over the valid tokens (semantics check of RoPE / mask / GQA / RMSNorm)."""
    import torch
    from transformers import LlamaConfig, LlamaModel
    cfg = LlamaConfig(hidden_size=TD, intermediate_size=TFF, num_hidden_layers=layers, num_attention_heads=TH, num_key_value_heads=TKV,
                      head_dim=TDH, rms_norm_eps=RMS_EPS, rope_theta=ROPE_BASE, vocab_size=8, attn_implementation="eager",
                      mlp_bias=False, attention_bias=False, tie_word_embeddings=False)
    m = LlamaModel(cfg).eval()
    sd = {}
    for L in range(layers):
        for k, v in tw.layer(L).items():
            sd[f"layers.{L}.{TextTower.LAYER[k]}"] = torch.from_numpy(v)
    sd["norm.weight"] = torch.from_numpy(tw.norm)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k and "embed_tokens" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    lay = prefix_layout(inp["vis_out"].shape[0], inp["lang_mask"])
    mask = torch.where(torch.from_numpy(lay["att2d"]), 0.0, torch.finfo(torch.float32).min)[None, None]
    pos = torch.from_numpy(lay["pos"])[None]
    with torch.no_grad():
        o = m(inputs_embeds=torch.from_numpy(ref["EMB"])[None], attention_mask=mask, position_ids=pos, output_hidden_states=True)
    valid = lay["pad"]
    worst = 0.0
    for n in range(1, layers):          # transformers 5 records the LAST entry of hidden_states after the final norm
        worst = max(worst, float(np.abs(o.hidden_states[n][0].numpy()[valid] - ref[f"L{n}"][valid]).max()))
    last = ref[f"L{layers}"]
    normed = last / np.sqrt((last * last).mean(1, keepdims=True) + RMS_EPS) * tw.norm
    worst = max(worst, float(np.abs(o.last_hidden_state[0].numpy()[valid] - normed[valid]).max()))
    return worst


def vlm_library_layout(tw: TextTower, layers: int = TLAYERS) -> dict[str, np.ndarray]:
    """Text-tower tensors in library layout (fp16): GEMM weights [in, out] (HF Linear transposed), norm weights as rows.
    wc (connector), per layer wq/wk/wv/wo/wg/wu/wd + g1/g2, gfin."""
    f16 = lambda a: np.ascontiguousarray(a, dtype=np.float16)  # noqa: E731
    p = {"wc": f16(tw.connector.T), "gfin": f16(tw.norm.reshape(1, -1))}
    for L in range(layers):
        w = tw.layer(L)
        for nm in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
            p[f"{nm}{L}"] = f16(w[nm].T)
        p[f"g1{L}"], p[f"g2{L}"] = f16(w["g1"].reshape(1, -1)), f16(w["g2"].reshape(1, -1))
    return p


def numpy_reference_vlm(p: dict[str, np.ndarray], inp: dict, layers: int) -> dict[str, np.ndarray]:
    """The fp16 program's floor: fp32 math with every stored intermediate rounded to fp16, in the device's op order
    (connector GEMM, sqrt(960) scale, RMSNorm, projections, RoPE with the fp16 table, per-head attention, ...)."""
    f = lambda k: p[k].astype(np.float32)  # noqa: E731
    r16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    lay = prefix_layout(inp["vis_out"].shape[0], inp["lang_mask"])
    S, n_img = lay["n"], lay["n_img"]
    sq = math.sqrt(TD)
    img = [r16(r16(pixel_shuffle(r16(v))) @ f("wc")) for v in inp["vis_out"]]
    x = np.zeros((S, TD), np.float32)
    x[:n_img] = np.concatenate(img, 0)
    x[n_img:n_img + LANG_LEN] = r16(p["lang"].astype(np.float32))
    x[:n_img + LANG_LEN] = r16(x[:n_img + LANG_LEN] * np.float32(sq))
    x[-1] = r16(p["state_emb"].astype(np.float32))
    table = p["rope"].astype(np.float32)
    out = {"EMB": x}
    tok = lay["tok"].astype(np.uint32)
    allowed = (tok[None, :] <= tok[:, None]) & (tok[:, None] != TOK_PAD)

    def rms(a, g):
        return a * (1 / np.sqrt((a * a).mean(1, keepdims=True) + RMS_EPS)) * g

    for L in range(layers):
        ln = r16(rms(x, f(f"g1{L}")))
        q = r16(ln @ f(f"wq{L}")); k = r16(ln @ f(f"wk{L}")); v = r16(ln @ f(f"wv{L}"))
        q = r16(apply_rope(q.reshape(S, TH, TDH), table)).reshape(S, TD)
        k = r16(apply_rope(k.reshape(S, TKV, TDH), table)).reshape(S, TKVD)
        out[f"K{L + 1}"], out[f"V{L + 1}"] = k, v
        o = np.zeros((S, TD), np.float32)
        grp = TH // TKV
        for hd in range(TH):
            sl = slice(hd * TDH, (hd + 1) * TDH); kv = slice((hd // grp) * TDH, (hd // grp + 1) * TDH)
            s = r16(q[:, sl] @ k[:, kv].T)
            z = np.where(allowed, s / math.sqrt(TDH), -np.inf)
            m = np.where(allowed.any(1, keepdims=True), z.max(1, keepdims=True), 0.0)
            e = np.where(allowed, np.exp(z - m), 0.0); ssum = e.sum(1, keepdims=True)
            pr = np.where(ssum > 0, e / np.where(ssum > 0, ssum, 1), 1.0 / S)
            o[:, sl] = r16(r16(pr) @ v[:, kv])
        h = r16(x + r16(o @ f(f"wo{L}")))
        ln2 = r16(rms(h, f(f"g2{L}")))
        g = r16(ln2 @ f(f"wg{L}")); u = r16(ln2 @ f(f"wu{L}"))
        a = r16(g / (1 + np.exp(-g)) * u)
        x = r16(h + r16(a @ f(f"wd{L}")))
        out[f"L{L + 1}"] = x
    out["OUT"] = r16(rms(x, f("gfin")))
    return out


def prepare_vlm(ckpt: str | Path, n_cams: int, out: str | Path, task_ids: list[int] | None = None, layers: int = TLAYERS,
                image_seed: int = 0, state_seed: int = 0, crosscheck: bool = True) -> Path:
    """Checkpoint -> npz: text-tower weights in library layout (fp16), the host-prepared inputs (vision outputs per
    camera, language embeddings, state embedding, token classes, RoPE table), the fp32 lerobot-semantics reference
    and the fp16-floor twin. n_cams = 1 -> 113 tokens, 3 -> the full 241-token prefix."""
    inp = vlm_inputs(ckpt, n_cams, task_ids or DEFAULT_TASK_IDS, image_seed, state_seed)
    tw = TextTower(ckpt)
    for v in inp["vis_out"]:
        assert np.array_equal(pixel_shuffle(v), pixel_shuffle_gather(v)), "pixel shuffle gather formula != HF"
    ref = prefix_reference(tw, inp, layers)
    lay = prefix_layout(n_cams, inp["lang_mask"])
    if crosscheck:
        ncc = min(2, layers)
        worst = hf_llama_crosscheck(tw, inp, ref, ncc)
        print(f"[prepare-vlm] lerobot-semantics reference vs transformers LlamaModel (rope 10000, 4-D mask): max |diff| {worst:.2e} over L1..L{ncc}")
    p = vlm_library_layout(tw, layers)
    p["lang"] = tw.embed_rows(inp["ids"]).astype(np.float16)                                   # raw rows: the device scales by sqrt(960)
    p["state_emb"] = (inp["state"] @ tw.state_w.T + tw.state_b).reshape(1, -1).astype(np.float16)
    p["rope"] = rope_table(lay["pos"]).astype(np.float16)
    npref = numpy_reference_vlm(p, inp, layers)
    arrays = {f"p_{k}": v for k, v in p.items()}
    arrays["vis_out"] = inp["vis_out"].astype(np.float16)
    arrays["tok"] = lay["tok"]; arrays["pos"] = lay["pos"].astype(np.int64); arrays["ids"] = inp["ids"]; arrays["lang_mask"] = inp["lang_mask"]
    arrays.update({f"ref_{k}": v for k, v in ref.items()})
    arrays.update({f"np_{k}": v for k, v in npref.items()})
    arrays["meta"] = np.array([lay["n"], layers, n_cams, image_seed, state_seed], dtype=np.int64)
    out = Path(out)
    np.savez(out, **arrays)
    valid = lay["pad"]
    for tag in ("EMB", "L1", f"L{layers}", "K1", f"K{layers}", "OUT"):
        if tag in ref:
            d = np.abs(npref[tag][valid] - ref[tag][valid])
            print(f"[prepare-vlm] fp16-program floor vs fp32 reference  {tag:<4}: max abs {d.max():.4f} median {np.median(d):.4f} (|ref| max {np.abs(ref[tag][valid]).max():.2f})")
    print(f"[prepare-vlm] {out}  tokens={lay['n']} (img {lay['n_img']}, lang {LANG_LEN} of which {int(inp['lang_mask'].sum())} valid, state 1) "
          f"layers={layers}  params {sum(v.nbytes for v in p.values()) / 2 ** 20:.1f} MiB fp16")
    return out


# per-layer parameter family (name, rows, cols) of the text tower
VLM_PARAMS = [("wq", TD, TD), ("wk", TD, TKVD), ("wv", TD, TKVD), ("wo", TD, TD), ("wg", TD, TFF), ("wu", TD, TFF), ("wd", TFF, TD),
              ("g1", 1, TD), ("g2", 1, TD)]
# RedMulE tiles (tm, tn, tk) per GEMM; tm is replaced by min(tm, S_pad)
VLM_TILES = {"wc": (64, 320, 256), "wq": (128, 192, 320), "wk": (128, 320, 320), "wv": (128, 320, 320), "wo": (128, 192, 320),
             "wg": (128, 256, 320), "wu": (128, 256, 320), "wd": (128, 192, 256)}


def emit_vlm(npz: str | Path | dict, layers: int | None = None, cluster: int = -1, attn: int = -1,
             dumps: tuple[str, ...] = ("EMB", "L1", "OUT"), nsamples: int = 64, marks: bool = True, layer0: int = 0) -> tuple[str, dict[int, np.ndarray]]:
    """-> (mlir, {hbm_offset: array to preload}) of the VLM prefix program: pixel shuffle + connector GEMM per camera,
    sqrt(960) scaling, then one scf.for over the decoder layers (RMSNorm, q/k/v GEMMs with k/v written straight
    into the per-layer KV cache, RoPE on q and k, masked GQA attention, o-proj + residual, RMSNorm, gate/up GEMMs,
    silu_mul, down GEMM + residual), final RMSNorm. The sequence is padded to S_pad (multiple of 128; padding
    rows carry TOK_PAD and are never attended). Weights live in two HBM regions (west, then south for the layers
    that do not fit in the first 256 MiB): the layer index selects the region through arith.divui.
    KV cache layout (what the action expert reads): layer L's keys at kv_base + L * kv_stride as [S_pad, 320]
    fp16 row-major (head h = columns [64h, 64h+64), RoPE applied), its values right after at + S_pad*320*2.
    layer0 > 0: the program runs layers [layer0, layer0 + layers) only, starting from the reference's L<layer0>
    rounded to fp16 (no connector / embedding stage): the simulator's host memory is ~400 MB + 2x the preload image,
    which the shared 7.8 GB host cannot always spare for all 16 layers (324 MiB) at once."""
    data = np.load(npz) if isinstance(npz, (str, Path)) else npz
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    n, npz_layers, n_cams = int(data["meta"][0]), int(data["meta"][1]), int(data["meta"][2])
    layers = (npz_layers - layer0) if layers is None else layers
    assert layer0 + layers <= npz_layers, (layer0, layers, npz_layers)
    last = layer0 + layers == npz_layers
    n_img = n_cams * IMG_TOKENS
    S = ((n + 127) // 128) * 128
    e = _Emitter()
    e.next_off = HBM_DATA_START
    T: dict[str, str] = {}
    pre: dict[int, np.ndarray] = {}
    cl = f"cluster = {cluster} : i32"
    hc = f"cluster = {attn} : i32"
    hv = lambda rows, cols, ld, eoff: f'memref<{rows}x{cols}xf16, strided<[{ld}, 1], offset: {eoff}>, "{e.space}">'  # noqa: E731

    def alloc(rows, cols, at=None):
        if at is not None:
            e.next_off = at
        off = e.next_off
        e.next_off += (rows * cols * 2 + 4095) & ~4095
        return off

    def B(name, rows, cols, arr=None):
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            pre[e.next_off] = arr
        T[name] = e.buf(name, rows, cols)
        return name

    def mark(tag, idx=None):
        if marks:
            e.op(f'softhier.mark {idx + " " if idx else ""}{{tag = "{tag}"}}')

    def dump(tag, name, seed, view=None, idx=None, out_tag=None):
        if tag in dumps:
            e.op(f'softhier.dump_samples %{name}{", " + idx if idx else ""} {{seed = {seed} : i32, n = {nsamples} : i32, '
                 f'tag = "{out_tag or tag}"}} : {view or T[name]}')
            return True
        return False

    def dump_mark(tag, dumped, idx=None):
        if dumped and marks:
            e.op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
            mark(tag, idx)

    gem = lambda nm, M: (lambda t: f"tile_m = {min(t[0], M)} : i32, tile_n = {t[1]} : i32, tile_k = {t[2]} : i32, pipeline")(VLM_TILES[nm])  # noqa: E731

    # ---- inputs: vision outputs per camera, the residual stream with the language / state rows, mask + RoPE table
    x0 = np.zeros((S, TD), np.float16)
    if layer0 == 0:
        x0[n_img:n_img + LANG_LEN] = P["lang"]
        x0[n_img + LANG_LEN] = P["state_emb"][0]
    else:
        x0[:n] = data[f"ref_L{layer0}"].astype(np.float16)
    tok = np.full(S, TOK_PAD, np.uint16); tok[:n] = data["tok"]
    rope = np.zeros((S, TDH), np.float16); rope[:n] = P["rope"]
    for c in range(n_cams):
        B(f"vis{c}", GRID * GRID, D, data["vis_out"][c] if layer0 == 0 else None)
    B("ps", n_img, D * PIX_SCALE * PIX_SCALE)
    B("x", S, TD, x0)
    pre[e.next_off] = tok; T["tok"] = f'memref<{S}xi16, "{e.space}">'
    e.op(f"%tok = softhier.hbm_buffer {{offset = {e.next_off} : i32}} : {T['tok']}"); e.next_off += (S * 2 + 4095) & ~4095
    B("rope", S, TDH, rope)
    for nm, r, c in [("ln1", S, TD), ("q", S, TD), ("o", S, TD), ("ao", S, TD), ("h", S, TD), ("ln2", S, TD),
                     ("g", S, TFF), ("u", S, TFF), ("a", S, TFF), ("f2", S, TD), ("fin", S, TD)]:
        B(nm, r, c)
    # KV cache: [K_L | V_L] per layer, contiguous family
    kv_base = e.next_off
    kv_stride = 2 * S * TKVD * 2
    e.next_off += layers * kv_stride
    T["kc"] = T["vc"] = f'memref<{S}x{TKVD}xf16, "{e.space}">'
    # ---- parameters: connector + final norm in the west region, the per-layer family split over west / south
    B("wc", D * PIX_SCALE * PIX_SCALE, TD, P["wc"] if layer0 == 0 else None); B("gfin", 1, TD, P["gfin"])
    fam_base = e.next_off
    layer0_off, stride = {}, 0
    probe = fam_base
    for nm, r, c in VLM_PARAMS:
        layer0_off[nm] = probe - fam_base
        probe += (r * c * 2 + 4095) & ~4095
    stride = probe - fam_base
    n_west = min(layers, (HBM_WEST_END - fam_base) // stride)
    gap = (HBM_SOUTH - fam_base) - n_west * stride if n_west < layers else 0       # layer n_west lands at HBM_SOUTH
    for L in range(layers):
        base = fam_base + L * stride + (gap if L >= n_west else 0)
        for nm, r, c in VLM_PARAMS:
            arr = P[f"{nm}{L + layer0}"]
            assert arr.shape == (r, c), (nm, L, arr.shape)
            pre[base + layer0_off[nm]] = arr
        if L == 0:
            for nm, r, c in VLM_PARAMS:
                T[nm] = f'memref<{r}x{c}xf16, "{e.space}">'
    last_end = max(off + a.nbytes for off, a in pre.items())
    sent = sentinel_array()
    sent_off = (last_end + 0xFFFF) & ~0xFFFF
    pre[sent_off] = sent; T["sentinel"] = f'memref<{sent.shape[0]}x{sent.shape[1]}xf16, "{e.space}">'
    e.op(f"%sentinel = softhier.hbm_buffer {{offset = {sent_off} : i32}} : {T['sentinel']}")
    mark("preload")
    e.op(f"softhier.preload_wait %sentinel : {T['sentinel']}")

    # ---- connector: pixel shuffle + projection per camera, then the sqrt(960) scaling of image + language rows
    mark("start")
    if layer0 == 0:
        for c in range(n_cams):
            psv = hv(IMG_TOKENS, D * PIX_SCALE * PIX_SCALE, D * PIX_SCALE * PIX_SCALE, c * IMG_TOKENS * D * PIX_SCALE * PIX_SCALE)
            e.op(f"%ps{c} = softhier.view %ps : {T['ps']} -> {psv}")
            e.op(f"softhier.pixel_shuffle %vis{c} -> %ps{c} {{scale = {PIX_SCALE} : i32, {cl}}} : {T[f'vis{c}']} -> {psv}")
        xi = hv(n_img, TD, TD, 0)
        e.op(f"%ximg = softhier.view %x : {T['x']} -> {xi}")
        e.op(f"softhier.gemm %ps, %wc into %ximg {{fmt = \"fp16\", {gem('wc', n_img)}, {cl}}} : {T['ps']}, {T['wc']}, {xi}")
        xs = hv(n_img + LANG_LEN, TD, TD, 0)
        e.op(f"%xsc = softhier.view %x : {T['x']} -> {xs}")
        e.op(f"softhier.scale %xsc -> %xsc {{scale = {math.sqrt(TD)!r} : f32, {cl}}} : {xs} -> {xs}")
    xv = hv(n, TD, TD, 0)
    e.op(f"%xval = softhier.view %x : {T['x']} -> {xv}")
    mark("emb")
    if layer0 == 0:
        dump_mark("embdump", dump("EMB", "xval", 200, xv))

    # ---- the decoder layers
    e.op("%c0 = arith.constant 0 : index")
    e.op("%c1 = arith.constant 1 : index")
    e.op(f"%cL = arith.constant {layers} : index")
    e.op(f"%cW = arith.constant {max(n_west, 1)} : index")
    e.op(f"%cS = arith.constant {stride} : index")
    e.op(f"%cG = arith.constant {gap} : index")
    e.op(f"%cL0 = arith.constant {layer0 + 1} : index")
    e.op("scf.for %L = %c0 to %cL step %c1 {")
    e.op("%L1 = arith.addi %L, %cL0 : index")
    e.op("%Lr = arith.divui %L, %cW : index")
    e.op("%Lo = arith.muli %L, %cS : index")
    e.op("%Lg = arith.muli %Lr, %cG : index")
    e.op("%Le = arith.addi %Lo, %Lg : index")
    for nm, r, c in VLM_PARAMS:
        e.op(f"%{nm} = softhier.hbm_buffer %Le {{offset = {fam_base + layer0_off[nm]} : i32, stride = 1 : i32}} : {T[nm]}")
    e.op(f"%kc = softhier.hbm_buffer %L {{offset = {kv_base} : i32, stride = {kv_stride} : i32}} : {T['kc']}")
    e.op(f"%vc = softhier.hbm_buffer %L {{offset = {kv_base + S * TKVD * 2} : i32, stride = {kv_stride} : i32}} : {T['vc']}")
    e.op(f"softhier.rmsnorm %x, %g1 -> %ln1 {{eps = {RMS_EPS:.1e} : f32, {cl}}} : {T['x']}, {T['g1']} -> {T['ln1']}")
    for dst, w in (("q", "wq"), ("kc", "wk"), ("vc", "wv")):
        e.op(f"softhier.gemm %ln1, %{w} into %{dst} {{fmt = \"fp16\", {gem(w, S)}, {cl}}} : {T['ln1']}, {T[w]}, {T[dst]}")
    e.op(f"softhier.rope %q, %rope -> %q {{head_dim = {TDH} : i32, {cl}}} : {T['q']}, {T['rope']} -> {T['q']}")
    e.op(f"softhier.rope %kc, %rope -> %kc {{head_dim = {TDH} : i32, {cl}}} : {T['kc']}, {T['rope']} -> {T['kc']}")
    e.op(f"softhier.attention %q, %kc, %vc, %tok -> %o {{scale = {1 / math.sqrt(TDH)!r} : f32, heads = {TH} : i32, kv_heads = {TKV} : i32, {hc}}} "
         f": {T['q']}, {T['kc']}, {T['vc']}, {T['tok']} -> {T['o']}")
    mark("attn", "%L1")
    e.op(f"softhier.gemm %o, %wo into %ao {{fmt = \"fp16\", {gem('wo', S)}, {cl}}} : {T['o']}, {T['wo']}, {T['ao']}")
    e.op(f"softhier.add %x, %ao -> %h {{{cl}}} : {T['x']}, {T['ao']} -> {T['h']}")
    e.op(f"softhier.rmsnorm %h, %g2 -> %ln2 {{eps = {RMS_EPS:.1e} : f32, {cl}}} : {T['h']}, {T['g2']} -> {T['ln2']}")
    e.op(f"softhier.gemm %ln2, %wg into %g {{fmt = \"fp16\", {gem('wg', S)}, {cl}}} : {T['ln2']}, {T['wg']}, {T['g']}")
    e.op(f"softhier.gemm %ln2, %wu into %u {{fmt = \"fp16\", {gem('wu', S)}, {cl}}} : {T['ln2']}, {T['wu']}, {T['u']}")
    e.op(f"softhier.silu_mul %g, %u -> %a {{{cl}}} : {T['g']}, {T['u']} -> {T['a']}")
    e.op(f"softhier.gemm %a, %wd into %f2 {{fmt = \"fp16\", {gem('wd', S)}, {cl}}} : {T['a']}, {T['wd']}, {T['f2']}")
    e.op(f"softhier.add %h, %f2 -> %x {{{cl}}} : {T['h']}, {T['f2']} -> {T['x']}")
    mark("layer", "%L1")
    kvv = hv(n, TKVD, TKVD, 0)
    d = False
    if any(t.startswith("L") and t[1:].isdigit() for t in dumps):
        e.op(f'softhier.dump_samples %xval, %L1 {{seed = 220 : i32, n = {nsamples} : i32, tag = "L"}} : {xv}'); d = True
    if any(t.startswith("K") and t[1:].isdigit() for t in dumps):
        e.op(f"%kcv = softhier.view %kc : {T['kc']} -> {kvv}")
        e.op(f'softhier.dump_samples %kcv, %L1 {{seed = 230 : i32, n = {nsamples} : i32, tag = "K"}} : {kvv}'); d = True
    if any(t.startswith("V") and t[1:].isdigit() for t in dumps):
        e.op(f"%vcv = softhier.view %vc : {T['vc']} -> {kvv}")
        e.op(f'softhier.dump_samples %vcv, %L1 {{seed = 240 : i32, n = {nsamples} : i32, tag = "V"}} : {kvv}'); d = True
    dump_mark("ldump", d, "%L1")
    e.op("}")
    if last:
        e.op(f"softhier.rmsnorm %x, %gfin -> %fin {{eps = {RMS_EPS:.1e} : f32, {cl}}} : {T['x']}, {T['gfin']} -> {T['fin']}")
    mark("end")
    if last:
        fv = hv(n, TD, TD, 0)
        e.op(f"%finv = softhier.view %fin : {T['fin']} -> {fv}")
        dump("OUT", "finv", 205, fv)
    body = "\n".join(e.lines)
    mlir = f"builtin.module {{\n  func.func @smolvla_vlm_prefix() {{\n{body}\n    func.return\n  }}\n}}\n"
    info = {"S_pad": S, "n": n, "n_img": n_img, "layer0": layer0, "kv_base": kv_base, "kv_stride": kv_stride, "fam_base": fam_base, "stride": stride,
            "n_west": n_west, "gap": gap, "image_bytes": sum(a.nbytes for a in pre.values())}
    emit_vlm.last_info = info
    return mlir, pre


emit_vlm.last_info = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("prepare-vlm", help="checkpoint -> npz of the VLM text prefix (weights, host-prepared inputs, references)")
    pv.add_argument("--ckpt", default="/app/models/smolvla_base/model.safetensors")
    pv.add_argument("--cams", type=int, default=1, help="cameras (64 image tokens each): 1 -> 113 tokens, 3 -> the full 241")
    pv.add_argument("--layers", type=int, default=TLAYERS)
    pv.add_argument("--task-ids", type=int, nargs="*", help=f"language token ids (default {DEFAULT_TASK_IDS})")
    pv.add_argument("--image-seed", type=int, default=0)
    pv.add_argument("--state-seed", type=int, default=0)
    pv.add_argument("--no-crosscheck", action="store_true")
    pv.add_argument("--out", required=True)
    pev = sub.add_parser("emit-vlm", help="npz -> MLIR of the prefix program on stdout")
    pev.add_argument("npz")
    pev.add_argument("--layers", type=int)
    pev.add_argument("--cluster", type=int, default=-1)
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
    pe.add_argument("--unroll", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.ckpt, a.seq, a.out, a.layers, a.image_seed)
    elif a.cmd == "prepare-vlm":
        prepare_vlm(a.ckpt, a.cams, a.out, a.task_ids, a.layers, a.image_seed, a.state_seed, not a.no_crosscheck)
    elif a.cmd == "emit-vlm":
        import sys
        sys.stdout.write(emit_vlm(a.npz, a.layers, a.cluster)[0])
    else:
        import sys
        sys.stdout.write(emit(a.npz, a.layers, a.attn, a.cluster, unroll=a.unroll)[0])


if __name__ == "__main__":
    main()
