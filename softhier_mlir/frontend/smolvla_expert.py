"""SmolVLA's action expert (`model.vlm_with_expert.lm_expert.*`, 16 layers, width 720) and its flow-matching
sampling loop as a softhier program; the ledger is docs/SMOLVLA_EXPERT.md.

  ref      (lerobot venv)  python -m softhier_mlir.frontend.smolvla_expert_ref --out expert_ref.npz
           runs lerobot's SmolVLAPolicy in fp32 on a fixed synthetic observation and dumps the prefix KV cache,
           the per-step x_t / v_t, the final action chunk and step-0 intermediates.
  prepare  (system python: torch + safetensors)
           python3 -m softhier_mlir.frontend.smolvla_expert prepare --ckpt /app/models/smolvla_base/model.safetensors \
                   --ref /app/models/smolvla_base/expert_ref.npz --out /app/models/smolvla_base/expert.npz
           expert weights in library layout + the host-side tables (time table, RoPE tables, validity row) + the
           lerobot reference arrays, fp16 where the device reads them.
  emit     (numpy only; tests/gvsoc/expert.py)
           emit_layer_test(seed)       one self-attention layer + one cross-attention layer on LCG data (step 1)
           emit_flow(npz, ...)         the 10-step Euler loop with real weights and the host's prefix KV (step 2/3)

The expert as lerobot runs it at inference (VLMWithExpertModel.forward with fill_kv_cache=False, modeling code
in lerobot/policies/smolvla): Llama decoder layers (RMSNorm eps 1e-5, no biases, SwiGLU 720 -> 2048 -> 720),
15 query heads x 64 = 960, 5 kv heads x 64 = 320 (GQA, query head h uses kv head h // 3), scale 1/8.
  layer l even (self_attn_every_n_layers = 2): q,k,v = proj(rmsnorm(h)); q,k rotated at positions n_valid + i
      (lerobot apply_rope: half-split, base 1e4); keys = [VLM prefix K of layer l (already rotated, 241 rows) ; k],
      values = [prefix V ; v]; mask = prefix padding on the first 241 columns, causal (j <= i) on the 50 own columns.
  layer l odd: q rotated at positions i (0..49); keys = prefix K_l @ Wk_l^T, values = prefix V_l @ Wv_l^T
      (k/v_proj are 320 -> 320 on these layers; the projected KV is step-invariant, so it is computed once per
      chunk before the step loop); mask = prefix padding only.
  o -> o_proj -> residual; rmsnorm -> down(silu(gate) * up) -> residual; final rmsnorm; action_out_proj.
Suffix embedding per step s (t_s = 1 - s/10): e = action_in_proj(x_t); [e | time_emb(t_s)] -> action_time_mlp_in
-> silu -> action_time_mlp_out. The time half of action_time_mlp_in is folded on the host into a per-step bias
table tb[s] = time_emb(t_s) @ W_in[:, 720:]^T + b_in (exact algebra). Euler: x_{s+1} = x_s - 0.1 v_s, x_0 = noise.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from softhier_mlir.frontend.siglip import _Emitter
from softhier_mlir.sim.preload import sentinel_array
from softhier_mlir.testing import lcg

PREFIX = "model.vlm_with_expert.lm_expert."
D, DQ, DKV, FF, H, HKV, DH, LAYERS = 720, 960, 320, 2048, 15, 5, 64, 16
S, LP, AD, STEPS = 50, 241, 32, 10            # chunk_size, prefix length (3 x 64 image + 48 language + 1 state), action dim, num_steps
RMS_EPS, SCALE, ROPE_BASE = 1e-5, DH ** -0.5, 10000.0
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0
HBM_DATA_START = 0x10000
GROUP = H // HKV


# ----------------------------------------------------------------------------- host-side tables
def rope_table(positions: np.ndarray, dh: int = DH) -> np.ndarray:
    """[len(positions), dh] fp16: [cos(dh/2) | sin(dh/2)] of lerobot's apply_rope (timescale base^(2i/dh)) per row,
    the table every head of the row is rotated with (softhier.rope's cos_sin operand)."""
    half = dh // 2
    timescale = ROPE_BASE ** ((2.0 / dh) * np.arange(half, dtype=np.float64))
    rad = positions.astype(np.float64)[:, None] / timescale[None, :]
    return np.ascontiguousarray(np.concatenate([np.cos(rad), np.sin(rad)], axis=1).astype(np.float16))


def time_embedding(t: float, dim: int = D) -> np.ndarray:
    """create_sinusoidal_pos_embedding(t, dim, min_period, max_period): [sin(w t) | cos(w t)], float64."""
    frac = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** frac
    x = (2 * math.pi / period) * t
    return np.concatenate([np.sin(x), np.cos(x)])


def flow_times(steps: int = STEPS) -> list[float]:
    return [1.0 + s * (-1.0 / steps) for s in range(steps)]


# ----------------------------------------------------------------------------- weights
def load_expert(ckpt: str | Path) -> dict[str, np.ndarray]:
    """{name: fp32} of the 145 lm_expert tensors (bf16 -> fp32 via torch) + the 10 fp32 projection tensors."""
    import torch
    from safetensors.torch import load_file
    sd = load_file(str(ckpt))
    out = {k[len(PREFIX):]: v.to(torch.float32).numpy() for k, v in sd.items() if k.startswith(PREFIX)}
    for k in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        out[k + ".weight"] = sd[f"model.{k}.weight"].to(torch.float32).numpy()
        out[k + ".bias"] = sd[f"model.{k}.bias"].to(torch.float32).numpy()
    assert len(out) == 153, len(out)   # 145 lm_expert tensors + 4 projections x (weight, bias)
    return out


def to_library_layout(w: dict[str, np.ndarray], time_embs: np.ndarray) -> dict[str, np.ndarray]:
    """HF tensors -> {library name: fp16}. GEMM weights are [in, out] (Linear weight transposed); rows for
    gamma / bias. Self layers: wqkv [720, 1600] = [Wq^T | Wk^T | Wv^T]; cross layers: wq [720, 960] and the
    320 -> 320 wkx / wvx. wgu [720, 4096] = [Wgate^T | Wup^T]. tb [10, 720] = the folded time half of
    action_time_mlp_in + its bias for the 10 flow times."""
    T = lambda a: np.ascontiguousarray(a.T)  # noqa: E731
    row = lambda v: v.reshape(1, -1)  # noqa: E731
    p: dict[str, np.ndarray] = {}
    for L in range(LAYERS):
        g = f"layers.{L}."
        wq, wk, wv = w[g + "self_attn.q_proj.weight"], w[g + "self_attn.k_proj.weight"], w[g + "self_attn.v_proj.weight"]
        if L % 2 == 0:
            assert wk.shape == (DKV, D)
            p[f"wqkv{L}"] = np.concatenate([T(wq), T(wk), T(wv)], axis=1)        # [720, 1600]
        else:
            assert wk.shape == (DKV, DKV)
            p[f"wq{L}"], p[f"wkx{L}"], p[f"wvx{L}"] = T(wq), T(wk), T(wv)
        p[f"wo{L}"] = T(w[g + "self_attn.o_proj.weight"])                        # [960, 720]
        p[f"wgu{L}"] = np.concatenate([T(w[g + "mlp.gate_proj.weight"]), T(w[g + "mlp.up_proj.weight"])], axis=1)   # [720, 4096]
        p[f"wd{L}"] = T(w[g + "mlp.down_proj.weight"])                            # [2048, 720]
        p[f"g1{L}"], p[f"g2{L}"] = row(w[g + "input_layernorm.weight"]), row(w[g + "post_attention_layernorm.weight"])
    p["gf"] = row(w["norm.weight"])
    p["wa"], p["ba"] = T(w["action_in_proj.weight"]), row(w["action_in_proj.bias"])            # [32, 720]
    win, bin_ = w["action_time_mlp_in.weight"], w["action_time_mlp_in.bias"]                   # [720, 1440]
    p["wti"] = T(win[:, :D])                                                                   # action half [720, 720]
    p["tb"] = (time_embs.astype(np.float64) @ win[:, D:].T.astype(np.float64) + bin_).astype(np.float32)   # [10, 720]
    p["wto"], p["bto"] = T(w["action_time_mlp_out.weight"]), row(w["action_time_mlp_out.bias"])
    p["wout"], p["bout"] = T(w["action_out_proj.weight"]), row(w["action_out_proj.bias"])     # [720, 32]
    return {k: np.ascontiguousarray(v, dtype=np.float16) for k, v in p.items()}


def prepare(ckpt: str | Path, ref: str | Path, out: str | Path) -> Path:
    """expert.npz = weights in library layout (p_*), host tables (p_*), lerobot reference arrays (ref_*)."""
    r = dict(np.load(ref))
    te = np.stack([time_embedding(t) for t in flow_times()])
    assert np.abs(te - r["time_emb"]).max() < 1e-3, "time embedding twin differs from lerobot"   # lerobot feeds fp32 t_s
    te = r["time_emb"].astype(np.float64)                                                          # the table uses lerobot own values
    p = to_library_layout(load_expert(ckpt), te)
    n_valid = int(r["n_valid"])
    p["valid"] = r["prefix_valid"].reshape(1, LP).astype(np.float16)
    p["rq_self"] = rope_table(n_valid + np.arange(S))           # self layers: positions prefix_offset + i (q and own k)
    p["rq_cross"] = rope_table(np.arange(S))                    # cross layers: positions i
    for L in range(LAYERS):
        p[f"kp{L}"], p[f"vp{L}"] = r[f"kv_k_{L}"].astype(np.float16), r[f"kv_v_{L}"].astype(np.float16)
    p["x0"] = r["noise"].astype(np.float16)
    arrays = {f"p_{k}": v for k, v in p.items()}
    arrays.update({f"ref_{k}": v for k, v in r.items() if k not in ("images",)})
    np.savez(out, **arrays)
    print(f"[prepare] {out}: params {sum(v.nbytes for k, v in p.items() if not k.startswith(('kp', 'vp'))) / 2 ** 20:.1f} MiB fp16, "
          f"KV {sum(v.nbytes for k, v in p.items() if k.startswith(('kp', 'vp'))) / 2 ** 20:.1f} MiB, n_valid {n_valid}")
    return Path(out)


# ----------------------------------------------------------------------------- numpy twin (fp16-floor model)
def _r16(a):
    return np.asarray(a, np.float32).astype(np.float16).astype(np.float32)


def np_rmsnorm(x, g):
    x = x.astype(np.float32)
    return _r16(x * (1.0 / np.sqrt((x * x).mean(1, keepdims=True) + RMS_EPS)) * g.astype(np.float32))


def np_rope(x, tab, dh=DH):
    x, tab = x.astype(np.float32), tab.astype(np.float32)
    y = np.empty_like(x); hh = dh // 2
    c, s = tab[:, :hh], tab[:, hh:dh]
    for h0 in range(0, x.shape[1], dh):
        x1, x2 = x[:, h0:h0 + hh], x[:, h0 + hh:h0 + dh]
        y[:, h0:h0 + hh] = x1 * c - x2 * s
        y[:, h0 + hh:h0 + dh] = x2 * c + x1 * s
    return _r16(y)


def np_attention(q, kp, vp, ko, vo, valid, scale=SCALE):
    """GQA attention with the prefix validity mask and causal own keys; fp16 rounding where the device stores fp16."""
    q = q.astype(np.float32); Sq = q.shape[0]
    K = kp.astype(np.float32) if ko is None else np.concatenate([kp, ko]).astype(np.float32)
    V = vp.astype(np.float32) if vo is None else np.concatenate([vp, vo]).astype(np.float32)
    Lp, L = kp.shape[0], K.shape[0]
    allowed = np.ones((Sq, L), bool)
    if valid is not None:
        allowed[:, :Lp] &= valid.reshape(-1)[:Lp] != 0
    if ko is not None:
        allowed[:, Lp:] &= np.tril(np.ones((Sq, L - Lp), bool))
    o = np.zeros((Sq, q.shape[1]), np.float32)
    for h in range(q.shape[1] // DH):
        kv = h // GROUP
        s = _r16(q[:, h * DH:(h + 1) * DH] @ K[:, kv * DH:(kv + 1) * DH].T) * scale
        s = np.where(allowed, s, -np.inf)
        e = np.exp(s - s.max(1, keepdims=True)); e = _r16(e)
        o[:, h * DH:(h + 1) * DH] = _r16(_r16(e @ V[:, kv * DH:(kv + 1) * DH]) / e.sum(1, keepdims=True))
    return o


def np_layer(h, P, L, kp, vp, valid, rq, rk, kx=None, vx=None):
    """One expert layer on the fp16-floor model; P holds fp16 arrays named as in to_library_layout. Returns
    (h_out, intermediates dict)."""
    f = lambda k: P[k].astype(np.float32)  # noqa: E731
    xn = np_rmsnorm(h, f(f"g1{L}"))
    if L % 2 == 0:
        qkv = _r16(xn @ f(f"wqkv{L}"))
        q, k, v = np_rope(qkv[:, :DQ], rq), np_rope(qkv[:, DQ:DQ + DKV], rk), qkv[:, DQ + DKV:]
        o = np_attention(q, kp, vp, k, v, valid)
    else:
        q = np_rope(_r16(xn @ f(f"wq{L}")), rq)
        o = np_attention(q, kx, vx, None, None, valid)
    h1 = _r16(h + _r16(o @ f(f"wo{L}")))
    xn2 = np_rmsnorm(h1, f(f"g2{L}"))
    gu = _r16(xn2 @ f(f"wgu{L}"))
    g, u = gu[:, :FF], gu[:, FF:]
    m = _r16(g / (1 + np.exp(-g)) * u)
    h2 = _r16(h1 + _r16(m @ f(f"wd{L}")))
    return h2, {"XN": xn, "Q": q, "O": o, "H1": h1, "M": m, "H": h2}


def np_kv_proj(P, L, kp, vp):
    return _r16(kp.astype(np.float32) @ P[f"wkx{L}"].astype(np.float32)), _r16(vp.astype(np.float32) @ P[f"wvx{L}"].astype(np.float32))


def np_flow(P, steps=STEPS, layers=LAYERS, record=False):
    """The device program on the fp16-floor model: returns x_t per step [steps+1, 50, 32] (+ step-0 intermediates)."""
    f = lambda k: P[k].astype(np.float32)  # noqa: E731
    valid, rq_s, rq_c = P["valid"], P["rq_self"], P["rq_cross"]
    kx = {L: np_kv_proj(P, L, P[f"kp{L}"], P[f"vp{L}"]) for L in range(1, layers, 2)}
    x = f("x0"); xs = [x.copy()]; inter = {}
    for s in range(steps):
        e = _r16(_r16(x @ f("wa")) + f("ba"))
        e1 = _r16(_r16(e @ f("wti")) + f("tb")[s])
        e1 = _r16(e1 / (1 + np.exp(-e1)))
        h = _r16(_r16(e1 @ f("wto")) + f("bto"))
        if s == 0:
            inter["EMB"] = h
        for L in range(layers):
            if L % 2 == 0:
                h, it = np_layer(h, P, L, P[f"kp{L}"], P[f"vp{L}"], valid, rq_s, rq_s)
            else:
                h, it = np_layer(h, P, L, None, None, valid, rq_c, None, *kx[L])
            if s == 0 and record:
                inter[f"H{L}"] = it["H"]; inter[f"O{L}"] = it["O"]
        fin = np_rmsnorm(h, f("gf"))
        if s == 0:
            inter["FIN"] = fin
        v = _r16(_r16(fin @ f("wout")) + f("bout"))
        x = _r16(x + (-1.0 / steps) * v)
        xs.append(x.copy())
    return np.stack(xs), inter


# ----------------------------------------------------------------------------- programs
class _Prog:
    """Shared emission helpers over frontend.siglip._Emitter: HBM buffers (+ optional preload array), marks, dumps."""

    def __init__(self, cluster: int = -1, nsamples: int = 64, marks: bool = True):
        self.e = _Emitter(); self.e.next_off = HBM_DATA_START
        self.T: dict[str, str] = {}
        self.pre: dict[int, np.ndarray] = {}
        self.off: dict[str, int] = {}
        self.cl = f"cluster = {cluster} : i32"
        self.nsamples, self.marks = nsamples, marks
        self.sp = self.e.space

    def alloc(self, rows, cols):
        off = self.e.next_off
        self.e.next_off += (rows * cols * 2 + 4095) & ~4095
        return off

    def B(self, name, rows, cols, arr=None):
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            self.pre[self.e.next_off] = arr
        self.off[name] = self.e.next_off
        self.T[name] = self.e.buf(name, rows, cols)
        return name

    def slot(self, name, rows, cols, arr=None):
        """an offset in a parameter slab without a declaration; the declaration is an indexed hbm_buffer"""
        off = self.alloc(rows, cols)
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            self.pre[off] = arr
        self.T[name] = f'memref<{rows}x{cols}xf16, "{self.sp}">'
        return off

    def op(self, text):
        self.e.op(text)

    def mt(self, rows, cols, ld, eoff):
        return f'memref<{rows}x{cols}xf16, strided<[{ld}, 1], offset: {eoff}>, "{self.sp}">'

    def view(self, name, src, rows, cols, ld, eoff):
        self.T[name] = self.e.view(name, src, self.T[src], rows, cols, ld, eoff)
        return name

    def mark(self, tag, idx=None):
        if self.marks:
            self.op(f'softhier.mark {idx + " " if idx else ""}{{tag = "{tag}"}}')

    def dump(self, name, seed, tag, idx=None, all_=False):
        if all_:
            self.op(f'softhier.dump_all %{name}{", " + idx if idx else ""} {{tag = "{tag}"}} : {self.T[name]}')
        else:
            self.op(f'softhier.dump_samples %{name}{", " + idx if idx else ""} {{seed = {seed} : i32, n = {self.nsamples} : i32, tag = "{tag}"}} : {self.T[name]}')

    def gemm(self, x, w, z, tm, tn, tk, step=None, fmt_steps=None, cl=None):
        fs = f', fmt_steps = [{", ".join(chr(34) + f + chr(34) for f in fmt_steps)}]' if fmt_steps else ""
        st = f" step {step}" if step and fmt_steps else ""
        self.op(f'softhier.gemm %{x}, %{w} into %{z}{st} {{fmt = "fp16"{fs}, tile_m = {tm} : i32, tile_n = {tn} : i32, '
                f'tile_k = {tk} : i32, pipeline, {cl or self.cl}}} : {self.T[x]}, {self.T[w]}, {self.T[z]}')

    def module(self, name):
        body = "\n".join(self.e.lines)
        return f"builtin.module {{\n  func.func @{name}() {{\n{body}\n    func.return\n  }}\n}}\n"


# tile shapes (tm, tn, tk) per GEMM family; overridable for experiments (docs/SMOLVLA_EXPERT.md)
TILES = {"qkv": (S, 64, D), "o": (S, 48, DQ), "gu": (S, 256, 240), "d": (S, 48, 512), "kv": (LP, 64, DKV),
         "a": (S, 48, AD), "t": (S, 48, D), "out": (S, 32, D)}


def _layer_ops(P: _Prog, L_tag: str, kind: str, names: dict, step=None, fmt_steps=None, tiles=TILES, prof=False, pidx=None):
    """Emit one expert layer. names: parameter / KV SSA names for this layer; kind 'self' | 'cross'.
    Activations: h (residual, in place), xn, qkv (q | k | v for self; q for cross), o, ao, gu, m, f2."""
    T, op, cl = P.T, P.op, P.cl
    m = lambda tag: P.mark(tag, pidx) if prof else None  # noqa: E731
    op(f"softhier.rmsnorm %h, %{names['g1']} -> %xn {{eps = {RMS_EPS:.1e} : f32, {cl}}} : {T['h']}, {T[names['g1']]} -> {T['xn']}")
    if kind == "self":
        P.gemm("xn", names["wqkv"], "qkv", *tiles["qkv"], step=step, fmt_steps=fmt_steps)
        m("qkv")
        op(f"softhier.rope %q, %rq_self -> %q {{head_dim = {DH} : i32, {cl}}} : {T['q']}, {T['rq_self']} -> {T['q']}")
        op(f"softhier.rope %k, %rq_self -> %k {{head_dim = {DH} : i32, {cl}}} : {T['k']}, {T['rq_self']} -> {T['k']}")
        m("rope")
        op(f"softhier.cross_attention %q, %{names['kp']}, %{names['vp']} own %k, %v valid %valid -> %o "
           f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : {T['q']}, {T[names['kp']]}, {T[names['vp']]} "
           f"own {T['k']}, {T['v']} valid {T['valid']} -> {T['o']}")
    else:
        P.gemm("xn", names["wq"], "q", *tiles["qkv"], step=step, fmt_steps=fmt_steps)
        m("qkv")
        op(f"softhier.rope %q, %rq_cross -> %q {{head_dim = {DH} : i32, {cl}}} : {T['q']}, {T['rq_cross']} -> {T['q']}")
        m("rope")
        op(f"softhier.cross_attention %q, %{names['kx']}, %{names['vx']} valid %valid -> %o "
           f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : {T['q']}, {T[names['kx']]}, {T[names['vx']]} "
           f"valid {T['valid']} -> {T['o']}")
    m("attn")
    P.gemm("o", names["wo"], "ao", *tiles["o"], step=step, fmt_steps=fmt_steps)
    op(f"softhier.add %h, %ao -> %h {{{cl}}} : {T['h']}, {T['ao']} -> {T['h']}")
    m("oproj")
    op(f"softhier.rmsnorm %h, %{names['g2']} -> %xn {{eps = {RMS_EPS:.1e} : f32, {cl}}} : {T['h']}, {T[names['g2']]} -> {T['xn']}")
    P.gemm("xn", names["wgu"], "gu", *tiles["gu"], step=step, fmt_steps=fmt_steps)
    m("gateup")
    op(f"softhier.silu_mul %ga, %up -> %m {{{cl}}} : {T['ga']}, {T['up']} -> {T['m']}")
    m("silu")
    P.gemm("m", names["wd"], "f2", *tiles["d"], step=step, fmt_steps=fmt_steps)
    op(f"softhier.add %h, %f2 -> %h {{{cl}}} : {T['h']}, {T['f2']} -> {T['h']}")
    m("down")


def _activations(P: _Prog):
    for nm, r, c in [("x", S, AD), ("e", S, D), ("e1", S, D), ("h", S, D), ("xn", S, D), ("qkv", S, DQ + 2 * DKV), ("o", S, DQ),
                     ("ao", S, D), ("gu", S, 2 * FF), ("m", S, FF), ("f2", S, D), ("fin", S, D), ("vt", S, AD)]:
        P.B(nm, r, c)
    P.view("q", "qkv", S, DQ, DQ + 2 * DKV, 0)
    P.view("k", "qkv", S, DKV, DQ + 2 * DKV, DQ)
    P.view("v", "qkv", S, DKV, DQ + 2 * DKV, DQ + DKV)
    P.view("ga", "gu", S, FF, 2 * FF, 0)
    P.view("up", "gu", S, FF, 2 * FF, FF)


def emit_layer_test(seed: int = 1, cluster: int = -1, nsamples: int = 128, tiles=TILES, device_fill: bool = False) -> tuple[str, dict, dict]:
    """Step 1: one self-attention layer (L=0) followed by one cross-attention layer (L=1) on LCG data (weights, input,
    prefix KV, a random 0/1 validity row; the RoPE tables are host tables). The LCG data is generated by the host twin
    (softhier_mlir.testing.lcg) and preloaded; device_fill=True generates it on the device instead (hbm_fill_lcg, same
    numbers, ~13 M scalar stores of simulated time). Returns (mlir, preload, reference dict of the dumped tensors on
    the fp16-floor numpy twin)."""
    P = _Prog(cluster, nsamples)
    T, op = P.T, P.op
    data: dict[str, np.ndarray] = {}

    def F(name, rows, cols, sd, lo, hi, scale):
        data[name] = lcg.fill_fp16(rows, cols, sd, lo, hi, scale)
        if device_fill:
            P.B(name, rows, cols)
            op(f"softhier.hbm_fill_lcg %{name} {{seed = {sd} : i32, lo = {lo} : i32, hi = {hi} : i32, scale = {scale!r} : f32}} : {T[name]}")
        else:
            P.B(name, rows, cols, data[name])
    _activations(P)
    s0 = seed * 100
    F("hin", S, D, s0 + 1, -16, 16, 0.125)
    F("wqkv0", D, DQ + 2 * DKV, s0 + 2, -8, 8, 1 / 128); F("wo0", DQ, D, s0 + 3, -8, 8, 1 / 128)
    F("wgu0", D, 2 * FF, s0 + 4, -8, 8, 1 / 128); F("wd0", FF, D, s0 + 5, -8, 8, 1 / 256)
    F("g10", 1, D, s0 + 6, 2, 6, 0.25); F("g20", 1, D, s0 + 7, 2, 6, 0.25)
    F("kp0", LP, DKV, s0 + 8, -8, 8, 0.125); F("vp0", LP, DKV, s0 + 9, -8, 8, 0.125)
    F("wq1", D, DQ, s0 + 12, -8, 8, 1 / 128); F("wkx1", DKV, DKV, s0 + 13, -8, 8, 1 / 128); F("wvx1", DKV, DKV, s0 + 14, -8, 8, 1 / 128)
    F("wo1", DQ, D, s0 + 15, -8, 8, 1 / 128); F("wgu1", D, 2 * FF, s0 + 16, -8, 8, 1 / 128); F("wd1", FF, D, s0 + 17, -8, 8, 1 / 256)
    F("g11", 1, D, s0 + 18, 2, 6, 0.25); F("g21", 1, D, s0 + 19, 2, 6, 0.25)
    F("kp1", LP, DKV, s0 + 20, -8, 8, 0.125); F("vp1", LP, DKV, s0 + 21, -8, 8, 0.125)
    F("valid", 1, LP, s0 + 22, 0, 1, 1.0)
    P.B("kx1", LP, DKV); P.B("vx1", LP, DKV)
    n_valid = 200
    data["rq_self"], data["rq_cross"] = rope_table(n_valid + np.arange(S)), rope_table(np.arange(S))
    P.B("rq_self", S, DH, data["rq_self"]); P.B("rq_cross", S, DH, data["rq_cross"])
    sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
    op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
    # program: KV projection of the cross layer (once per chunk), self layer, cross layer
    P.mark("start")
    P.gemm("kp1", "wkx1", "kx1", *tiles["kv"]); P.gemm("vp1", "wvx1", "vx1", *tiles["kv"])
    P.mark("kvproj")
    op(f"softhier.add %hin, %hin -> %h {{{P.cl}}} : {T['hin']}, {T['hin']} -> {T['h']}")   # h = 2 hin (a copy with a known scale)
    names0 = dict(g1="g10", g2="g20", wqkv="wqkv0", wo="wo0", wgu="wgu0", wd="wd0", kp="kp0", vp="vp0")
    _layer_ops(P, "0", "self", names0, tiles=tiles)
    P.mark("layer0")
    P.dump("h", 301, "H0"); P.dump("o", 302, "O0"); P.dump("qkv", 303, "QKV0")
    names1 = dict(g1="g11", g2="g21", wq="wq1", wkx="wkx1", wvx="wvx1", wo="wo1", wgu="wgu1", wd="wd1", kx="kx1", vx="vx1")
    _layer_ops(P, "1", "cross", names1, tiles=tiles)
    P.mark("layer1")
    P.dump("h", 304, "H1"); P.dump("o", 305, "O1"); P.dump("kx1", 306, "KX1"); P.dump("m", 307, "M1")
    # host reference on the numpy twin
    h = _r16(2 * data["hin"].astype(np.float32))
    h, it0 = np_layer(h, data, 0, data["kp0"], data["vp0"], data["valid"], data["rq_self"], data["rq_self"])
    ref = {"H0": it0["H"], "O0": it0["O"]}
    kx1, vx1 = np_kv_proj(data, 1, data["kp1"], data["vp1"])
    h, it1 = np_layer(h, data, 1, None, None, data["valid"], data["rq_cross"], None, kx1, vx1)
    ref.update({"H1": it1["H"], "O1": it1["O"], "KX1": kx1, "M1": it1["M"]})
    # QKV0 after RoPE (q, k rotated in place, v untouched)
    xn0 = np_rmsnorm(_r16(2 * data["hin"].astype(np.float32)), data["g10"].astype(np.float32))
    qkv = _r16(xn0 @ data["wqkv0"].astype(np.float32))
    ref["QKV0"] = np.concatenate([np_rope(qkv[:, :DQ], data["rq_self"]), np_rope(qkv[:, DQ:DQ + DKV], data["rq_self"]), qkv[:, DQ + DKV:]], axis=1)
    return P.module("expert_layer_test"), P.pre, ref


def emit_op_test(which: str = "attn", seed: int = 3, cluster: int = -1, nsamples: int = 128, cols: int = DQ) -> tuple[str, dict, dict]:
    """Kernel-level test of one op on preloaded LCG data (bisection aid): which = attn (self: prefix + own causal
    keys + validity row) | xattn (cross: prefix only) | rmsnorm | rope | silu | axpy. Returns (mlir, preload, ref)."""
    P = _Prog(cluster, nsamples)
    T, op = P.T, P.op
    s0 = seed * 100
    d = {}

    def F(name, rows, cols, sd, lo, hi, scale):
        d[name] = lcg.fill_fp16(rows, cols, sd, lo, hi, scale)
        P.B(name, rows, cols, d[name])
    ref = {}
    if which in ("attn", "xattn"):
        F("q", S, DQ, s0 + 1, -8, 8, 0.125); F("kp", LP, DKV, s0 + 2, -8, 8, 0.125); F("vp", LP, DKV, s0 + 3, -8, 8, 0.125)
        F("k", S, DKV, s0 + 4, -8, 8, 0.125); F("v", S, DKV, s0 + 5, -8, 8, 0.125); F("valid", 1, LP, s0 + 6, 0, 1, 1.0)
        P.B("o", S, DQ)
        sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
        op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
        P.mark("start")
        if which == "attn":
            op(f"softhier.cross_attention %q, %kp, %vp own %k, %v valid %valid -> %o {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} "
               f": {T['q']}, {T['kp']}, {T['vp']} own {T['k']}, {T['v']} valid {T['valid']} -> {T['o']}")
            ref["O"] = np_attention(d["q"], d["kp"], d["vp"], d["k"], d["v"], d["valid"])
        else:
            op(f"softhier.cross_attention %q, %kp, %vp valid %valid -> %o {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} "
               f": {T['q']}, {T['kp']}, {T['vp']} valid {T['valid']} -> {T['o']}")
            ref["O"] = np_attention(d["q"], d["kp"], d["vp"], None, None, d["valid"])
        P.mark("op")
        P.dump("o", 320, "O")
    else:
        C = cols   # default 960 columns: a multiple of the head size for the RoPE test
        F("x", S, C, s0 + 1, -16, 16, 0.125); F("b", S, C, s0 + 2, -16, 16, 0.125); F("g", 1, C, s0 + 3, 2, 6, 0.25)
        if which == "rope":
            d["tab"] = rope_table(np.arange(S) + 7); P.B("tab", S, DH, d["tab"])
        P.B("y", S, C)
        sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
        op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
        P.mark("start")
        x, b, g = d["x"].astype(np.float32), d["b"].astype(np.float32), d["g"].astype(np.float32)
        if which == "rmsnorm":
            op(f"softhier.rmsnorm %x, %g -> %y {{eps = {RMS_EPS:.1e} : f32, {P.cl}}} : {T['x']}, {T['g']} -> {T['y']}")
            ref["Y"] = np_rmsnorm(x, g)
        elif which == "rope":
            op(f"softhier.rope %x, %tab -> %y {{head_dim = {DH} : i32, {P.cl}}} : {T['x']}, {T['tab']} -> {T['y']}")
            ref["Y"] = np_rope(x, d["tab"], DH)
        elif which == "silu":
            op(f"softhier.silu_mul %x, %b -> %y {{{P.cl}}} : {T['x']}, {T['b']} -> {T['y']}")
            ref["Y"] = x / (1 + np.exp(-x)) * b
        elif which == "silu_view":   # the layer's shapes: gate | up halves of a [50, 4096] buffer, output [50, 2048]
            F("gu", S, 2 * FF, s0 + 9, -16, 16, 0.125); P.B("m", S, FF)
            P.view("ga", "gu", S, FF, 2 * FF, 0); P.view("up", "gu", S, FF, 2 * FF, FF)
            op(f"softhier.silu_mul %ga, %up -> %m {{{P.cl}}} : {T['ga']}, {T['up']} -> {T['m']}")
            gu = d["gu"].astype(np.float32); g_, u_ = gu[:, :FF], gu[:, FF:]
            ref["M"] = g_ / (1 + np.exp(-g_)) * u_
            P.mark("op"); P.dump("m", 322, "M")
            return P.module("expert_op_silu_view"), P.pre, ref
        elif which == "axpy":
            op(f"softhier.axpy %x, %b -> %y {{alpha = -0.1 : f32, {P.cl}}} : {T['x']}, {T['b']} -> {T['y']}")
            ref["Y"] = x - 0.1 * b
        P.mark("op")
        P.dump("y", 321, "Y")
    return P.module(f"expert_op_{which}"), P.pre, ref


def emit_flow(npz, steps: int = STEPS, layers: int = LAYERS, cluster: int = -1, fmt_steps: list[str] | None = None,
              profile: bool = False, dumps: tuple[str, ...] = ("X",), nsamples: int = 64, tiles=TILES) -> tuple[str, dict]:
    """Step 2/3: the flow loop. Returns (mlir, preload). layers must be even (self/cross pairs).
    dumps: X (x_t after every step, all 1600 elements), A (final actions), EMB / H / O (step-loop intermediates, sampled,
    tagged with step*16+layer so the host can pick step 0). fmt_steps: per-step RedMulE format of every weight GEMM
    inside the step loop (R4's experiment hook; fp8 steps are plumbing only, the operands stay fp16 in memory).
    profile: a mark after every op group of every layer (per-op timing breakdown)."""
    data = np.load(npz) if isinstance(npz, (str, Path)) else npz
    W = {k[2:]: data[k] for k in data if k.startswith("p_")}
    assert layers % 2 == 0 and layers <= LAYERS      # layers == 0: the step tail only (debugging)
    if fmt_steps is not None:
        assert len(fmt_steps) == steps, (len(fmt_steps), steps)
    P = _Prog(cluster, nsamples)
    T, op = P.T, P.op
    _activations(P)
    for nm in ("x0", "wa", "ba", "wti", "tb", "wto", "bto", "gf", "wout", "bout", "valid", "rq_self", "rq_cross"):
        P.B(nm, *W[nm].shape, W[nm])
    P.B("zero", S, AD, np.zeros((S, AD), np.float16))
    # parameter slabs: one per (self, cross) pair at a constant stride; the cross layer's projected KV lives in the slab too
    SELF = [("wqkv", D, DQ + 2 * DKV), ("wo", DQ, D), ("wgu", D, 2 * FF), ("wd", FF, D), ("g1", 1, D), ("g2", 1, D), ("kp", LP, DKV), ("vp", LP, DKV)]
    CROSS = [("wq", D, DQ), ("wkx", DKV, DKV), ("wvx", DKV, DKV), ("wo", DQ, D), ("wgu", D, 2 * FF), ("wd", FF, D), ("g1", 1, D), ("g2", 1, D),
             ("kp", LP, DKV), ("vp", LP, DKV), ("kx", LP, DKV), ("vx", LP, DKV)]
    pair0, stride = {}, 0
    for p in range(layers // 2):
        begin = P.e.next_off
        for L, spec, sfx in ((2 * p, SELF, "s"), (2 * p + 1, CROSS, "c")):
            for nm, r, c in spec:
                arr = W.get(f"{nm}{L}")
                off = P.slot(f"{nm}_{sfx}", r, c, arr)
                if p == 0:
                    pair0[f"{nm}_{sfx}"] = off
        if p == 0:
            stride = P.e.next_off - begin
    sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
    op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
    op("%c0 = arith.constant 0 : index"); op("%c1 = arith.constant 1 : index"); op("%c2 = arith.constant 2 : index")
    op(f"%c16 = arith.constant 16 : index"); op(f"%cP = arith.constant {layers // 2} : index"); op(f"%cS = arith.constant {steps} : index")

    def slab(sfx_names):
        for nm in sfx_names:
            op(f"%{nm} = softhier.hbm_buffer %p {{offset = {pair0[nm]} : i32, stride = {stride} : i32}} : {T[nm]}")
    P.mark("start")
    # once per chunk: the cross layers' prefix KV through their 320 -> 320 k/v projections (step-invariant)
    if layers:
        op("scf.for %p = %c0 to %cP step %c1 {")
        slab(["kp_c", "vp_c", "wkx_c", "wvx_c", "kx_c", "vx_c"])
        P.gemm("kp_c", "wkx_c", "kx_c", *tiles["kv"]); P.gemm("vp_c", "wvx_c", "vx_c", *tiles["kv"])
        op("}")
    P.mark("kvproj")
    op(f"softhier.add %x0, %zero -> %x {{{P.cl}}} : {T['x0']}, {T['zero']} -> {T['x']}")     # x = x0 (copy: x0 stays intact)
    op("scf.for %s = %c0 to %cS step %c1 {")
    op("%s16 = arith.muli %s, %c16 : index")
    fs = fmt_steps
    # suffix embedding: e = x Wa + ba; e1 = silu(e Wti + tb[s]); h = e1 Wto + bto
    P.gemm("x", "wa", "e", *tiles["a"], step="%s", fmt_steps=fs)
    op(f"softhier.add_bias %e, %ba -> %e {{{P.cl}}} : {T['e']}, {T['ba']} -> {T['e']}")
    P.gemm("e", "wti", "e1", *tiles["t"], step="%s", fmt_steps=fs)
    op(f"%tbs = softhier.view %tb, %s {{stride = {D} : i32}} : {T['tb']} -> {P.mt(1, D, D, 0)}")
    op(f"softhier.add_bias %e1, %tbs -> %e1 {{{P.cl}}} : {T['e1']}, {P.mt(1, D, D, 0)} -> {T['e1']}")
    op(f"softhier.silu_mul %e1 -> %e1 {{{P.cl}}} : {T['e1']} -> {T['e1']}")
    P.gemm("e1", "wto", "h", *tiles["t"], step="%s", fmt_steps=fs)
    op(f"softhier.add_bias %h, %bto -> %h {{{P.cl}}} : {T['h']}, {T['bto']} -> {T['h']}")
    if profile:
        P.mark("emb", "%s")
    if "EMB" in dumps:
        P.dump("h", 310, "EMB", "%s")
    if layers:
        op("scf.for %p = %c0 to %cP step %c1 {")
        op("%L0 = arith.muli %p, %c2 : index"); op("%L1 = arith.addi %L0, %c1 : index")
        op("%i0 = arith.addi %s16, %L0 : index"); op("%i1 = arith.addi %s16, %L1 : index")
        slab([f"{nm}_s" for nm, _, _ in SELF] + [f"{nm}_c" for nm, _, _ in CROSS if nm not in ("wkx", "wvx", "kp", "vp")])
        _layer_ops(P, "s", "self", {k: f"{k}_s" for k in ("g1", "g2", "wqkv", "wo", "wgu", "wd", "kp", "vp")}, step="%s", fmt_steps=fs,
                   tiles=tiles, prof=profile, pidx="%i0")
        if "O" in dumps:
            P.dump("o", 311, "O", "%i0")
        if "H" in dumps:
            P.dump("h", 312, "H", "%i0")
        _layer_ops(P, "c", "cross", {k: f"{k}_c" for k in ("g1", "g2", "wq", "wo", "wgu", "wd", "kx", "vx")}, step="%s", fmt_steps=fs,
                   tiles=tiles, prof=profile, pidx="%i1")
        if "O" in dumps:
            P.dump("o", 311, "O", "%i1")
        if "H" in dumps:
            P.dump("h", 312, "H", "%i1")
        op("}")
    op(f"softhier.rmsnorm %h, %gf -> %fin {{eps = {RMS_EPS:.1e} : f32, {P.cl}}} : {T['h']}, {T['gf']} -> {T['fin']}")
    P.gemm("fin", "wout", "vt", *tiles["out"], step="%s", fmt_steps=fs)
    op(f"softhier.add_bias %vt, %bout -> %vt {{{P.cl}}} : {T['vt']}, {T['bout']} -> {T['vt']}")
    op(f"softhier.axpy %x, %vt -> %x {{alpha = {-1.0 / steps!r} : f32, {P.cl}}} : {T['x']}, {T['vt']} -> {T['x']}")
    P.mark("step", "%s")
    if "X" in dumps:
        P.dump("x", 0, "X", "%s", all_=True)
        op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
        P.mark("xdump", "%s")
    op("}")
    P.mark("end")
    if "A" in dumps:
        P.dump("x", 0, "A", all_=True)
    return P.module("smolvla_expert_flow"), P.pre


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--ckpt", default="/app/models/smolvla_base/model.safetensors")
    pp.add_argument("--ref", default="/app/models/smolvla_base/expert_ref.npz")
    pp.add_argument("--out", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("npz", nargs="?")
    pe.add_argument("--layer-test", action="store_true")
    pe.add_argument("--steps", type=int, default=STEPS)
    pe.add_argument("--layers", type=int, default=LAYERS)
    pe.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.ckpt, a.ref, a.out)
    else:
        import sys
        if a.layer_test:
            sys.stdout.write(emit_layer_test()[0])
        else:
            sys.stdout.write(emit_flow(a.npz, a.steps, a.layers, profile=a.profile)[0])


if __name__ == "__main__":
    main()
