"""Test-time adaptation of a LoRA adapter on SmolVLA's action expert: one SGD / Adam step as a softhier program
(forward with LoRA, MSE loss on the predicted velocity, backward through the expert into the LoRA A / B only, optimizer
on fp32 masters), the data-parallel variant (one sample per cluster, gradients summed with the in-network REDADD) and
the float64 torch references. Ledger: docs/TTT.md. Runner: tests/gvsoc/ttt.py.

LoRA (r = 16, scale s = alpha / r = 2) on three projections of every expert layer (the TTT plan's choice):
    q  = xn Wq + s (xn Aq) Bq          Aq [720, 16]   Bq [16, 960]
    ao = o  Wo + s (o  Ao) Bo          Ao [960, 16]   Bo [16, 720]
    f2 = m  Wd + s (m  Ad) Bd          Ad [2048, 16]  Bd [16, 720]
98 048 parameters per layer, 1.57 M for 16 layers, kept as one flat arena (fp16 copy for the GEMMs, fp32 master,
fp16 gradients, Adam moments in fp32), layer L at element L * 98048 in the order Aq Bq Ao Bo Ad Bd.

Backward of one layer (dh2 = gradient of the layer output; every frozen-weight product is formed TRANSPOSED so RedMulE
reads the weights in their stored [in, out] layout and only the 50-row activations are transposed: dX^T = W dY^T):
    dh2^T;  dBd = td^T dh2;  ud^T = s Bd dh2^T;  dAd = (ud^T m)^T;  dm^T = Wd dh2^T + Ad ud^T;  dm
    dg, du = silu_mul_bwd(g, u, dm);  dxn2^T = Wgu [dg | du]^T;  dh1 = rmsnorm_bwd(h1, g2, dxn2) + dh2
    dBo = to^T dh1;  uo^T = s Bo dh1^T;  dAo = (uo^T o)^T;  do^T = Wo dh1^T + Ao uo^T;  do
    dq, dk, dv = attention_bwd(q, [kp; k], [vp; v], o, do)   (prefix KV frozen; cross layers: dq only)
    dq, dk = rope^T (the RoPE table with -sin);  dBq = tq^T dq;  uq^T = s Bq dq^T;  dAq = (uq^T xn)^T
    dxn^T = Wqkv [dq | dk | dv]^T + Aq uq^T;  dh = rmsnorm_bwd(h, g1, dxn) + dh1
(td = s m Ad, to = s o Ao, tq = s xn Aq are kept from the forward, as are h, xn, q|k|v after RoPE, o, h1, g|u, m.)
"""
from __future__ import annotations

import numpy as np

from softhier_mlir.frontend import smolvla_expert as E
from softhier_mlir.frontend.smolvla_expert import (AD, D, DH, DKV, DQ, FF, GROUP, H, HKV, LAYERS, LP, RMS_EPS, S, SCALE,
                                                    _Prog, prefix_tok, rope_table)
from softhier_mlir.sim.preload import sentinel_array
from softhier_mlir.testing import lcg


def _torch():
    """torch for the references: the system interpreter's CPU build when this venv has none"""
    try:
        import torch
    except ImportError:
        import sys
        sys.path.append("/usr/local/lib/python3.10/dist-packages")
        import torch
    return torch


R, ALPHA = 16, 32.0
LS = ALPHA / R                                     # LoRA scale s
LORA = [("aq", D, R), ("bq", R, DQ), ("ao", DQ, R), ("bo", R, D), ("ad", FF, R), ("bd", R, D)]
LORA_OFF, _o = {}, 0
for _n, _r, _c in LORA:
    LORA_OFF[_n] = _o
    _o += _r * _c
LAYER_N = _o                                       # 98048 parameters per layer
ARENA_COLS = 64


def arena_rows(layers: int = LAYERS) -> int:
    return layers * LAYER_N // ARENA_COLS


# saved activations of one layer (a "slab", one per layer + the final residual): name, cols
SLAB = [("hin", D), ("xn", D), ("tq", R), ("qkv", DQ + 2 * DKV), ("o", DQ), ("to", R), ("h1", D), ("gu", 2 * FF), ("m", FF), ("td", R)]
# GEMM tiles (tm, tn, tk) of the training program; forward ones from smolvla_expert.TILES
TT = {"lq": (S, R, D), "lqb": (S, 64, R), "lo": (S, R, DQ), "lob": (S, 48, R), "ld": (S, R, 512), "ldb": (S, 48, R),
      "dB720": (R, 48, S), "dB960": (R, 64, S), "uT720": (R, S, D), "uT960": (R, S, DQ),
      "aT2048": (R, 128, S), "aT960": (R, 64, S), "aT720": (R, 48, S),
      "dmT": (128, S, 240), "dmTa": (128, S, R), "dxn2T": (48, S, 512), "doT": (64, S, 240), "doTa": (64, S, R),
      "dxnT": (48, S, 320), "dxnTa": (48, S, R), "dfin": (S, 48, AD)}


class TProg(_Prog):
    """_Prog with f32 buffers, indexed buffers / views and the op spellings of the training ops."""

    def B(self, name, rows, cols, arr=None, elem="f16"):
        esz = 4 if elem == "f32" else 2
        off = self.e.next_off
        if arr is not None:
            assert arr.shape == (rows, cols), (name, arr.shape, rows, cols)
            self.pre[off] = np.ascontiguousarray(arr)
        self.e.next_off += (rows * cols * esz + 4095) & ~4095
        self.off[name] = off
        self.T[name] = f'memref<{rows}x{cols}x{elem}, "{self.sp}">'
        self.e.lines.append(f"    %{name} = softhier.hbm_buffer {{offset = {off} : i32}} : {self.T[name]}")
        return name

    def ib(self, name, idx, offset, stride, rows, cols, elem="f16"):
        """indexed buffer (declared in place): offset + idx * stride"""
        self.T[name] = f'memref<{rows}x{cols}x{elem}, "{self.sp}">'
        self.op(f"%{name} = softhier.hbm_buffer %{idx} {{offset = {offset} : i32, stride = {stride} : i32}} : {self.T[name]}")
        return name

    def vw(self, name, src, rows, cols, ld, eoff, idx=None, stride=0, elem="f16"):
        t = f'memref<{rows}x{cols}x{elem}, strided<[{ld}, 1], offset: {eoff}>, "{self.sp}">'
        ix = f", %{idx}" if idx else ""
        st = f" {{stride = {stride} : i32}}" if idx else ""
        self.op(f"%{name} = softhier.view %{src}{ix}{st} : {self.T[src]} -> {t}")
        self.T[name] = t
        return name

    def g(self, x, w, z, tiles, acc=False, cl=None):
        tm, tn, tk = tiles
        self.op(f'softhier.gemm %{x}, %{w} into %{z} {{fmt = "fp16", tile_m = {tm} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, '
                f'pipeline{", accumulate" if acc else ""}, {cl or self.cl}}} : {self.T[x]}, {self.T[w]}, {self.T[z]}')

    def gt(self, x, w, z, scr, tiles, tx=False, tw=False, acc=False):
        tm, tn, tk = tiles
        fl = ", ".join(f for f, on in (("trans_x", tx), ("trans_w", tw), ("accumulate", acc)) if on)
        self.op(f"softhier.gemm_t %{x}, %{w} into %{z} scratch %{scr} {{{fl}, tile_m = {tm} : i32, tile_n = {tn} : i32, "
                f"tile_k = {tk} : i32, pipeline, {self.cl}}} : {self.T[x]}, {self.T[w]}, {self.T[z]}, {self.T[scr]}")

    def tr(self, src, dst):
        self.op(f"softhier.transpose %{src} -> %{dst} {{{self.cl}}} : {self.T[src]} -> {self.T[dst]}")

    def scale(self, x, y, s):
        self.op(f"softhier.scale %{x} -> %{y} {{scale = {s!r} : f32, {self.cl}}} : {self.T[x]} -> {self.T[y]}")

    def add(self, a, b, y):
        self.op(f"softhier.add %{a}, %{b} -> %{y} {{{self.cl}}} : {self.T[a]}, {self.T[b]} -> {self.T[y]}")

    def rms(self, x, g, y):
        self.op(f"softhier.rmsnorm %{x}, %{g} -> %{y} {{eps = {RMS_EPS:.1e} : f32, {self.cl}}} : {self.T[x]}, {self.T[g]} -> {self.T[y]}")

    def rms_bwd(self, x, g, dy, dx, res=None):
        r, rt = (f" res %{res}", f" res {self.T[res]}") if res else ("", "")
        self.op(f"softhier.rmsnorm_bwd %{x}, %{g}, %{dy}{r} -> %{dx} {{eps = {RMS_EPS:.1e} : f32, {self.cl}}} : "
                f"{self.T[x]}, {self.T[g]}, {self.T[dy]}{rt} -> {self.T[dx]}")

    def rope(self, x, tab):
        self.op(f"softhier.rope %{x}, %{tab} -> %{x} {{head_dim = {DH} : i32, {self.cl}}} : {self.T[x]}, {self.T[tab]} -> {self.T[x]}")

    def dumps(self, name, seed, tag, n=None):
        n = n or self.nsamples
        self.op(f'softhier.dump_samples %{name} {{seed = {seed} : i32, n = {n} : i32, tag = "{tag}"}} : {self.T[name]}')


# ----------------------------------------------------------------------------- program pieces
def _layer_fwd(P: TProg, kind: str, sl: dict, nxt_hin: str, W: dict, Lw: dict, kv: dict):
    """Forward of one layer with LoRA, writing the saved activations into the slab `sl` (name map) and the layer output
    into `nxt_hin`. W: frozen weight names, Lw: LoRA (fp16 arena views), kv: prefix / cross KV names."""
    op, T, cl = P.op, P.T, P.cl
    P.rms(sl["hin"], W["g1"], sl["xn"])
    if kind == "self":
        P.g(sl["xn"], W["wqkv"], sl["qkv"], E.TILES["qkv"])
    else:
        P.g(sl["xn"], W["wq"], sl["q"], E.TILES["qkv"])
    P.g(sl["xn"], Lw["aq"], sl["tq"], TT["lq"]); P.scale(sl["tq"], sl["tq"], LS)
    P.g(sl["tq"], Lw["bq"], sl["q"], TT["lqb"], acc=True)
    if kind == "self":
        P.rope(sl["q"], "rq_self"); P.rope(sl["k"], "rq_self")
        op(f"softhier.cross_attention %{sl['q']}, %{kv['kp']}, %{kv['vp']} own %{sl['k']}, %{sl['v']} mask %tok -> %{sl['o']} "
           f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : {T[sl['q']]}, {T[kv['kp']]}, {T[kv['vp']]} "
           f"own {T[sl['k']]}, {T[sl['v']]} mask {T['tok']} -> {T[sl['o']]}")
    else:
        P.rope(sl["q"], "rq_cross")
        op(f"softhier.cross_attention %{sl['q']}, %{kv['kx']}, %{kv['vx']} mask %tok -> %{sl['o']} "
           f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : {T[sl['q']]}, {T[kv['kx']]}, {T[kv['vx']]} "
           f"mask {T['tok']} -> {T[sl['o']]}")
    P.g(sl["o"], W["wo"], "ao", E.TILES["o"])
    P.g(sl["o"], Lw["ao"], sl["to"], TT["lo"]); P.scale(sl["to"], sl["to"], LS)
    P.g(sl["to"], Lw["bo"], "ao", TT["lob"], acc=True)
    P.add(sl["hin"], "ao", sl["h1"])
    P.rms(sl["h1"], W["g2"], "xn2")
    P.g("xn2", W["wgu"], sl["gu"], E.TILES["gu"])
    op(f"softhier.silu_mul %{sl['ga']}, %{sl['up']} -> %{sl['m']} {{{cl}}} : {T[sl['ga']]}, {T[sl['up']]} -> {T[sl['m']]}")
    P.g(sl["m"], W["wd"], "f2", E.TILES["d"])
    P.g(sl["m"], Lw["ad"], sl["td"], TT["ld"]); P.scale(sl["td"], sl["td"], LS)
    P.g(sl["td"], Lw["bd"], "f2", TT["ldb"], acc=True)
    P.add(sl["h1"], "f2", nxt_hin)


def _layer_bwd(P: TProg, kind: str, sl: dict, W: dict, Lw: dict, Gw: dict, kv: dict, dh_out: str, dh_in: str, marks: bool = False, pidx=None):
    """Backward of one layer: dh_out (gradient of the layer output) -> dh_in (of its input); LoRA grads into Gw."""
    op, T, cl = P.op, P.T, P.cl
    mk = (lambda t: P.mark(t, pidx)) if marks else (lambda t: None)
    nq = DQ + 2 * DKV if kind == "self" else DQ
    # MLP down projection (LoRA d)
    P.tr(dh_out, "dhT")
    P.gt(sl["td"], dh_out, Gw["bd"], "gscr", TT["dB720"], tx=True)
    P.g(Lw["bd"], "dhT", "uT", TT["uT720"]); P.scale("uT", "uT", LS)
    P.g("uT", sl["m"], "aT2048", TT["aT2048"]); P.tr("aT2048", Gw["ad"])
    P.g(W["wd"], "dhT", "dmT", TT["dmT"]); P.g(Lw["ad"], "uT", "dmT", TT["dmTa"], acc=True)
    P.tr("dmT", "dm")
    mk("bdown")
    op(f"softhier.silu_mul_bwd %{sl['ga']}, %{sl['up']}, %dm -> %dga, %dup {{{cl}}} : {T[sl['ga']]}, {T[sl['up']]}, {T['dm']} -> {T['dga']}, {T['dup']}")
    mk("bsilu")
    P.tr("dgu", "dguT")
    P.g(W["wgu"], "dguT", "dxnT", TT["dxn2T"]); P.tr("dxnT", "dxn")
    P.rms_bwd(sl["h1"], W["g2"], "dxn", "dh1", res=dh_out)
    mk("bgateup")
    # o projection (LoRA o)
    P.tr("dh1", "dhT")
    P.gt(sl["to"], "dh1", Gw["bo"], "gscr", TT["dB720"], tx=True)
    P.g(Lw["bo"], "dhT", "uT", TT["uT720"]); P.scale("uT", "uT", LS)
    P.g("uT", sl["o"], "aT960", TT["aT960"]); P.tr("aT960", Gw["ao"])
    P.g(W["wo"], "dhT", "doT", TT["doT"]); P.g(Lw["ao"], "uT", "doT", TT["doTa"], acc=True)
    P.tr("doT", "do")
    mk("boproj")
    # attention
    if kind == "self":
        op(f"softhier.attention_bwd %{sl['q']}, %{kv['kp']}, %{kv['vp']} own %{sl['k']}, %{sl['v']} mask %tok, %{sl['o']}, %do -> %dq "
           f"grads %dk, %dv scratch %ascr {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : "
           f"{T[sl['q']]}, {T[kv['kp']]}, {T[kv['vp']]} own {T[sl['k']]}, {T[sl['v']]} mask {T['tok']}, {T[sl['o']]}, {T['do']} -> {T['dq']} "
           f"grads {T['dk']}, {T['dv']} scratch {T['ascr']}")
    else:
        op(f"softhier.attention_bwd %{sl['q']}, %{kv['kx']}, %{kv['vx']} mask %tok, %{sl['o']}, %do -> %dq "
           f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {cl}}} : "
           f"{T[sl['q']]}, {T[kv['kx']]}, {T[kv['vx']]} mask {T['tok']}, {T[sl['o']]}, {T['do']} -> {T['dq']}")
    mk("battn")
    # RoPE^T, q projection (LoRA q), input norm
    if kind == "self":
        P.rope("dq", "rb_self"); P.rope("dk", "rb_self")
        P.tr("dqkv", "dqkvT")
    else:
        P.rope("dq", "rb_cross")
        P.tr("dq", "dqT")
    P.gt(sl["tq"], "dq", Gw["bq"], "gscr", TT["dB960"], tx=True)
    P.g(Lw["bq"], "dqT", "uT", TT["uT960"]); P.scale("uT", "uT", LS)
    P.g("uT", sl["xn"], "aT720", TT["aT720"]); P.tr("aT720", Gw["aq"])
    if kind == "self":
        P.g(W["wqkv"], "dqkvT", "dxnT", TT["dxnT"])
    else:
        P.g(W["wq"], "dqT", "dxnT", TT["dxnT"])
    P.g(Lw["aq"], "uT", "dxnT", TT["dxnTa"], acc=True)
    P.tr("dxnT", "dxn")
    P.rms_bwd(sl["hin"], W["g1"], "dxn", dh_in, res="dh1")
    mk("bqproj")
    del nq


def _bwd_buffers(P: TProg):
    for nm, r, c in [("dhT", D, S), ("uT", R, S), ("aT2048", R, FF), ("aT960", R, DQ), ("aT720", R, D), ("dmT", FF, S), ("dm", S, FF),
                     ("dgu", S, 2 * FF), ("dguT", 2 * FF, S), ("dxnT", D, S), ("dxn", S, D), ("dh1", S, D), ("doT", DQ, S), ("do", S, DQ),
                     ("dqkv", S, DQ + 2 * DKV), ("dqkvT", DQ + 2 * DKV, S), ("ascr", S, 2 * H * DH), ("gscr", 2 * FF, S)]:
        P.B(nm, r, c)
    P.vw("dga", "dgu", S, FF, 2 * FF, 0); P.vw("dup", "dgu", S, FF, 2 * FF, FF)
    P.vw("dq", "dqkv", S, DQ, DQ + 2 * DKV, 0); P.vw("dk", "dqkv", S, DKV, DQ + 2 * DKV, DQ); P.vw("dv", "dqkv", S, DKV, DQ + 2 * DKV, DQ + DKV)
    P.vw("dqT", "dqkvT", DQ, S, S, 0)


def _slab_layout():
    off, out = 0, {}
    for nm, c in SLAB:
        out[nm] = off
        off += (S * c * 2 + 4095) & ~4095
    return out, off


def _slab_decl(P: TProg, idx: str, sfx: str, base: int, stride: int, lay: dict) -> dict:
    """indexed slab buffers of layer `idx` (an index SSA name) -> name map"""
    sl = {}
    for nm, c in SLAB:
        sl[nm] = P.ib(f"{nm}{sfx}", idx, base + lay[nm], stride, S, c)
    nq = DQ + 2 * DKV
    sl["q"] = P.vw(f"q{sfx}", sl["qkv"], S, DQ, nq, 0); sl["k"] = P.vw(f"k{sfx}", sl["qkv"], S, DKV, nq, DQ)
    sl["v"] = P.vw(f"v{sfx}", sl["qkv"], S, DKV, nq, DQ + DKV)
    sl["ga"] = P.vw(f"ga{sfx}", sl["gu"], S, FF, 2 * FF, 0); sl["up"] = P.vw(f"up{sfx}", sl["gu"], S, FF, 2 * FF, FF)
    return sl


def _lora_views(P: TProg, arena: str, idx: str, sfx: str, elem="f16") -> dict:
    return {nm: P.vw(f"{arena}_{nm}{sfx}", arena, r, c, c, LORA_OFF[nm], idx=idx, stride=LAYER_N, elem=elem) for nm, r, c in LORA}


def lora_init(layers: int, seed: int) -> np.ndarray:
    """fp16 arena [layers * 98048] of LCG LoRA weights: A in [-8, 8] / 256, B in [-8, 8] / 512 (B != 0 so that both A and B
    receive gradients at the first step; standard LoRA init has B = 0 and dA = 0)."""
    out = np.zeros(layers * LAYER_N, np.float16)
    for L in range(layers):
        for i, (nm, r, c) in enumerate(LORA):
            sc = 1 / 256 if nm[0] == "a" else 1 / 512
            a = lcg.fill_fp16(r, c, seed * 1000 + L * 10 + i, -8, 8, sc)
            out[L * LAYER_N + LORA_OFF[nm]: L * LAYER_N + LORA_OFF[nm] + r * c] = a.reshape(-1)
    return out


def emit_ttt(Wt: dict, layers: int, head: bool, lora16: np.ndarray, *, lr: float = 1e-3, opt: str = "sgd", loss_scale: float = 1.0,
             step: int = 0, nsamples: int = 128, cluster: int = -1, profile: bool = False, dump_layers=(0,), x_in=None, target=None,
             h_in=None, dh_in=None, extra_dumps=()) -> tuple[str, dict, dict]:
    """The TTT step program. Wt: weights in library layout (smolvla_expert.to_library_layout names, numpy fp16) incl. the
    host tables (rq_self, rq_cross, tok, kp{L}, vp{L}, and for head=True: wa ba wti tb wto bto gf wout bout).
    head=True: x_t (x_in [50, 32]) -> suffix embedding (time row `step`) -> layers -> final norm -> v; MSE(v, target) with the
    gradient scaled by loss_scale; head=False: h_in [50, 720] -> layers, the output gradient dh_in [50, 720] is given.
    Then backward into the LoRA arena and one optimizer step (opt = sgd | adam, lr; gradients divided by loss_scale).
    Returns (mlir, preload, info) with info = buffer offsets for the host-side checks."""
    assert layers % 2 == 0
    P = TProg(cluster, nsamples)
    T, op = P.T, P.op
    N = layers * LAYER_N
    rows = N // ARENA_COLS
    # ---- buffers: activations shared between layers, backward scratch, LoRA arenas, tables, weights, slabs
    for nm, r, c in [("ao", S, D), ("xn2", S, D), ("f2", S, D), ("dhA", S, D), ("dhB", S, D)]:
        P.B(nm, r, c)
    _bwd_buffers(P)
    w32 = np.ascontiguousarray(lora16.astype(np.float32).reshape(rows, ARENA_COLS))
    P.B("w16", rows, ARENA_COLS, lora16.reshape(rows, ARENA_COLS))
    P.B("w32", rows, ARENA_COLS, w32, elem="f32")
    P.B("g16", rows, ARENA_COLS)
    if opt == "adam":
        P.B("m32", rows, ARENA_COLS, np.zeros_like(w32), elem="f32"); P.B("v32", rows, ARENA_COLS, np.zeros_like(w32), elem="f32")
    for nm in ("rq_self", "rq_cross"):
        P.B(nm, S, DH, Wt[nm])
    rb = lambda t: np.concatenate([t[:, :DH // 2], -t[:, DH // 2:]], axis=1).astype(np.float16)  # noqa: E731
    P.B("rb_self", S, DH, rb(Wt["rq_self"])); P.B("rb_cross", S, DH, rb(Wt["rq_cross"]))
    P.B("tok", 1, LP, Wt["tok"].astype(np.uint16), elem="i16")
    if head:
        for nm in ("wa", "ba", "wti", "tb", "wto", "bto", "gf", "wout", "bout"):
            P.B(nm, *Wt[nm].shape, Wt[nm])
        P.B("x", S, AD, x_in.astype(np.float16)); P.B("tgt", S, AD, target.astype(np.float16))
        for nm, r, c in [("e", S, D), ("e1", S, D), ("fin", S, D), ("vt", S, AD), ("dvt", S, AD), ("dfin", S, D)]:
            P.B(nm, r, c)
        P.B("lrows", S, 1, elem="f32")
    # slabs: one per layer + the final residual
    lay, sstride = _slab_layout()
    slab0 = P.e.next_off
    if not head:
        P.pre[slab0 + lay["hin"]] = h_in.astype(np.float16)
        P.B("dh_out", S, D, dh_in.astype(np.float16))
    P.e.next_off = slab0 + (layers + 1) * sstride
    for nm, off in (("hfin", slab0 + layers * sstride + lay["hin"]), ("h0", slab0 + lay["hin"])):
        T[nm] = f'memref<{S}x{D}xf16, "{P.sp}">'
        op(f"%{nm} = softhier.hbm_buffer {{offset = {off} : i32}} : {T[nm]}")
    # weights: per (self, cross) pair slabs at a constant stride (the flow program's layout)
    SELF = [("wqkv", D, DQ + 2 * DKV), ("wo", DQ, D), ("wgu", D, 2 * FF), ("wd", FF, D), ("g1", 1, D), ("g2", 1, D), ("kp", LP, DKV), ("vp", LP, DKV)]
    CROSS = [("wq", D, DQ), ("wkx", DKV, DKV), ("wvx", DKV, DKV), ("wo", DQ, D), ("wgu", D, 2 * FF), ("wd", FF, D), ("g1", 1, D), ("g2", 1, D),
             ("kp", LP, DKV), ("vp", LP, DKV), ("kx", LP, DKV), ("vx", LP, DKV)]
    pair0, pstride = {}, 0
    for p in range(layers // 2):
        begin = P.e.next_off
        for L, spec, sfx in ((2 * p, SELF, "s"), (2 * p + 1, CROSS, "c")):
            for nm, r, c in spec:
                off = P.slot(f"{nm}_{sfx}", r, c, Wt.get(f"{nm}{L}"))
                if p == 0:
                    pair0[f"{nm}_{sfx}"] = off
        if p == 0:
            pstride = P.e.next_off - begin
    sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
    op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
    for c in (0, 1, 2):
        op(f"%c{c} = arith.constant {c} : index")
    op(f"%cP = arith.constant {layers // 2} : index"); op(f"%cPm1 = arith.constant {layers // 2 - 1} : index")

    def wslab(names, sfx):
        out = {}
        for nm in names:
            op(f"%{nm}_{sfx} = softhier.hbm_buffer %p {{offset = {pair0[f'{nm}_{sfx}']} : i32, stride = {pstride} : i32}} : {T[f'{nm}_{sfx}']}")
            out[nm] = f"{nm}_{sfx}"
        return out
    P.mark("start")
    # ---- once per chunk: cross-layer KV projection (frozen)
    op("scf.for %p = %c0 to %cP step %c1 {")
    w = wslab(["kp", "vp", "wkx", "wvx", "kx", "vx"], "c")
    P.g(w["kp"], w["wkx"], w["kx"], E.TILES["kv"]); P.g(w["vp"], w["wvx"], w["vx"], E.TILES["kv"])
    op("}")
    P.mark("kvproj")
    # ---- forward
    if head:
        P.g("x", "wa", "e", E.TILES["a"])
        op(f"softhier.add_bias %e, %ba -> %e {{{P.cl}}} : {T['e']}, {T['ba']} -> {T['e']}")
        P.g("e", "wti", "e1", E.TILES["t"])
        P.vw("tbs", "tb", 1, D, D, step * D)
        op(f"softhier.add_bias %e1, %tbs -> %e1 {{{P.cl}}} : {T['e1']}, {T['tbs']} -> {T['e1']}")
        op(f"softhier.silu_mul %e1 -> %e1 {{{P.cl}}} : {T['e1']} -> {T['e1']}")
        P.g("e1", "wto", "h0", E.TILES["t"])
        op(f"softhier.add_bias %h0, %bto -> %h0 {{{P.cl}}} : {T['h0']}, {T['bto']} -> {T['h0']}")
        P.mark("emb")
    op("scf.for %p = %c0 to %cP step %c1 {")
    op("%L0 = arith.muli %p, %c2 : index"); op("%L1 = arith.addi %L0, %c1 : index"); op("%L2 = arith.addi %L0, %c2 : index")
    ws = wslab([n for n, _, _ in SELF], "s"); wc = wslab([n for n, _, _ in CROSS if n not in ("wkx", "wvx")], "c")
    s0, s1 = _slab_decl(P, "L0", "_0", slab0, sstride, lay), _slab_decl(P, "L1", "_1", slab0, sstride, lay)
    hnext = P.ib("hin_2", "L2", slab0 + lay["hin"], sstride, S, D)
    l0, l1 = _lora_views(P, "w16", "L0", "_f0"), _lora_views(P, "w16", "L1", "_f1")
    _layer_fwd(P, "self", s0, s1["hin"], ws, l0, {"kp": ws["kp"], "vp": ws["vp"]})
    if profile:
        P.mark("fself", "%L0")
    _layer_fwd(P, "cross", s1, hnext, wc, l1, {"kx": wc["kx"], "vx": wc["vx"]})
    if profile:
        P.mark("fcross", "%L1")
    op("}")
    if head:
        P.rms("hfin", "gf", "fin")
        P.g("fin", "wout", "vt", E.TILES["out"])
        op(f"softhier.add_bias %vt, %bout -> %vt {{{P.cl}}} : {T['vt']}, {T['bout']} -> {T['vt']}")
    P.mark("fwd")
    # ---- loss and the gradient entering the last layer
    if head:
        gs = 2.0 / (S * AD) * loss_scale
        op(f"softhier.mse_grad %vt, %tgt -> %dvt loss %lrows {{gscale = {gs!r} : f32, {P.cl}}} : {T['vt']}, {T['tgt']} -> {T['dvt']} loss {T['lrows']}")
        P.gt("dvt", "wout", "dfin", "gscr", TT["dfin"], tw=True)
        P.rms_bwd("hfin", "gf", "dfin", "dhA")
    else:
        P.scale("dh_out", "dhA", loss_scale)
    P.mark("loss")
    # ---- backward, pairs in reverse: cross 2p+1 (dhA -> dhB), self 2p (dhB -> dhA)
    op("scf.for %pr = %c0 to %cP step %c1 {")
    op("%p = arith.subi %cPm1, %pr : index")
    op("%L0 = arith.muli %p, %c2 : index"); op("%L1 = arith.addi %L0, %c1 : index")
    ws = wslab([n for n, _, _ in SELF], "s"); wc = wslab([n for n, _, _ in CROSS if n not in ("wkx", "wvx")], "c")
    s0, s1 = _slab_decl(P, "L0", "_b0", slab0, sstride, lay), _slab_decl(P, "L1", "_b1", slab0, sstride, lay)
    l0, l1 = _lora_views(P, "w16", "L0", "_b0"), _lora_views(P, "w16", "L1", "_b1")
    g0, g1 = _lora_views(P, "g16", "L0", "_g0"), _lora_views(P, "g16", "L1", "_g1")
    _layer_bwd(P, "cross", s1, wc, l1, g1, {"kx": wc["kx"], "vx": wc["vx"]}, "dhA", "dhB", marks=profile, pidx="%L1")
    if profile:
        P.mark("bcross", "%L1")
    _layer_bwd(P, "self", s0, ws, l0, g0, {"kp": ws["kp"], "vp": ws["vp"]}, "dhB", "dhA", marks=profile, pidx="%L0")
    if profile:
        P.mark("bself", "%L0")
    op("}")
    P.mark("bwd")
    # ---- optimizer on the whole arena
    inv = 1.0 / loss_scale
    if opt == "adam":
        op(f'softhier.optim_step %g16, %w32 -> %w16 moments %m32, %v32 {{kind = "adam", lr = {lr!r} : f32, inv_scale = {inv!r} : f32, '
           f'b1 = 0.9 : f32, b2 = 0.999 : f32, eps = 1.0e-8 : f32, bc1 = 0.1 : f32, bc2 = 0.001 : f32, {P.cl}}} : '
           f"{T['g16']}, {T['w32']} -> {T['w16']} moments {T['m32']}, {T['v32']}")
    else:
        op(f'softhier.optim_step %g16, %w32 -> %w16 {{kind = "sgd", lr = {lr!r} : f32, inv_scale = {inv!r} : f32, {P.cl}}} : '
           f"{T['g16']}, {T['w32']} -> {T['w16']}")
    P.mark("opt")
    # ---- dumps (after the timed region where possible: they read buffers nothing writes later)
    if head:
        P.dump("vt", 0, "V", all_=True)
        P.dumps("lrows", 5, "LOSS", n=64)
    for L in dump_layers:
        for i, (nm, r, c) in enumerate(LORA):
            off = L * LAYER_N + LORA_OFF[nm]
            vg = P.vw(f"dg_{nm}{L}", "g16", r, c, c, off)
            P.dumps(vg, 400 + 10 * L + i, f"G{nm.upper()}{L}")
            vw32 = P.vw(f"dw_{nm}{L}", "w32", r, c, c, off, elem="f32")
            P.dumps(vw32, 500 + 10 * L + i, f"W{nm.upper()}{L}")
    for nm in extra_dumps:
        P.dumps(nm, 600, nm.upper())
    info = {"rows": rows, "slab0": slab0, "sstride": sstride, "lay": lay, "w16": P.off["w16"], "g16": P.off["g16"]}
    return P.module("smolvla_ttt"), P.pre, info


# ----------------------------------------------------------------------------- the data a test runs on
def lcg_weights(layers: int, seed: int) -> dict:
    """Frozen expert weights + tables on LCG data in library layout (the scales of smolvla_expert.emit_layer_test)."""
    s0 = seed * 100
    W = {}
    for L in range(layers):
        b = s0 + 30 * L
        if L % 2 == 0:
            W[f"wqkv{L}"] = lcg.fill_fp16(D, DQ + 2 * DKV, b + 2, -8, 8, 1 / 128)
        else:
            W[f"wq{L}"] = lcg.fill_fp16(D, DQ, b + 12, -8, 8, 1 / 128)
            W[f"wkx{L}"] = lcg.fill_fp16(DKV, DKV, b + 13, -8, 8, 1 / 128); W[f"wvx{L}"] = lcg.fill_fp16(DKV, DKV, b + 14, -8, 8, 1 / 128)
        W[f"wo{L}"] = lcg.fill_fp16(DQ, D, b + 3, -8, 8, 1 / 128); W[f"wgu{L}"] = lcg.fill_fp16(D, 2 * FF, b + 4, -8, 8, 1 / 128)
        W[f"wd{L}"] = lcg.fill_fp16(FF, D, b + 5, -8, 8, 1 / 256)
        W[f"g1{L}"] = lcg.fill_fp16(1, D, b + 6, 2, 6, 0.25); W[f"g2{L}"] = lcg.fill_fp16(1, D, b + 7, 2, 6, 0.25)
        W[f"kp{L}"] = lcg.fill_fp16(LP, DKV, b + 8, -8, 8, 0.125); W[f"vp{L}"] = lcg.fill_fp16(LP, DKV, b + 9, -8, 8, 0.125)
    W["tok"] = prefix_tok((np.random.default_rng(seed).random(LP) > 0.15).astype(np.float32))
    W["rq_self"], W["rq_cross"] = rope_table(200 + np.arange(S)), rope_table(np.arange(S))
    return W


# ----------------------------------------------------------------------------- torch reference (float64)
def torch_ttt(Wt: dict, layers: int, head: bool, lora16: np.ndarray, *, lr: float, opt: str = "sgd", step: int = 0,
              x_in=None, target=None, h_in=None, dh_in=None, dtype=None) -> dict:
    """The same step in torch autograd (float64 by default) on the fp16 values the device reads: returns loss, v, the
    LoRA gradient arena (unscaled) and the updated fp32 arena."""
    torch = _torch()
    dt = dtype or torch.float64
    t = lambda a: torch.tensor(np.asarray(a, np.float32), dtype=dt)  # noqa: E731
    lora = torch.tensor(lora16.astype(np.float32), dtype=dt, requires_grad=True)

    def lv(L, nm):
        r, c = next((r, c) for n, r, c in LORA if n == nm)
        o = L * LAYER_N + LORA_OFF[nm]
        return lora[o:o + r * c].view(r, c)

    def rms(x, g):
        return x * torch.rsqrt((x * x).mean(1, keepdim=True) + RMS_EPS) * g

    def rope(x, tab):
        c, s = tab[:, :DH // 2], tab[:, DH // 2:]
        out = []
        for h0 in range(0, x.shape[1], DH):
            x1, x2 = x[:, h0:h0 + DH // 2], x[:, h0 + DH // 2:h0 + DH]
            out += [x1 * c - x2 * s, x2 * c + x1 * s]
        return torch.cat(out, 1)

    tok = np.asarray(Wt["tok"]).reshape(-1)[:LP].astype(np.uint16)

    def attn(q, K, V, own):
        Lp = LP
        allowed = np.ones((S, K.shape[0]), bool)
        allowed[:, :Lp] &= tok != 0xFFFF
        if own:
            allowed[:, Lp:] &= np.tril(np.ones((S, S), bool))
        mask = torch.tensor(np.where(allowed, 0.0, -np.inf), dtype=dt)
        outs = []
        for h in range(H):
            kv = h // GROUP
            sc = q[:, h * DH:(h + 1) * DH] @ K[:, kv * DH:(kv + 1) * DH].T * SCALE + mask
            outs.append(torch.softmax(sc, 1) @ V[:, kv * DH:(kv + 1) * DH])
        return torch.cat(outs, 1)

    rq_s, rq_c = t(Wt["rq_self"]), t(Wt["rq_cross"])
    if head:
        x = t(x_in)
        e = x @ t(Wt["wa"]) + t(Wt["ba"])
        e1 = e @ t(Wt["wti"]) + t(Wt["tb"])[step]
        e1 = e1 * torch.sigmoid(e1)
        h = e1 @ t(Wt["wto"]) + t(Wt["bto"])
    else:
        h = t(h_in)
    for L in range(layers):
        g = lambda nm: t(Wt[f"{nm}{L}"])  # noqa: E731
        xn = rms(h, g("g1"))
        if L % 2 == 0:
            qkv = xn @ g("wqkv")
            q = qkv[:, :DQ] + LS * (xn @ lv(L, "aq")) @ lv(L, "bq")
            k, v = rope(qkv[:, DQ:DQ + DKV], rq_s), qkv[:, DQ + DKV:]
            q = rope(q, rq_s)
            o = attn(q, torch.cat([g("kp"), k]), torch.cat([g("vp"), v]), True)
        else:
            q = rope(xn @ g("wq") + LS * (xn @ lv(L, "aq")) @ lv(L, "bq"), rq_c)
            o = attn(q, g("kp") @ g("wkx"), g("vp") @ g("wvx"), False)
        h1 = h + o @ g("wo") + LS * (o @ lv(L, "ao")) @ lv(L, "bo")
        gu = rms(h1, g("g2")) @ g("wgu")
        m = torch.nn.functional.silu(gu[:, :FF]) * gu[:, FF:]
        h = h1 + m @ g("wd") + LS * (m @ lv(L, "ad")) @ lv(L, "bd")
    out = {}
    if head:
        v = rms(h, t(Wt["gf"])) @ t(Wt["wout"]) + t(Wt["bout"])
        loss = ((v - t(target)) ** 2).mean()
        out["v"] = v.detach().numpy(); out["loss"] = float(loss)
    else:
        loss = (h * t(dh_in)).sum()
        out["h"] = h.detach().numpy()
    loss.backward()
    gr = lora.grad.detach().numpy()
    out["grad"] = gr
    w = lora16.astype(np.float64)
    if opt == "adam":
        m_, v_ = 0.1 * gr, 0.001 * gr * gr
        out["w_new"] = w - lr * (m_ / 0.1) / (np.sqrt(v_ / 0.001) + 1e-8)
    else:
        out["w_new"] = w - lr * gr
    return out


def lora_tensor(arena: np.ndarray, L: int, nm: str) -> np.ndarray:
    r, c = next((r, c) for n, r, c in LORA if n == nm)
    o = L * LAYER_N + LORA_OFF[nm]
    return arena[o:o + r * c].reshape(r, c)


# ----------------------------------------------------------------------------- step 1: the backward primitives alone
def emit_ops_test(seed: int = 5, cluster: int = -1, nsamples: int = 128) -> tuple[str, dict, dict]:
    """Every training primitive once on LCG data at the expert's shapes, a mark after each (timing), sampled dumps.
    Returns (mlir, preload, reference dict {tag: array}) with float64 torch / numpy references."""
    torch = _torch()
    P = TProg(cluster, nsamples)
    T, op = P.T, P.op
    s0 = seed * 100
    d = {}

    def F(name, rows, cols, sd, lo, hi, scale):
        d[name] = lcg.fill_fp16(rows, cols, s0 + sd, lo, hi, scale)
        P.B(name, rows, cols, d[name])
    F("x", S, D, 1, -16, 16, 0.125); F("dy", S, D, 2, -16, 16, 1 / 32); F("dr", S, D, 3, -16, 16, 1 / 32); F("g", 1, D, 4, 2, 6, 0.25)
    F("u", S, R, 5, -8, 8, 1 / 16); F("bt", D, R, 6, -8, 8, 1 / 64)
    F("gu", S, 2 * FF, 7, -32, 32, 1 / 8); F("dm", S, FF, 8, -16, 16, 1 / 32)
    F("pin", S, 320, 9, 0, 16, 1 / 64); F("dp", S, 320, 10, -16, 16, 1 / 32)
    F("vp", S, AD, 11, -16, 16, 1 / 8); F("tg", S, AD, 12, -16, 16, 1 / 8)
    F("q", S, DQ + 2 * DKV, 13, -8, 8, 0.125); F("kp", LP, DKV, 14, -8, 8, 0.125); F("vpp", LP, DKV, 15, -8, 8, 0.125)
    F("dout", S, DQ, 16, -16, 16, 1 / 64)
    F("wbig", FF, D, 17, -8, 8, 1 / 256)
    d["tok"] = prefix_tok((np.random.default_rng(seed).random(LP) > 0.15).astype(np.float32)); P.B("tok", 1, LP, d["tok"], elem="i16")
    F("gw", 64, 64, 18, -16, 16, 1 / 64)
    w0 = lcg.fill_fp16(64, 64, s0 + 19, -16, 16, 1 / 64).astype(np.float32); d["w0"] = w0
    P.B("w32", 64, 64, w0, elem="f32"); P.B("w16", 64, 64, w0.astype(np.float16))
    P.B("a32", 64, 64, w0, elem="f32"); P.B("a16", 64, 64, w0.astype(np.float16))
    P.B("m32", 64, 64, np.zeros((64, 64), np.float32), elem="f32"); P.B("v32", 64, 64, np.zeros((64, 64), np.float32), elem="f32")
    for nm, r, c in [("da", D, R), ("dx", S, D), ("dgu", S, 2 * FF), ("dpx", S, 320), ("dv", S, AD), ("o", S, DQ), ("dq", S, DQ + 2 * DKV),
                     ("dqx", S, DQ), ("oc", S, DQ), ("ascr", S, 2 * H * DH), ("scr", 2 * FF, S), ("wbigT", D, FF), ("ub", S, D)]:
        P.B(nm, r, c)
    P.B("lrows", S, 1, elem="f32")
    P.vw("ga", "gu", S, FF, 2 * FF, 0); P.vw("up", "gu", S, FF, 2 * FF, FF)
    P.vw("dga", "dgu", S, FF, 2 * FF, 0); P.vw("dup", "dgu", S, FF, 2 * FF, FF)
    nq = DQ + 2 * DKV
    P.vw("qq", "q", S, DQ, nq, 0); P.vw("kk", "q", S, DKV, nq, DQ); P.vw("vv", "q", S, DKV, nq, DQ + DKV)
    P.vw("dqq", "dq", S, DQ, nq, 0); P.vw("dkk", "dq", S, DKV, nq, DQ); P.vw("dvv", "dq", S, DKV, nq, DQ + DKV)
    sent = sentinel_array(); P.B("sentinel", *sent.shape, sent)
    op(f"softhier.preload_wait %sentinel : {T['sentinel']}")
    P.mark("start")
    P.gt("x", "u", "da", "scr", (48, R, S), tx=True); P.mark("gemm_tx")           # da [720,16] = x^T u (x stored [50, 720])
    P.gt("u", "bt", "ub", "scr", (S, 48, R), tw=True); P.mark("gemm_tw")          # ub [50,720] = u bt^T (bt stored [720, 16])
    P.tr("wbig", "wbigT"); P.mark("wT2048x720")                                    # what transposing one frozen weight would cost
    P.rms_bwd("x", "g", "dy", "dx", res="dr"); P.mark("rms_bwd")
    op(f"softhier.silu_mul_bwd %ga, %up, %dm -> %dga, %dup {{{P.cl}}} : {T['ga']}, {T['up']}, {T['dm']} -> {T['dga']}, {T['dup']}"); P.mark("silu_bwd")
    op(f"softhier.softmax_bwd %pin, %dp -> %dpx {{scale = 0.125 : f32, {P.cl}}} : {T['pin']}, {T['dp']} -> {T['dpx']}"); P.mark("softmax_bwd")
    op(f"softhier.mse_grad %vp, %tg -> %dv loss %lrows {{gscale = 1.0 : f32, {P.cl}}} : {T['vp']}, {T['tg']} -> {T['dv']} loss {T['lrows']}"); P.mark("mse")
    op(f"softhier.cross_attention %qq, %kp, %vpp own %kk, %vv mask %tok -> %o {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} "
       f": {T['qq']}, {T['kp']}, {T['vpp']} own {T['kk']}, {T['vv']} mask {T['tok']} -> {T['o']}"); P.mark("attn_fwd")
    op(f"softhier.attention_bwd %qq, %kp, %vpp own %kk, %vv mask %tok, %o, %dout -> %dqq grads %dkk, %dvv scratch %ascr "
       f"{{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} : {T['qq']}, {T['kp']}, {T['vpp']} own {T['kk']}, {T['vv']} "
       f"mask {T['tok']}, {T['o']}, {T['dout']} -> {T['dqq']} grads {T['dkk']}, {T['dvv']} scratch {T['ascr']}"); P.mark("attn_bwd_self")
    op(f"softhier.cross_attention %qq, %kp, %vpp mask %tok -> %oc {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} "
       f": {T['qq']}, {T['kp']}, {T['vpp']} mask {T['tok']} -> {T['oc']}"); P.mark("xattn_fwd")
    op(f"softhier.attention_bwd %qq, %kp, %vpp mask %tok, %oc, %dout -> %dqx {{scale = {SCALE!r} : f32, heads = {H} : i32, kv_heads = {HKV} : i32, {P.cl}}} : "
       f"{T['qq']}, {T['kp']}, {T['vpp']} mask {T['tok']}, {T['oc']}, {T['dout']} -> {T['dqx']}"); P.mark("attn_bwd_cross")
    op(f'softhier.optim_step %gw, %w32 -> %w16 {{kind = "sgd", lr = 0.5 : f32, inv_scale = 0.25 : f32, {P.cl}}} : {T["gw"]}, {T["w32"]} -> {T["w16"]}'); P.mark("sgd")
    op(f'softhier.optim_step %gw, %a32 -> %a16 moments %m32, %v32 {{kind = "adam", lr = 0.01 : f32, inv_scale = 0.25 : f32, b1 = 0.9 : f32, b2 = 0.999 : f32, '
       f'eps = 1.0e-8 : f32, bc1 = 0.1 : f32, bc2 = 0.001 : f32, {P.cl}}} : {T["gw"]}, {T["a32"]} -> {T["a16"]} moments {T["m32"]}, {T["v32"]}'); P.mark("adam")
    for i, nm in enumerate(["da", "ub", "wbigT", "dx", "dgu", "dpx", "dv", "lrows", "o", "dq", "dqx", "w32", "w16", "a32", "a16", "m32", "v32"]):
        P.dumps(nm, 700 + i, nm.upper())
    # ---- references (float64)
    f = {k: v.astype(np.float64) for k, v in d.items() if k != "tok"}
    ref = {"DA": f["x"].T @ f["u"], "UB": f["u"] @ f["bt"].T, "WBIGT": f["wbig"].T}
    x = torch.tensor(f["x"], requires_grad=True)
    y = x * torch.rsqrt((x * x).mean(1, keepdim=True) + RMS_EPS) * torch.tensor(f["g"])
    y.backward(torch.tensor(f["dy"])); ref["DX"] = x.grad.numpy() + f["dr"]
    a = torch.tensor(f["gu"][:, :FF], requires_grad=True); b = torch.tensor(f["gu"][:, FF:], requires_grad=True)
    (torch.nn.functional.silu(a) * b).backward(torch.tensor(f["dm"])); ref["DGU"] = np.concatenate([a.grad.numpy(), b.grad.numpy()], 1)
    pin = f["pin"]                                                                      # softmax_bwd takes y = pin and dy = dp
    ref["DPX"] = 0.125 * pin * (f["dp"] - (pin * f["dp"]).sum(1, keepdims=True))
    ref["DV"] = f["vp"] - f["tg"]; ref["LROWS"] = ((f["vp"] - f["tg"]) ** 2).sum(1, keepdims=True)
    tok = d["tok"].reshape(-1).astype(np.uint16)
    qv = torch.tensor(f["q"][:, :DQ], requires_grad=True)
    ko = torch.tensor(f["q"][:, DQ:DQ + DKV], requires_grad=True); vo = torch.tensor(f["q"][:, DQ + DKV:], requires_grad=True)
    K = torch.cat([torch.tensor(f["kp"]), ko]); V = torch.cat([torch.tensor(f["vpp"]), vo])
    allowed = np.ones((S, LP + S), bool); allowed[:, :LP] &= tok != 0xFFFF; allowed[:, LP:] &= np.tril(np.ones((S, S), bool))
    mask = torch.tensor(np.where(allowed, 0.0, -np.inf))
    o = torch.cat([torch.softmax(qv[:, h * DH:(h + 1) * DH] @ K[:, (h // GROUP) * DH:(h // GROUP + 1) * DH].T * SCALE + mask, 1)
                   @ V[:, (h // GROUP) * DH:(h // GROUP + 1) * DH] for h in range(H)], 1)
    o.backward(torch.tensor(f["dout"]))
    ref["O"] = o.detach().numpy()
    ref["DQ"] = np.concatenate([qv.grad.numpy(), ko.grad.numpy(), vo.grad.numpy()], 1)
    qc = torch.tensor(f["q"][:, :DQ], requires_grad=True)
    Kc, Vc = torch.tensor(f["kp"]), torch.tensor(f["vpp"])
    mc = torch.tensor(np.where(np.broadcast_to(tok != 0xFFFF, (S, LP)), 0.0, -np.inf))
    oc = torch.cat([torch.softmax(qc[:, h * DH:(h + 1) * DH] @ Kc[:, (h // GROUP) * DH:(h // GROUP + 1) * DH].T * SCALE + mc, 1)
                    @ Vc[:, (h // GROUP) * DH:(h // GROUP + 1) * DH] for h in range(H)], 1)
    oc.backward(torch.tensor(f["dout"])); ref["DQX"] = qc.grad.numpy()
    g = f["gw"] * 0.25
    ref["W32"] = f["w0"] - 0.5 * g; ref["W16"] = ref["W32"]
    m_, v_ = 0.1 * g, 0.001 * g * g
    ref["A32"] = f["w0"] - 0.01 * (m_ / 0.1) / (np.sqrt(v_ / 0.001) + 1e-8); ref["A16"] = ref["A32"]; ref["M32"] = m_; ref["V32"] = v_
    return P.module("ttt_ops"), P.pre, ref
