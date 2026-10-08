"""Emit a SigLIP / ViT encoder (N layers) in the softhier dialect.

The emitted module declares every activation and parameter as an HBM buffer, fills the
parameters (and the input) from the runtime LCG when `test=True`, and dumps samples of the
tensors listed in `dumps` so the host can compare against softhier_mlir.testing.siglip_ref.
Weights are [in, out] (X . W convention), biases/gamma/beta are 1 x cols rows.

    python -m softhier_mlir.frontend.siglip --seq 256 --layers 1 > siglip1.mlir
"""
from __future__ import annotations

import argparse
import sys


class _Emitter:
    def __init__(self, space: str = "hbm_west") -> None:
        self.lines: list[str] = []
        self.next_off = 0
        self.n = 0
        self.space = space

    def buf(self, name: str, rows: int, cols: int) -> str:
        off = self.next_off
        self.next_off += (rows * cols * 2 + 4095) & ~4095
        self.lines.append(f'    %{name} = softhier.hbm_buffer {{offset = {off} : i32}} : memref<{rows}x{cols}xf16, "{self.space}">')
        return f"memref<{rows}x{cols}xf16, \"{self.space}\">"

    def view(self, name: str, src: str, src_t: str, rows: int, cols: int, ld: int, eoff: int) -> str:
        t = f'memref<{rows}x{cols}xf16, strided<[{ld}, 1], offset: {eoff}>, "{self.space}">'
        self.lines.append(f"    %{name} = softhier.view %{src} : {src_t} -> {t}")
        return t

    def op(self, text: str) -> None:
        self.lines.append("    " + text)


def emit(seq: int, d: int, ff: int, heads: int, layers: int = 1, cluster: int = -1,
         test: bool = True, dumps: tuple[str, ...] = ("X", "LN1", "Q", "K", "KT", "P0", "V", "O", "H", "G", "OUT"),
         nsamples: int = 64, fused_attention: bool = False) -> str:
    """`fused_attention=True` emits one `softhier.attention` op (every head inside one cluster's TCDM) instead
    of the K transpose + per-head gemm / softmax / gemm through HBM; the kT / sc buffers and the KT / P0 dumps
    then do not exist."""
    dh = d // heads
    e = _Emitter()
    T: dict[str, str] = {}
    cl = f"cluster = {cluster} : i32"
    if fused_attention:
        dumps = tuple(t for t in dumps if t not in ("KT", "P0"))

    def B(name, rows, cols):
        T[name] = e.buf(name, rows, cols)
        return name

    def fill(name, seed, lo, hi, scale):
        if test:
            e.op(f"softhier.hbm_fill_lcg %{name} {{seed = {seed} : i32, lo = {lo} : i32, hi = {hi} : i32, scale = {scale!r} : f32}} : {T[name]}")

    # activations (shared across layers)
    for nm, r, c in [("x", seq, d), ("ln1", seq, d), ("q", seq, d), ("k", seq, d), ("v", seq, d), ("kT", d, seq),
                     ("sc", heads * seq, seq), ("o", seq, d), ("ao", seq, d), ("h", seq, d), ("ln2", seq, d),
                     ("f1", seq, ff), ("g", seq, ff), ("f2", seq, d), ("out", seq, d)]:
        if fused_attention and nm in ("kT", "sc"):
            continue
        B(nm, r, c)
    fill("x", 1, -16, 16, 0.125)
    # parameters per layer (seeds continue the layer-0 numbering: siglip_ref uses the same scheme)
    for L in range(layers):
        s0 = 100 * L
        for nm, r, c, seed, lo, hi, scale in [
            (f"wq{L}", d, d, 2 + s0, -8, 8, 1 / 128), (f"wk{L}", d, d, 3 + s0, -8, 8, 1 / 128),
            (f"wv{L}", d, d, 4 + s0, -8, 8, 1 / 128), (f"wo{L}", d, d, 5 + s0, -8, 8, 1 / 128),
            (f"w1{L}", d, ff, 6 + s0, -8, 8, 1 / 128), (f"w2{L}", ff, d, 7 + s0, -8, 8, 1 / 256),
            (f"bq{L}", 1, d, 8 + s0, -4, 4, 0.0625), (f"bk{L}", 1, d, 9 + s0, -4, 4, 0.0625),
            (f"bv{L}", 1, d, 10 + s0, -4, 4, 0.0625), (f"bo{L}", 1, d, 11 + s0, -4, 4, 0.0625),
            (f"b1{L}", 1, ff, 12 + s0, -4, 4, 0.0625), (f"b2{L}", 1, d, 13 + s0, -4, 4, 0.0625),
            (f"g1{L}", 1, d, 14 + s0, 2, 6, 0.25), (f"be1{L}", 1, d, 15 + s0, -4, 4, 0.125),
            (f"g2{L}", 1, d, 16 + s0, 2, 6, 0.25), (f"be2{L}", 1, d, 17 + s0, -4, 4, 0.125)]:
            B(nm, r, c)
            fill(nm, seed, lo, hi, scale)

    gem = lambda tm, tn, tk: f"tile_m = {tm} : i32, tile_n = {tn} : i32, tile_k = {tk} : i32, pipeline"  # noqa: E731
    big, qk, pv = gem(256, 256, 256), gem(256, 256, dh), gem(256, dh, 256)
    xin = "x"
    for L in range(layers):
        W = lambda nm: f"{nm}{L}"  # noqa: E731
        e.op(f"softhier.layernorm %{xin}, %{W('g1')}, %{W('be1')} -> %ln1 {{eps = 1.0e-6 : f32, {cl}}} : {T[xin]}, {T[W('g1')]}, {T[W('be1')]} -> {T['ln1']}")
        for dst, w, bias in (("q", "wq", "bq"), ("k", "wk", "bk"), ("v", "wv", "bv")):
            e.op(f"softhier.gemm %ln1, %{W(w)} into %{dst} {{fmt = \"fp16\", {big}, {cl}}} : {T['ln1']}, {T[W(w)]}, {T[dst]}")
            e.op(f"softhier.add_bias %{dst}, %{W(bias)} -> %{dst} {{{cl}}} : {T[dst]}, {T[W(bias)]} -> {T[dst]}")
        if fused_attention:
            e.op(f"softhier.attention %q, %k, %v -> %o {{scale = {1 / dh ** 0.5!r} : f32, heads = {heads} : i32, {cl}}} "
                 f": {T['q']}, {T['k']}, {T['v']} -> {T['o']}")
        else:
            e.op(f"softhier.transpose %k -> %kT {{{cl}}} : {T['k']} -> {T['kT']}")
            for hd in range(heads):
                hc = f"cluster = {hd % 16 if cluster < 0 else cluster} : i32"
                qh = e.view(f"q{L}_{hd}", "q", T["q"], seq, dh, d, hd * dh)
                kh = e.view(f"kT{L}_{hd}", "kT", T["kT"], dh, seq, seq, hd * dh * seq)
                sh = e.view(f"s{L}_{hd}", "sc", T["sc"], seq, seq, seq, hd * seq * seq)
                vh = e.view(f"v{L}_{hd}", "v", T["v"], seq, dh, d, hd * dh)
                oh = e.view(f"o{L}_{hd}", "o", T["o"], seq, dh, d, hd * dh)
                e.op(f"softhier.gemm %q{L}_{hd}, %kT{L}_{hd} into %s{L}_{hd} {{fmt = \"fp16\", {qk}, {hc}}} : {qh}, {kh}, {sh}")
                e.op(f"softhier.softmax %s{L}_{hd} -> %s{L}_{hd} {{scale = {1 / dh ** 0.5!r} : f32, {hc}}} : {sh} -> {sh}")
                e.op(f"softhier.gemm %s{L}_{hd}, %v{L}_{hd} into %o{L}_{hd} {{fmt = \"fp16\", {pv}, {hc}}} : {sh}, {vh}, {oh}")
        e.op("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}")
        e.op(f"softhier.gemm %o, %{W('wo')} into %ao {{fmt = \"fp16\", {big}, {cl}}} : {T['o']}, {T[W('wo')]}, {T['ao']}")
        e.op(f"softhier.add_bias %ao, %{W('bo')} -> %ao {{{cl}}} : {T['ao']}, {T[W('bo')]} -> {T['ao']}")
        e.op(f"softhier.add %{xin}, %ao -> %h {{{cl}}} : {T[xin]}, {T['ao']} -> {T['h']}")
        e.op(f"softhier.layernorm %h, %{W('g2')}, %{W('be2')} -> %ln2 {{eps = 1.0e-6 : f32, {cl}}} : {T['h']}, {T[W('g2')]}, {T[W('be2')]} -> {T['ln2']}")
        e.op(f"softhier.gemm %ln2, %{W('w1')} into %f1 {{fmt = \"fp16\", {big}, {cl}}} : {T['ln2']}, {T[W('w1')]}, {T['f1']}")
        e.op(f"softhier.add_bias %f1, %{W('b1')} -> %f1 {{{cl}}} : {T['f1']}, {T[W('b1')]} -> {T['f1']}")
        e.op(f"softhier.gelu %f1 -> %g {{{cl}}} : {T['f1']} -> {T['g']}")
        e.op(f"softhier.gemm %g, %{W('w2')} into %f2 {{fmt = \"fp16\", {big}, {cl}}} : {T['g']}, {T[W('w2')]}, {T['f2']}")
        e.op(f"softhier.add_bias %f2, %{W('b2')} -> %f2 {{{cl}}} : {T['f2']}, {T[W('b2')]} -> {T['f2']}")
        e.op(f"softhier.add %h, %f2 -> %out {{{cl}}} : {T['h']}, {T['f2']} -> {T['out']}")
        xin = "out"   # next layer reads this layer's output (out is re-used as the residual stream)
    if test:
        tagmap = {"X": "x", "LN1": "ln1", "Q": "q", "K": "k", "KT": "kT", "V": "v", "O": "o", "H": "h", "G": "g", "OUT": "out"}
        seeds = {"X": 200, "LN1": 206, "Q": 201, "K": 207, "KT": 208, "P0": 209, "V": 211, "O": 202, "H": 203, "G": 204, "OUT": 205}
        for tag in dumps:
            if tag == "P0":
                p0 = e.view("p0view", "sc", T["sc"], seq, seq, seq, 0)
                e.op(f'softhier.dump_samples %p0view {{seed = 209 : i32, n = {nsamples} : i32, tag = "P0"}} : {p0}')
            else:
                nm = tagmap[tag]
                e.op(f'softhier.dump_samples %{nm} {{seed = {seeds[tag]} : i32, n = {nsamples} : i32, tag = "{tag}"}} : {T[nm]}')
    body = "\n".join(e.lines)
    return f"builtin.module {{\n  func.func @siglip_encoder() {{\n{body}\n    func.return\n  }}\n}}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--ff", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--cluster", type=int, default=-1, help="-1 = all clusters")
    ap.add_argument("--no-test", action="store_true", help="no LCG fills / sample dumps")
    ap.add_argument("--fused", action="store_true", help="fused softhier.attention instead of per-head gemm/softmax/gemm")
    a = ap.parse_args()
    sys.stdout.write(emit(a.seq, a.d, a.ff, a.heads, a.layers, a.cluster, not a.no_test, fused_attention=a.fused))


if __name__ == "__main__":
    main()
