"""numpy reference of tests/gvsoc/siglip_layer/main.c (same LCG data, fp32 math)."""
from __future__ import annotations

import numpy as np

from softhier_mlir.testing import lcg


def layer_reference(S: int, D: int, F: int, H: int) -> dict[str, np.ndarray]:
    f = lambda *a, **k: lcg.fill_fp16(*a, **k).astype(np.float32)  # noqa: E731
    dh = D // H
    x = f(S, D, 1, -16, 16, 0.125)
    wq, wk, wv, wo = (f(D, D, s, -8, 8, 1 / 128) for s in (2, 3, 4, 5))
    w1 = f(D, F, 6, -8, 8, 1 / 128); w2 = f(F, D, 7, -8, 8, 1 / 256)
    bq, bk, bv, bo = (f(1, D, s, -4, 4, 0.0625) for s in (8, 9, 10, 11))
    b1 = f(1, F, 12, -4, 4, 0.0625); b2 = f(1, D, 13, -4, 4, 0.0625)
    g1 = f(1, D, 14, 2, 6, 0.25); be1 = f(1, D, 15, -4, 4, 0.125)
    g2 = f(1, D, 16, 2, 6, 0.25); be2 = f(1, D, 17, -4, 4, 0.125)

    def ln(a, g, b, eps=1e-6):
        m = a.mean(1, keepdims=True); v = a.var(1, keepdims=True)
        return (a - m) / np.sqrt(v + eps) * g + b

    def r16(a):  # device stores every intermediate as fp16
        return a.astype(np.float16).astype(np.float32)

    ln1 = r16(ln(x, g1, be1))
    q = r16(r16(ln1 @ wq) + bq); k = r16(r16(ln1 @ wk) + bk); v = r16(r16(ln1 @ wv) + bv)
    o = np.zeros((S, D), np.float32)
    p0 = None
    for hd in range(H):
        sl = slice(hd * dh, (hd + 1) * dh)
        s = r16(q[:, sl] @ k[:, sl].T) * 0.125
        p = np.exp(s - s.max(1, keepdims=True)); p = r16(p / p.sum(1, keepdims=True))
        if hd == 0:
            p0 = p
        o[:, sl] = r16(p @ v[:, sl])
    ao = r16(r16(o @ wo) + bo)
    h = r16(x + ao)
    ln2 = r16(ln(h, g2, be2))
    f1 = r16(r16(ln2 @ w1) + b1)
    g = r16(0.5 * f1 * (1 + np.tanh(0.7978845608 * (f1 + 0.044715 * f1 ** 3))))
    f2 = r16(r16(g @ w2) + b2)
    out = r16(h + f2)
    return {"X": x, "LN1": ln1, "Q": q, "K": k, "KT": k.T, "P0": p0, "V": v, "O0": o[:, :dh], "O6": o[:, 6 * dh:7 * dh], "O": o, "H": h, "G": g, "OUT": out}
