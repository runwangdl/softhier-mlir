"""Host twins of the on-device test helpers in runtime/sh_test.inc.c.

The device generates its own test data from a 32-bit LCG and prints sampled outputs; the host
regenerates the same data here, computes a numpy reference and compares the samples. No data
ever has to be transferred into the simulator for a correctness test.
"""
from __future__ import annotations

import re

import numpy as np

_A, _C, _M = 1664525, 1013904223, 1 << 32


_LANES = 4096


def _lcg_stream(seed: int, n: int) -> np.ndarray:
    """n successive values of sh_lcg() (state >> 8) for the given seed.

    Leapfrog form so a 7 M-element tensor costs ~50 ms instead of seconds: the first `_LANES`
    states are stepped sequentially, then every lane advances by `_LANES` steps at once with
    the composed affine map (A^k, C (A^k - 1) / (A - 1)) in wrapping uint32 arithmetic."""
    s = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    k = min(n, _LANES)
    lane = np.empty(k, dtype=np.uint32)
    for j in range(k):
        s = (s * _A + _C) % _M
        lane[j] = s
    nblk = -(-n // k) if k else 0
    if nblk <= 1:
        return lane[:n] >> 8
    ak, ck = 1, 0
    for _ in range(k):
        ak, ck = (ak * _A) % _M, (ck * _A + _C) % _M
    out = np.empty((nblk, k), dtype=np.uint32)
    out[0] = lane
    ak32, ck32 = np.uint32(ak), np.uint32(ck)
    with np.errstate(over="ignore"):
        for m in range(1, nblk):
            out[m] = out[m - 1] * ak32 + ck32
    return out.reshape(-1)[:n] >> 8


def _lcg_stream_reference(seed: int, n: int) -> np.ndarray:
    """The scalar form (== the C loop); kept for the self-test of the leapfrog version."""
    s = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        s = (s * _A + _C) % _M
        out[i] = s >> 8
    return out


def fill_fp16(rows: int, cols: int, seed: int, lo: int, hi: int, scale: float = 1.0) -> np.ndarray:
    """== sh_test_fill_fp16: integers in [lo, hi] times scale, rounded to fp16, row-major."""
    v = _lcg_stream(seed, rows * cols) % (hi - lo + 1)
    return ((v.astype(np.int64) + lo).astype(np.float32) * np.float32(scale)).astype(np.float16).reshape(rows, cols)


def const_fp16(rows: int, cols: int, bits: int) -> np.ndarray:
    """== sh_test_fill_const_fp16: every element is the fp16 code `bits` (returned as fp16)."""
    return np.full((rows, cols), bits & 0xFFFF, dtype=np.uint16).view(np.float16)


def col_parity_fp16(rows: int, cols: int, even_bits: int, odd_bits: int) -> np.ndarray:
    """== sh_test_fill_colparity_fp16: even columns `even_bits`, odd columns `odd_bits` (fp16 codes)."""
    row = np.where(np.arange(cols) & 1, odd_bits & 0xFFFF, even_bits & 0xFFFF).astype(np.uint16)
    return np.ascontiguousarray(np.broadcast_to(row, (rows, cols))).view(np.float16)


def sample_positions(rows: int, cols: int, seed: int, nsamples: int) -> list[tuple[int, int]]:
    """== the (i, j) sequence of sh_test_dump_samples."""
    v = _lcg_stream(seed, 2 * nsamples)
    return [(int(v[2 * n] % rows), int(v[2 * n + 1] % cols)) for n in range(nsamples)]


_SAMPLE_RE = re.compile(r"^(\S+) (\d+) (\d+) ([0-9a-fA-F]{4})$")


def parse_samples(stdout: str) -> dict[str, list[tuple[int, int, float]]]:
    """Collect '<tag> r c hex' lines into {tag: [(r, c, value)]}."""
    out: dict[str, list[tuple[int, int, float]]] = {}
    for ln in stdout.splitlines():
        m = _SAMPLE_RE.match(ln.strip())
        if m:
            val = np.frombuffer(bytes.fromhex(m.group(4)), dtype=">f2")[0]
            out.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), float(val)))
    return out


def compare_samples(samples: list[tuple[int, int, float]], ref: np.ndarray, atol: float, rtol: float,
                    show: int = 0) -> tuple[int, float]:
    """Return (mismatches, max abs error) of device samples against the reference array;
    print the first `show` mismatches."""
    bad, maxerr = 0, 0.0
    for r, c, got in samples:
        want = float(ref[r, c])
        err = abs(got - want)
        maxerr = max(maxerr, err)
        if err > atol + rtol * abs(want):
            if bad < show:
                print(f"       mismatch [{r},{c}] got {got:.4f} want {want:.4f}")
            bad += 1
    return bad, maxerr
