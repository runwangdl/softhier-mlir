"""Host twins of the on-device test helpers in runtime/sh_test.inc.c.

The device generates its own test data from a 32-bit LCG and prints sampled outputs; the host
regenerates the same data here, computes a numpy reference and compares the samples. No data
ever has to be transferred into the simulator for a correctness test.
"""
from __future__ import annotations

import re

import numpy as np

_A, _C, _M = 1664525, 1013904223, 1 << 32


def _lcg_stream(seed: int, n: int) -> np.ndarray:
    """n successive values of sh_lcg() (state >> 8) for the given seed."""
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
