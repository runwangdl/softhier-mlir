"""Tile-shape policy for the frontends (the compiler decides, the library applies).

GEMM: the SoftHier GEMMs on the gvsoc model are HBM-bound, not RedMulE-bound (docs/DSE.md): the
aggregate HBM -> TCDM rate is ~85-125 B/cycle whatever the number of streaming clusters, so the
total traffic ``tiles * (tm*K + K*tn) * 2 B = 2 M N K (1/tm + 1/tn)`` sets the time, and splitting a
small-M GEMM into more output tiles to occupy more clusters makes it *slower* (256x768x768 at S=256:
3 tiles of 256x256 = 35 us, 12 tiles of 128x128 = 56 us, 12 tiles of 256x64 = 70 us, 2 tiles of
256x384 = 32 us). ``gemm_tiles`` therefore ranks the candidate tiles with the analytic model
(``softhier_mlir.dse.cost.gemm_est``: RedMulE FSM, double-buffered DMA, the measured HBM sharing
curve, per-link cap) and returns the cheapest one that fits TCDM, which picks the largest tiles the
L1 budget allows unless a single cluster would become compute- or link-bound.

Attention: ``attention_q_block`` mirrors the library's ``sh_attention_q_block`` (runtime/sh_attention.inc.c)
so the frontend can set the ``q_block`` attribute explicitly.
"""
from __future__ import annotations

import math

from softhier_mlir.dse.cost import CostParams, gemm_est
from softhier_mlir.sim.gvsoc import Arch

ELEM = 2
TM_CAND = (64, 128, 256, 512)
TN_CAND = (64, 128, 256, 384, 512, 768, 1024)
TK_CAND = (128, 256, 512)


def gemm_l1_bytes(tm: int, tn: int, tk: int, pipeline: int = 1) -> int:
    """TCDM bytes sh_gemm needs (sh_gemm_l1_bytes with l1_base 0)."""
    return (2 if pipeline else 1) * (tm * tk + tk * tn) * ELEM + tm * tn * ELEM


def gemm_candidates(M: int, N: int, K: int, l1: int, pipeline: int = 1) -> list[tuple[int, int, int]]:
    out = []
    for tm in TM_CAND:
        for tn in TN_CAND:
            for tk in TK_CAND:
                if tm > M or tn > N or tk > K or M % tm or N % tn or K % tk:
                    continue
                if gemm_l1_bytes(tm, tn, tk, pipeline) > l1:
                    continue
                out.append((tm, tn, tk))
    return out


def gemm_tiles(M: int, N: int, K: int, all_clusters: bool = True, arch: Arch | None = None,
               prm: CostParams | None = None, pipeline: int = 1, l1_reserve: int = 0) -> tuple[int, int, int]:
    """Cheapest (tm, tn, tk) under the analytic model; (256, 256, 256) when nothing else divides the shape.
    l1_reserve: TCDM bytes to keep free (0: the whole TCDM, as sh_gemm assumes)."""
    arch = arch or Arch()
    prm = prm or CostParams()
    best, best_c = None, math.inf
    for tm, tn, tk in gemm_candidates(M, N, K, arch.cluster_tcdm_size - l1_reserve, pipeline):
        c = gemm_est(arch, prm, M, N, K, tm, tn, tk, pipeline, 0, all_clusters).cycles
        # ties (within 1 %): prefer the squarer / larger tile, i.e. fewer, bigger DMA bursts
        if c < best_c * 0.99 or (best is not None and abs(c - best_c) <= 0.01 * best_c and tm * tn > best[0] * best[1]):
            best, best_c = (tm, tn, tk), c
    return best or (256, 256, 256)


def gemm_rank(M: int, N: int, K: int, all_clusters: bool = True, arch: Arch | None = None,
              prm: CostParams | None = None, top: int = 8) -> list[tuple[tuple[int, int, int], float, int]]:
    """[(tile, modelled cycles, tiles)] sorted by cost, for reports."""
    arch = arch or Arch()
    prm = prm or CostParams()
    rows = []
    for tm, tn, tk in gemm_candidates(M, N, K, arch.cluster_tcdm_size):
        rows.append(((tm, tn, tk), gemm_est(arch, prm, M, N, K, tm, tn, tk, 1, 0, all_clusters).cycles, (M // tm) * (N // tn)))
    rows.sort(key=lambda r: r[1])
    return rows[:top]


def attention_l1_bytes(S: int, dh: int, sq: int, base: int = 0x1000, nprof: int = 8) -> int:
    """sh_attention_l1_bytes_q: q | k | kT | s | v | o | sums | stamps from SH_ATTN_L1_BASE."""
    kb, qb, sb = S * dh * ELEM, sq * dh * ELEM, sq * S * ELEM
    return base + qb + kb + kb + sb + kb + qb + sq * 4 + nprof * 4


def attention_q_block(S: int, dh: int, H: int, P: int, l1: int = 0x100000) -> int:
    """The library's default rule (sh_attention_q_block): the largest sq in {S, 256, 128, 64} that divides S,
    fits L1 and minimises the rows per cluster ceil(H * S/sq / P) * sq. 0 if nothing fits."""
    best, best_rows = 0, 0
    for sq in (S, 256, 128, 64):
        if sq == 0 or sq > S or S % sq or sq % 4 or attention_l1_bytes(S, dh, sq) > l1:
            continue
        rows = math.ceil(H * (S // sq) / P) * sq
        if not best or rows < best_rows:
            best, best_rows = sq, rows
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="rank GEMM tile shapes with the analytic model")
    ap.add_argument("shapes", nargs="+", help="MxNxK ...")
    ap.add_argument("--cluster0", action="store_true", help="single cluster instead of SH_ALL")
    a = ap.parse_args()
    for s in a.shapes:
        M, N, K = (int(v) for v in s.lower().split("x"))
        print(f"{M}x{N}x{K}: pick {gemm_tiles(M, N, K, not a.cluster0)}")
        for tile, c, n in gemm_rank(M, N, K, not a.cluster0):
            print(f"   {tile[0]:>4}x{tile[1]:<4}x{tile[2]:<4} tiles={n:<4} model {c / 1e3:8.1f}k cycles")
