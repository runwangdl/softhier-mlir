#!/usr/bin/env python3
"""Bytes moved by the iDMAs of a gvsoc run, from its `--trace=idma` log (every finished burst lists its transactions:
1-D {src, dst, size} and 2-D {src, dst, size per repeat, repeats}), by kind and by program segment.

    python tests/gvsoc/dma_bytes.py <log>          (or tests/gvsoc/expert.py flow ... --trace-dma)

Kinds, by the source / destination address (flex_cluster address map):
    hbm_rd   source in HBM (>= 0xC0000000)                       HBM -> cluster over the NoC
    hbm_wr   destination in HBM                                  cluster -> HBM over the NoC
    c2c      source or destination in another cluster's TCDM     cluster <-> cluster over the NoC (a multicast is
             (0x30000000 + cid * 1 MB)                           counted once: the bytes its source injects)
    zero     source in the zero memory (0x18000000)              TCDM clearing, no NoC
    local    TCDM -> TCDM of the same cluster                    (K transposes, query stacking), no NoC
Segments: a transaction belongs to the program segment (between two `[mark]` lines of cluster 0, whose mcycle equals the
simulated ns at 1 GHz) in which its burst finished; segment names drop the trailing index like expert.py's breakdown.
"""
from __future__ import annotations

import re
import sys
from bisect import bisect_right
from collections import defaultdict

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FIN = re.compile(r"\[iDMA\] Finished : (\d+) ns ---> (\d+) ns")
_TXN = re.compile(r"Txn \d+ = \{type: (1D|2D), src: 0x([0-9a-f]+)(?:, src_stride: 0x[0-9a-f]+)?, dst: 0x([0-9a-f]+)"
                  r"(?:, dst_stride: 0x[0-9a-f]+, repeats: 0x([0-9a-f]+))?, size: 0x([0-9a-f]+)")
HBM, ZOMEM, REMOTE, REMOTE_END = 0xC0000000, 0x18000000, 0x30000000, 0x30000000 + 64 * 0x100000


def kind(src: int, dst: int) -> str:
    if src >= HBM:
        return "hbm_rd"
    if dst >= HBM:
        return "hbm_wr"
    if REMOTE <= src < REMOTE_END or REMOTE <= dst < REMOTE_END:
        return "c2c"
    if ZOMEM <= src < ZOMEM + 0x1000000:
        return "zero"
    return "local"


def parse(stdout: str):
    """-> (marks [(tag, ns)], txns [(end_ns, kind, bytes)])"""
    marks, txns = [], []
    for ln in stdout.splitlines():
        if ln.startswith("[mark] "):
            _, tag, c = ln.split()
            marks.append((tag, int(c)))
            continue
        if "[iDMA] Finished" not in ln:
            continue
        ln = _ANSI.sub("", ln)
        m = _FIN.search(ln)
        t1 = int(m[2])
        for t in _TXN.finditer(ln):
            src, dst, size = int(t[2], 16), int(t[3], 16), int(t[5], 16)
            reps = int(t[4], 16) if t[1] == "2D" else 1
            txns.append((t1, kind(src, dst), size * reps))
    return marks, txns


def by_segment(stdout: str) -> dict[str, dict[str, int]]:
    """{segment name: {kind: bytes}}; 'pre' = before the first mark. Assumes the 32-bit mcycle did not wrap."""
    marks, txns = parse(stdout)
    times = [c for _, c in marks]
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t1, k, b in txns:
        i = bisect_right(times, t1)          # the segment ends at mark i (the first mark after the burst finished)
        name = re.sub(r"\d+$", "", marks[i][0]) if i < len(marks) else "after"
        out[name][k] += b
    return out


def report(stdout: str) -> dict[str, dict[str, int]]:
    seg = by_segment(stdout)
    kinds = ("hbm_rd", "hbm_wr", "c2c", "local", "zero")
    print("     iDMA bytes per segment (KB):  " + "  ".join(f"{k:>9}" for k in kinds))
    tot = defaultdict(int)
    for name, d in seg.items():
        print(f"       {name:<10}" + " " * 20 + "  ".join(f"{d.get(k, 0) / 1e3:9.1f}" for k in kinds))
        for k in kinds:
            tot[k] += d.get(k, 0)
    print(f"       {'total':<10}" + " " * 20 + "  ".join(f"{tot[k] / 1e3:9.1f}" for k in kinds))
    return seg


if __name__ == "__main__":
    report(open(sys.argv[1]).read())
