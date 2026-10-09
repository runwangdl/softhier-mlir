#!/usr/bin/env python3
"""Cluster sets (SH_GROUP, docs/SPATIAL_SPLIT.md) on gvsoc: every op that deals work over SH_ALL gives the SAME numbers
when it is dealt over a set of 8 or 4 clusters, and still matches its numpy / fp16-twin reference.

    python tests/gvsoc/sets.py                       # all tests x {all, set:0xa5a5 (8, a checkerboard), set:0x8421 (4, the diagonal)}
    python tests/gvsoc/sets.py --tests gemm rowops --sets all set:0x0fff set:0xf000

For each test the existing harness (tests/gvsoc/run.py gemm / rowops / llmops / attention, tests/gvsoc/expert.py op
attn / xattn) runs once per set; its reference check must pass, and every sampled output element must be bit-identical
to the SH_ALL run. The ROI per set is printed (a set of P clusters is expected to take longer than 16; how much is
what docs/SPATIAL_SPLIT.md measures on the real stages)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tests.gvsoc.expert as X  # noqa: E402
import tests.gvsoc.run as R  # noqa: E402
from softhier_mlir.frontend.clusters import parse  # noqa: E402
from softhier_mlir.testing import lcg  # noqa: E402

_last: dict = {}


def _capture(mod):
    real = mod.run_sim

    def wrapped(*a, **k):
        r = real(*a, **k)
        _last["r"] = r
        return r
    mod.run_sim = wrapped


_capture(R)
_capture(X)
R.GEMM_EXTRA = "#define DUMP_Z 256\n"     # the gemm test self-checks on the device; also dump Z samples for the bitwise check

TESTS = {
    "gemm": lambda cl: R.run_gemm([f"1024x768x768:256,256,256,1,0,{cl}", f"256x3072x768:128,256,256,1,0,{cl}"], 256),
    "rowops": lambda cl: R.run_rowops(256, 768, R.cluster_c(cl), 64),
    "llmops": lambda cl: R.run_llmops(128, 960, 15, 5, R.cluster_c(cl), 256),
    "attention": lambda cl: R.run_attention(256, 768, 12, R.cluster_c(cl), False, 128),
    "xattn": lambda cl: X.run_op("xattn", 3, parse(cl), 128),
    "attn": lambda cl: X.run_op("attn", 3, parse(cl), 128),
}


def run(tests: list[str], sets: list[str]) -> bool:
    all_ok = True
    rows = []
    for t in tests:
        base = None
        for cl in sets:
            print(f"=== {t} cluster={cl}", flush=True)
            if t == "gemm":                      # two shapes = two simulations: keep the samples of both
                outs = []
                real = R.run_sim

                def keep(*a, _real=real, **k):
                    r = _real(*a, **k)
                    outs.append(r["stdout"])
                    return r
                R.run_sim = keep
                ok = TESTS[t](cl)
                R.run_sim = real
                stdout = "\n".join(outs)
                roi = None
            else:
                ok = TESTS[t](cl)
                stdout = _last["r"]["stdout"]
                roi = _last["r"]["rois"]
            got = lcg.parse_samples(stdout)
            if base is None:
                base, same = got, True
            else:
                same = got == base
            print(f"    {t} cluster={cl}: reference {'PASS' if ok else 'FAIL'}; samples vs {sets[0]}: "
                  f"{'bit-identical' if same else 'DIFFERENT'} ({sum(len(v) for v in got.values())} values)")
            rows.append((t, cl, ok, same, roi))
            all_ok &= ok and same
    print("\nsummary")
    for t, cl, ok, same, roi in rows:
        print(f"  {t:<10} {cl:<14} ref={'PASS' if ok else 'FAIL'} same={'yes' if same else 'NO'} roi_ns={roi}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", default=list(TESTS))
    ap.add_argument("--sets", nargs="*", default=["all", "set:0xa5a5", "set:0x8421"])
    a = ap.parse_args()
    sys.exit(0 if run(a.tests, a.sets) else 1)
