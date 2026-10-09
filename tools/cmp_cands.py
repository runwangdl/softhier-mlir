"""Compare an N-candidate expert run with N independent N=1 runs (tests/gvsoc/expert.py flow --save-x): python tools/cmp_cands.py nN.npz n1_a.npz n1_b.npz ..."""
import sys
import numpy as np

c = np.load(sys.argv[1])["x"]
singles = [np.load(f)["x"] for f in sys.argv[2:]]
S = 50
for s in range(c.shape[0]):
    parts = []
    for i, a in enumerate(singles):
        blk = c[s, i * S:(i + 1) * S]
        parts.append(f"cand{i}: maxdiff {np.abs(blk - a[s]).max():.6f} bit-identical {np.array_equal(blk, a[s])}")
    print(f"step {s}: " + " | ".join(parts))
