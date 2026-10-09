#!/usr/bin/env python3
"""Tables of docs/FLOW_DATAFLOW.md from saved device runs (tests/gvsoc/expert.py flow --save-x / logs).

    python tests/gvsoc/flow_tables.py schedules --dir <runs dir> [--npz /app/models/smolvla_base/expert.npz]
        x_<name>.npz files (device x_t after every step, 10 steps x 50 x 32) -> action-chunk error of every schedule
        vs lerobot fp32, vs the device all-fp16 run (x_fp16.npz) and vs its own fp16-floor twin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCHEDULES = {   # name -> per-step format (the f_<name> runs of the W2 measurement)
    "fp16": ["fp16"] * 10,
    "fp8_0_2": ["fp8"] * 3 + ["fp16"] * 7,
    "fp8_0_5": ["fp8"] * 6 + ["fp16"] * 4,
    "fp8_all": ["fp8"] * 10,
    "fp8_7_9": ["fp16"] * 7 + ["fp8"] * 3,
}


def schedules(d: Path, npz: str) -> None:
    from softhier_mlir.frontend import smolvla_expert as E
    data = dict(np.load(npz))
    P = {k[2:]: data[k] for k in data if k.startswith("p_")}
    ref = data["ref_xt"][10].astype(np.float64)
    base = np.load(d / "x_fp16.npz")["x"][9].astype(np.float64) if (d / "x_fp16.npz").exists() else None
    q8 = E.quantize_expert(P)
    print("| schedule (layer GEMMs) | x_10 vs lerobot: max / mean / rms | vs device all-fp16: max / mean | vs own fp16-floor twin: max |")
    print("|---|---|---|---|")
    for name, sc in SCHEDULES.items():
        f = d / f"x_{name}.npz"
        if not f.exists():
            print(f"| {name} | (missing) | | |")
            continue
        a = np.load(f)["x"][9].astype(np.float64)
        twin = E.np_flow(P, fmt_steps=sc, q8=q8)[0][10].astype(np.float64)
        e = np.abs(a - ref)
        vb = f"{np.abs(a - base).max():.4f} / {np.abs(a - base).mean():.5f}" if base is not None else "-"
        print(f"| {name} | {e.max():.4f} / {e.mean():.5f} / {np.sqrt((e ** 2).mean()):.5f} | {vb} | {np.abs(a - twin).max():.4f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["schedules"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--npz", default="/app/models/smolvla_base/expert.npz")
    a = ap.parse_args()
    schedules(Path(a.dir), a.npz)
