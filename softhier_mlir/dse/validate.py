"""Model-vs-simulation check on whole kernels (sh_gemm / sh_gemm_mesh / row ops).

    python -m softhier_mlir.dse.validate --home /app/softhier_dse --params docs/dse/params.json
    python -m softhier_mlir.dse.validate --from-json docs/dse/validation_default.json --params docs/dse/params.json

Prints one line per kernel: simulated cycles (barrier subtracted), model cycles, error, and the
model's view of what bounds the kernel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from softhier_mlir.dse import calibrate as cb
from softhier_mlir.dse import cost
from softhier_mlir.dse.workload import OpRec
from softhier_mlir.sim import gvsoc

REFERENCE = [   # the kernels quoted in AI_AGENT/SoftHier/README.md section 6 + the SigLIP S=256 shapes
    cb.Case.gemm(256, 256, 256, mode="one"), cb.Case.gemm(512, 768, 768, mode="one"),
    cb.Case.gemm(1024, 768, 768), cb.Case.gemm(1024, 3072, 768),
    cb.Case.gemm(256, 768, 768), cb.Case.gemm(256, 3072, 768), cb.Case.gemm(256, 768, 3072),
    cb.Case.gemm(256, 256, 64, 256, 256, 64, mode="one"), cb.Case.gemm(256, 64, 256, 256, 64, 256, mode="one"),
    cb.Case.summa(1024, 1024, 1024), cb.Case.gemm(512, 768, 768, pipeline=0, mode="one"),
    cb.Case.gemm(256, 768, 768, 256, 256, 128),
]


def op_of(case: cb.Case) -> OpRec:
    p = case.p
    if case.kind == "gemm":
        return OpRec("gemm", (p[0], p[1], p[2]), (p[3], p[4], p[5]), pipeline=p[6], cluster=-1 if p[7] else 0)
    if case.kind == "summa":
        return OpRec("summa", (p[0], p[1], p[2]), (p[3], p[3], p[4]), pipeline=p[5], cluster=-1)
    return OpRec(case.kind, (p[0], p[1]), cluster=-1 if p[2] else 0)


def table(rows: list[dict], arch: gvsoc.Arch, prm: cost.CostParams) -> str:
    base = min(r["cycles"] for r in rows if r["case"].kind == "barrier" and r["cycles"] is not None)
    out = ["| kernel | sim cycles | model cycles | err % | bound | model note |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["case"].kind == "barrier" or r["cycles"] is None:
            continue
        op = op_of(r["case"])
        e = cost.op_est(arch, prm, op, 1)
        s = r["cycles"] - base
        out.append(f"| {op} | {s:,} | {e.cycles:,.0f} | {100 * (e.cycles - s) / s:+.1f} | {e.bound} | {e.note} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.environ.get("SOFTHIER_HOME"))
    ap.add_argument("--params", default="docs/dse/params.json")
    ap.add_argument("--from-json")
    ap.add_argument("-o", "--out", help="save the simulated rows as JSON")
    a = ap.parse_args()
    if a.home:
        gvsoc.set_home(a.home)
    prm = cost.params_from_json(a.params) if Path(a.params).exists() else cost.CostParams()
    rows = cb.load_rows(a.from_json) if a.from_json else cb.run_cases([cb.Case.barrier()] + REFERENCE)
    if a.out:
        Path(a.out).write_text(json.dumps([{"kind": r["case"].kind, "p": list(r["case"].p), "cycles": r["cycles"]} for r in rows], indent=1))
    print(table(rows, gvsoc.Arch(), prm))


if __name__ == "__main__":
    main()
