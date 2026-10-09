#!/usr/bin/env python3
"""On-simulator test of the on-chip RSSM (runtime/sh_wm.inc.c, docs/WORLD_MODEL.md).

    python3 -m softhier_mlir.frontend.wm_rssm prepare --model step --K 128 --H 10 --out /app/models/wm_rssm/rssm_step.npz
    python tests/gvsoc/wm.py --npz /app/models/wm_rssm/rssm_step.npz --K 32 --H 10 [--cluster all] [--posterior]
    python tests/gvsoc/wm.py ... --sweep "32,64,128" --cluster 0,all          # cycles per (K, H, clusters), no dumps

Compares the device's h / z / a (sampled positions; per step) with the torch module (RSSM.prior + actor for
imagination, RSSM.step for --posterior) and with the fp16-floor numpy twin of the kernel (Twin(fp16=True)).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from softhier_mlir.frontend import wm_rssm as W  # noqa: E402
from softhier_mlir.sim.gvsoc import build_sw, run_sim  # noqa: E402
from softhier_mlir.sim.preload import make_preload_elf, sentinel_array  # noqa: E402
from softhier_mlir.testing import lcg  # noqa: E402

HERE = Path(__file__).resolve().parent
APP = HERE / "wm"
HBM_START = 0x1000
PH = ["load", "gemm", "ln", "gru", "softmax", "head", "io"]
WM_RE = re.compile(r"\[sh_wm_rssm\] (.*)")


def _al(n: int) -> int:
    return (n + 4095) & ~4095


def run(npz: str, K: int, H: int, cluster: str = "0", posterior: bool = False, record: bool = True, kb: int = 0,
        nsamples: int = 512, check: bool = True, timeout: int = 7200, debug: bool = False) -> dict:
    c, sd, d = W.load(npz)
    assert K <= d["h0"].shape[0] and H <= d["obs"].shape[0], "prepare with a larger K / H"
    blob, _ = W.pack(c, sd, posterior=posterior)
    h0, z0, a0, obs = d["h0"][:K], d["z0"][:K], d["a0"][:K], d["obs"][:H, :K]
    a0p = np.zeros((K, c.AP), np.float16); a0p[:, :c.act] = a0
    obp = np.zeros((H, K, c.OP), np.float16); obp[:, :, :c.obs] = obs
    rows = (H if record else 1) * K
    off, cur = {}, HBM_START
    pre = {}
    for name, arr, nbytes in (("BLOB", blob, blob.nbytes), ("H0", h0.astype(np.float16), None), ("Z0", z0.astype(np.float16), None),
                              ("A0", a0p, None), ("OBS", obp.reshape(H * K, c.OP), None),
                              ("HS", None, rows * c.deter * 2), ("ZS", None, rows * c.S * 2), ("AS", None, rows * c.AP * 2)):
        off[name] = cur
        if arr is not None:
            pre[cur] = arr if arr.ndim == 2 else arr.reshape(1, -1)
            nbytes = arr.nbytes
        cur += _al(nbytes)
    if debug:
        off["DBG"] = cur; cur += 10 * 0x10000
    sent = cur
    APP.mkdir(exist_ok=True)
    elf = make_preload_elf(APP / "preload.elf", {**pre, sent: sentinel_array()})
    cl = "SH_ALL" if cluster == "all" else str(int(cluster))
    hdr = (f"#define DETER {c.deter}\n#define STOCH {c.stoch}\n#define CLASSES {c.classes}\n#define HIDDEN {c.hidden}\n"
           f"#define UNITS {c.units}\n#define OBS {c.obs}\n#define ACT {c.act}\n#define K_TRAJ {K}\n#define H_STEPS {H}\n"
           f"#define POSTERIOR {int(posterior)}\n#define RECORD {int(record)}\n#define KB {kb}\n#define CLUSTER {cl}\n"
           f"#define NSAMPLES {nsamples if check else 1}\n#define SH_PRELOAD 0x{sent:x}\n"
           + "".join(f"#define OFF_{k} 0x{v:x}\n" for k, v in off.items()))
    if debug:
        widths = [c.S + c.AP, c.hidden, c.hidden + c.deter, 3 * c.deter, 3 * c.deter, c.deter + (c.units if posterior else 0),
                  c.S, c.S + c.deter, c.AO, c.S + c.AP]
        m0 = min(K, kb) if kb else K
        hdr += f"#define DBG_M {m0}\n" + "".join(f"#define DBG_W{i} {w}\n" for i, w in enumerate(widths))
    (APP / "shape.h").write_text(hdr)
    build_sw(APP)
    r = run_sim(timeout=timeout, preload=elf)
    (APP / "last_run.log").write_text(r["stdout"])
    out = {"ok": r["ok"] and "WM_DONE" in r["stdout"], "roi": r["roi_ns"], "wall": r["wall_s"], "K": K, "H": H,
           "cluster": cluster, "posterior": posterior}
    m = WM_RE.search(r["stdout"])
    if m:
        out["line"] = m.group(1)
        nums = dict(re.findall(r"(\w+) (\d+)", m.group(1).split("|")[1]))
        out["phases"] = {k: int(nums[k]) for k in PH if k in nums}
        out["rows"] = int(re.search(r"rows/cluster=(\d+)", m.group(1)).group(1))
        out["chunk"] = int(re.search(r"chunk=(\d+)", m.group(1)).group(1))
    if not out["ok"]:
        print(r["stdout"][-3000:])
        return out
    ph = out.get("phases", {})
    step = sum(v for k, v in ph.items() if k not in ("load", "io")) / H
    out["step_cycles"] = step
    print(f"{'PASS' if out['ok'] else 'FAIL'} wm K={K} H={H} cluster={cluster} posterior={int(posterior)} roi={r['roi_ns']} ns wall={r['wall_s']}s")
    print(f"     {out.get('line', '')}")
    print(f"     per step (cluster 0, compute phases): {step:.0f} cycles; " +
          " ".join(f"{k} {100 * ph[k] / (step * H):.1f}%" for k in ("gemm", "ln", "gru", "softmax", "head") if k in ph) +
          f"; weight staging {ph.get('load', 0)} cycles, state io {ph.get('io', 0)} cycles")
    if not check:
        return out
    # ---- accuracy
    got = lcg.parse_samples(r["stdout"])
    if debug:
        _debug(c, sd, got, h0, z0, a0)
    tw16 = W.Twin(c, sd, fp16=True)
    th = tw16.rollout(h0, z0, a0, H, obs=obs if posterior else None)
    pre_ = "ref_post_" if posterior else "ref_"
    refs = {"HS": d[pre_ + "hs"][:H, :K], "ZS": d[pre_ + "zs"][:H, :K], "AS": d[pre_ + "as"][:H, :K]}
    twin = {"HS": th[0], "ZS": th[1], "AS": th[2]}
    ok = out["ok"]
    out["err"] = {}
    for tag in ("HS", "ZS", "AS"):
        if tag not in got:
            print(f"     {tag} MISSING"); ok = False; continue
        ref, tw = refs[tag], twin[tag]
        if not record:
            ref, tw = ref[-1:], tw[-1:]
        cols = ref.shape[-1]
        e_ref = np.zeros(ref.shape[0]); e_tw = np.zeros(ref.shape[0]); n = np.zeros(ref.shape[0], int)
        for row, col, v in got[tag]:
            if col >= cols:
                continue                                  # the action's padding lane
            t, k = divmod(row, K)
            e_ref[t] = max(e_ref[t], abs(v - ref[t, k, col])); e_tw[t] = max(e_tw[t], abs(v - tw[t, k, col])); n[t] += 1
        out["err"][tag] = (e_ref.tolist(), e_tw.tolist())
        tol = 0.01 + 0.006 * np.arange(ref.shape[0])            # fp16 floor: ~1e-3 per step, growing along the open loop
        good = bool((e_ref <= tol).all())
        ok &= good
        print(f"     {tag} ({int(n.sum())} samples) max abs per step vs torch: " + " ".join(f"{v:.4f}" for v in e_ref) +
              "\n        vs fp16 twin: " + " ".join(f"{v:.4f}" for v in e_tw) + f"  {'PASS' if good else 'FAIL'}")
    out["ok"] = ok
    return out


def _debug(c, sd, got, h0, z0, a0):
    """step-0 intermediates (prior path) of the device vs the fp16 twin, slot by slot"""
    tw = W.Twin(c, sd, fp16=True)
    r = tw.r
    h, z, a = r(h0), r(z0), r(a0)
    za = tw._za(z, a)
    t1 = tw.mm(za, "w_in"); x = tw.ln(t1, "g_in", "b_in")
    xh = np.concatenate([x, h], -1)
    pg = tw.mm(xh, "w_g"); pl = tw.ln(pg, "g_g", "b_g", relu=False)
    hn = tw.gru(x, h)
    y = tw.ln(tw.mm(hn, "w_io"), "g_io", "b_io"); lg = tw.mm(y, "w_is")
    zn = tw.latent(lg + tw.w["b_is"])
    f = np.concatenate([zn, hn], -1)
    q = tw.ln(tw.mm(f, "w_a1"), "g_a1", "b_a1"); q = tw.ln(tw.mm(q, "w_a2"), "g_a2", "b_a2"); o = tw.mm(q, "w_ao")
    an = tw.actor(zn, hn)
    za2 = tw._za(zn, an)
    refs = [za, t1, xh, pg, pl, hn, lg, f, o, za2]
    names = ["ZA in", "img_in GEMM", "XH (x|h)", "GRU GEMM", "GRU LN", "h'", "logits", "F (z'|h')", "actor out", "ZA out"]
    for i, (nm, ref) in enumerate(zip(names, refs)):
        smp = got.get(f"D{i}", [])
        err = [abs(v - ref[rr, cc]) for rr, cc, v in smp if cc < ref.shape[1] and rr < ref.shape[0]]
        worst = max(((abs(v - ref[rr, cc]), rr, cc, v, ref[rr, cc]) for rr, cc, v in smp if cc < ref.shape[1]), default=None)
        print(f"     dbg {i} {nm:<12} n={len(err)} max abs {max(err) if err else float('nan'):.4f} (|ref| max {np.abs(ref).max():.3f}) worst {worst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/app/models/wm_rssm/rssm_step.npz")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--cluster", default="0", help="0 | all (comma list with --sweep)")
    ap.add_argument("--posterior", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--kb", type=int, default=0)
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--debug", action="store_true", help="dump step-0 intermediates (prior path) and compare slot by slot")
    ap.add_argument("--sweep", help="comma list of K: timing only (no sample dumps)")
    a = ap.parse_args()
    if a.sweep:
        res = []
        for cl in a.cluster.split(","):
            for K in (int(v) for v in a.sweep.split(",")):
                r = run(a.npz, K, a.H, cl, a.posterior, not a.no_record, a.kb, check=False)
                res.append(r)
        print("K,H,clusters,rows_per_cluster,chunk,roi_ns,step_cycles," + ",".join(PH))
        for r in res:
            ph = r.get("phases", {})
            print(f"{r['K']},{r['H']},{r['cluster']},{r.get('rows')},{r.get('chunk')},{r['roi']},{r.get('step_cycles', 0):.0f}," + ",".join(str(ph.get(k, '')) for k in PH))
        sys.exit(0 if all(r["ok"] for r in res) else 1)
    r = run(a.npz, a.K, a.H, a.cluster, a.posterior, not a.no_record, a.kb, a.nsamples, debug=a.debug)
    sys.exit(0 if r["ok"] else 1)
