"""Analytic cost model of the softhier-ops library on a flex_cluster architecture.

Every estimate is cycles at the cluster clock (1 GHz on gvsoc: cycles == ns). The model follows
the library's control flow (runtime/sh_gemm.inc.c, sh_rowops.inc.c) and the gvsoc component
models (light_redmule.cpp, the iDMA / FlooNoC / ideal-HBM rates); the handful of constants it
cannot derive are in ``CostParams`` and are fitted by ``softhier_mlir.dse.calibrate`` on the
micro-benchmarks (docs/DSE.md has the fitted values and the model-vs-simulation errors).

Model structure
  RedMulE tile   max(TCDM block accesses, array runtime) per (i, j, k) buffer iteration, with
                 ce_height x ce_width MAC/cycle; a contraction tile < buffer_n or an output-column
                 tile < ce_width*(ce_pipe+1) leaves part of the array idle (attention dh = 64).
  DMA            fixed latency + per-row cost + bytes / (noc_link_width / 8); clusters streaming
                 at the same time share the HBM channel(s) the buffers live in.
  sh_gemm        per output tile: prologue load, then per K step max(compute, next load) when
                 pipelined (sum otherwise), store; output tiles dealt round-robin over clusters.
  sh_gemm_mesh   SUMMA: diagonal load + 2 serialized broadcasts (measured fixed cost + line rate)
                 per K step overlapped with compute; group barriers.
  row ops        sequential stage-in / scalar compute / stage-out per row block, blocks round-robin
                 over clusters; the scalar cost per element is a per-op parameter.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from softhier_mlir.dse.workload import OpRec, Workload
from softhier_mlir.sim.gvsoc import Arch

ELEM = 2


@dataclass
class CostParams:
    """Calibrated constants (cycles unless noted). Defaults: gvsoc flex_cluster, ideal HBM,
    fitted 2026-10-08 (docs/DSE.md). Everything a hardware change does not alter lives here."""
    # RedMulE (light_redmule.cpp with queue_depth = 1): exact to 0.4% on 13 tile shapes
    redmule_block: float = 1.0          # cycles per TCDM block request (bandwidth-wide) in the FSM
    redmule_fixed: float = 125.0        # config + trigger + wait + intra-cluster sync, per trigger
    # iDMA / NoC
    dma_fixed: float = 189.0            # per 2-D load: issue, NoC round trip, wait
    dma_row: float = 0.0                # per 2-D row (burst) of a load: not visible
    store_row: float = 24.0             # per 1-D row store (issued one by one; issue-bound for rows <= 1.5 KB)
    hbm_channels: int = 1               # HBM channels the kernel's buffers hit (all in node 0 today)
    hbm_bw_scale: float = 1.32          # 16 clusters streaming: 84.7 B/cycle aggregate = 1.32 x one 512 b link
    # measured aggregate HBM->TCDM rate (B/cycle at 512 b links) vs number of concurrently streaming
    # clusters 0..n-1; NOT monotonic (depends on which mesh rows are busy). Interpolated; scaled with
    # the link width. Empty -> hbm_bw_scale is used instead.
    hbm_agg: dict = field(default_factory=lambda: {1: 64.0, 2: 116.8, 4: 83.0, 8: 125.4, 16: 84.7})
    l1_zero_bw: float = 64.0            # bytes/cycle of the ZOMEM -> TCDM clear
    # synchronisation / collectives (collectives_measured.md, 4x4 mesh, 512 b links)
    barrier: float = 259.0              # global barrier as timed around an empty region (118 + wake-up)
    group_barrier: float = 259.0
    cluster_sync: float = 30.0          # flex_intra_cluster_sync
    bcast_fixed: float = 106.0
    redadd_fixed: float = 127.0
    # scalar row ops: cycles per element on the first core (software fp16 conversion; another
    # agent is replacing these). layernorm also reads gamma/beta from HBM per element.
    elem: dict = field(default_factory=lambda: {
        "layernorm": 296.0, "softmax": 346.0, "gelu": 146.0, "add": 92.0, "add_bias": 92.0,
        "scale": 92.0, "transpose": 16.0})
    rowop_fixed: float = 300.0          # per row block (3 syncs + DMA waits)


# ----------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------
def link_bw(arch: Arch) -> float:
    """Bytes/cycle of one NoC link (= one cluster's HBM->TCDM stream, = one ideal HBM channel)."""
    return arch.noc_link_width / 8


def hbm_bw(arch: Arch, prm: CostParams) -> float:
    return link_bw(arch) * prm.hbm_channels * prm.hbm_bw_scale


def stream_bw(arch: Arch, prm: CostParams, n_active: int) -> float:
    """Per-cluster HBM rate when clusters 0..n_active-1 stream at once: the measured aggregate
    curve (linear interpolation in n, scaled with the link width), capped by one link."""
    n = max(1, n_active)
    if prm.hbm_agg:
        pts = sorted((int(k), float(v)) for k, v in prm.hbm_agg.items())
        scale = link_bw(arch) / 64.0
        if n <= pts[0][0]:
            agg = pts[0][1]
        elif n >= pts[-1][0]:
            agg = pts[-1][1]
        else:
            for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
                if n0 <= n <= n1:
                    agg = a0 + (a1 - a0) * (n - n0) / (n1 - n0)
                    break
        return min(link_bw(arch), agg * scale * prm.hbm_channels / n)
    return min(link_bw(arch), hbm_bw(arch, prm) / n)


def dma_load(arch: Arch, prm: CostParams, rows: int, cols: int, n_active: int = 1) -> float:
    return prm.dma_fixed + prm.dma_row * rows + rows * cols * ELEM / stream_bw(arch, prm, n_active)


def dma_store(arch: Arch, prm: CostParams, rows: int, cols: int, n_active: int = 1) -> float:
    """Per-row 1-D stores issued back to back then waited: issue-bound for short rows
    (~25 cycles per bare_dma_start_1d), bandwidth-bound for long ones."""
    return prm.dma_fixed + max(prm.store_row * rows, rows * cols * ELEM / stream_bw(arch, prm, n_active))


def redmule_buffers(arch: Arch) -> tuple[int, int, int]:
    """(buffer_h, buffer_w, buffer_n): rows of X per pass, output columns per pass, contraction
    elements per pass (one TCDM-bandwidth-wide block)."""
    bh = arch.redmule_ce_height
    bw = arch.redmule_ce_width * (arch.redmule_ce_pipe + 1)
    bn = (arch.cluster_tcdm_bank_width // 8) * arch.cluster_tcdm_bank_nb // arch.redmule_elem_size
    return bh, bw, bn


def redmule_cycles(arch: Arch, prm: CostParams, tm: int, tn: int, tk: int) -> float:
    """One trigger of Y[tm,tn] += X[tm,tk] . W[tk,tn] (library call sh_redmule / one K step of sh_gemm).
    Mirrors LightRedmule: PRELOAD (X, Y blocks) -> ROUTINE over (i, j, k) buffer tiles, each
    max(block accesses, array runtime) -> STORING (Z blocks)."""
    bh, bw, bn = redmule_buffers(arch)
    pipe1 = arch.redmule_ce_pipe + 1
    ni, nj, nk = math.ceil(tm / bh), math.ceil(tn / bw), math.ceil(tk / bn)
    lefts = tk % bn
    cb = prm.redmule_block
    h_first = min(tm, bh)
    cycles = (2 * h_first) * cb                                   # preload: X + Y blocks
    first = True
    for i in range(ni):
        hi = min(bh, tm - i * bh)
        for j in range(nj):
            for k in range(nk):
                last_k = k == nk - 1
                n_here = lefts if (last_k and lefts) else bn
                runtime = n_here * pipe1
                runtime = math.ceil(runtime / bw) * bw
                is_last = (i == ni - 1) and (j == nj - 1) and last_k
                blocks = n_here                                   # W rows
                if not is_last:
                    # X block of the next (i, k) and, at the last k, the Y block of the next (i, j)
                    ni2 = i if not (last_k and j == nj - 1) else i + 1
                    blocks += min(bh, tm - ni2 * bh)
                    if last_k:
                        blocks += min(bh, tm - ni2 * bh)
                if k == 0 and not first:
                    blocks += hi                                   # Z store of the previous (i, j)
                cycles += max(runtime, blocks * cb)
                first = False
    cycles += min(bh, tm - (ni - 1) * bh) * cb + bw               # storing + routine->storing latency
    return cycles + prm.redmule_fixed


def redmule_utilization(arch: Arch, prm: CostParams, tm: int, tn: int, tk: int) -> float:
    return tm * tn * tk / (arch.macs_per_cycle * redmule_cycles(arch, prm, tm, tn, tk))


# ----------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------
@dataclass
class Est:
    cycles: float
    compute: float = 0.0        # RedMulE / scalar busy time on the critical cluster
    dma: float = 0.0            # DMA busy time on the critical cluster
    bound: str = ""
    note: str = ""

    def __float__(self) -> float:
        return self.cycles


def gemm_est(arch: Arch, prm: CostParams, M: int, N: int, K: int, tm: int, tn: int, tk: int,
             pipeline: int = 1, accumulate: int = 0, all_clusters: bool = True, n_lanes: int = 1) -> Est:
    """sh_gemm. all_clusters: tiles dealt round-robin over the whole mesh (SH_ALL). Otherwise one
    cluster runs it while n_lanes clusters in total stream from HBM (concurrent attention heads)."""
    tiles = (M // tm) * (N // tn)
    KT = K // tk
    P = arch.n_clusters if all_clusters else 1
    n_active = min(P, tiles) if all_clusters else n_lanes
    my_tiles = math.ceil(tiles / P)
    comp = redmule_cycles(arch, prm, tm, tn, tk)
    load_k = dma_load(arch, prm, tm, tk, n_active) + dma_load(arch, prm, tk, tn, n_active)
    store = dma_store(arch, prm, tm, tn, n_active)
    pre = dma_load(arch, prm, tm, tn, n_active) if accumulate else prm.dma_fixed + tm * tn * ELEM / prm.l1_zero_bw
    if pipeline:
        per_tile = pre + load_k + (KT - 1) * max(comp, load_k) + comp + store + (KT + 2) * prm.cluster_sync
    else:
        per_tile = pre + load_k + KT * comp + (KT - 1) * load_k + store + (2 * KT + 2) * prm.cluster_sync
    total = my_tiles * per_tile + (prm.barrier if all_clusters else 0)
    c, d = my_tiles * KT * comp, my_tiles * (pre + KT * load_k + store)
    return Est(total, c, d, "compute" if c >= d else "dma",
               f"{my_tiles} tiles/cluster, {n_active} clusters streaming, tile {comp:.0f} vs load {load_k:.0f}")


def summa_est(arch: Arch, prm: CostParams, M: int, N: int, K: int, T: int, tk: int, pipeline: int = 1,
              accumulate: int = 0) -> Est:
    """sh_gemm_mesh on the P x P mesh (M == N == P*T)."""
    P = arch.num_cluster_x
    if arch.num_cluster_y != P or M != P * T or N != P * T:
        return Est(float("inf"), note="SUMMA needs a square mesh and M == N == P*T")
    KT = K // tk
    comp = redmule_cycles(arch, prm, T, T, tk)
    xb, wb = T * tk * ELEM, tk * T * ELEM
    diag = dma_load(arch, prm, T, tk, P) + dma_load(arch, prm, tk, T, P)        # P diagonals share HBM
    bc = 2 * prm.bcast_fixed + (xb + wb) / link_bw(arch)                          # two serialized multicasts
    step_feed = diag + bc
    if pipeline:
        per_step = max(comp, step_feed) + prm.group_barrier
    else:
        per_step = comp + step_feed + 2 * prm.group_barrier
    pre = (dma_load(arch, prm, T, T, P) if accumulate else prm.dma_fixed + T * T * ELEM / prm.l1_zero_bw) + step_feed + prm.group_barrier
    total = prm.barrier + pre + KT * per_step + dma_store(arch, prm, T, T, P * P) + prm.barrier
    return Est(total, KT * comp, KT * step_feed, "compute" if comp >= step_feed else "feed",
               f"tile {comp:.0f} vs diag load+bcast {step_feed:.0f}")


XM_L1_LIMIT = 0x90000          # runtime/sh_gemm.inc.c SH_XM_L1_LIMIT
XM_CHUNK = 32768               # bytes per multicast collective
XM_WHOLE_MAX = 0x50000         # SH_XM_WHOLE_MAX: auto picks the whole-X schedule only up to this X row block


def xmcast_plan(arch: Arch, M: int, N: int, K: int, tm: int = 0, tn: int = 0, tk: int = 0, mode: str = "auto",
                l1_base: int = 0) -> dict | None:
    """Python twin of sh_xm_make_plan (runtime/sh_gemm.inc.c): rows per block, column slice Nc per cluster, K-panel tk,
    whole (X multicast once) or panel schedule, TCDM bytes. None when nothing fits."""
    P = arch.n_clusters
    g = tn or 4
    rows = tm or M
    if M % rows:
        return None
    Nc = -(-(-(-N // P)) // g) * g
    lim = XM_L1_LIMIT - l1_base

    def need(t, whole):
        return (rows * K * ELEM if whole else 2 * rows * t * ELEM) + 2 * t * Nc * ELEM + rows * Nc * ELEM
    auto_whole = rows * K * ELEM <= XM_WHOLE_MAX
    for whole in ((False,) if mode == "panel" else (True,) if mode == "whole" else (True, False) if auto_whole else (False,)):
        if tk:
            t = tk if K % tk == 0 and need(tk, whole) <= lim else 0
        else:
            t = next((d for d in range(min(K, 256), 0, -1) if K % d == 0 and need(d, whole) <= lim), 0)
        if t:
            nact = min(P, -(-N // Nc))
            return {"rows": rows, "Nc": Nc, "tk": t, "KT": K // t, "whole": whole, "l1": l1_base + need(t, whole),
                    "active": nact, "last_nc": N - (nact - 1) * Nc}
    return None


def gemm_xmcast_traffic(M: int, N: int, K: int, rows: int | None = None) -> dict:
    """HBM bytes of sh_gemm_xmcast_ex: X once, W once per row block, Z once; NoC multicast bytes injected (X once)."""
    MT = M // (rows or M)
    return {"x": M * K * ELEM, "w": MT * K * N * ELEM, "z": M * N * ELEM, "mcast": M * K * ELEM}


def gemm_xmcast_est(arch: Arch, prm: CostParams, M: int, N: int, K: int, tm: int = 0, tn: int = 0, tk: int = 0,
                    mode: str = "auto", accumulate: int = 0) -> Est:
    """sh_gemm_xmcast_ex. Per row block: Y clear; whole schedule = cluster 0 loads X (KT 2-D loads) while every active
    cluster loads its W panel 0, multicast (32 KB collectives at line rate + fixed cost each), global barrier, then a
    local double-buffered K loop max(RedMulE, W panel load); panel schedule = per K-panel max(RedMulE, source cluster's
    X panel load + its W panel load + multicast) + a global barrier. W panels: `active` clusters stream at once."""
    p = xmcast_plan(arch, M, N, K, tm, tn, tk, mode)
    if p is None:
        return Est(float("inf"), note="no xmcast plan fits TCDM")
    rows, Nc, t, KT, n = p["rows"], p["Nc"], p["tk"], p["KT"], p["active"]
    MT = M // rows
    comp = redmule_cycles(arch, prm, rows, Nc, t)
    wload = dma_load(arch, prm, t, Nc, n)
    pre = (dma_load(arch, prm, rows, Nc, n) if accumulate else prm.dma_fixed + rows * Nc * ELEM / prm.l1_zero_bw)
    store = dma_store(arch, prm, rows, Nc, n)

    def mcast(b):
        return math.ceil(b / XM_CHUNK) * prm.bcast_fixed + b / link_bw(arch)
    if p["whole"]:
        xload = KT * prm.dma_fixed + rows * K * ELEM / link_bw(arch)
        head = max(xload + mcast(rows * K * ELEM), wload) + prm.barrier
        loop = (KT - 1) * (max(comp, wload) + prm.cluster_sync) + comp + prm.cluster_sync
    else:
        xload = prm.dma_fixed + rows * t * ELEM / link_bw(arch)
        head = xload + mcast(rows * t * ELEM) + wload + prm.barrier
        feed = xload + wload + mcast(rows * t * ELEM)
        loop = (KT - 1) * (max(comp, feed) + prm.barrier) + comp + prm.barrier
    per_block = pre + head + loop + store + prm.cluster_sync
    total = 2 * prm.barrier + MT * per_block + (MT - 1) * prm.barrier
    c, d = MT * KT * comp, MT * (KT * wload + pre + store)
    return Est(total, c, d, "compute" if c >= d else "dma",
               f"{'whole' if p['whole'] else 'panel'} tk={t} KT={KT} Nc={Nc} x{n} clusters, tile {comp:.0f} vs W panel {wload:.0f}")


def rowop_est(arch: Arch, prm: CostParams, kind: str, rows: int, cols: int, all_clusters: bool = True,
              n_lanes: int = 1) -> Est:
    """sh_rowop family + sh_transpose. Sequential load / scalar / store per block; blocks dealt
    round-robin over the clusters (SH_ALL) or run by one cluster."""
    P = arch.n_clusters if all_clusters else 1
    ce = prm.elem[kind]
    if kind == "transpose":
        B = 64
        nblk = math.ceil(rows / B) * math.ceil(cols / B)
        n_active = min(P, nblk) if all_clusters else n_lanes
        per = dma_load(arch, prm, B, B, n_active) + B * B * ce + dma_store(arch, prm, B, B, n_active) + prm.rowop_fixed
        my = math.ceil(nblk / P)
        total = my * per + (prm.barrier if all_clusters else 0)
        return Est(total, my * B * B * ce, my * (per - B * B * ce - prm.rowop_fixed), "scalar", f"{my} blocks/cluster")
    nin = 2 if kind == "add" else 1
    rowb = cols * ELEM
    rpb = max(1, min(0x40000 // (rowb * (nin + 1)), rows))
    if all_clusters:
        rpb = min(rpb, max(1, math.ceil(rows / P)))
    sizes = [rpb] * (rows // rpb) + ([rows % rpb] if rows % rpb else [])   # the last block is partial
    nblk = len(sizes)
    n_active = min(P, nblk) if all_clusters else n_lanes
    # blocks are dealt round-robin: the critical cluster is the one with the most rows
    lanes = [sizes[c::P] for c in range(min(P, nblk))]
    mine = max(lanes, key=sum)

    def block(nr: int) -> tuple[float, float]:
        load = dma_load(arch, prm, nr, cols, n_active) + (dma_load(arch, prm, nr if kind == "add" else 1, cols, n_active) if kind in ("add", "add_bias") else 0)
        return nr * cols * ce, load + dma_store(arch, prm, nr, cols, n_active) + prm.rowop_fixed

    comp = sum(block(nr)[0] for nr in mine)
    dma = sum(block(nr)[1] for nr in mine)
    total = comp + dma + (prm.barrier if all_clusters else 0)
    return Est(total, comp, dma, "scalar" if comp >= dma else "dma", f"{len(mine)} blocks, {sum(mine)} rows on the critical cluster (block {rpb} rows)")


def op_est(arch: Arch, prm: CostParams, op: OpRec, n_lanes: int = 1) -> Est:
    if op.kind == "barrier":
        return Est(prm.barrier, bound="sync")
    if op.kind == "gemm":
        M, N, K = op.shape
        tm, tn, tk = op.tiles
        if M % tm or N % tn or K % tk:
            return Est(float("inf"), note="shape not divisible by tile")
        if 2 * (tm * tk + tk * tn) * ELEM * (1 if op.pipeline else 0.5) + tm * tn * ELEM > arch.cluster_tcdm_size:
            return Est(float("inf"), note="tiles exceed TCDM")
        return gemm_est(arch, prm, M, N, K, tm, tn, tk, op.pipeline, op.accumulate, op.cluster < 0, n_lanes)
    if op.kind == "xmcast":
        M, N, K = op.shape
        tm, tn, tk = (tuple(op.tile) + (0, 0, 0))[:3]
        return gemm_xmcast_est(arch, prm, M, N, K, tm, tn, tk, accumulate=op.accumulate)
    if op.kind == "summa":
        M, N, K = op.shape
        return summa_est(arch, prm, M, N, K, op.tiles[0], op.tiles[2], op.pipeline, op.accumulate)
    return rowop_est(arch, prm, op.kind, op.shape[0], op.shape[1], op.cluster < 0, n_lanes)


# ----------------------------------------------------------------------------------------
# workload composition (timeline with per-cluster lanes)
# ----------------------------------------------------------------------------------------
def regions(wl: Workload) -> list[list[OpRec]]:
    """Split the program into maximal runs of per-cluster (asynchronous) ops; sync ops are
    singleton regions. Within an async region, ops on different clusters overlap."""
    out: list[list[OpRec]] = []
    cur: list[OpRec] = []
    for op in wl:
        if op.kind in ("barrier", "summa", "xmcast") or op.cluster < 0:
            if cur:
                out.append(cur)
                cur = []
            out.append([op])
        else:
            cur.append(op)
    if cur:
        out.append(cur)
    return out


def compose(wl: Workload, cost_of, prm: CostParams | None = None) -> dict:
    """Timeline composition: sync ops add their cost; an async region adds the longest cluster
    lane. ``cost_of(op, n_lanes) -> float`` supplies per-op cycles (analytic or simulated).
    Returns total cycles plus per-kind and per-op breakdowns."""
    total = 0.0
    by_kind: dict[str, float] = OrderedDict()
    per_op: list[tuple[OpRec, float, float]] = []    # (op, own cycles, contribution to total)
    for reg in regions(wl):
        if len(reg) == 1 and (reg[0].kind in ("barrier", "summa", "xmcast") or reg[0].cluster < 0):
            op = reg[0]
            c = cost_of(op, 1)
            total += c
            by_kind[op.kind] = by_kind.get(op.kind, 0.0) + c
            per_op.append((op, c, c))
            continue
        lanes: dict[int, float] = {}
        n_lanes = len({o.cluster for o in reg})
        own = []
        for op in reg:
            c = cost_of(op, n_lanes)
            lanes[op.cluster] = lanes.get(op.cluster, 0.0) + c
            own.append(c)
        crit = max(lanes.values())
        crit_cluster = max(lanes, key=lanes.get)
        total += crit
        for op, c in zip(reg, own):
            contrib = c if op.cluster == crit_cluster else 0.0
            by_kind[op.kind] = by_kind.get(op.kind, 0.0) + contrib
            per_op.append((op, c, contrib))
    return {"cycles": total, "by_kind": by_kind, "per_op": per_op}


def estimate(wl: Workload, arch: Arch, prm: CostParams | None = None) -> dict:
    prm = prm or CostParams()
    cache: dict[tuple, float] = {}

    def cost_of(op: OpRec, n_lanes: int) -> float:
        key = (op.sig, n_lanes)
        if key not in cache:
            cache[key] = op_est(arch, prm, op, n_lanes).cycles
        return cache[key]

    r = compose(wl, cost_of, prm)
    r["macs_per_cycle"] = wl.total_macs / r["cycles"] if r["cycles"] else 0.0
    return r


def breakdown_table(r: dict) -> str:
    tot = r["cycles"]
    lines = [f"{'kind':<10} {'cycles':>14} {'share':>7}"]
    for k, v in sorted(r["by_kind"].items(), key=lambda kv: -kv[1]):
        lines.append(f"{k:<10} {v:>14,.0f} {100 * v / tot:>6.1f}%")
    lines.append(f"{'total':<10} {tot:>14,.0f}   ({tot / 1e6:.3f} ms @1GHz, {r.get('macs_per_cycle', 0):.0f} MAC/cycle)")
    return "\n".join(lines)


def params_to_json(prm: CostParams, path: str) -> None:
    import json
    from dataclasses import asdict
    with open(path, "w") as f:
        json.dump(asdict(prm), f, indent=1)


def params_from_json(path: str) -> CostParams:
    import json
    d = json.load(open(path))
    prm = CostParams()
    for k, v in d.items():
        if k == "elem":
            prm.elem.update(v)
        elif k == "hbm_agg":
            prm.hbm_agg = {int(n): float(r) for n, r in v.items()}
        elif hasattr(prm, k):
            setattr(prm, k, v)
    return prm


# ----------------------------------------------------------------------------------------
# world-model kernels (docs/WORLD_MODEL.md): SmolVLA expert with N candidate chunks, on-chip RSSM
# ----------------------------------------------------------------------------------------
def gemm_traffic(M: int, N: int, K: int, tm: int, tn: int, tk: int) -> dict:
    """HBM bytes of one sh_gemm (no accumulate): every output tile streams its X row panel and W column panel."""
    MT, NT = M // tm, N // tn
    return {"x": NT * M * K * ELEM, "w": MT * K * N * ELEM, "z": M * N * ELEM}


def expert_step_traffic(nb: int = 1, layers: int = 16, tiles: dict | None = None, gemm: str = "tiles") -> OrderedDict:
    """HBM bytes per flow step of the SmolVLA expert program (frontend.smolvla_expert.emit_flow) with nb candidate
    chunks (S_q = 50 nb): weights (W panels), activations through the GEMMs (X panels, Z), attention staging
    (q, prefix K/V per query head, own K/V, o) and the row ops (rmsnorm / rope / silu_mul / adds / embedding / tail).
    The prefix K/V and the weights do not grow with nb; everything else does.
    gemm="xmcast": the step GEMMs as sh_gemm_xmcast_ex (X read once per GEMM; docs/XPANEL_MCAST.md)."""
    from softhier_mlir.frontend import smolvla_expert as E
    t = E.batch_tiles(nb, tiles or E.TILES)
    R, D, DQ, DKV, FF, AD, LP, H, DH = E.S * nb, E.D, E.DQ, E.DKV, E.FF, E.AD, E.LP, E.H, E.DH
    out = OrderedDict((k, 0) for k in ("weights", "gemm_act", "attn_kv", "attn_qo", "rowops"))

    def gm(M, N, K, fam):
        tr = gemm_xmcast_traffic(M, N, K) if gemm.startswith("xmcast") else gemm_traffic(M, N, K, *t[fam])
        out["weights"] += tr["w"]; out["gemm_act"] += tr["x"] + tr["z"]

    def rows(nread, nwrite, cols, params=0):
        out["rowops"] += (nread + nwrite) * R * cols * ELEM + params * cols * ELEM

    gm(R, D, AD, "a"); gm(R, D, D, "t"); gm(R, D, D, "t")                      # suffix embedding
    rows(1, 1, D, 1); rows(1, 1, D, 1); rows(1, 1, D); rows(1, 1, D, 1)        # 3 biases + SiLU
    for L in range(layers):
        self_l = L % 2 == 0
        rows(1, 1, D, 1)                                                       # rmsnorm
        gm(R, DQ + 2 * DKV if self_l else DQ, D, "qkv")
        rows(1, 1, DQ, 0); out["rowops"] += R * DH * ELEM                      # rope q (+ table)
        if self_l:
            rows(1, 1, DKV, 0)                                                 # rope own k
        # attention: per query head: q, prefix K + V (Lp rows of dh), own K + V (self), o; the token mask
        own = 2 * R * DH * ELEM if self_l else 0
        out["attn_kv"] += H * (2 * LP * DH * ELEM + LP * 2)
        out["attn_qo"] += H * (2 * R * DH * ELEM + own)
        gm(R, D, DQ, "o"); rows(2, 1, D)                                       # o_proj + residual
        rows(1, 1, D, 1); gm(R, 2 * FF, D, "gu")                               # rmsnorm + gate/up
        rows(2, 1, FF)                                                         # silu * up
        gm(R, D, FF, "d"); rows(2, 1, D)                                       # down + residual
    rows(1, 1, D, 1); gm(R, AD, D, "out"); rows(1, 1, AD, 1); rows(2, 1, AD)    # final norm, out proj, bias, axpy
    out["total"] = sum(out.values())
    return out


# Per-op gvsoc time of the expert at full depth (us per layer call; emb / step per flow step; the GEMM entries include
# their rmsnorm / residual add), 16 clusters, ideal HBM, 1 GHz, tests/gvsoc/expert.py flow --profile --cands N
# (2026-10-08/09, docs/WORLD_MODEL.md). N = 1 and N = 4: 16-layer 10-step runs. N = 2 and N = 8: 4-layer 2-step runs
# with their GEMM entries scaled by the 16-layer / 4-layer ratio measured at N = 1 and N = 4 (interpolated for
# N = 2, the N = 4 ratio for N = 8; attention / silu / rope / emb / step agree within 1 % between the depths).
# The analytic GEMM / row-op models above do not cover the expert's fused attention and the fetch-bound SIMD row
# kernels, so the expert curve is this measured table (linear in N between the points).
EXPERT_OP_US = {
    1: {"attn": 242.0, "gateup": 96.0, "down": 66.0, "silu": 78.0, "qkv": 64.0, "oproj": 41.0, "rope": 22.0, "emb": 143.0, "step": 42.0},
    2: {"attn": 432.1, "gateup": 124.9, "down": 101.3, "silu": 138.2, "qkv": 95.5, "oproj": 61.0, "rope": 33.5, "emb": 200.7, "step": 59.7},
    4: {"attn": 813.8, "gateup": 187.7, "down": 184.0, "silu": 257.9, "qkv": 169.3, "oproj": 109.3, "rope": 59.1, "emb": 319.9, "step": 100.1},
    8: {"attn": 1584.3, "gateup": 311.0, "down": 326.9, "silu": 750.5, "qkv": 305.9, "oproj": 189.3, "rope": 112.4, "emb": 562.5, "step": 179.8},
}
EXPERT_LAYER_OPS = ("attn", "gateup", "down", "silu", "qkv", "oproj", "rope")


# The same per-op table with every step GEMM as sh_gemm_xmcast_ex (emit_flow(gemm="xmcast"), docs/XPANEL_MCAST.md):
# 16-layer runs, N = 1 (10 steps) and N = 4 (3 steps), 2026-10-09. Attention / silu / rope are unchanged by construction.
EXPERT_OP_US_XMCAST = {
    1: {"attn": 238.9, "gateup": 90.7, "down": 53.0, "silu": 78.5, "qkv": 57.7, "oproj": 32.4, "rope": 22.5, "emb": 116.1, "step": 44.3},
    4: {"attn": 813.9, "gateup": 155.8, "down": 80.1, "silu": 257.9, "qkv": 115.5, "oproj": 53.0, "rope": 59.2, "emb": 223.7, "step": 107.0},
}


def expert_step_est(nb: int, layers: int = 16, gemm: str = "tiles") -> dict:
    """Per flow step of the expert with nb candidates: cycles (1 GHz) from the measured per-op table, the
    attention / GEMM / row-op split, HBM bytes (expert_step_traffic) and per-candidate figures.
    gemm="xmcast": the X-multicast step GEMMs (measured at N = 1 and 4; linear in between, scaled beyond)."""
    tab = EXPERT_OP_US_XMCAST if gemm.startswith("xmcast") else EXPERT_OP_US
    pts = sorted(tab)
    lo = max(p for p in pts if p <= nb) if nb >= pts[0] else pts[0]
    hi = min((p for p in pts if p >= nb), default=pts[-1])
    w = 0.0 if hi == lo else (nb - lo) / (hi - lo)
    us = {k: tab[lo][k] + w * (tab[hi][k] - tab[lo][k]) for k in tab[1]}
    if nb > pts[-1]:
        us = {k: tab[pts[-1]][k] * nb / pts[-1] for k in us}     # beyond the table: linear in N (row-op bound)
        if tab is EXPERT_OP_US_XMCAST:   # extrapolated: only the GEMMs scale from N = 4; every other op as measured at nb
            base = expert_step_est(nb, layers)["us"]
            us = {k: (us[k] if k in ("gateup", "down", "qkv", "oproj", "emb") else base[k]) for k in us}
    split = {"attention": layers * us["attn"],
             "gemm": layers * (us["gateup"] + us["down"] + us["qkv"] + us["oproj"]),
             "rowops": layers * (us["silu"] + us["rope"]) + us["emb"] + us["step"]}
    tot = sum(split.values())
    tr = expert_step_traffic(nb, layers, gemm=gemm)
    return {"cycles": tot * 1e3, "split_us": split, "us": us, "hbm_bytes": tr["total"], "traffic": tr,
            "per_cand_cycles": tot * 1e3 / nb, "per_cand_bytes": tr["total"] / nb}


# On-chip RSSM (runtime/sh_wm.inc.c). Row-kernel costs fitted on gvsoc (2026-10-08, tests/gvsoc/wm.py): cycles per
# output element of the cluster, independent of the rows per cluster from 8 to 32 (fetch-bound cores, SIMULATOR_NOTES #7).
WM_ELEM = {"ln": 11.0, "gru": 27.5, "softmax": 21.5}
WM_HEAD_ROW = 64.0          # actor tanh head per row


def wm_gemms(c, posterior: bool = False) -> list[tuple[int, int]]:
    """(contraction, output columns) of the RedMulE calls of one RSSM step (frontend.wm_rssm.Cfg)."""
    S, D, Hd, U = c.S, c.deter, c.hidden, c.units
    g = [(S + c.AP, Hd), (Hd + D, 3 * D)]
    g += [(D + U, Hd), (Hd, S)] if posterior else [(D, Hd), (Hd, S)]
    if posterior:
        g += [(c.OP, U), (U, U)]
    g += [(S + D, U), (U, U), (U, c.AO)]
    return g


def wm_rssm_est(c, K: int, H: int, clusters: int = 1, arch: Arch | None = None, prm: CostParams | None = None,
                posterior: bool = False) -> dict:
    """Cycles of sh_wm_rssm: weight staging + H steps of the busiest cluster (ceil(K / clusters) rows); GEMMs with
    redmule_cycles (m = rows), row kernels with the fitted per-element costs. No HBM traffic inside the steps."""
    arch = arch or Arch()
    prm = prm or CostParams()
    m = math.ceil(K / clusters)
    gemm = sum(redmule_cycles(arch, prm, m, k, n) + prm.dma_fixed + m * k * ELEM / prm.l1_zero_bw + 2 * prm.cluster_sync
               for n, k in wm_gemms(c, posterior))
    lncols = c.hidden * 3 + 3 * c.deter + c.units * 2 + (2 * c.units if posterior else 0)    # img_in, GRU, head, actor x2 (+ enc x2)
    ln = WM_ELEM["ln"] * m * lncols
    gru = WM_ELEM["gru"] * m * c.deter
    sm = WM_ELEM["softmax"] * m * c.S
    head = WM_HEAD_ROW * m
    step = gemm + ln + gru + sm + head
    from softhier_mlir.frontend.wm_rssm import pack_bytes
    blob = pack_bytes(c, posterior)
    load = dma_load(arch, prm, 1, blob // ELEM, clusters)
    macs = K * H * _wm_macs(c, posterior)
    return {"cycles": load + H * step, "load": load, "step": step, "gemm": gemm, "ln": ln, "gru": gru, "softmax": sm,
            "head": head, "rows_per_cluster": m, "macs": macs, "mac_per_cycle": macs / (load + H * step)}


def _wm_macs(c, posterior: bool) -> int:
    return sum(n * k for n, k in wm_gemms(c, posterior))


if __name__ == "__main__":
    import argparse
    from softhier_mlir.dse import workload as wlm
    ap = argparse.ArgumentParser(description="analytic estimate of a softhier module")
    ap.add_argument("input", nargs="?", default=None, help=".mlir file, - for stdin; default: SigLIP S=256 1 layer")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    a = ap.parse_args()
    wl = wlm.from_file(a.input) if a.input else wlm.siglip(seq=a.seq, layers=a.layers)
    arch = Arch()
    r = estimate(wl, arch)
    print(breakdown_table(r))
    for op, own, contrib in r["per_op"]:
        if contrib:
            print(f"  {str(op):<50} {own:>12,.0f}")
