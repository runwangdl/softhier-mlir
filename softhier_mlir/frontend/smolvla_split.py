"""SmolVLA with structural prefill / decode disaggregation on one SoftHier chip (H3, docs/SPATIAL_SPLIT.md).

Stage A = vision tower + connector + VLM prefix (writes the KV cache of chunk t+1), stage B = the action expert's flow
(10 Euler steps over the KV cache of chunk t). Both stages live in ONE program, inside a loop over periods:

  time-shared ("time"):  period p = A on all 16 clusters, then B on all 16 clusters
  spatial ("split"):     period p = A on cluster set mA  ||  B on cluster set mB (softhier.on_clusters, SH_GROUP), then a
                         global barrier; A writes KV slot block (p + 1) % 2 while B reads block p % 2 (double-buffered KV)

Block 0 is preloaded with lerobot's KV of the first layers, so B has a chunk to work on in period 0 and every period is a
steady-state period. The camera input is the same in every period (timing does not depend on the pixels), so every
chunk of A produces the same KV and the final action chunk (B of the last period, on A's KV from the period before) is
the same number in every mode: the cross-mode check is bit-for-bit. `stages` = ("A",) or ("B",) runs one stage alone on
its set (the isolated stage times of the period model).

Depth: the HBM image and the host memory of gvsoc limit one program (docs/SMOLVLA_E2E.md section 2), so the program runs
`vlayers` of the 12 SigLIP layers, `players` of the 16 prefix layers and `xlayers` (= players: the expert reads one KV
slot per layer) of the 16 expert layers; the stage times are scaled to full depth with the per-layer times the marks
measure (softhier_mlir.dse.cost.spatial_split_period).
"""
from __future__ import annotations

import re

import numpy as np

from softhier_mlir.frontend import smolvla as V
from softhier_mlir.frontend import smolvla_e2e as E2E
from softhier_mlir.frontend import smolvla_expert as X
from softhier_mlir.frontend.clusters import cluster_set
from softhier_mlir.sim.preload import sentinel_array

_HOIST = re.compile(r"^\s*%[\w]+ = softhier\.hbm_buffer \{offset")
_SSA = re.compile(r"%([A-Za-z_]\w*)")
_KEEP = {"KP", "KBA", "KBB"}


def _body(mlir: str, pfx: str) -> tuple[list[str], list[str]]:
    """(hoisted top-level hbm_buffer declarations, remaining body lines) of an emitter's module, the preload wait and the
    sentinel dropped (the combined program has one), every SSA name prefixed with `pfx` (except the period loop's)."""
    lines = mlir.splitlines()
    i = next(k for k, ln in enumerate(lines) if "func.func" in ln)
    j = max(k for k, ln in enumerate(lines) if "func.return" in ln)
    decls, body = [], []
    for ln in lines[i + 1:j]:
        if "softhier.preload_wait" in ln or "%sentinel = " in ln or 'tag = "preload"' in ln:
            continue
        ln = _SSA.sub(lambda m: m[0] if m[1] in _KEEP else f"%{pfx}{m[1]}", ln)
        (decls if _HOIST.match(ln) else body).append(ln.strip())
    return decls, body


def _drop_sentinel(pre: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    s = sentinel_array()
    return {o: a for o, a in pre.items() if not (a.shape == s.shape and a.dtype == s.dtype and np.array_equal(a.view(np.uint16), s.view(np.uint16)))}


def _align(off: int, a: int = 0x100000) -> int:
    return (off + a - 1) & ~(a - 1)


def emit_pipeline(e2e, mode: str = "split", mask_a: int = 0x0FFF, mask_b: int = 0xF000, stages=("A", "B"), periods: int = 2,
                  vlayers: int = 3, players: int = 4, xlayers: int = 4, steps: int = 10, chunk: int = 50,
                  inner_marks: bool | None = None, optimize: str | None = "Os", far: bool = True,
                  dump_kv: str = "all") -> tuple[str, dict, dict]:
    """-> (mlir, {hbm offset: preload array}, info). mode "time" ignores the masks (all ops SH_ALL, A then B per period);
    "split" runs A on mask_a and B on mask_b concurrently. inner_marks (default: on unless both stages run concurrently):
    the emitters' per-layer / per-step marks; concurrent stages print stage-end marks only (both sets' leaders print
    through the same character UART, and per-layer marks of the two sets would interleave). Marks: A<p> / B<p> (stage
    end, printed by the stage's set leader), period<p> (after the global barrier, cluster 0).
    The three stages' library code does not fit the 64 KB instruction memory together; optimize = "Os" and far = True
    compile the generated functions for size and place them in HBM (`sh.far_code`, runtime -DSH_FAR_CODE; build flags
    in tests/gvsoc/split.py)."""
    assert mode in ("time", "split") and set(stages) <= {"A", "B"} and stages
    assert players == xlayers and xlayers % 2 == 0, "the expert reads one prefix KV slot per layer"
    assert mask_a & mask_b == 0 or mode == "time" or len(stages) == 1, "the two sets must be disjoint"
    concurrent = mode == "split" and len(stages) == 2
    if inner_marks is None:
        inner_marks = not concurrent
    cl_a = -1 if mode == "time" else cluster_set(mask_a)
    cl_b = -1 if mode == "time" else cluster_set(mask_b)
    it = E2E.info(e2e)
    assert it["cams"] == 1, "one camera (the KV hand-over layout of the 1-camera chain)"
    pre: dict[int, np.ndarray] = {}
    info: dict = {"mode": mode, "mask_a": mask_a, "mask_b": mask_b, "stages": list(stages), "periods": periods, "vlayers": vlayers,
                  "players": players, "xlayers": xlayers, "steps": steps, "chunk": chunk}

    # ---- stage A: vision (SigLIP only) + prefix (pixel shuffle + connector + layers), KV double-buffered
    base = V.HBM_DATA_START
    mv, prv = E2E.emit_vision(e2e, cluster=cl_a, layers=vlayers, hbm_base=base, marks=inner_marks, connector=False)
    vi = E2E.emit_vision.last_info
    pre.update(_drop_sentinel(prv))
    data = E2E.vlm_data(e2e, layers_from=0, layers=players)
    base = _align(vi["end"])
    mp, prp = V.emit_vlm(data, layers=players, cluster=cl_a, attn=cl_a, dumps=(), marks=inner_marks, pad_rows_zero=True,
                         hbm_base=base, vis_off=vi["vis_off"], x_reset=True, kv_buf="%KBA")
    pi = V.emit_vlm.last_info
    pre.update(_drop_sentinel(prp))
    info.update(vis_off=vi["vis_off"], kv_base=pi["kv_base"], kv_stride=pi["kv_stride"], S_pad=pi["S_pad"], n=pi["n"])
    # KV block 0 = lerobot's KV of the first `players` layers (B's chunk in period 0)
    S, n = pi["S_pad"], pi["n"]
    for L in range(players):
        for j, kv in enumerate(("k", "v")):
            a = np.zeros((S, V.TKVD), np.float16)
            a[:n] = e2e[f"ref_kv_{kv}_{L}"].astype(np.float16)
            pre[pi["kv_base"] + L * pi["kv_stride"] + j * S * V.TKVD * 2] = a
    # ---- stage B: the expert flow on KV block KBB
    base = _align(max(o + a.nbytes for o, a in pre.items()))
    xdata = E2E.expert_data(e2e, {})
    mx, prx = X.emit_flow(xdata, steps=steps, layers=xlayers, cluster=cl_b, dumps=(), kv_base=pi["kv_base"], kv_stride=pi["kv_stride"],
                          s_pad=S, num_steps=steps, chunk=chunk, hbm_base=base, marks=inner_marks, kv_buf="%KBB",
                          pair_marks=inner_marks)
    pre.update(_drop_sentinel(prx))
    end = max(o + a.nbytes for o, a in pre.items())
    assert end <= V.HBM_WEST_END, f"image {end / 2 ** 20:.0f} MiB exceeds the west HBM region"
    info["image_end"] = end
    info["image_bytes"] = sum(a.nbytes for a in pre.values())

    dv, bv = _body(mv, "v_")
    dp, bp = _body(mp, "p_")
    dx, bx = _body(mx, "x_")
    sent = sentinel_array()
    sent_off = _align(end, 0x10000)
    pre[sent_off] = sent
    st = f'memref<{sent.shape[0]}x{sent.shape[1]}xf16, "hbm_west">'
    out: list[str] = dv + dp + dx
    out += [f"%sentinel = softhier.hbm_buffer {{offset = {sent_off} : i32}} : {st}",
            f"softhier.preload_wait %sentinel : {st}",
            "%KC0 = arith.constant 0 : index", "%KC1 = arith.constant 1 : index", "%KC2 = arith.constant 2 : index",
            f"%KNP = arith.constant {periods} : index",
            'softhier.mark {tag = "start"}',
            "scf.for %KP = %KC0 to %KNP step %KC1 {",
            "%KP1 = arith.addi %KP, %KC1 : index",
            "%KBA = arith.remui %KP1, %KC2 : index",
            "%KBB = arith.remui %KP, %KC2 : index"]
    sa = "" if mode == "time" else f", cluster_set = {mask_a} : i32"
    sb = "" if mode == "time" else f", cluster_set = {mask_b} : i32"
    for stg in stages:
        body, m, s = (bv + bp, mask_a, sa) if stg == "A" else (bx, mask_b, sb)
        if mode == "split":
            out.append(f"softhier.on_clusters attributes {{cluster_set = {m} : i32}} {{")
        out += body
        out.append(f'softhier.mark %KP {{tag = "{stg}"{s}}}')
        if mode == "split":
            out.append("}")
    out += ["softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}", 'softhier.mark %KP {tag = "period"}', "}",
            'softhier.mark {tag = "end"}']
    # ---- outputs: the last chunk's actions (B's x) and the KV block A wrote last
    if "B" in stages:
        out.append('softhier.dump_all %x_x {tag = "ACT"}' + f' : memref<{chunk}x{X.AD}xf16, "hbm_west">')
    if "A" in stages and dump_kv in ("all", "samples"):
        last = periods % 2      # A of period p wrote block (p + 1) % 2; the last period: periods % 2
        kvt = f'memref<{n}x{V.TKVD}xf16, strided<[{V.TKVD}, 1], offset: 0>, "hbm_west">'
        full = f'memref<{S}x{V.TKVD}xf16, "hbm_west">'
        for L in range(players):
            for j, kv in enumerate(("K", "V")):
                off = pi["kv_base"] + (last * players + L) * pi["kv_stride"] + j * S * V.TKVD * 2
                out += [f"%dkv{L}{kv} = softhier.hbm_buffer {{offset = {off} : i32}} : {full}",
                        f"%dkv{L}{kv}v = softhier.view %dkv{L}{kv} : {full} -> {kvt}",
                        (f'softhier.dump_all %dkv{L}{kv}v {{tag = "{kv}C{L}"}} : {kvt}' if dump_kv == "all" else
                         f'softhier.dump_samples %dkv{L}{kv}v {{seed = {700 + 2 * L + j} : i32, n = 256 : i32, tag = "{kv}C{L}"}} : {kvt}')]
    fa = [f'sh.optimize = "{optimize}"'] if optimize else []
    if far:
        fa.append("sh.far_code")
    attrs = f" attributes {{{', '.join(fa)}}}" if fa else ""
    name = f"smolvla_{mode}_{''.join(stages)}"
    mlir = (f"builtin.module {{\n  func.func @{name}(){attrs} {{\n" + "\n".join("    " + ln for ln in out)
            + "\n    func.return\n  }\n}\n")
    return mlir, pre, info


