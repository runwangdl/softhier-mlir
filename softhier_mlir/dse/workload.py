"""Workload IR for design-space exploration.

Extracts a flat, shape-level description of a ``softhier``-dialect module: one record per
library call the backend would emit (GEMM / SUMMA GEMM / row ops / transpose), with shapes,
bytes moved, MACs, the tiling attributes and the cluster mapping. Program order is kept
because the cost model composes a timeline: ops with an explicit ``cluster = c`` run on that
cluster concurrently with the other clusters' ops until the next synchronisation point (an op
with ``cluster = -1``, a SUMMA GEMM or a ``group_barrier``).

    python -m softhier_mlir.frontend.siglip --layers 12 --no-test | python -m softhier_mlir.dse.workload -
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Iterator

from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp
from xdsl.parser import Parser

from softhier_mlir.backend.emit_c import _Buffers
from softhier_mlir.dialects import softhier as sh

ROW_OPS = {sh.LayerNormOp: "layernorm", sh.SoftmaxOp: "softmax", sh.GeluOp: "gelu",
           sh.AddOp: "add", sh.AddBiasOp: "add_bias"}
ELEM = 2  # fp16 everywhere in the library today
SYNC_KINDS = ("barrier", "summa", "xmcast")   # plus any op with cluster == -1


@dataclass(frozen=True)
class OpRec:
    """One library call. ``shape`` is (M, N, K) for gemm/summa, (rows, cols) otherwise."""
    kind: str                       # gemm | summa | xmcast | layernorm | softmax | gelu | add | add_bias | transpose | barrier
    shape: tuple = ()
    tile: tuple = ()                # (tm, tn, tk) for gemm/summa (0 = library default 256)
    pipeline: int = 1
    accumulate: int = 0
    cluster: int = -1               # -1 = all clusters (SH_ALL); >= 0 one cluster
    fmt: str = "fp16"
    name: str = ""                  # free-form label (result buffer), not part of the key

    # ---- derived ---------------------------------------------------------------------
    @property
    def is_gemm(self) -> bool:
        return self.kind in ("gemm", "summa", "xmcast")

    @property
    def tiles(self) -> tuple[int, int, int]:
        tm, tn, tk = (self.tile + (0, 0, 0))[:3]
        return (tm or 256, tn or 256, tk or 256)

    @property
    def macs(self) -> int:
        if self.is_gemm:
            m, n, k = self.shape
            return m * n * k
        return 0

    @property
    def elements(self) -> int:
        if self.is_gemm:
            return self.shape[0] * self.shape[1]
        return self.shape[0] * self.shape[1]

    @property
    def bytes_hbm(self) -> int:
        """Bytes the library moves between HBM and TCDM (loads + stores), following sh_gemm /
        sh_gemm_mesh / sh_rowop exactly (no reuse across output tiles)."""
        if self.kind == "barrier":
            return 0
        if self.kind == "gemm":
            m, n, k = self.shape
            tm, tn, tk = self.tiles
            ntiles = (m // tm) * (n // tn)
            per_tile = (tm * k + k * tn) * ELEM + tm * tn * ELEM * (2 if self.accumulate else 1)
            return ntiles * per_tile
        if self.kind == "xmcast":                      # X once, W once per row block (tile_m), Z once
            m, n, k = self.shape
            mt = m // (self.tile[0] if self.tile and self.tile[0] else m)
            return (m * k + mt * k * n) * ELEM + m * n * ELEM * (2 if self.accumulate else 1)
        if self.kind == "summa":
            m, n, k = self.shape
            t = self.tiles[0]
            p = m // t                                    # mesh side: diagonal clusters load panels
            return p * (t * k + k * t) * ELEM + m * n * ELEM * (2 if self.accumulate else 1)
        rows, cols = self.shape
        nin = {"add": 2, "add_bias": 1 + 0, "layernorm": 1, "softmax": 1, "gelu": 1, "transpose": 1}[self.kind]
        return (nin + 1) * rows * cols * ELEM

    @property
    def sig(self) -> tuple:
        """Identity for caching / counting: everything that changes the kernel's runtime,
        with per-cluster placement folded to 'one cluster' (which cluster does not matter)."""
        return (self.kind, self.shape, self.tiles if self.is_gemm else (), self.pipeline,
                self.accumulate, -1 if self.cluster < 0 else 0, self.fmt)

    def __str__(self) -> str:
        if self.kind == "barrier":
            return "barrier"
        shp = "x".join(str(d) for d in self.shape)
        cl = "all" if self.cluster < 0 else f"c{self.cluster}"
        if self.is_gemm:
            tl = "x".join(str(d) for d in self.tiles)
            return f"{self.kind} {shp} tile {tl} pipe={self.pipeline} acc={self.accumulate} {cl}"
        return f"{self.kind} {shp} {cl}"


@dataclass
class Workload:
    ops: list[OpRec] = field(default_factory=list)
    name: str = ""

    def __iter__(self) -> Iterator[OpRec]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def total_macs(self) -> int:
        return sum(o.macs for o in self.ops)

    @property
    def total_bytes(self) -> int:
        return sum(o.bytes_hbm for o in self.ops)

    def unique(self) -> "OrderedDict[tuple, tuple[OpRec, int]]":
        """sig -> (representative op, count), in first-occurrence order. Barriers excluded."""
        out: OrderedDict[tuple, tuple[OpRec, int]] = OrderedDict()
        for o in self.ops:
            if o.kind == "barrier":
                continue
            rec, n = out.get(o.sig, (o, 0))
            out[o.sig] = (rec, n + 1)
        return out

    def retile(self, tile: tuple[int, int, int] | None = None, pipeline: int | None = None,
               only_default: bool = True) -> "Workload":
        """Kernel-knob variant: new (tm, tn, tk) / pipeline for every gemm the tile divides.
        ``only_default`` leaves gemms whose tiling was set explicitly below 256 (attention
        heads) alone, so a kernel sweep does not break their shape constraints."""
        ops = []
        for o in self.ops:
            if o.kind == "gemm":
                m, n, k = o.shape
                new = o
                if tile and (not only_default or o.tiles == (256, 256, 256)) and m % tile[0] == 0 and n % tile[1] == 0 and k % tile[2] == 0:
                    new = replace(new, tile=tuple(tile))
                if pipeline is not None:
                    new = replace(new, pipeline=pipeline)
                ops.append(new)
            else:
                ops.append(o)
        return Workload(ops, self.name)

    def remap_clusters(self, n_clusters: int) -> "Workload":
        """Fold explicit cluster ids onto a mesh of n_clusters (the frontend pins attention heads
        to ``hd % 16``; on a smaller mesh those ops would otherwise never run)."""
        return Workload([replace(o, cluster=o.cluster % n_clusters) if o.cluster >= 0 else o for o in self.ops], self.name)

    def table(self) -> str:
        lines = [f"{'count':>5}  {'kind':<9} {'shape':<16} {'tile':<12} {'cl':<4} {'MMAC':>9} {'MB moved':>9}"]
        for rec, n in self.unique().values():
            shp = "x".join(str(d) for d in rec.shape)
            tl = "x".join(str(d) for d in rec.tiles) if rec.is_gemm else "-"
            cl = "all" if rec.cluster < 0 else "one"
            lines.append(f"{n:>5}  {rec.kind:<9} {shp:<16} {tl:<12} {cl:<4} {n * rec.macs / 1e6:>9.1f} {n * rec.bytes_hbm / 2**20:>9.2f}")
        lines.append(f"total: {len(self.ops)} ops, {self.total_macs / 1e9:.3f} GMAC, {self.total_bytes / 2**20:.1f} MB HBM traffic")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------
def _int_attr(op, name: str, default: int) -> int:
    a = op.attributes.get(name)
    return a.value.data if isinstance(a, IntegerAttr) else default


def parse_module(text: str, name: str = "<mlir>") -> ModuleOp:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(sh.SoftHier)
    return Parser(ctx, text, name).parse_module()


def extract(module: ModuleOp, name: str = "") -> Workload:
    fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    bufs = _Buffers()
    for op in fn.body.block.ops:
        if isinstance(op, sh.HbmBufferOp):
            bufs.add_hbm(op)
        elif isinstance(op, sh.L1BufferOp):
            bufs.add_l1(op)
        elif isinstance(op, sh.ViewOp):
            bufs.add_view(op)
    ops: list[OpRec] = []
    for op in fn.body.block.ops:
        if isinstance(op, sh.GemmOp):
            m, k, _, _ = bufs.geom(op.x)
            _, n, _, _ = bufs.geom(op.w)
            kind = "summa" if "summa" in op.attributes else "xmcast" if "xmcast" in op.attributes else "gemm"
            ops.append(OpRec(kind=kind, shape=(m, n, k),
                             tile=(_int_attr(op, "tile_m", 0), _int_attr(op, "tile_n", 0), _int_attr(op, "tile_k", 0)),
                             pipeline=1 if "pipeline" in op.attributes else 0,
                             accumulate=1 if "accumulate" in op.attributes else 0,
                             cluster=_int_attr(op, "cluster", 0), fmt=op.fmt.data, name=bufs.name(op.z)))
        elif type(op) in ROW_OPS:
            src = op.x if hasattr(op, "x") else op.a
            rows, cols, _, _ = bufs.geom(src)
            ops.append(OpRec(kind=ROW_OPS[type(op)], shape=(rows, cols), cluster=_int_attr(op, "cluster", 0),
                             name=bufs.name(op.y)))
        elif isinstance(op, sh.TransposeOp) and bufs.space(op.src) != "tcdm":
            rows, cols, _, _ = bufs.geom(op.src)
            ops.append(OpRec(kind="transpose", shape=(rows, cols), cluster=_int_attr(op, "cluster", 0), name=bufs.name(op.dst)))
        elif isinstance(op, sh.GroupBarrierOp):
            ops.append(OpRec(kind="barrier"))
        # fills / dumps / declarations: not part of the workload
    return Workload(ops, name or fn.sym_name.data)


def from_mlir(text: str, name: str = "") -> Workload:
    return extract(parse_module(text), name)


def from_file(path: str) -> Workload:
    text = sys.stdin.read() if path == "-" else open(path).read()
    return from_mlir(text, path)


def siglip(seq: int = 256, d: int = 768, ff: int = 3072, heads: int = 12, layers: int = 1, cluster: int = -1) -> Workload:
    """The SigLIP encoder workload straight from the frontend (no test fills)."""
    from softhier_mlir.frontend import siglip as fe
    return from_mlir(fe.emit(seq, d, ff, heads, layers, cluster, test=False), f"siglip_S{seq}_L{layers}")


def main() -> None:
    ap = argparse.ArgumentParser(description="workload summary of a softhier-dialect module")
    ap.add_argument("input", help=".mlir file or - for stdin")
    ap.add_argument("--ops", action="store_true", help="also list every op in program order")
    a = ap.parse_args()
    w = from_file(a.input)
    if a.ops:
        for i, o in enumerate(w.ops):
            print(f"{i:4d}  {o}")
    print(w.table())


if __name__ == "__main__":
    main()
