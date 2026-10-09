"""Cluster sets in the frontends (spatial split, docs/SPATIAL_SPLIT.md).

The emitters take a `cluster` int: an id >= 0, -1 (SH_ALL) or -2 (SH_SELF). A cluster set is encoded as
``cluster_set(mask) = -(0x10000 + mask)``: still negative, so every emitter decision that asks "is this split over
several clusters" (``cluster < 0``: tile choice, q blocks) treats it like SH_ALL, and ``cl_attr`` turns it into the
op attributes ``cluster = -1 : i32, cluster_set = <mask> : i32`` (emit_c lowers that to ``SH_GROUP(mask)``).
"""
from __future__ import annotations

SET_BASE = 0x10000
NCL = 16                    # 4 x 4 mesh; cluster id = y * 4 + x


def cluster_set(mask: int) -> int:
    """The emitter `cluster` value of the set of clusters whose bit is set in `mask` (bit i = cluster id i)."""
    assert 0 < mask < (1 << NCL), mask
    return -(SET_BASE + mask)


def set_mask(cluster: int) -> int | None:
    """mask of an encoded set, None for an id / SH_ALL / SH_SELF."""
    return (-cluster - SET_BASE) if cluster <= -SET_BASE else None


def set_size(cluster: int) -> int:
    m = set_mask(cluster)
    return bin(m).count("1") if m is not None else NCL if cluster == -1 else 1


def cl_attr(cluster: int) -> str:
    """The `cluster` attribute(s) of a library op."""
    m = set_mask(cluster)
    return f"cluster = -1 : i32, cluster_set = {m} : i32" if m is not None else f"cluster = {cluster} : i32"


def set_attr(cluster: int) -> str:
    """`, cluster_set = <mask> : i32` for marks / dumps / barriers of a set ('' otherwise)."""
    m = set_mask(cluster)
    return f", cluster_set = {m} : i32" if m is not None else ""


def barrier_op(cluster: int) -> str:
    """The barrier after a group of ops: the global one, or the set's."""
    m = set_mask(cluster)
    return ("softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}" if m is None
            else f"softhier.group_barrier {{cluster_set = {m} : i32}}")


def c_cluster(cluster: int) -> str:
    """The C `cluster` argument (for softhier.call argument templates)."""
    m = set_mask(cluster)
    return f"SH_GROUP(0x{m:04x}u)" if m is not None else "SH_ALL" if cluster == -1 else "SH_SELF" if cluster == -2 else str(cluster)


def parse(spec: str) -> int:
    """CLI spelling: 'all' | '<id>' | 'set:<mask>' (mask in any int base, e.g. set:0x00ff) | 'rows:<y0>-<y1>'."""
    if spec == "all":
        return -1
    if spec.startswith("set:"):
        return cluster_set(int(spec[4:], 0))
    if spec.startswith("rows:"):
        a, b = (int(v) for v in spec[5:].split("-"))
        return cluster_set(sum(0xF << (4 * y) for y in range(a, b + 1)))
    return int(spec)
