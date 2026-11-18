"""Candidate arcs per GPS point.

Each trip has its OWN directed road network, and ~92% of arcs have a reverse twin with
identical geometry. A twin pair is therefore always tied on distance, so both twins are
kept together; only the heading feature can tell them apart later.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from common.config import K, RADIUS_M
from common.geo import point_segment_distance

_TIE_DECIMALS = 3  # distances equal to the mm are ties (twins differ only by float noise)
_MAX_EXPANSIONS = 12


@dataclass
class ArcIndex:
    tree: STRtree
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    from_node: np.ndarray
    to_node: np.ndarray
    length: np.ndarray
    twin: np.ndarray  # arc id of the reverse arc (to -> from), -1 if none


def twin_ids(from_node, to_node) -> np.ndarray:
    """Reverse-arc id per arc (-1 if the reverse arc does not exist), vectorised."""
    f, t = np.asarray(from_node, np.int64), np.asarray(to_node, np.int64)
    n = int(max(f.max(), t.max())) + 1
    keys = f * n + t
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    rev = t * n + f
    pos = np.clip(np.searchsorted(sorted_keys, rev), 0, len(keys) - 1)
    found = sorted_keys[pos] == rev
    return np.where(found, order[pos], -1)


def build_arc_index(nodes_xy, arcs) -> ArcIndex:
    """STRtree over arc LineStrings in metres (built once per trip, geometry vectorised)."""
    nodes_xy = np.asarray(nodes_xy, float)
    fn = arcs["from_node"].to_numpy()
    tn = arcs["to_node"].to_numpy()
    a, b = nodes_xy[fn], nodes_xy[tn]
    geoms = shapely.linestrings(np.stack([a, b], axis=1))
    return ArcIndex(STRtree(geoms), a[:, 0], a[:, 1], b[:, 0], b[:, 1], fn, tn,
                    np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]), twin_ids(fn, tn))


def _pairs(track_xy, index: ArcIndex, point_ids, radius) -> pd.DataFrame:
    pts = shapely.points(track_xy[point_ids])
    q = index.tree.query(pts, predicate="dwithin", distance=radius)
    pi, ai = point_ids[q[0]], q[1]
    d, t = point_segment_distance(track_xy[pi, 0], track_xy[pi, 1], index.ax[ai],
                                  index.ay[ai], index.bx[ai], index.by[ai])
    return pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist_m": d, "proj_t": t})


def get_candidates(track_xy, index: ArcIndex, k: int = K, radius: float = RADIUS_M) -> pd.DataFrame:
    """DataFrame[point_idx, arc_id, dist_m, proj_t, fallback].

    Arcs within `radius`, nearest `k` kept. If a point has none in range, the nearest `k`
    are kept and fallback=True. The reverse twin of every kept arc is kept as well, so
    a tie between twins never splits them (a point can have up to 2k rows).
    """
    track_xy = np.asarray(track_xy, float)
    all_ids = np.arange(len(track_xy))
    pool = _pairs(track_xy, index, all_ids, radius)
    pool["fallback"] = False
    missing = np.setdiff1d(all_ids, pool.point_idx.unique())
    r = radius
    for _ in range(_MAX_EXPANSIONS):
        if len(missing) == 0:
            break
        r *= 2
        extra = _pairs(track_xy, index, missing, r)
        counts = extra.groupby("point_idx").size()
        done = counts[counts >= k].index.to_numpy()
        if r > 1e5:  # nothing nearer than 100 km: take whatever exists
            done = extra.point_idx.unique()
        pool = pd.concat([pool, extra[extra.point_idx.isin(done)].assign(fallback=True)],
                         ignore_index=True)
        missing = np.setdiff1d(missing, done)

    pool["_d"] = pool.dist_m.round(_TIE_DECIMALS)
    pool = pool.sort_values(["point_idx", "_d", "arc_id"], kind="stable", ignore_index=True)
    kept = pool[pool.groupby("point_idx").cumcount() < k]

    want = pd.DataFrame({"point_idx": kept.point_idx.to_numpy(),
                         "arc_id": index.twin[kept.arc_id.to_numpy()]})
    twins = pool.merge(want[want.arc_id >= 0], on=["point_idx", "arc_id"], how="inner")
    out = pd.concat([kept, twins], ignore_index=True).drop_duplicates(["point_idx", "arc_id"])
    out = out.sort_values(["point_idx", "_d", "arc_id"], kind="stable", ignore_index=True)
    return out[["point_idx", "arc_id", "dist_m", "proj_t", "fallback"]]
