"""One row per (GPS point, candidate arc). Never reads .route or any labels.

This module is label-free by design (it is imported by predict_trip). Training tables
with labels are made by features/make_tables.py.
"""
import numpy as np
import pandas as pd

from common.config import K, RADIUS_M
from common.geo import angle_diff_deg, bearing_deg, make_projector, to_metres
from common.io import load_arcs, load_nodes, load_track
from features.candidates import build_arc_index, get_candidates

LOW_SPEED_MPS = 1.0  # below this the GPS heading is just noise
NEUTRAL_HEADING = 90.0
COLUMNS = ["trip_id", "point_idx", "t", "lon", "lat", "arc_id", "dist_m", "dist_rank",
           "dist_minus_min", "heading_diff", "speed_mps", "dt_s", "arc_len_m", "proj_t",
           "fallback", "low_speed", "has_twin", "connects_prev_best"]
# Identifiers / location: kept for joining and mapping, NEVER model inputs (lon, lat, t and
# trip_id would let a model memorise places and trips instead of learning matching).
METADATA_COLUMNS = ["trip_id", "point_idx", "t", "lon", "lat", "arc_id"]
FEATURE_COLUMNS = [c for c in COLUMNS if c not in METADATA_COLUMNS]


def _point_motion(xy, t):
    """GPS heading (prev -> next point), speed and dt per point."""
    T = len(xy)
    prev = np.maximum(np.arange(T) - 1, 0)
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    dt2 = t[nxt] - t[prev]
    dist = np.hypot(xy[nxt, 0] - xy[prev, 0], xy[nxt, 1] - xy[prev, 1])
    speed = np.where(dt2 > 0, dist / np.where(dt2 > 0, dt2, 1.0), 0.0)
    heading = bearing_deg(xy[prev, 0], xy[prev, 1], xy[nxt, 0], xy[nxt, 1])
    dt = np.ones(T)
    if T > 1:
        dt[1:] = np.diff(t)
        dt[0] = dt[1]
    return heading, speed, dt


def build_features(trip_dir, k: int = K, radius: float = RADIUS_M) -> pd.DataFrame:
    """Unlabelled feature table for one trip (see attach_labels for the label column)."""
    track, nodes, arcs = load_track(trip_dir), load_nodes(trip_dir), load_arcs(trip_dir)
    tr = make_projector(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx, ny = to_metres(tr, nodes.lon, nodes.lat)
    px, py = to_metres(tr, track.lon, track.lat)
    xy = np.column_stack([px, py])
    t = track.t.to_numpy(float)

    index = build_arc_index(np.column_stack([nx, ny]), arcs)
    df = get_candidates(xy, index, k, radius)
    heading, speed, dt = _point_motion(xy, t)

    pi, ai = df.point_idx.to_numpy(), df.arc_id.to_numpy()
    low = speed < LOW_SPEED_MPS
    arc_bearing = bearing_deg(index.ax[ai], index.ay[ai], index.bx[ai], index.by[ai])
    hd = angle_diff_deg(heading[pi], arc_bearing)
    hd = np.where(low[pi] | (index.length[ai] == 0), NEUTRAL_HEADING, hd)

    name = getattr(trip_dir, "name", str(trip_dir))
    df.insert(0, "trip_id", name)
    df["t"] = t[pi]
    df["lon"] = track.lon.to_numpy()[pi]
    df["lat"] = track.lat.to_numpy()[pi]
    d_round = df.dist_m.round(3)
    df["dist_rank"] = d_round.groupby(df.point_idx).rank(method="min").astype(int) - 1
    df["dist_minus_min"] = df.dist_m - df.dist_m.groupby(df.point_idx).transform("min")
    df["heading_diff"] = hd
    df["speed_mps"] = speed[pi]
    df["dt_s"] = dt[pi]
    df["arc_len_m"] = index.length[ai]
    df["low_speed"] = low[pi].astype(int)
    df["has_twin"] = (index.twin[ai] >= 0).astype(int)

    # previous point's best arc: nearest correctly-headed candidate, else nearest overall.
    # df is already sorted by (point_idx, distance), so drop_duplicates keeps the nearest.
    ok = df[(df.heading_diff < 90) & (df.low_speed == 0)]
    best = df.drop_duplicates("point_idx").set_index("point_idx").arc_id
    best_ok = ok.drop_duplicates("point_idx").set_index("point_idx").arc_id
    prev_best = best_ok.combine_first(best).reindex(np.arange(len(xy))).to_numpy()
    pb = np.concatenate([[-1], prev_best[:-1]])[pi]
    has_pb = pb >= 0
    pbs = np.where(has_pb, pb, 0).astype(int)
    connects = has_pb & ((ai == pbs) | (index.to_node[pbs] == index.from_node[ai]))
    df["connects_prev_best"] = connects.astype(int)
    return df[COLUMNS]


def attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add label = 1 where arc_id is the point's reference arc (labels: point_idx, arc_id)."""
    ref = labels.set_index("point_idx").arc_id
    out = df.copy()
    out["label"] = (out.arc_id.to_numpy() == out.point_idx.map(ref).to_numpy()).astype(int)
    return out
