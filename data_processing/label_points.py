"""Per-point reference labels: map each GPS point to an arc of its own .route.

Rule: point i gets route[j_i] with j_i non-decreasing in i (the route index may stay the
same, or move forward and skip arcs, never go back). Among all such sequences we pick the
one with the smallest total point-to-arc distance (DP), not the greedy nearest arc, so a
noisy point or a nearby parallel road cannot drag the index forward or back.

Uses the trip's .route, so it is for train/student_test labels and server-side evaluation
only; never call it from build_features/predict_trip.

Run: python -m data_processing.label_points --splits train student_test
"""
import argparse

import numpy as np
import pandas as pd

from common.config import RESULTS
from common.geo import make_projector, point_segment_distance, to_metres
from common.io import list_trips, load_arcs, load_nodes, load_route, load_track

P95_FLAG_M = 30.0


def _route_indices(track_xy, route_arcs, arc_geoms):
    """DP over (point, route index). Returns (route index per point, distance per point)."""
    r = np.asarray(route_arcs)
    seg = np.asarray(arc_geoms, float)[r]  # (R, 4): ax, ay, bx, by of the route arcs only
    px, py = track_xy[:, 0:1], track_xy[:, 1:2]
    dist, _ = point_segment_distance(px, py, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])
    T, R = dist.shape
    idx = np.arange(R)
    cost = dist[0].copy()
    back = np.zeros((T, R), dtype=np.int32)
    for i in range(1, T):
        # best predecessor of j is the cheapest k <= j: running prefix-min with its argmin
        run_min = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost <= run_min, idx, 0))
        cost = dist[i] + run_min
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(cost))
    for i in range(T - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j, dist[np.arange(T), j]


def label_points(track_xy, route_arcs, arc_geoms):
    """Return (arc_id per point, distance_m per point).

    track_xy: (T, 2) projected metres; route_arcs: arc ids in travel order;
    arc_geoms: (n_arcs, 4) array of ax, ay, bx, by in metres, indexed by arc_id.
    """
    j, d = _route_indices(np.asarray(track_xy, float), route_arcs, arc_geoms)
    return np.asarray(route_arcs)[j], d


def label_trip(trip_dir) -> pd.DataFrame:
    """Load a trip and return DataFrame[point_idx, arc_id, route_idx, dist_m]."""
    track, nodes, arcs, route = (load_track(trip_dir), load_nodes(trip_dir),
                                 load_arcs(trip_dir), load_route(trip_dir))
    tr = make_projector(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx, ny = to_metres(tr, nodes.lon, nodes.lat)
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    geoms = np.column_stack([nx[fn], ny[fn], nx[tn], ny[tn]])
    px, py = to_metres(tr, track.lon, track.lat)
    j, d = _route_indices(np.column_stack([px, py]), route, geoms)
    return pd.DataFrame({"point_idx": np.arange(len(j)), "arc_id": np.asarray(route)[j],
                         "route_idx": j, "dist_m": d})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    if "eval_test" in args.splits:
        raise SystemExit("eval_test labels may only be read in api/evaluation.py")
    rows = []
    for split in args.splits:
        for trip in list_trips(split):
            lab = label_trip(trip)
            assert (np.diff(lab.route_idx) >= 0).all()
            rows.append({"split": split, "trip": trip.name, "n_points": len(lab),
                         "mean_dist_m": lab.dist_m.mean(),
                         "p95_dist_m": np.percentile(lab.dist_m, 95)})
    df = pd.DataFrame(rows)
    df["flag_p95_gt_30"] = df.p95_dist_m > P95_FLAG_M
    out = RESULTS / "eda" / "label_quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out, index=False)
    print(df.round(2).to_string(index=False))
    print(f"\n{int(df.flag_p95_gt_30.sum())}/{len(df)} trips flagged (p95 > {P95_FLAG_M:.0f} m)")
    print(df[df.flag_p95_gt_30][["split", "trip", "mean_dist_m", "p95_dist_m"]].round(1)
          .to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
