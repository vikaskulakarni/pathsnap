"""Per-point reference labels: map each GPS point to an arc of its own .route.

Rule: the label of point i is route[j_i] where j_i is non-decreasing in i (the route
index may stay the same or move forward, never back). Among all such sequences we pick
the one with the smallest total point-to-arc distance (dynamic programming), not the
greedy nearest arc per point, so noise or a nearby parallel/return road cannot make the
label jump to an earlier or later part of the route.

Run: python -m data_processing.label_points
"""
import numpy as np
import pandas as pd

from common.config import RADIUS_M
from common.geo import make_projector, point_segment_distance, to_metres
from common.io import list_trips, load_arcs, load_nodes, load_route, load_track


def label_points(trip_dir) -> pd.DataFrame:
    """Return DataFrame[point_idx, arc_id, route_idx, dist_m] for one trip.

    Needs the trip's .route, so it is for train/student_test labels and for the
    server-side evaluation only; never call it from build_features/predict_trip.
    """
    track, nodes, arcs, route = (load_track(trip_dir), load_nodes(trip_dir),
                                 load_arcs(trip_dir), load_route(trip_dir))
    tr = make_projector(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx, ny = to_metres(tr, nodes.lon, nodes.lat)
    px, py = to_metres(tr, track.lon, track.lat)

    r = np.asarray(route)
    fn, tn = arcs.from_node.to_numpy()[r], arcs.to_node.to_numpy()[r]
    # (T, R) distance from every point to every route arc segment
    dist, _ = point_segment_distance(px[:, None], py[:, None], nx[fn][None, :],
                                     ny[fn][None, :], nx[tn][None, :], ny[tn][None, :])

    T, R = dist.shape
    cost = dist[0].copy()
    back = np.zeros((T, R), dtype=np.int32)
    idx = np.arange(R)
    for i in range(1, T):
        # best predecessor of j is the cheapest k <= j: a running minimum with its argmin
        run_min = np.minimum.accumulate(cost)
        is_new = cost <= run_min
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        back[i] = arg
        cost = dist[i] + run_min
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(cost))
    for i in range(T - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return pd.DataFrame({"point_idx": np.arange(T), "arc_id": r[j], "route_idx": j,
                         "dist_m": dist[np.arange(T), j]})


def main():
    rows = []
    for split in ("train", "student_test"):
        for trip in list_trips(split):
            lab = label_points(trip)
            assert (np.diff(lab.route_idx) >= 0).all()
            rows.append((split, trip.name, len(lab), lab.dist_m.median(),
                         (lab.dist_m > RADIUS_M).mean()))
    df = pd.DataFrame(rows, columns=["split", "trip", "n_points", "median_dist_m", "share_over_radius"])
    print(df.groupby("split")[["n_points", "median_dist_m", "share_over_radius"]].mean().round(3))
    print(f"trips with >5% of labelled points farther than {RADIUS_M:.0f} m from their arc:")
    print(df[df.share_over_radius > 0.05].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
