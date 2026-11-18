"""Candidate miss rate for K x radius, on TRAIN only.

Run: python -m features.candidate_sweep
"""
import itertools
import time

import numpy as np
import pandas as pd

from common.config import RESULTS
from common.geo import make_projector, to_metres
from common.io import list_trips, load_arcs, load_nodes, load_track
from data_processing.label_points import label_trip
from features.candidates import build_arc_index, get_candidates

KS = (8, 12, 16)
RADII = (30, 60, 100)


def main():
    acc = {c: [0, 0, 0] for c in itertools.product(KS, RADII)}  # points, misses, rows
    t0 = time.time()
    for trip in list_trips("train"):
        track, nodes, arcs = load_track(trip), load_nodes(trip), load_arcs(trip)
        tr = make_projector(nodes.lon.to_numpy(), nodes.lat.to_numpy())
        nx, ny = to_metres(tr, nodes.lon, nodes.lat)
        px, py = to_metres(tr, track.lon, track.lat)
        xy = np.column_stack([px, py])
        index = build_arc_index(np.column_stack([nx, ny]), arcs)
        ref = label_trip(trip).arc_id.to_numpy()
        for k, r in acc:
            c = get_candidates(xy, index, k, r)
            pi = c.point_idx.to_numpy()
            hit = np.bincount(pi, weights=(c.arc_id.to_numpy() == ref[pi]),
                              minlength=len(xy)) > 0
            a = acc[(k, r)]
            a[0] += len(xy)
            a[1] += int((~hit).sum())
            a[2] += len(c)
        print(f"{trip.name} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame([{"k": k, "radius_m": r, "points": p, "miss_rate": m / p,
                        "mean_rows_per_point": n / p} for (k, r), (p, m, n) in acc.items()])
    out = RESULTS / "eda" / "candidate_sweep.csv"
    df.round(5).to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
