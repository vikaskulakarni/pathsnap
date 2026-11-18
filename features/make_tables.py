"""Build labelled feature tables for train / student_test (never eval_test).

Labels come from data_processing.label_points and are attached after the label-free
build_features(), so features never see them.
Run: python -m features.make_tables --splits train student_test
"""
import argparse
import time

import numpy as np
import pandas as pd

from common.config import DATA_PROCESSED, K, RADIUS_M, RESULTS
from common.io import list_trips, load_arcs
from data_processing.label_points import label_trip
from features.build_features import attach_labels, build_features
from features.candidates import twin_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    if "eval_test" in args.splits:
        raise SystemExit("refusing to build an eval_test feature table "
                         "(predict_trip builds eval features on the fly)")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eda").mkdir(parents=True, exist_ok=True)
    summary = []
    for split in args.splits:
        frames, n_pts, n_miss, n_twin_only, secs = [], 0, 0, 0, []
        for trip in list_trips(split):
            t0 = time.time()
            lab = label_trip(trip)
            df = attach_labels(build_features(trip), lab)
            secs.append(time.time() - t0)
            hit = (df.groupby("point_idx").label.max()
                   .reindex(lab.point_idx, fill_value=0).to_numpy() == 1)
            miss_pts = lab.point_idx.to_numpy()[~hit]
            arcs = load_arcs(trip)
            ref_twin = twin_ids(arcs.from_node, arcs.to_node)[lab.arc_id.to_numpy()[~hit]]
            keys = set(zip(df.point_idx.tolist(), df.arc_id.tolist()))
            n_twin_only += sum((p, a) in keys
                               for p, a in zip(miss_pts.tolist(), ref_twin.tolist()) if a >= 0)
            n_pts += len(lab)
            n_miss += len(miss_pts)
            frames.append(df)
        full = pd.concat(frames, ignore_index=True)
        out = DATA_PROCESSED / f"{split}_features.parquet"
        full.to_parquet(out, index=False)
        row = {"split": split, "trips": len(secs), "points": n_pts, "rows": len(full),
               "miss_rate": n_miss / n_pts, "miss_twin_only_share": n_twin_only / max(n_miss, 1),
               "mean_rows_per_point": len(full) / n_pts, "positive_share": full.label.mean(),
               "sec_per_trip": float(np.mean(secs)), "k": K, "radius_m": RADIUS_M}
        summary.append(row)
        print(f"\n[{split}] saved {out}")
        for key, val in row.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    pd.DataFrame(summary).to_csv(RESULTS / "eda" / "feature_build_stats.csv", index=False)


if __name__ == "__main__":
    main()
