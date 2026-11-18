"""Candidate miss rate per trip (share of GPS points whose reference arc is not a candidate).

Shows how much of the train miss rate comes from the trips whose reference labels fail the
quality check. Reads the labelled feature tables written by features.make_tables.

Run: python -m features.miss_report
"""
import pandas as pd

from common.config import RESULTS
from models.common import flagged_train_trips, load_table


def main():
    flagged = flagged_train_trips()
    rows = []
    for split in ("train", "student_test"):
        t = load_table(split)
        hit = t.groupby(["trip_id", "point_idx"]).label.max()
        per = hit.groupby("trip_id").agg(points="size", hits="sum")
        per["miss_rate"] = 1 - per.hits / per.points
        per["split"] = split
        per["label_flagged"] = per.index.isin(flagged)
        rows.append(per.reset_index())
    out = pd.concat(rows, ignore_index=True)[["split", "trip_id", "points", "miss_rate", "label_flagged"]]
    out.round(5).to_csv(RESULTS / "eda" / "candidate_miss_by_trip.csv", index=False)

    def pooled(d):
        return float((d.miss_rate * d.points).sum() / d.points.sum())

    tr, st = out[out.split == "train"], out[out.split == "student_test"]
    print(f"train, all 80 trips:           {pooled(tr):.4f}")
    print(f"train, 73 trips (unflagged):   {pooled(tr[~tr.label_flagged]):.4f}")
    print(f"train, 7 flagged trips only:   {pooled(tr[tr.label_flagged]):.4f}")
    print(f"student_test (10 trips):       {pooled(st):.4f}")
    print(tr.sort_values("miss_rate", ascending=False).head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
