"""Measure the decoder's corrections and regressions against aligned student-test labels."""
import argparse
from pathlib import Path

import pandas as pd

from common.config import DATA_RAW, RESULTS
from data_processing.label_points import label_trip


def compare(data_root: Path) -> pd.DataFrame:
    rows = []
    for trip in sorted((data_root / "student_test").iterdir()):
        if not trip.is_dir():
            continue
        predictions = pd.read_csv(RESULTS / "predictions" / f"{trip.name}.csv")
        truth = label_trip(trip)[["point_idx", "arc_id"]]
        joined = truth.merge(predictions[["point_idx", "classifier_arc", "hmm_arc"]],
                             on="point_idx", how="outer", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all() or joined.empty:
            raise ValueError(f"Observation mismatch for {trip.name}")
        initial = joined.classifier_arc.eq(joined.arc_id)
        decoded = joined.hmm_arc.eq(joined.arc_id)
        rows.append({"trip": trip.name, "points": len(joined),
                     "both_correct": int((initial & decoded).sum()),
                     "corrected_by_hmm": int((~initial & decoded).sum()),
                     "lost_by_hmm": int((initial & ~decoded).sum()),
                     "both_wrong": int((~initial & ~decoded).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_RAW)
    result = compare(parser.parse_args().data_root)
    if len(result) != 10:
        parser.error("Expected the 10 student-test trips.")
    result.to_csv(RESULTS / "decoder_changes.csv", index=False)
