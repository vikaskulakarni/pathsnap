"""Check the starter classifier against a local development trip."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from common.config import ARTIFACTS
from features.build_features import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", type=Path)
    trip = parser.parse_args().trip.resolve()
    if trip.parent.name not in {"train", "student_test"}:
        parser.error("Choose a development trip from train or student_test.")
    starter = ARTIFACTS / "starter"
    model = joblib.load(starter / "logreg.joblib")
    columns = json.loads((starter / "feature_list.json").read_text())
    candidates = build_features(trip)
    if columns != list(model.feature_names_in_):
        raise ValueError("The saved feature order differs from the fitted model.")
    scores = model.predict_proba(candidates[columns])[:, list(model.classes_).index(1)]
    if not len(scores) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Invalid candidate probabilities.")
    print(f"{trip.name}: scored {len(scores)} candidates for {candidates.point_idx.nunique()} observations.")


if __name__ == "__main__":
    main()
