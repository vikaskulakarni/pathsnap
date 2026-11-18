"""Capture table identity and software versions for the full-partition experiment."""
import hashlib
import json
import platform
from importlib.metadata import version

import pandas as pd

from common.config import DATA_PROCESSED, RESULTS, SEED

def record():
    inputs = {}
    for split in ("train", "student_test"):
        file = DATA_PROCESSED / f"{split}_features.parquet"
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        ids = pd.read_parquet(file, columns=["trip_id", "point_idx"])
        inputs[split] = {"sha256": digest.hexdigest(), "trips": int(ids.trip_id.nunique()),
                         "observations": len(ids.drop_duplicates()), "candidate_rows": len(ids)}
    packages = ("scikit-learn", "numpy", "pandas", "scipy", "optuna", "pyarrow", "joblib")
    run = {"seed": SEED, "python": platform.python_version(),
           "packages": {name: version(name) for name in packages}, "feature_tables": inputs}
    folder = RESULTS / "all80"
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "environment.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    record()
