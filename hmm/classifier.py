"""Load the selected classifier artifacts and score candidate tables with them."""
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from common.config import ARTIFACTS


def inference_directory():
    release = os.environ.get("PATHSNAP_MODEL_RELEASE", "demonstrated")
    if release not in {"demonstrated", "all80"}:
        raise ValueError("Unknown PATHSNAP_MODEL_RELEASE. Choose demonstrated or all80.")
    return ARTIFACTS / "all80" if release == "all80" else ARTIFACTS


def load_artifacts(alt: bool = False, directory=None):
    """(model, feature list, version string). Loaded once at API start-up."""
    directory = inference_directory() if directory is None else directory
    model = joblib.load(directory / ("model_alt.joblib" if alt else "model.joblib"))
    features = json.loads((directory / "feature_list.json").read_text())
    meta = json.loads((directory / "best_params.json").read_text())
    version = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:8]
    return model, features, version


def score_candidates(model, features: list[str], df: pd.DataFrame) -> np.ndarray:
    """P(candidate arc is the reference arc) for every row; labels are never used."""
    X = df[features].astype(float).to_numpy()
    return model.predict_proba(X)[:, 1]
