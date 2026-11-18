"""Load the selected classifier artifacts and score candidate tables with them."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from common.config import ARTIFACTS


def load_artifacts(alt: bool = False):
    """(model, feature list, version string). Loaded once at API start-up."""
    model = joblib.load(ARTIFACTS / ("model_alt.joblib" if alt else "model.joblib"))
    features = json.loads((ARTIFACTS / "feature_list.json").read_text())
    meta = json.loads((ARTIFACTS / "best_params.json").read_text())
    version = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:8]
    return model, features, version


def score_candidates(model, features: list[str], df: pd.DataFrame) -> np.ndarray:
    """P(candidate arc is the reference arc) for every row; labels are never used."""
    X = df[features].astype(float).to_numpy()
    return model.predict_proba(X)[:, 1]
