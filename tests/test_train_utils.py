import csv
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import train_utils


def _candidates(trip_count: int, offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "trip_id": f"{trip:08d}", "point_idx": point, "arc_id": arc,
            "label": int(arc == 1), "dist_m": 10.0 if arc == 1 else 60.0,
            "trip_marker": trip,
        }
        for trip in range(offset, offset + trip_count)
        for point in range(2)
        for arc in range(2)
    ])


class _CheckUnseenTrips(ClassifierMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_CheckUnseenTrips":
        self.training_trips_ = set(X["trip_marker"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.training_trips_.isdisjoint(X["trip_marker"])
        scores = 1 - X["dist_m"].to_numpy() / 100
        return np.column_stack((1 - scores, scores))


def test_grouped_cv_keeps_trips_out_of_training_and_aligns_oof_rows() -> None:
    df = _candidates(5).sample(frac=1, random_state=0)
    result = train_utils.cv_evaluate(_CheckUnseenTrips(), df, ["trip_marker", "dist_m"])
    assert len(result["folds"]) == 5
    assert result["cv_top1"] == result["cv_macro_f1_mean"] == 1
    assert result["cv_macro_f1_std"] == 0
    assert result["oof"].index.equals(df.index)
    np.testing.assert_allclose(result["oof"]["proba"], 1 - df["dist_m"] / 100)


def test_fit_and_test_returns_a_fitted_clone_and_metrics() -> None:
    estimator = make_pipeline(StandardScaler(), LogisticRegression())
    train, test = _candidates(5), _candidates(2, offset=5)
    model, metrics, scores = train_utils.fit_and_test(estimator, train, test, ["dist_m"])
    assert model is not estimator
    assert metrics["test_top1"] == metrics["train_macro_f1"] == 1
    assert metrics["test_candidate_misses"] == 0
    assert metrics["infer_ms_per_point"] >= 0
    np.testing.assert_allclose(scores, model.predict_proba(test[["dist_m"]])[:, 1])
    with pytest.raises(ValueError, match="disjoint"):
        train_utils.fit_and_test(estimator, train, train, ["dist_m"])


def test_load_split_preserves_ids_and_filters_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_utils, "ROOT", tmp_path)
    folder = tmp_path / "data" / "processed"
    folder.mkdir(parents=True)
    df = _candidates(1).drop(columns="trip_marker").assign(
        t=12.0, lon=77.0, lat=13.0, fallback=False,
    )
    df.to_parquet(folder / "train_features.parquet", index=False)
    loaded, features = train_utils.load_split("train")
    assert loaded["trip_id"].tolist() == ["00000000"] * 4
    assert features == ["dist_m", "fallback"]
    with pytest.raises(ValueError, match="only"):
        train_utils.load_split("eval_test")


def test_log_experiment_retains_old_rows_and_records_shared_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_utils, "ROOT", tmp_path)
    config = ModuleType("common.config")
    config.SEED = 42
    monkeypatch.setitem(sys.modules, "common.config", config)

    train_utils.log_experiment({"model": "baseline", "params": {"C": 1.0}})
    train_utils.log_experiment({"model": "tuned", "test_top1": 0.75})
    with (tmp_path / "results" / "experiments.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        first, second = list(csv.DictReader(stream))

    assert first["model"] == "baseline"
    assert first["seed"] == second["seed"] == "42"
    assert first["timestamp"]
    assert json.loads(first["params"]) == {"C": 1.0}
    assert first["test_top1"] == ""
    assert second["test_top1"] == "0.75"
