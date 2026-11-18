"""Shared training routines for trip-grouped candidate classification."""

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold

from models.metrics import POINT_KEYS, pair_metrics, top1_accuracy

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_KEYS = POINT_KEYS + ["arc_id"]
METADATA = {*CANDIDATE_KEYS, "label", "t", "lon", "lat"}
DEV_SPLITS = ("train", "student_test")


def load_split(split: str) -> tuple[pd.DataFrame, list[str]]:
    """Load development candidates and keep trip and position metadata out of the model."""
    if split not in DEV_SPLITS:
        raise ValueError(f"Training utilities accept only {DEV_SPLITS}.")

    path = ROOT / "data" / "processed" / f"{split}_features.parquet"
    df = pd.read_parquet(path)
    required = {*CANDIDATE_KEYS, "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    if df.empty or df[CANDIDATE_KEYS].isna().any().any():
        raise ValueError(f"{path.name} must contain candidates with complete identifiers.")
    if df.duplicated(CANDIDATE_KEYS).any() or not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path.name} must contain unique candidates and binary labels.")

    feature_cols = [
        column for column in df.select_dtypes(include=["number", "bool"])
        if column not in METADATA
    ]
    if not feature_cols:
        raise ValueError(f"{path.name} has no numeric or boolean model features.")
    return df, feature_cols


def _positive_probabilities(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    return np.asarray(estimator.predict_proba(features))[:, classes.index(1)]


def cv_evaluate(
    estimator: BaseEstimator,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Fit a fresh estimator per fold so preprocessing cannot see validation trips."""
    features = df[list(feature_cols)]
    labels = df["label"]
    probabilities = np.empty(len(df), dtype=float)
    fold_rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(features, labels, groups=df["trip_id"]), start=1
    ):
        model = clone(estimator).fit(features.iloc[train_idx], labels.iloc[train_idx])
        scores = _positive_probabilities(model, features.iloc[validation_idx])
        probabilities[validation_idx] = scores
        validation = df.iloc[validation_idx]
        metrics = pair_metrics(validation["label"], scores)
        top1, _ = top1_accuracy(validation, scores)
        fold_rows.append({
            "fold": fold, "macro_f1": metrics["macro_f1"],
            "logloss": metrics["logloss"], "top1": top1,
        })

    folds = pd.DataFrame(fold_rows)
    return {
        "folds": folds,
        "cv_macro_f1_mean": float(folds["macro_f1"].mean()),
        "cv_macro_f1_std": float(folds["macro_f1"].std(ddof=0)),
        "cv_logloss": float(folds["logloss"].mean()),
        "cv_top1": float(folds["top1"].mean()),
        "oof": df[CANDIDATE_KEYS].assign(proba=probabilities),
    }


def fit_and_test(
    estimator: BaseEstimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[Any, dict[str, float | int], np.ndarray]:
    """Return a refitted model, student-test metrics, and aligned test probabilities."""
    if set(train_df["trip_id"]).intersection(test_df["trip_id"]):
        raise ValueError("Training and student-test trips must be disjoint.")
    test_points = len(test_df[POINT_KEYS].drop_duplicates())
    if test_points == 0:
        raise ValueError("Student testing requires observations with candidates.")

    columns = list(feature_cols)
    train_features = train_df[columns]
    test_features = test_df[columns]
    start = perf_counter()
    model = clone(estimator).fit(train_features, train_df["label"])
    train_s = perf_counter() - start

    start = perf_counter()
    test_scores = _positive_probabilities(model, test_features)
    infer_ms_per_point = (perf_counter() - start) * 1000 / test_points

    train_scores = _positive_probabilities(model, train_features)
    pair_results = pair_metrics(test_df["label"], test_scores)
    metrics = {f"test_{name}": value for name, value in pair_results.items()}
    test_top1, misses = top1_accuracy(test_df, test_scores)
    return model, {
        **metrics,
        "test_top1": test_top1,
        "test_candidate_misses": misses,
        "test_candidate_miss_rate": misses / test_points,
        "train_macro_f1": pair_metrics(train_df["label"], train_scores)["macro_f1"],
        "train_s": train_s,
        "infer_ms_per_point": infer_ms_per_point,
    }, test_scores


def log_experiment(row: Mapping[str, Any]) -> None:
    """Retain earlier runs when a later experiment adds a new metric column."""
    from common.config import SEED

    path = ROOT / "results" / "experiments.csv"
    record = {
        **row,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "params": json.dumps(row.get("params", {}), sort_keys=True, allow_nan=False),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    columns = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing)
            rows = list(reader)
            columns = reader.fieldnames or []
    columns = list(dict.fromkeys([*columns, *record]))

    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temp:
        temp_path = Path(temp.name)
        try:
            writer = csv.DictWriter(temp, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(record)
            temp.close()
            temp_path.replace(path)
        finally:
            temp.close()
            temp_path.unlink(missing_ok=True)
