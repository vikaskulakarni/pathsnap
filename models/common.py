"""Shared helpers for model training/tuning: data loading, trip-grouped folds, metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from common.config import DATA_PROCESSED, RESULTS
from features.build_features import FEATURE_COLUMNS


def load_table(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / f"{split}_features.parquet")
    df["fallback"] = df["fallback"].astype(int)
    return df.reset_index(drop=True)


def flagged_train_trips() -> set[str]:
    """Train trips whose reference labels are unreliable (p95 > 30 m from label_points).
    Derived from train labels only; excluded from fitting, CV and decoder tuning."""
    q = pd.read_csv(RESULTS / "eda" / "label_quality.csv", dtype={"trip": str})
    return set(q[(q.split == "train") & q.flag_p95_gt_30].trip)


def load_train() -> pd.DataFrame:
    df = load_table("train")
    return df[~df.trip_id.isin(flagged_train_trips())].reset_index(drop=True)


def load_experiment_tables(all_trips: bool):
    """Choose the labelled development tables and validate the full-partition protocol."""
    train = load_table("train") if all_trips else load_train()
    test = load_table("student_test")
    if all_trips:
        train_ids, test_ids = set(train.trip_id), set(test.trip_id)
        if len(train_ids) != 80 or len(test_ids) != 10 or train_ids & test_ids:
            raise ValueError("Expected 80 training and 10 separate student-test trips.")
    return train, test


def point_top1(df: pd.DataFrame, proba: np.ndarray) -> float:
    """Observation-level Top-1: per GPS point take the highest-scoring candidate and check
    it is the reference arc. Points whose reference arc is not a candidate count as wrong."""
    s = pd.Series(proba, index=df.index)
    idx = s.groupby([df.trip_id, df.point_idx]).idxmax()
    return float(df.label.loc[idx].sum() / idx.shape[0])


def row_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))}


def chosen_arcs(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point): DataFrame[trip_id, point_idx, arc_id]."""
    s = pd.Series(proba, index=df.index)
    idx = s.groupby([df.trip_id, df.point_idx]).idxmax()
    return df.loc[idx, ["trip_id", "point_idx", "arc_id"]].reset_index(drop=True)


__all__ = ["FEATURE_COLUMNS", "load_table", "load_train", "flagged_train_trips", "point_top1",
           "row_metrics", "chosen_arcs"]
