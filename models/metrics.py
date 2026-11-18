"""Metrics for candidate pairs and reconstructed routes."""

from collections.abc import Collection, Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

POINT_KEYS = ["trip_id", "point_idx"]


def _point_reference_present(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("Expected at least one observation with candidates.")
    return df.groupby(POINT_KEYS, sort=False)["label"].any()


def top1_accuracy(df: pd.DataFrame, scores: ArrayLike) -> tuple[float, int]:
    """Count a candidate miss as wrong and resolve ties in candidate row order."""
    probabilities = np.asarray(scores, dtype=float)
    if probabilities.shape != (len(df),) or not np.isfinite(probabilities).all():
        raise ValueError("Expected one finite score per candidate row.")
    reference_present = _point_reference_present(df)
    candidates = df[POINT_KEYS + ["label"]].reset_index(drop=True)
    candidates["score"] = probabilities
    winners = candidates.groupby(POINT_KEYS, sort=False)["score"].idxmax()
    return float(candidates.loc[winners, "label"].mean()), int((~reference_present).sum())


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """Measure the share of represented observations without a reference candidate."""
    return float((~_point_reference_present(df)).mean())


def pair_metrics(
    y: ArrayLike, proba: ArrayLike, threshold: float = 0.5
) -> dict[str, float]:
    """Report binary candidate metrics without requiring both classes in validation."""
    predictions = np.asarray(proba, dtype=float) >= threshold
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "macro_f1": float(f1_score(y, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "logloss": float(log_loss(y, proba, labels=[0, 1])),
    }


def route_continuity(
    arcs_in_order: Sequence[int], successors: Mapping[int, Collection[int]]
) -> float | None:
    """Measure same-arc or directed successor steps within one trip."""
    if len(arcs_in_order) < 2:
        return None
    connected = sum(
        current == following or following in successors.get(current, ())
        for current, following in pairwise(arcs_in_order)
    )
    return connected / (len(arcs_in_order) - 1)
