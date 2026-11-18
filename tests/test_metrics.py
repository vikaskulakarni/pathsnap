import pandas as pd
import pytest

from models.metrics import candidate_miss_rate, pair_metrics, route_continuity, top1_accuracy


def test_top1_keeps_trips_separate_and_counts_misses() -> None:
    candidates = pd.DataFrame({
        "trip_id": ["00000001"] * 6 + ["00000002"] * 3,
        "point_idx": [0] * 3 + [1] * 3 + [0] * 3,
        "arc_id": [0, 1, 2] * 3,
        "label": [0, 1, 0, 1, 0, 0, 0, 0, 0],
    }, index=[0, 1, 2] * 3)
    scores = [0.1, 0.8, 0.2, 0.3, 0.9, 0.1, 0.7, 0.4, 0.1]
    # One correct winner, one wrong winner, and one point without a reference candidate.
    assert top1_accuracy(candidates, scores) == (pytest.approx(1 / 3), 1)
    assert candidate_miss_rate(candidates) == pytest.approx(1 / 3)


def test_pair_metrics_work_with_one_class_in_validation() -> None:
    metrics = pair_metrics([0, 0], [0.1, 0.2])
    assert metrics["accuracy"] == 1
    assert metrics["precision"] == metrics["recall"] == 0
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["logloss"] == pytest.approx(0.1642520335)


def test_route_continuity_uses_directed_successors() -> None:
    assert route_continuity([4, 4, 5, 4], {4: {5}}) == pytest.approx(2 / 3)
    assert route_continuity([4], {4: {5}}) is None
