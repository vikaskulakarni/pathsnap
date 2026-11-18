"""Train and compare five PathSnap candidate classifiers from Laptop 1's feature tables."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import SEED
from common.io import list_trips, load_arcs
from models.metrics import POINT_KEYS, route_continuity
from models.train_utils import ROOT, cv_evaluate, fit_and_test, load_split, log_experiment

MODELS = ("logreg", "knn", "rf", "lgbm", "mlp")


def build_model(name: str) -> tuple[str, BaseEstimator, dict]:
    """Apply scaling inside each fold for models that depend on feature scale."""
    if name == "logreg":
        params = {"class_weight": "balanced", "max_iter": 2000, "random_state": SEED}
        return "Linear", make_pipeline(StandardScaler(), LogisticRegression(**params)), params
    if name == "knn":
        params = {"n_neighbors": 25, "weights": "distance", "n_jobs": -1}
        return "Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(**params)), params
    if name == "rf":
        params = {
            "n_estimators": 300, "min_samples_leaf": 5, "class_weight": "balanced_subsample",
            "random_state": SEED, "n_jobs": -1,
        }
        return "Tree ensemble", RandomForestClassifier(**params), params
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        # Balanced weights are computed by the estimator from its current fit partition.
        params = {
            "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
            "class_weight": "balanced", "random_state": SEED, "n_jobs": -1, "verbosity": -1,
        }
        return "Boosting", LGBMClassifier(**params), params
    if name == "mlp":
        params = {
            "hidden_layer_sizes": (64, 32), "early_stopping": True, "max_iter": 200,
            "random_state": SEED,
        }
        return "Neural network", make_pipeline(StandardScaler(), MLPClassifier(**params)), params
    raise ValueError(f"Unknown model: {name}")


def load_student_graphs(data_root: Path | None) -> dict[str, dict[int, set[int]]]:
    """Index directed successor arcs for student-test continuity scoring."""
    if data_root is None:
        trips = list_trips("student_test")
    else:
        trips = sorted(path for path in (data_root / "student_test").iterdir() if path.is_dir())
    graphs = {}
    for trip in trips:
        arcs = load_arcs(trip)
        outgoing = arcs.groupby("from_node")["arc_id"].agg(set).to_dict()
        graphs[trip.name] = {
            arc.arc_id: outgoing.get(arc.to_node, set()) for arc in arcs.itertuples(index=False)
        }
    return graphs


def predicted_continuity(
    candidates: pd.DataFrame, scores: np.ndarray, graphs: dict[str, dict[int, set[int]]],
) -> float | None:
    """Average connected transitions without joining the end of one trip to another."""
    scored = candidates[[*POINT_KEYS, "arc_id"]].reset_index(drop=True).assign(score=scores)
    best = scored.loc[scored.groupby(POINT_KEYS, sort=False)["score"].idxmax()]
    steps = 0
    connected = 0.0
    for trip, rows in best.sort_values("point_idx").groupby("trip_id"):
        arcs = rows["arc_id"].tolist()
        fraction = route_continuity(arcs, graphs[str(trip)])
        if fraction is not None:
            count = len(arcs) - 1
            connected += count * fraction
            steps += count
    return connected / steps if steps else None


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_diagnostics(
    name: str, fitted: BaseEstimator, columns: list[str],
    labels: pd.Series, scores: np.ndarray, figure_dir: Path,
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(labels, scores >= 0.5, labels=[0, 1])
    display.ax_.set_title(f"{name} candidate confusion matrix · student_test")
    _save_figure(display.figure_, figure_dir / f"cm_{name}.png")
    if name in {"rf", "lgbm"}:
        importance = pd.Series(fitted.feature_importances_, index=columns).sort_values()
        fig, ax = plt.subplots()
        importance.plot.barh(ax=ax)
        ax.set(title=f"{name} feature importance · train", xlabel="Importance", ylabel="Feature")
        _save_figure(fig, figure_dir / f"fi_{name}.png")
    if name == "mlp":
        classifier = fitted[-1]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(np.arange(1, len(classifier.loss_curve_) + 1), classifier.loss_curve_)
        axes[0].set(title="MLP train loss", xlabel="Epoch", ylabel="Loss")
        validation = classifier.validation_scores_
        axes[1].plot(np.arange(1, len(validation) + 1), validation)
        axes[1].set(title="MLP internal validation", xlabel="Epoch", ylabel="Accuracy")
        _save_figure(fig, figure_dir / "mlp_training_curve.png")


def run_baselines(names: list[str], data_root: Path | None = None, folds: int = 5) -> None:
    train, columns = load_split("train")
    test, test_columns = load_split("student_test")
    if set(columns) != set(test_columns):
        raise ValueError("Training and student-test feature columns differ.")
    graphs = load_student_graphs(data_root)
    missing = set(test["trip_id"].astype(str)) - graphs.keys()
    if missing:
        raise FileNotFoundError("Student-test road graphs are missing. Check --data-root.")

    figure_dir = ROOT / "results" / "figures"
    oof_dir = ROOT / "results" / "oof"
    model_dir = ROOT / "artifacts" / "candidates"
    for directory in (figure_dir, oof_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"Training {name} with {folds} trip-grouped folds…", flush=True)
        family, estimator, params = build_model(name)
        cv = cv_evaluate(estimator, train, columns, n_splits=folds)
        fitted, test_metrics, scores = fit_and_test(estimator, train, test, columns)
        row = {
            "model": name, "family": family, "params": params, "folds": folds,
            **{key: value for key, value in cv.items() if key.startswith("cv_")},
            **test_metrics, "test_route_continuity": predicted_continuity(test, scores, graphs),
        }
        _save_diagnostics(name, fitted, columns, test["label"], scores, figure_dir)
        joblib.dump(fitted, model_dir / f"{name}.joblib")
        (model_dir / f"{name}_features.json").write_text(
            json.dumps(columns, indent=2) + "\n", encoding="utf-8",
        )
        cv["oof"].to_parquet(oof_dir / f"{name}.parquet", index=False)
        log_experiment(row)
        print(json.dumps(row, indent=2), flush=True)


def main(default_model: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default=default_model or "all")
    parser.add_argument("--data-root", type=Path, help="Parent directory of train and student_test.")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run_baselines(list(MODELS) if args.model == "all" else [args.model], args.data_root, args.folds)


if __name__ == "__main__":
    main()
