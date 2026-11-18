"""Train and compare five model families on the candidate-feature table.

Fit/CV on ALL 80 train trips (GroupKFold by trip; the 7 unreliable-label trips are only studied
in models/ablation_trips.py), development comparison on the 10 student_test trips. Saves out-of-fold probabilities
(for decoder tuning), every fitted model, the comparison table and figures.

Run: python -m models.train_models
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common.config import ARTIFACTS, FIGURES, RESULTS, SEED
from models.common import FEATURE_COLUMNS, load_table, load_train, point_top1, row_metrics

N_FOLDS = 5


def model_zoo() -> dict:
    """name -> (algorithm family, configuration string, factory)."""
    return {
        "logistic_regression": ("Linear / probabilistic", "StandardScaler + LogisticRegression(C=1.0)",
                                lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
        "gaussian_naive_bayes": ("Probabilistic (generative)", "StandardScaler + GaussianNB",
                                 lambda: make_pipeline(StandardScaler(), GaussianNB())),
        "decision_tree": ("Tree-based", "DecisionTree(max_depth=12, min_samples_leaf=50)",
                          lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)),
        "random_forest": ("Ensemble (bagging)", "RandomForest(n_estimators=60, max_depth=14, min_samples_leaf=50)",
                          lambda: RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=50,
                                                         n_jobs=-1, random_state=SEED)),
        "hist_gradient_boosting": ("Boosting", "HistGradientBoosting(max_iter=150, learning_rate=0.1)",
                                   lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                                          random_state=SEED)),
    }


def connected_share(df: pd.DataFrame, chosen: pd.DataFrame) -> float:
    """Route continuity of one arc per point: share of consecutive points whose arcs are equal
    or directed-connected (to_node of one == from_node of the next). Needs the trips' arcs."""
    from common.io import list_trips, load_arcs
    paths = {p.name: p for p in list_trips("student_test")}
    conn = pairs = 0
    for tid, g in chosen.groupby("trip_id"):
        arcs = load_arcs(paths[tid])
        a = g.sort_values("point_idx").arc_id.to_numpy()
        fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
        conn += int(((a[:-1] == a[1:]) | (tn[a[:-1]] == fn[a[1:]])).sum())
        pairs += len(a) - 1
    return conn / pairs


def main():
    train, test = load_train(), load_table("student_test")
    X, y, groups = train[FEATURE_COLUMNS].to_numpy(float), train.label.to_numpy(), train.trip_id.to_numpy()
    Xt, yt = test[FEATURE_COLUMNS].to_numpy(float), test.label.to_numpy()
    print(f"train: {train.trip_id.nunique()} trips, {len(train):,} rows | "
          f"student_test: {test.trip_id.nunique()} trips, {len(test):,} rows", flush=True)
    (RESULTS / "oof").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))

    rows, test_proba = [], {}
    for name, (family, config, make) in model_zoo().items():
        oof = np.zeros(len(train))
        fold_top1, fold_f1, fold_ll = [], [], []
        t0 = time.perf_counter()
        for tr, va in folds:
            m = make().fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
            met = row_metrics(y[va], oof[va])
            fold_top1.append(point_top1(train.iloc[va], oof[va]))
            fold_f1.append(met["macro_f1"])
            fold_ll.append(met["log_loss"])
        cv_secs = time.perf_counter() - t0
        t0 = time.perf_counter()
        final = make().fit(X, y)
        fit_secs = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = final.predict_proba(Xt)[:, 1]
        infer_ms_per_1k = (time.perf_counter() - t0) / len(Xt) * 1e6
        test_proba[name] = p
        met = row_metrics(yt, p)
        from models.common import chosen_arcs
        rows.append({"model": name, "algorithm_family": family, "configuration": config,
                     "cv_macro_f1": np.mean(fold_f1), "cv_top1": np.mean(fold_top1),
                     "cv_top1_std": np.std(fold_top1), "cv_log_loss": np.mean(fold_ll),
                     "test_top1": point_top1(test, p), "test_macro_f1": met["macro_f1"],
                     "test_precision": met["precision"], "test_recall": met["recall"],
                     "test_log_loss": met["log_loss"],
                     "test_route_continuity": connected_share(test, chosen_arcs(test, p)),
                     "fit_seconds": fit_secs, "cv_seconds": cv_secs, "infer_ms_per_1k_rows": infer_ms_per_1k})
        pd.DataFrame({"trip_id": train.trip_id, "point_idx": train.point_idx, "arc_id": train.arc_id,
                      "proba": oof.astype("float32")}).to_parquet(RESULTS / "oof" / f"{name}.parquet", index=False)
        joblib.dump(final, ARTIFACTS / "models" / f"{name}.joblib")
        print(f"{name}: cv_top1={rows[-1]['cv_top1']:.4f} test_top1={rows[-1]['test_top1']:.4f} "
              f"cv_f1={rows[-1]['cv_macro_f1']:.4f} ({cv_secs:.0f}s cv)", flush=True)

    df = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
    df.round(5).to_csv(RESULTS / "model_comparison.csv", index=False)
    (ARTIFACTS / "feature_list.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2) + "\n")

    for col, ttl, fn in (("cv_top1", "CV Top-1 (observation level), GroupKFold by trip", "cv_top1"),
                         ("cv_macro_f1", "CV macro F1 (candidate rows)", "cv_macro_f1"),
                         ("test_top1", "student_test Top-1 (observation level)", "student_top1"),
                         ("test_log_loss", "student_test log loss (lower is better)", "student_log_loss")):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        d = df.sort_values(col, ascending=(col == "test_log_loss"))
        ax.barh(d.model, d[col], color="#3b6ea5")
        ax.set_xlabel(col)
        ax.set_title(ttl)
        fig.tight_layout()
        fig.savefig(FIGURES / f"model_{fn}.png", dpi=150)
        plt.close(fig)
    best = df.iloc[0].model
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(yt, (test_proba[best] >= 0.5).astype(int), ax=ax, colorbar=False)
    ax.set_title(f"{best}: student_test candidate rows")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_confusion_best.png", dpi=150)
    plt.close(fig)
    print("\n" + df[["model", "cv_top1", "cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss",
                     "test_route_continuity", "fit_seconds"]].round(4).to_string(index=False))
    print(f"\nbest by CV Top-1: {best}")


if __name__ == "__main__":
    main()
