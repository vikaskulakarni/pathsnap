"""Bayesian hyperparameter optimisation (Optuna TPE) of the best model from models.train_models.

Search: TPE on a seeded half of the train trips, 3-fold GroupKFold by trip, objective =
mean observation-level Top-1. The best configuration is then refit on all train trips and
compared with the untuned model on the SAME 5 trip-grouped folds and on student_test.
The final model is chosen by cross-validation, not by student_test.

Run: python -m optimisation.bayes_tune   (after models.train_models)
"""
import hashlib
import json
import shutil
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from common.config import ARTIFACTS, FIGURES, RESULTS, SEED
from models.common import FEATURE_COLUMNS, chosen_arcs, load_table, load_train, point_top1, row_metrics
from models.train_models import N_FOLDS, connected_share

N_TRIALS = 20
SEARCH_FOLDS = 3
SEARCH_TRIPS = 36


def build(name: str, p: dict):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=SEED, **p)
    return RandomForestClassifier(n_jobs=-1, random_state=SEED, **p)


def suggest(name: str, trial: optuna.Trial) -> dict:
    if name == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 60, 250),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)}
    return {"n_estimators": trial.suggest_int("n_estimators", 40, 120),
            "max_depth": trial.suggest_int("max_depth", 8, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0)}


def cv_eval(name, params, train, X, y, groups):
    """5-fold trip-grouped CV (same folds as models.train_models) -> metrics + OOF probabilities."""
    oof = np.zeros(len(train))
    top1, f1, ll = [], [], []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        oof[va] = build(name, params).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        m = row_metrics(y[va], oof[va])
        top1.append(point_top1(train.iloc[va], oof[va]))
        f1.append(m["macro_f1"])
        ll.append(m["log_loss"])
    return oof, {"cv_top1": np.mean(top1), "cv_macro_f1": np.mean(f1), "cv_log_loss": np.mean(ll)}


def main():
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    tunable = comp[comp.model.isin(["hist_gradient_boosting", "random_forest"])].sort_values("cv_top1", ascending=False)
    name = tunable.iloc[0].model
    print(f"best overall: {comp.sort_values('cv_top1', ascending=False).iloc[0].model}; tuning: {name}", flush=True)
    train, test = load_train(), load_table("student_test")
    X, y, groups = train[FEATURE_COLUMNS].to_numpy(float), train.label.to_numpy(), train.trip_id.to_numpy()

    rng = np.random.RandomState(SEED)
    trips = sorted(train.trip_id.unique())
    keep = set(rng.choice(trips, size=min(SEARCH_TRIPS, len(trips)), replace=False))
    m = train.trip_id.isin(keep).to_numpy()
    sub, Xs, ys, gs = train[m].reset_index(drop=True), X[m], y[m], groups[m]
    search_folds = list(GroupKFold(n_splits=SEARCH_FOLDS).split(Xs, ys, gs))

    def objective(trial):
        p = suggest(name, trial)
        scores = []
        for tr, va in search_folds:
            proba = build(name, p).fit(Xs[tr], ys[tr]).predict_proba(Xs[va])[:, 1]
            scores.append(point_top1(sub.iloc[va], proba))
        return float(np.mean(scores))

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)
    search_secs = time.perf_counter() - t0
    study.trials_dataframe().to_csv(RESULTS / "optuna_trials.csv", index=False)
    best = study.best_params
    print(f"search done in {search_secs:.0f}s, best objective {study.best_value:.4f}: {best}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    vals = [t.value for t in study.trials]
    ax.plot(range(1, len(vals) + 1), vals, "o-", label="trial")
    ax.plot(range(1, len(vals) + 1), np.maximum.accumulate(vals), label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV Top-1 (search sample)")
    ax.set_title(f"Optuna TPE search: {name}")
    ax.legend()
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "optuna_history.png", dpi=150)
    plt.close(fig)

    untuned_row = comp[comp.model == name].iloc[0]
    from models.train_models import model_zoo
    untuned_params = model_zoo()[name][2]().get_params()
    oof, cvm = cv_eval(name, best, train, X, y, groups)
    t0 = time.perf_counter()
    tuned = build(name, best).fit(X, y)
    fit_secs = time.perf_counter() - t0
    Xt, yt = test[FEATURE_COLUMNS].to_numpy(float), test.label.to_numpy()
    t0 = time.perf_counter()
    p = tuned.predict_proba(Xt)[:, 1]
    infer = (time.perf_counter() - t0) / len(Xt) * 1e6
    met = row_metrics(yt, p)
    tuned_row = {"configuration": "tuned (Optuna TPE)", **cvm, "test_top1": point_top1(test, p),
                 "test_macro_f1": met["macro_f1"], "test_log_loss": met["log_loss"],
                 "test_route_continuity": connected_share(test, chosen_arcs(test, p)),
                 "fit_seconds": fit_secs, "infer_ms_per_1k_rows": infer}
    untuned = {"configuration": "untuned", "cv_top1": untuned_row.cv_top1, "cv_macro_f1": untuned_row.cv_macro_f1,
               "cv_log_loss": untuned_row.cv_log_loss, "test_top1": untuned_row.test_top1,
               "test_macro_f1": untuned_row.test_macro_f1, "test_log_loss": untuned_row.test_log_loss,
               "test_route_continuity": untuned_row.test_route_continuity, "fit_seconds": untuned_row.fit_seconds,
               "infer_ms_per_1k_rows": untuned_row.infer_ms_per_1k_rows}
    cmp = pd.DataFrame([untuned, tuned_row])
    cmp.insert(0, "model", name)
    cmp.round(5).to_csv(RESULTS / "tuning_comparison.csv", index=False)
    print(cmp.round(4).to_string(index=False), flush=True)

    pd.DataFrame({"trip_id": train.trip_id, "point_idx": train.point_idx, "arc_id": train.arc_id,
                  "proba": oof.astype("float32")}).to_parquet(RESULTS / "oof" / f"{name}_tuned.parquet", index=False)
    joblib.dump(tuned, ARTIFACTS / "models" / f"{name}_tuned.joblib")

    # final selection by cross-validation (never by student_test)
    use_tuned = tuned_row["cv_top1"] > untuned["cv_top1"]
    chosen = f"{name}_tuned" if use_tuned else name
    shutil.copy(ARTIFACTS / "models" / f"{chosen}.joblib", ARTIFACTS / "model.joblib")
    alt = comp[comp.model != name].sort_values("cv_top1", ascending=False).iloc[0].model
    shutil.copy(ARTIFACTS / "models" / f"{alt}.joblib", ARTIFACTS / "model_alt.joblib")
    record = {"model": name, "tuned": bool(use_tuned), "selected_file": chosen, "alt_model": alt,
              "best_params": best, "untuned_params": {k: v for k, v in untuned_params.items()
                                                      if k in best},
              "search": {"sampler": "TPE", "n_trials": N_TRIALS, "folds": SEARCH_FOLDS,
                         "trips": len(keep), "objective": "mean observation-level Top-1",
                         "best_objective": study.best_value, "seconds": search_secs, "seed": SEED}}
    (ARTIFACTS / "best_params.json").write_text(json.dumps(record, indent=2) + "\n")
    version = hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()[:8]
    print(f"\nselected: {chosen} (tuned CV Top-1 {tuned_row['cv_top1']:.4f} vs untuned {untuned['cv_top1']:.4f}); "
          f"alt: {alt}; version {version}")


if __name__ == "__main__":
    main()
