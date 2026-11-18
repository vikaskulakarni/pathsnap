"""Train five baselines, rank them, tune the leading model and export the measured results."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import optuna
import pandas as pd

from common.config import SEED
from models.compare_models import compare_models
from models.train_baselines import MODELS, build_model, load_student_graphs, predicted_continuity, run_baselines
from models.train_utils import ROOT, cv_evaluate, fit_and_test, load_split, log_experiment


def search_space(name: str, trial: optuna.Trial, max_neighbors: int) -> dict:
    if name == "logreg":
        return {"logisticregression__C": trial.suggest_float("logisticregression__C", 0.01, 100, log=True),
                "logisticregression__class_weight": trial.suggest_categorical("logisticregression__class_weight", [None, "balanced"])}
    if name == "knn":
        return {"kneighborsclassifier__n_neighbors": trial.suggest_int("kneighborsclassifier__n_neighbors", 1, min(51, max_neighbors)),
                "kneighborsclassifier__weights": trial.suggest_categorical("kneighborsclassifier__weights", ["distance", "uniform"]),
                "kneighborsclassifier__p": trial.suggest_int("kneighborsclassifier__p", 1, 2)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
    if name == "lgbm":
        return {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 80)}
    return {"mlpclassifier__alpha": trial.suggest_float("mlpclassifier__alpha", 1e-5, 0.1, log=True),
            "mlpclassifier__learning_rate_init": trial.suggest_float("mlpclassifier__learning_rate_init", 1e-4, 0.02, log=True),
            "mlpclassifier__batch_size": trial.suggest_categorical("mlpclassifier__batch_size", [32, 64, 128])}


def finish(data_root: Path | None, trials: int, timeout: int, folds: int = 5) -> None:
    from sklearn.model_selection import GroupKFold
    if trials < 1 or timeout < 1:
        raise ValueError("Trials and timeout must be positive.")
    # Resolve both inputs before starting any real experiment.
    train, features = load_split("train")
    test, _ = load_split("student_test")
    run_baselines(list(MODELS), data_root, folds)
    comparison = compare_models()
    leader = comparison.iloc[0]
    name = str(leader.model)
    family, estimator, initial = build_model(name)
    max_neighbors = min(len(indices) for indices, _ in GroupKFold(folds).split(train, groups=train.trip_id))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    def objective(trial: optuna.Trial) -> float:
        params = search_space(name, trial, max_neighbors)
        return cv_evaluate(estimator.set_params(**params), train, features, folds)["cv_macro_f1_mean"]
    start = perf_counter()
    results = ROOT / "results"
    def record_trials(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        study.trials_dataframe().to_csv(results / "optuna_trials.csv", index=False)
    study.optimize(objective, n_trials=trials, timeout=timeout, callbacks=[record_trials])
    elapsed = perf_counter() - start
    tuned = estimator.set_params(**study.best_params)
    cv = cv_evaluate(tuned, train, features, folds)
    fitted, scores, proba = fit_and_test(tuned, train, test, features)
    scores["test_route_continuity"] = predicted_continuity(test, proba, load_student_graphs(data_root))
    row = {"model": name + "_tuned", "family": family, "params": {**initial, **study.best_params},
           "folds": folds, **{k: v for k, v in cv.items() if k.startswith("cv_")}, **scores}
    log_experiment(row)
    pd.DataFrame([leader.to_dict(), row]).to_csv(results / "tuned_vs_untuned.csv", index=False)
    config = {"model": name, "features": features, "params": study.best_params,
              "objective": "mean trip-grouped CV macro F1", "folds": folds,
              "seed": SEED, "trials": len(study.trials), "search_seconds": elapsed}
    artifacts = ROOT / "artifacts"
    (artifacts / "best_params.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    joblib.dump(fitted, artifacts / "model.joblib")
    cv["oof"].to_parquet(results / "oof" / "final.parquet", index=False)
    import matplotlib.pyplot as plt
    history = study.trials_dataframe()
    ax = history.plot(x="number", y="value", marker="o", legend=False)
    ax.set(xlabel="Trial", ylabel="Mean CV macro F1", title=f"Bayesian search · {name}")
    ax.figure.tight_layout()
    ax.figure.savefig(results / "figures" / "optuna_history.png", dpi=150)
    plt.close(ax.figure)
    if history.value.nunique() > 1:
        importance = optuna.importance.get_param_importances(study)
        if importance:
            ax = pd.Series(importance).sort_values().plot.barh()
            ax.set(title="Search parameter importance", xlabel="Estimated importance", ylabel="Parameter")
            ax.figure.tight_layout()
            ax.figure.savefig(results / "figures" / "optuna_param_importance.png", dpi=150)
            plt.close(ax.figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    finish(args.data_root, args.trials, args.timeout)
