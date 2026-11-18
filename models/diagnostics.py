"""Diagnostics of the saved models on student_test: confusion matrices for every model,
a reliability (calibration) diagram, and permutation importance of the selected model.

Run: python -m models.diagnostics   (after models.train_models and optimisation.bayes_tune)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

from common.config import ARTIFACTS, FIGURES, RESULTS, SEED
from models.common import FEATURE_COLUMNS, load_table, point_top1

MODEL_FILES = ["logistic_regression", "gaussian_naive_bayes", "decision_tree", "random_forest",
               "hist_gradient_boosting", "hist_gradient_boosting_tuned"]


def main():
    test = load_table("student_test")
    X, y = test[FEATURE_COLUMNS].to_numpy(float), test.label.to_numpy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    proba = {}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, name in zip(axes.ravel(), MODEL_FILES):
        model = joblib.load(ARTIFACTS / "models" / f"{name}.joblib")
        proba[name] = model.predict_proba(X)[:, 1]
        ConfusionMatrixDisplay.from_predictions(y, (proba[name] >= 0.5).astype(int), ax=ax, colorbar=False)
        ax.set_title(name, fontsize=9)
    fig.suptitle("student_test candidate rows: confusion matrices (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_confusion_all.png", dpi=150)
    plt.close(fig)

    selected = json.loads((ARTIFACTS / "best_params.json").read_text())["selected_file"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in ("logistic_regression", "random_forest", selected):
        frac, mean = calibration_curve(y, proba[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title("student_test reliability diagram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_calibration.png", dpi=150)
    plt.close(fig)

    model = joblib.load(ARTIFACTS / "models" / f"{selected}.joblib")
    base = point_top1(test, proba[selected])
    rng = np.random.RandomState(SEED)
    rows = []
    for j, col in enumerate(FEATURE_COLUMNS):  # permute one column, measure the drop in observation-level Top-1
        drops = []
        for _ in range(3):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - point_top1(test, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": col, "top1_drop_mean": float(np.mean(drops)), "top1_drop_std": float(np.std(drops))})
    imp = pd.DataFrame(rows).sort_values("top1_drop_mean", ascending=False)
    imp.round(5).to_csv(RESULTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.feature[::-1], imp.top1_drop_mean[::-1], xerr=imp.top1_drop_std[::-1], color="#3b6ea5")
    ax.set_xlabel("drop in student_test Top-1 when the feature is shuffled")
    ax.set_title(f"Permutation importance: {selected}")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_feature_importance.png", dpi=150)
    plt.close(fig)
    print(imp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
