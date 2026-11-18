"""Compare measured baselines, plot results and export both leaders for decoder testing."""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.train_baselines import MODELS
from models.train_utils import ROOT


def compare_models() -> pd.DataFrame:
    results = ROOT / "results"
    runs = pd.read_csv(results / "experiments.csv")
    table = runs[runs.model.isin(MODELS)].drop_duplicates("model", keep="last")
    missing = set(MODELS) - set(table.model)
    if missing:
        raise ValueError(f"Finish the missing baseline runs: {sorted(missing)}")
    table = table.sort_values(["test_top1", "cv_macro_f1_mean", "test_logloss"],
                              ascending=[False, False, True]).reset_index(drop=True)
    table.to_csv(results / "model_comparison.csv", index=False)
    indexed = table.set_index("model")
    plots = [
        ("cv_macro_f1", ["cv_macro_f1_mean"], "Training CV macro F1"),
        ("test_top1_f1", ["test_top1", "test_macro_f1"], "Student-test classification"),
        ("logloss", ["test_logloss"], "Student-test log loss"),
        ("infer_time", ["infer_ms_per_point"], "Student-test inference (ms per point)"),
        ("overfit_gap", ["train_macro_f1", "cv_macro_f1_mean"], "Train and CV macro F1"),
    ]
    for filename, columns, title in plots:
        ax = indexed[columns].plot.bar(rot=20, yerr=indexed["cv_macro_f1_std"] if filename == "cv_macro_f1" else None)
        ax.set(title=title, xlabel="Model", ylabel=title)
        ax.figure.tight_layout()
        ax.figure.savefig(results / "figures" / f"{filename}.png", dpi=150)
        plt.close(ax.figure)
    artifacts = ROOT / "artifacts"
    for name, destination in zip(table.model[:2], ["model.joblib", "model_alt.joblib"]):
        shutil.copyfile(artifacts / "candidates" / f"{name}.joblib", artifacts / destination)
    shutil.copyfile(artifacts / "candidates" / f"{table.model.iloc[0]}_features.json",
                    artifacts / "feature_list.json")
    return table


if __name__ == "__main__":
    print(compare_models().to_string(index=False))
