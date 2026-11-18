"""Compare fitted, out-of-fold, and held-out predictions for the full-partition experiment."""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from common.config import ARTIFACTS, RESULTS
from models.common import FEATURE_COLUMNS, load_table, point_top1, row_metrics


def main():
    folder = RESULTS / "all80"
    train, student = load_table("train"), load_table("student_test")
    keys = ["trip_id", "point_idx", "arc_id"]
    output = []
    for name in pd.read_csv(folder / "model_comparison.csv").model:
        estimator = joblib.load(ARTIFACTS / "all80" / "models" / f"{name}.joblib")
        train_probs = estimator.predict_proba(train[FEATURE_COLUMNS].to_numpy(float))[:, 1]
        held_out = pd.read_parquet(folder / "oof" / f"{name}.parquet")
        if not held_out[keys].equals(train[keys]):
            raise ValueError(f"Out-of-fold candidate identifiers differ for {name}")
        test_probs = estimator.predict_proba(student[FEATURE_COLUMNS].to_numpy(float))[:, 1]
        binary = test_probs >= 0.5
        tn, fp, fn, tp = confusion_matrix(student.label, binary, labels=[0, 1]).ravel()
        output.append({"model": name, "train_top1": point_top1(train, train_probs),
                       "pooled_oof_top1": point_top1(train, held_out.proba.to_numpy()),
                       "train_macro_f1": row_metrics(train.label, train_probs)["macro_f1"],
                       "pooled_oof_macro_f1": row_metrics(train.label, held_out.proba)["macro_f1"],
                       "test_candidate_accuracy": accuracy_score(student.label, binary),
                       "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp})
    pd.DataFrame(output).to_csv(folder / "fit_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
