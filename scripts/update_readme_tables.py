"""Fill the RESULTS block of README.md from results/*.csv (so the README never holds typed-in numbers).

Run: python scripts/update_readme_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build() -> str:
    hmm = pd.read_csv(RES / "hmm_results.csv")
    h = pd.DataFrame({"Method": hmm.method, "Top-1": hmm.top1.map("{:.3f}".format),
                      "Macro F1": hmm.macro_f1.map("{:.3f}".format),
                      "Route continuity": hmm.route_continuity.map("{:.3f}".format),
                      "Wrong direction": hmm.wrong_direction_rate.map("{:.3f}".format),
                      "Trips >= 90% correct": (hmm.trips_ge90_share * 100).map("{:.0f}%".format)})
    mc = pd.read_csv(RES / "model_comparison.csv")
    m = pd.DataFrame({"Model": mc.model, "Family": mc.algorithm_family,
                      "CV Top-1": mc.cv_top1.map("{:.3f}".format), "CV macro F1": mc.cv_macro_f1.map("{:.3f}".format),
                      "Student Top-1": mc.test_top1.map("{:.3f}".format),
                      "Student macro F1": mc.test_macro_f1.map("{:.3f}".format),
                      "Student log loss": mc.test_log_loss.map("{:.3f}".format),
                      "Route continuity": mc.test_route_continuity.map("{:.3f}".format)})
    tc = pd.read_csv(RES / "tuning_comparison.csv")
    t = pd.DataFrame({"Model": tc.model, "Configuration": tc.configuration,
                      "CV Top-1": tc.cv_top1.map("{:.4f}".format), "CV macro F1": tc.cv_macro_f1.map("{:.4f}".format),
                      "Student Top-1": tc.test_top1.map("{:.4f}".format),
                      "Student macro F1": tc.test_macro_f1.map("{:.4f}".format),
                      "Log loss": tc.test_log_loss.map("{:.4f}".format)})
    cm = pd.read_csv(RES / "eda" / "candidate_miss_by_trip.csv", dtype={"trip_id": str})

    def pooled(d):
        return (d.miss_rate * d.points).sum() / d.points.sum()

    tr = cm[cm.split == "train"]
    c = pd.DataFrame({"Partition": ["train, all 80 trips", "train, 73 trips with reliable labels",
                                    "train, 7 flagged trips", "student_test (10 trips)"],
                      "Candidate miss rate": [f"{pooled(tr):.4f}", f"{pooled(tr[~tr.label_flagged]):.4f}",
                                              f"{pooled(tr[tr.label_flagged]):.4f}",
                                              f"{pooled(cm[cm.split == 'student_test']):.4f}"]})
    return "\n\n".join([
        "**Route decoding** (`results/hmm_results.csv`; reference labels from `data_processing/label_points.py`):\n\n" + md_table(h),
        "**Model comparison** (`results/model_comparison.csv`; CV = 5-fold GroupKFold by trip on the 80 train trips):\n\n" + md_table(m),
        "**Bayesian tuning** (`results/tuning_comparison.csv`; same folds and partitions for both rows):\n\n" + md_table(t),
        "**Candidate miss rate** (reference arc absent from the candidates; `results/eda/candidate_miss_by_trip.csv`):\n\n" + md_table(c),
    ])


def main():
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{START}\n{build()}\n{END}{tail}", encoding="utf-8")
    print("README results block updated")


if __name__ == "__main__":
    main()
