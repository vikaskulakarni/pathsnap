"""Classifier-only vs distance-only HMM vs classifier+HMM on student_test.

Metrics are measured against data_processing.label_points labels (reference route
labels, not guaranteed ground truth). Decoder params come from artifacts/hmm_config.json.
Writes results/hmm_results.csv, results/hmm_per_trip.csv, results/hmm_failures.csv and
results/figures/hmm_comparison.png.

Run: python -m hmm.evaluate
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common.config import FIGURES, RESULTS
from common.io import list_trips, load_arcs
from hmm.classifier import load_artifacts, score_candidates
from hmm.decode import decode_trip
from hmm.params import load_params
from hmm.road_graph import RoadGraph
from models.common import load_table

SPLIT = "student_test"


def argmax_arcs(df: pd.DataFrame, col: str, n_points: int) -> np.ndarray:
    """Per point, the candidate with the highest `col` (smallest distance if col is 'dist_m' inverted)."""
    best = df.sort_values(["point_idx", col], ascending=[True, col == "dist_m"], kind="stable")
    best = best.drop_duplicates("point_idx")
    return best.set_index("point_idx").arc_id.reindex(np.arange(n_points)).fillna(-1).astype(int).to_numpy()


def score(pred, ref, df, graph) -> dict:
    connected = (pred[:-1] == pred[1:]) | (graph.to_node[pred[:-1]] == graph.from_node[pred[1:]])
    rows_pred = (df.arc_id.to_numpy() == pred[df.point_idx.to_numpy()]).astype(int)
    return {"top1": float((pred == ref).mean()),
            "wrong_direction": float(((pred == graph.twin[ref]) & (pred != ref)).mean()),
            "macro_f1": float(f1_score(df.label.to_numpy(), rows_pred, average="macro")),
            "continuity": float(connected.mean())}


def ranges(mask: np.ndarray, top: int = 5) -> str:
    """Longest runs of True as 'start-end' strings."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append((j - i + 1, i, j))
            i = j + 1
        else:
            i += 1
    return "; ".join(f"{a}-{b}" for _, a, b in sorted(runs, reverse=True)[:top])


def main():
    params = load_params()
    table = load_table(SPLIT)
    models = {"model": load_artifacts()[:2], "alt": load_artifacts(alt=True)[:2]}
    method_names = ["nearest_arc", "classifier_only(model)", "classifier_only(alt)", "hmm_distance",
                    "hmm_classifier(model)", "hmm_classifier(alt)"]
    per_trip, failures, secs = [], [], {m: 0.0 for m in method_names}
    eps = {m: [0, 0] for m in method_names}
    for trip in list_trips(SPLIT):
        df = table[table.trip_id == trip.name].reset_index(drop=True)
        ref = df[df.label == 1].drop_duplicates("point_idx").set_index("point_idx").arc_id
        n = int(df.point_idx.max()) + 1
        ref = ref.reindex(np.arange(n)).fillna(-1).astype(int).to_numpy()
        graph = RoadGraph(load_arcs(trip))
        preds = {"nearest_arc": argmax_arcs(df, "dist_m", n)}
        for tag, (model, feats) in models.items():
            d = df.assign(proba=score_candidates(model, feats, df))
            t0 = time.perf_counter()
            preds[f"classifier_only({tag})"] = argmax_arcs(d, "proba", n)
            secs[f"classifier_only({tag})"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            out, st = decode_trip(d, params, graph, "classifier", n, return_stats=True)
            secs[f"hmm_classifier({tag})"] += time.perf_counter() - t0
            preds[f"hmm_classifier({tag})"] = out.arc_id.to_numpy()
            eps[f"hmm_classifier({tag})"][0] += st["eps_steps"]
            eps[f"hmm_classifier({tag})"][1] += st["steps"]
        t0 = time.perf_counter()
        out, st = decode_trip(df, params, graph, "distance", n, return_stats=True)
        secs["hmm_distance"] += time.perf_counter() - t0
        preds["hmm_distance"] = out.arc_id.to_numpy()
        eps["hmm_distance"][0] += st["eps_steps"]
        eps["hmm_distance"][1] += st["steps"]
        for m in method_names:
            per_trip.append({"trip": trip.name, "method": m, "points": n,
                             **score(preds[m], ref, df, graph)})
        # failure analysis: classifier+HMM vs classifier-only (model)
        a, b = preds["classifier_only(model)"], preds["hmm_classifier(model)"]
        failures.append({"trip": trip.name, "points": n,
                         "top1_loss_vs_classifier_only": float((a == ref).mean() - (b == ref).mean()),
                         "disagreement_point_ranges": ranges(a != b)})
        print(f"{trip.name}: " + "  ".join(f"{m}={np.mean(preds[m] == ref):.3f}" for m in method_names), flush=True)

    pt = pd.DataFrame(per_trip)
    # the 3 trips where classifier+HMM (model) loses most Top-1 vs classifier-only (negative = HMM gained)
    fail = pd.DataFrame(failures).sort_values("top1_loss_vs_classifier_only", ascending=False).head(3)
    fail.round(4).to_csv(RESULTS / "hmm_failures.csv", index=False)
    pt.round(5).to_csv(RESULTS / "hmm_per_trip.csv", index=False)

    rows = []
    for m in method_names:
        g = pt[pt.method == m]
        w = g.points.to_numpy()
        rows.append({"split": SPLIT, "method": m, "trips": len(g), "points": int(w.sum()),
                     "top1": float(np.average(g.top1, weights=w)),
                     "macro_f1": float(np.average(g.macro_f1, weights=w)),
                     "route_continuity": float(np.average(g.continuity, weights=w)),
                     "wrong_direction_rate": float(np.average(g.wrong_direction, weights=w)),
                     "trips_ge90_share": float((g.top1 >= 0.9).mean()),
                     "eps_transition_share": (eps[m][0] / eps[m][1]) if eps[m][1] else None,
                     "decode_s_per_trip": secs[m] / len(g), "params": json.dumps(params)})
    res = pd.DataFrame(rows)
    res.round(5).to_csv(RESULTS / "hmm_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(res))
    for k, (col, lab) in enumerate((("top1", "Top-1"), ("route_continuity", "route continuity"),
                                    ("trips_ge90_share", "trips >= 90% correct"))):
        ax.bar(xs + (k - 1) * 0.27, res[col], 0.27, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(res.method, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("student_test: classifier-only vs HMM decoding")
    ax.legend()
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "hmm_comparison.png", dpi=150)
    plt.close(fig)
    print("\n" + res.drop(columns=["params", "split"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
