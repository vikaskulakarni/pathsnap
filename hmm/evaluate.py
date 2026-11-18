"""Nearest-arc vs distance-only HMM vs distance+heading HMM on student_test.

Metrics are measured against data_processing.label_points labels (reference route
labels, not guaranteed ground truth). Params come from artifacts/hmm_config.json and
are NOT tuned here. Appends rows to results/hmm_results.csv.

Run: python -m hmm.evaluate
"""
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common.config import RESULTS
from common.io import list_trips, load_arcs
from data_processing.label_points import label_trip
from features.build_features import build_features
from hmm.decode import decode_trip
from hmm.params import load_params
from hmm.road_graph import RoadGraph

SPLIT = "student_test"


def nearest_arc(df: pd.DataFrame, n_points: int) -> np.ndarray:
    """Per point, the nearest candidate (twins tie on distance; first in table order wins)."""
    best = df.sort_values(["point_idx", "dist_m"], kind="stable").drop_duplicates("point_idx")
    return best.set_index("point_idx").arc_id.reindex(np.arange(n_points)).to_numpy()


def score(pred: np.ndarray, ref: np.ndarray, graph: RoadGraph) -> dict:
    twin_of_ref = graph.twin[ref]
    connected = (pred[:-1] == pred[1:]) | (graph.to_node[pred[:-1]] == graph.from_node[pred[1:]])
    return {"correct": int((pred == ref).sum()),
            "wrong_dir": int(((pred == twin_of_ref) & (pred != ref)).sum()),
            "connected": int(connected.sum()), "pairs": len(pred) - 1, "n": len(pred)}


def main():
    params = load_params()
    methods = ["nearest_arc", "hmm_distance", "hmm_distance+heading"]
    acc = {m: {"correct": 0, "wrong_dir": 0, "connected": 0, "pairs": 0, "n": 0,
               "trips_ge90": 0, "secs": 0.0, "eps": 0, "steps": 0} for m in methods}
    trips = list_trips(SPLIT)
    for trip in trips:
        lab = label_trip(trip)
        ref = lab.arc_id.to_numpy()
        df = build_features(trip)
        graph = RoadGraph(load_arcs(trip))
        preds = {}
        t0 = time.perf_counter()
        preds["nearest_arc"] = nearest_arc(df, len(ref))
        acc["nearest_arc"]["secs"] += time.perf_counter() - t0
        for m, emission in (("hmm_distance", "distance"), ("hmm_distance+heading", "distance+heading")):
            t0 = time.perf_counter()
            out, st = decode_trip(df, params, graph, emission, len(ref), return_stats=True)
            preds[m] = out.arc_id.to_numpy()
            acc[m]["secs"] += time.perf_counter() - t0
            acc[m]["eps"] += st["eps_steps"]
            acc[m]["steps"] += st["steps"]
        for m in methods:
            s = score(preds[m], ref, graph)
            for k in ("correct", "wrong_dir", "connected", "pairs", "n"):
                acc[m][k] += s[k]
            acc[m]["trips_ge90"] += int(s["correct"] / s["n"] >= 0.9)
        print(f"{trip.name}: " + "  ".join(f"{m}={np.mean(preds[m] == ref):.3f}" for m in methods), flush=True)
    now = datetime.now().isoformat(timespec="seconds")
    rows = [{"run_at": now, "split": SPLIT, "method": m, "trips": len(trips), "points": a["n"],
             "top1": a["correct"] / a["n"], "wrong_direction_rate": a["wrong_dir"] / a["n"],
             "route_continuity": a["connected"] / a["pairs"],
             "trips_ge90_share": a["trips_ge90"] / len(trips),
             "eps_transition_share": (a["eps"] / a["steps"]) if a["steps"] else None,
             "decode_s_per_trip": a["secs"] / len(trips), "params": json.dumps(params)}
            for m, a in acc.items()]
    out = pd.DataFrame(rows)
    path = RESULTS / "hmm_results.csv"
    out.round(5).to_csv(path, mode="a", header=not path.exists(), index=False)
    print("\n" + out.drop(columns=["run_at", "params"]).round(4).to_string(index=False))
    print(f"\nappended to {path}")


if __name__ == "__main__":
    main()
