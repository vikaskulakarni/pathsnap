"""Tune the distance/heading HMM decoder on TRAIN, then confirm on student_test.

One-factor-at-a-time search (coordinate descent) around the starting values: each
parameter is tried over a small list, the best value (by Top-1, ties by continuity) is
kept, then the next parameter is tried. Scored on a seeded sample of train trips that are
NOT flagged by data_processing.label_points (flagged trips have unreliable labels).
student_test is read only once at the end, to confirm; eval_test is never used.
The classifier weight `w` is added in the classifier stage.

Run: python -m hmm.tune
"""
import json
import time

import numpy as np
import pandas as pd

from common.config import DATA_PROCESSED, RESULTS, SEED
from common.io import list_trips, load_arcs
from hmm.decode import decode_trip
from hmm.params import CONFIG_PATH, DEFAULT_PARAMS, save_params
from hmm.road_graph import RoadGraph

N_TRIPS = 12
GRID = {  # tried in this order
    "heading_weight": [0.0, 0.25, 0.5, 1.0],
    "sigma_m": [4.0, 6.0, 8.0, 12.0, 20.0],
    "same": [0.4, 0.6, 0.8],
    "succ": [0.15, 0.3, 0.5],
    "uturn": [1e-4, 1e-3, 1e-2],
    "eps": [1e-8, 1e-6, 1e-4],
    "gap_s": [5.0, 10.0, 20.0],
}


def load_split(split: str, trip_ids: list[str]):
    table = pd.read_parquet(DATA_PROCESSED / f"{split}_features.parquet")
    paths = {p.name: p for p in list_trips(split)}
    out = []
    for tid in trip_ids:
        df = table[table.trip_id == tid]
        n = int(df.point_idx.max()) + 1
        pos = df[df.label == 1].drop_duplicates("point_idx").set_index("point_idx").arc_id
        ref = pos.reindex(np.arange(n)).fillna(-1).astype(int).to_numpy()  # -1: ref not a candidate
        out.append((df, ref, RoadGraph(load_arcs(paths[tid]))))
    return out


def score(trips, params: dict) -> dict:
    correct = n = conn = pairs = 0
    t0 = time.perf_counter()
    for df, ref, graph in trips:
        pred = decode_trip(df, params, graph, "distance+heading", len(ref)).arc_id.to_numpy()
        correct += int((pred == ref).sum())
        n += len(ref)
        conn += int(((pred[:-1] == pred[1:]) | (graph.to_node[pred[:-1]] == graph.from_node[pred[1:]])).sum())
        pairs += len(ref) - 1
    return {"top1": correct / n, "continuity": conn / pairs, "secs": time.perf_counter() - t0}


def main():
    quality = pd.read_csv(RESULTS / "eda" / "label_quality.csv", dtype={"trip": str})
    ok = sorted(quality[(quality.split == "train") & ~quality.flag_p95_gt_30].trip)
    sample = sorted(np.random.RandomState(SEED).choice(ok, size=N_TRIPS, replace=False).tolist())
    print(f"tuning on {N_TRIPS} train trips (non-flagged, seed {SEED}): {sample}", flush=True)
    trips = load_split("train", sample)

    best = dict(DEFAULT_PARAMS)
    cache: dict[str, dict] = {}

    def evaluate(p):
        key = json.dumps(p, sort_keys=True)
        if key not in cache:
            cache[key] = score(trips, p)
        return cache[key]

    rows = []
    best_score = evaluate(best)
    start = dict(best_score)
    print(f"start {best_score}", flush=True)
    for name, values in GRID.items():
        results = []
        for v in values:
            cand = {**best, name: v}
            s = evaluate(cand)
            rows.append({"param": name, "value": v, **s, "stage": "tune_train"})
            results.append((s["top1"], s["continuity"], v))
            print(f"  {name}={v}: top1={s['top1']:.4f} continuity={s['continuity']:.4f}", flush=True)
        top1, cont, v = max(results)
        best[name] = v
        best_score = evaluate(best)
        print(f"-> {name} = {v}", flush=True)

    pd.DataFrame(rows).round(5).to_csv(RESULTS / "hmm_tuning.csv", index=False)
    improved = (best_score["top1"], best_score["continuity"]) > (start["top1"], start["continuity"])
    if improved:
        save_params(best)
    print(f"\ntrain-sample top1 {start['top1']:.4f} -> {best_score['top1']:.4f}; "
          f"{'saved' if improved else 'kept defaults in'} {CONFIG_PATH}", flush=True)

    st = load_split("student_test", [p.name for p in list_trips("student_test")])
    a, b = score(st, DEFAULT_PARAMS), score(st, best)
    print(f"student_test confirm: default top1={a['top1']:.4f} cont={a['continuity']:.4f} | "
          f"tuned top1={b['top1']:.4f} cont={b['continuity']:.4f}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
