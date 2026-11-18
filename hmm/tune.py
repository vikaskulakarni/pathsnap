"""Tune the classifier+HMM decoder on TRAIN out-of-fold probabilities, confirm on student_test.

One-factor-at-a-time search (coordinate descent) from the starting values. Scored on a
seeded sample of N_TRIPS trips drawn from ALL 80 train trips (the protocol; the 7 unreliable-label
trips are only studied in models/ablation_trips.py), using the OUT-OF-FOLD probabilities of the
selected model (in-sample ones are over-confident). student_test is read once at the end, with
the fitted model, to confirm; eval_test is never used.

Run: python -m hmm.tune             (after optimisation.bayes_tune)
     python -m hmm.tune --baseline  (distance-only HMM, w fixed to 0: the fair baseline)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from common.config import RESULTS, SEED
from common.io import list_trips, load_arcs
from hmm.classifier import load_artifacts, score_candidates
from hmm.decode import decode_trip
from hmm.params import BASELINE_CONFIG_PATH, CONFIG_PATH, DEFAULT_PARAMS, save_params
from hmm.road_graph import RoadGraph
from models.common import load_table, load_train

N_TRIPS = 12
GRID = {  # tried in this order
    "w": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
    "heading_weight": [0.0, 0.25, 0.5, 1.0],  # baseline only: heading term of the distance-only HMM
    "sigma_m": [4.0, 8.0, 12.0, 20.0],
    "same": [0.4, 0.6, 0.8],
    "succ": [0.15, 0.3, 0.5],
    "uturn": [1e-4, 1e-3, 1e-2],
    "eps": [1e-8, 1e-6, 1e-4],
    "gap_s": [5.0, 10.0, 20.0],
}


def prepare(split: str, table: pd.DataFrame, trip_ids: list[str]):
    paths = {p.name: p for p in list_trips(split)}
    out = []
    for tid in trip_ids:
        df = table[table.trip_id == tid].reset_index(drop=True)
        n = int(df.point_idx.max()) + 1
        pos = df[df.label == 1].drop_duplicates("point_idx").set_index("point_idx").arc_id
        ref = pos.reindex(np.arange(n)).fillna(-1).astype(int).to_numpy()
        out.append((df, ref, RoadGraph(load_arcs(paths[tid]))))
    return out


def score(trips, params: dict, emission: str = "classifier") -> dict:
    correct = n = conn = pairs = 0
    t0 = time.perf_counter()
    for df, ref, graph in trips:
        pred = decode_trip(df, params, graph, emission, len(ref)).arc_id.to_numpy()
        correct += int((pred == ref).sum())
        n += len(ref)
        conn += int(((pred[:-1] == pred[1:]) | (graph.to_node[pred[:-1]] == graph.from_node[pred[1:]])).sum())
        pairs += len(ref) - 1
    return {"top1": correct / n, "continuity": conn / pairs, "secs": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="tune the distance-only HMM (w fixed to 0)")
    baseline = ap.parse_args().baseline
    grid = {k: v for k, v in GRID.items() if k != ("heading_weight" if not baseline else "w")}
    emission = "distance+heading" if baseline else "classifier"
    out_config = BASELINE_CONFIG_PATH if baseline else CONFIG_PATH
    meta = json.loads((CONFIG_PATH.parent / "best_params.json").read_text())
    oof = pd.read_parquet(RESULTS / "oof" / f"{meta['selected_file']}.parquet")
    train = load_train().merge(oof[["trip_id", "point_idx", "arc_id", "proba"]],
                               on=["trip_id", "point_idx", "arc_id"], how="left")
    assert train.proba.notna().all(), "out-of-fold probabilities do not cover the train table"
    all_trips = sorted(set(train.trip_id))  # protocol: the sample is drawn from ALL 80 train trips
    sample = sorted(np.random.RandomState(SEED).choice(all_trips, size=N_TRIPS, replace=False).tolist())
    print(f"model {meta['selected_file']}; tuning on {N_TRIPS} train trips (OOF probs, seed {SEED}): {sample}", flush=True)
    trips = prepare("train", train, sample)

    best = {**DEFAULT_PARAMS, **({"w": 0.0, "heading_weight": 1.0} if baseline else {})}
    cache: dict[str, dict] = {}

    def evaluate(p):
        key = json.dumps(p, sort_keys=True)
        if key not in cache:
            cache[key] = score(trips, p, emission)
        return cache[key]

    start = dict(evaluate(best))
    print(f"start {start}", flush=True)
    rows = []
    for name, values in grid.items():
        results = []
        for v in values:
            s = evaluate({**best, name: v})
            rows.append({"param": name, "value": v, **s, "stage": "tune_train_oof"})
            results.append((s["top1"], s["continuity"], v))
            print(f"  {name}={v}: top1={s['top1']:.4f} continuity={s['continuity']:.4f}", flush=True)
        best[name] = max(results)[2]
        print(f"-> {name} = {best[name]}", flush=True)
    final = evaluate(best)
    pd.DataFrame(rows).round(5).to_csv(RESULTS / ("hmm_tuning_distance.csv" if baseline else "hmm_tuning.csv"),
                                       index=False)
    improved = (final["top1"], final["continuity"]) > (start["top1"], start["continuity"])
    default_start = {**DEFAULT_PARAMS, **({"w": 0.0, "heading_weight": 0.0} if baseline else {})}
    save_params(best if improved else default_start, out_config)  # baseline never touches hmm_config.json
    print(f"\ntrain-OOF sample top1 {start['top1']:.4f} -> {final['top1']:.4f}; "
          f"{'saved tuned' if improved else 'saved default'} params to {out_config}", flush=True)

    model, feats, _ = load_artifacts()
    test = load_table("student_test")
    test["proba"] = score_candidates(model, feats, test)
    st = prepare("student_test", test, [p.name for p in list_trips("student_test")])
    a = score(st, {**DEFAULT_PARAMS, **({'w': 0.0, 'heading_weight': 0.0} if baseline else {})}, emission)
    b = score(st, best, emission)
    print(f"student_test confirm: default top1={a['top1']:.4f} cont={a['continuity']:.4f} | "
          f"tuned top1={b['top1']:.4f} cont={b['continuity']:.4f}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
