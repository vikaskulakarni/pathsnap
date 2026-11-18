"""Tune the classifier+HMM decoder on TRAIN out-of-fold probabilities, confirm on student_test.

One-factor-at-a-time search (coordinate descent) from the starting values. Scored on a
seeded sample of train trips that are not flagged by label_points, using the OUT-OF-FOLD
probabilities of the selected model (in-sample ones are over-confident). student_test is
read once at the end, with the fitted model, to confirm; eval_test is never used.

Run: python -m hmm.tune   (after optimisation.bayes_tune)
"""
import json
import time

import numpy as np
import pandas as pd

from common.config import RESULTS, SEED
from common.io import list_trips, load_arcs
from hmm.classifier import load_artifacts, score_candidates
from hmm.decode import decode_trip
from hmm.params import CONFIG_PATH, DEFAULT_PARAMS, save_params
from hmm.road_graph import RoadGraph
from models.common import flagged_train_trips, load_table, load_train

N_TRIPS = 12
GRID = {  # tried in this order
    "w": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
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


def score(trips, params: dict) -> dict:
    correct = n = conn = pairs = 0
    t0 = time.perf_counter()
    for df, ref, graph in trips:
        pred = decode_trip(df, params, graph, "classifier", len(ref)).arc_id.to_numpy()
        correct += int((pred == ref).sum())
        n += len(ref)
        conn += int(((pred[:-1] == pred[1:]) | (graph.to_node[pred[:-1]] == graph.from_node[pred[1:]])).sum())
        pairs += len(ref) - 1
    return {"top1": correct / n, "continuity": conn / pairs, "secs": time.perf_counter() - t0}


def main():
    meta = json.loads((CONFIG_PATH.parent / "best_params.json").read_text())
    oof = pd.read_parquet(RESULTS / "oof" / f"{meta['selected_file']}.parquet")
    train = load_train().merge(oof[["trip_id", "point_idx", "arc_id", "proba"]],
                               on=["trip_id", "point_idx", "arc_id"], how="left")
    assert train.proba.notna().all(), "out-of-fold probabilities do not cover the train table"
    flagged = flagged_train_trips()
    ok = sorted(set(train.trip_id) - flagged)
    sample = sorted(np.random.RandomState(SEED).choice(ok, size=N_TRIPS, replace=False).tolist())
    print(f"model {meta['selected_file']}; tuning on {N_TRIPS} train trips (OOF probs, seed {SEED}): {sample}", flush=True)
    trips = prepare("train", train, sample)

    best = dict(DEFAULT_PARAMS)
    cache: dict[str, dict] = {}

    def evaluate(p):
        key = json.dumps(p, sort_keys=True)
        if key not in cache:
            cache[key] = score(trips, p)
        return cache[key]

    start = dict(evaluate(best))
    print(f"start {start}", flush=True)
    rows = []
    for name, values in GRID.items():
        results = []
        for v in values:
            s = evaluate({**best, name: v})
            rows.append({"param": name, "value": v, **s, "stage": "tune_train_oof"})
            results.append((s["top1"], s["continuity"], v))
            print(f"  {name}={v}: top1={s['top1']:.4f} continuity={s['continuity']:.4f}", flush=True)
        best[name] = max(results)[2]
        print(f"-> {name} = {best[name]}", flush=True)
    final = evaluate(best)
    pd.DataFrame(rows).round(5).to_csv(RESULTS / "hmm_tuning.csv", index=False)
    improved = (final["top1"], final["continuity"]) > (start["top1"], start["continuity"])
    if improved:
        save_params(best)
    print(f"\ntrain-OOF sample top1 {start['top1']:.4f} -> {final['top1']:.4f}; "
          f"{'saved' if improved else 'kept defaults in'} {CONFIG_PATH}", flush=True)

    model, feats, _ = load_artifacts()
    test = load_table("student_test")
    test["proba"] = score_candidates(model, feats, test)
    st = prepare("student_test", test, [p.name for p in list_trips("student_test")])
    a, b = score(st, DEFAULT_PARAMS), score(st, best)
    print(f"student_test confirm: default top1={a['top1']:.4f} cont={a['continuity']:.4f} | "
          f"tuned top1={b['top1']:.4f} cont={b['continuity']:.4f}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
