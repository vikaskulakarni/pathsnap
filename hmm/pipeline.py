"""Single inference entry point: track + road network -> one arc per GPS point.

load track -> build_features (candidates) -> classifier probabilities -> Viterbi.
It never opens a .route file and never sees labels.

Run: python -m hmm.pipeline --split student_test --save
"""
import argparse
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from common.config import RESULTS
from common.io import list_trips, load_arcs, load_nodes, load_track
from features.build_features import build_features
from hmm.classifier import load_artifacts, score_candidates
from hmm.decode import decode_trip
from hmm.params import load_params
from hmm.road_graph import RoadGraph


def _argmax_arcs(df: pd.DataFrame, n_points: int) -> np.ndarray:
    """Classifier-only choice: highest-probability candidate per point (forward-filled)."""
    best = df.sort_values(["point_idx", "proba"], ascending=[True, False], kind="stable")
    best = best.drop_duplicates("point_idx").set_index("point_idx").arc_id
    out = best.reindex(np.arange(n_points)).ffill().bfill().to_numpy()
    return out.astype(int)


class Predictor:
    """Loads the model, feature list and decoder config once; caches each trip's RoadGraph."""

    def __init__(self, cache_size: int = 16):
        self.model, self.features, self.version = load_artifacts()
        self.params = load_params()
        self._graphs: OrderedDict[str, RoadGraph] = OrderedDict()
        self._cache_size = cache_size

    def _graph(self, trip_dir, arcs) -> RoadGraph:
        key = str(trip_dir)
        if key not in self._graphs:
            self._graphs[key] = RoadGraph(arcs)
            if len(self._graphs) > self._cache_size:
                self._graphs.popitem(last=False)
        return self._graphs[key]

    def predict_trip(self, trip_dir) -> dict:
        """{"points": [{point_idx, lon, lat, arc_id}], "route_coords": [[lon, lat], ...],
        "classifier_arcs": [...]} with exactly one arc per GPS point."""
        track, nodes, arcs = load_track(trip_dir), load_nodes(trip_dir), load_arcs(trip_dir)
        n = len(track)
        df = build_features(trip_dir)
        df["proba"] = score_candidates(self.model, self.features, df)
        graph = self._graph(trip_dir, arcs)
        hmm_arcs = decode_trip(df, self.params, graph, "classifier", n).arc_id.to_numpy()
        clf_arcs = _argmax_arcs(df, n)
        points = [{"point_idx": i, "lon": float(track.lon[i]), "lat": float(track.lat[i]),
                   "arc_id": int(a)} for i, a in enumerate(hmm_arcs)]
        keep = np.concatenate([[True], hmm_arcs[1:] != hmm_arcs[:-1]])  # collapse repeated arcs
        seq = hmm_arcs[keep]
        fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
        coords = [[float(nodes.lon[fn[a]]), float(nodes.lat[fn[a]])] for a in seq]
        coords.append([float(nodes.lon[tn[seq[-1]]]), float(nodes.lat[tn[seq[-1]]])])
        return {"points": points, "route_coords": coords, "classifier_arcs": [int(a) for a in clf_arcs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test", choices=["train", "student_test"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    predictor = Predictor()
    out_dir = RESULTS / "predictions"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
    for trip in list_trips(args.split):
        t0 = time.perf_counter()
        res = predictor.predict_trip(trip)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{trip.name}: {len(res['points'])} points, {ms:.0f} ms", flush=True)
        if args.save:
            pd.DataFrame({"point_idx": [p["point_idx"] for p in res["points"]],
                          "lon": [p["lon"] for p in res["points"]],
                          "lat": [p["lat"] for p in res["points"]],
                          "classifier_arc": res["classifier_arcs"],
                          "hmm_arc": [p["arc_id"] for p in res["points"]]}
                         ).to_csv(out_dir / f"{trip.name}.csv", index=False)


if __name__ == "__main__":
    main()
