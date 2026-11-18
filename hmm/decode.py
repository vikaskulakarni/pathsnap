import numpy as np
import pandas as pd

from hmm.emission import distance_logp, heading_logp
from hmm.road_graph import RoadGraph
from hmm.transition import transition_matrix
from hmm.viterbi import viterbi

EMISSIONS = ("distance", "distance+heading")


def decode_trip(features_df: pd.DataFrame, params: dict, graph: RoadGraph,
                emission: str = "distance", n_points: int | None = None) -> pd.DataFrame:
    """DataFrame[point_idx, arc_id], exactly one arc per GPS point.

    features_df is one trip's candidate table (labels not needed). The graph is that
    trip's own RoadGraph. A point without candidates takes the previous decoded arc
    (the first points take the first decoded arc); Viterbi runs over the rest, and the
    time gap across skipped points counts toward the gap rule.
    """
    if emission not in EMISSIONS:
        raise ValueError(f"emission must be one of {EMISSIONS}")
    if n_points is None:
        n_points = int(features_df.point_idx.max()) + 1 if len(features_df) else 0
    ids, arcs, emis, times = [], [], [], []
    for pid, g in features_df.groupby("point_idx", sort=True):
        e = distance_logp(g.dist_m.to_numpy(), params["sigma_m"])
        if emission == "distance+heading":
            e = e + params["heading_weight"] * heading_logp(g.heading_diff.to_numpy(),
                                                            params["heading_sigma_deg"])
        ids.append(int(pid))
        arcs.append(g.arc_id.to_numpy())
        emis.append(e)
        times.append(float(g.t.iloc[0]))

    def trans(t: int) -> np.ndarray:
        return transition_matrix(arcs[t - 1], arcs[t], graph, times[t] - times[t - 1], params)

    path = viterbi(emis, trans)
    out = np.full(n_points, -1, dtype=np.int64)
    for pid, a, k in zip(ids, arcs, path):
        out[pid] = a[k]
    if len(ids):  # fill points that had no candidates from the previous decoded arc
        last = out[ids[0]]
        for i in range(n_points):
            if out[i] < 0:
                out[i] = last
            last = out[i]
    return pd.DataFrame({"point_idx": np.arange(n_points), "arc_id": out})
