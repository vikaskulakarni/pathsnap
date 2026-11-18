import numpy as np

from hmm.road_graph import RoadGraph

BASE_HOPS = 3


def transition_matrix(prev_arcs, cur_arcs, graph: RoadGraph, dt_s: float, params: dict) -> np.ndarray:
    """(K_prev, K_cur) log-probabilities for moving between candidate arcs.

    Cases: same arc, U-turn to the reverse twin (rare), directed successor (1 hop),
    2 hops, 3 hops, anything else = eps (never -inf, so the path cannot die).
    After a time gap above gap_s the vehicle may have travelled further, so
    `gap_extra_hops` more hops are accepted (scored like 3 hops).
    """
    max_hops = BASE_HOPS + (int(params["gap_extra_hops"]) if dt_s > params["gap_s"] else 0)
    by_hops = {0: params["same"], 1: params["succ"], 2: params["hop2"]}
    log_eps = np.log(params["eps"])
    out = np.full((len(prev_arcs), len(cur_arcs)), log_eps)
    cur_index: dict[int, list[int]] = {}
    for j, b in enumerate(cur_arcs):
        cur_index.setdefault(int(b), []).append(j)
    for i, a in enumerate(prev_arcs):
        a = int(a)
        twin = int(graph.twin[a])
        for b, h in graph.reachable(a, max_hops).items():
            cols = cur_index.get(b)
            if cols is None:
                continue
            p = params["uturn"] if (b == twin and h == 1) else by_hops.get(h, params["hop3"])
            out[i, cols] = np.log(p)
    return out
