import numpy as np
import pandas as pd

from hmm.decode import decode_trip
from hmm.params import DEFAULT_PARAMS
from hmm.road_graph import RoadGraph
from hmm.transition import transition_matrix

# Straight road 0-1-2-3-4: arcs 0..3 go east (0->1 ... 3->4); arcs 4..7 are their reverse twins.
ARCS = pd.DataFrame({"from_node": [0, 1, 2, 3, 1, 2, 3, 4], "to_node": [1, 2, 3, 4, 0, 1, 2, 3]})
P = dict(DEFAULT_PARAMS)


def test_graph_twins_successors_and_hops():
    g = RoadGraph(ARCS)
    assert g.twin.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert set(g.successors(0)) == {1, 4}            # next arc east, and the U-turn twin
    assert g.hops(0, 0) == 0 and g.hops(0, 1) == 1 and g.hops(0, 3) == 3
    assert g.hops(0, 3, max_hops=2) is None          # too far
    assert g.hops(1, 0) is None or g.hops(1, 0) >= 2  # moving west needs the twin chain


def test_transition_cases_and_gap_rule():
    g = RoadGraph(ARCS)
    m = transition_matrix([0], [0, 1, 2, 3, 4, 7], g, dt_s=1.0, params=P)[0]
    want = np.log([P["same"], P["succ"], P["hop2"], P["hop3"], P["uturn"], P["eps"]])
    assert np.allclose(m, want)
    # a long gap allows 3 extra hops: arc 0 -> arc 3 stays 3 hops, and eastward far arcs get hop3 not eps
    far = ARCS.copy()
    far = pd.concat([far, pd.DataFrame({"from_node": [4, 5], "to_node": [5, 6]})], ignore_index=True)
    g2 = RoadGraph(far)  # arcs 8: 4->5, 9: 5->6 ; arc 9 is 5 hops from arc 0 (0,1,2,3,8,9)
    near = transition_matrix([0], [9], g2, 1.0, P)[0, 0]
    gap = transition_matrix([0], [9], g2, 30.0, P)[0, 0]
    assert near == np.log(P["eps"]) and gap == np.log(P["hop3"])


def _features(rows):
    """rows: (point_idx, t, arc_id, dist_m, heading_diff)"""
    return pd.DataFrame(rows, columns=["point_idx", "t", "arc_id", "dist_m", "heading_diff"])


def test_decode_prefers_connected_route_over_closest_jump():
    g = RoadGraph(ARCS)
    # Point 1 is slightly closer to arc 3 (a far-away arc) than to arc 1, but arc 1 follows arc 0.
    df = _features([(0, 0, 0, 3.0, 0), (1, 1, 1, 9.0, 0), (1, 1, 3, 8.0, 0), (2, 2, 1, 3.0, 0)])
    out = decode_trip(df, P, g)
    assert out.arc_id.tolist() == [0, 1, 1]
    nearest = df.sort_values("dist_m").drop_duplicates("point_idx").sort_values("point_idx")
    assert nearest.arc_id.tolist() == [0, 3, 1]  # the nearest-arc baseline jumps


def test_heading_variant_separates_twins():
    g = RoadGraph(ARCS)
    df = _features([(i, i, a, 2.0, h) for i in range(3) for a, h in ((1, 0.0), (5, 180.0))])
    assert decode_trip(df, P, g, "distance+heading").arc_id.tolist() == [1, 1, 1]


def test_points_without_candidates_take_previous_arc_and_one_arc_per_point():
    g = RoadGraph(ARCS)
    df = _features([(1, 1, 2, 3.0, 0), (3, 3, 2, 3.0, 0)])  # points 0 and 2 and 4 have no candidates
    out = decode_trip(df, P, g, n_points=5)
    assert len(out) == 5 and out.arc_id.tolist() == [2, 2, 2, 2, 2]
