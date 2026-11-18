import numpy as np
import pandas as pd
import pytest

from common.io import list_trips, load_arcs
from features.build_features import COLUMNS, FEATURE_COLUMNS, attach_labels, build_features
from features.candidates import build_arc_index, get_candidates, twin_ids

# Straight east-west road, 3 arcs each way: arcs 0,1,2 go east; 3,4,5 are their reverse twins.
NODES_XY = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
ARCS = pd.DataFrame({"from_node": [0, 1, 2, 1, 2, 3], "to_node": [1, 2, 3, 0, 1, 2]})


def test_twin_ids():
    assert twin_ids(ARCS.from_node, ARCS.to_node).tolist() == [3, 4, 5, 0, 1, 2]
    one_way = twin_ids([0, 1], [1, 2])
    assert one_way.tolist() == [-1, -1]


def test_twins_are_kept_together_even_with_k1():
    index = build_arc_index(NODES_XY, ARCS)
    c = get_candidates(np.array([[50.0, 5.0]]), index, k=1, radius=60.0)
    assert set(c.arc_id) == {0, 3}
    assert not c.fallback.any()
    assert np.allclose(c.dist_m, 5.0) and np.allclose(c.proj_t, 0.5)


def test_fallback_when_nothing_in_radius():
    index = build_arc_index(NODES_XY, ARCS)
    c = get_candidates(np.array([[50.0, 500.0]]), index, k=2, radius=60.0)
    assert c.fallback.all()
    assert {0, 3} <= set(c.arc_id)


def test_no_duplicate_rows_and_every_point_covered():
    index = build_arc_index(NODES_XY, ARCS)
    xy = np.array([[10.0, 3.0], [150.0, -4.0], [290.0, 2.0], [50.0, 900.0]])
    c = get_candidates(xy, index, k=3, radius=60.0)
    assert not c.duplicated(["point_idx", "arc_id"]).any()
    assert set(c.point_idx) == {0, 1, 2, 3}


def _make_trip(tmp_path, lons, lats):
    """Synthetic trip WITHOUT a .route file: build_features must not need one."""
    d = tmp_path / "t"
    d.mkdir()
    (d / "t.nodes").write_text("\n".join(f"{8.0 + k * 0.002} 49.0" for k in range(4)))
    (d / "t.arcs").write_text("\n".join(f"{a} {b}" for a, b in zip(ARCS.from_node, ARCS.to_node)))
    (d / "t.track").write_text(
        "\n".join(f"{lo} {la} {i + 1}.0" for i, (lo, la) in enumerate(zip(lons, lats))))
    return d


def test_heading_separates_twins_and_connectivity(tmp_path):
    lons = 8.0003 + np.arange(20) * 0.000137  # ~10 m/s east
    d = _make_trip(tmp_path, lons, np.full(20, 49.00001))
    df = build_features(d)
    assert list(df.columns) == COLUMNS
    mid = df[(df.point_idx == 5) & df.arc_id.isin([0, 3])].set_index("arc_id")
    assert mid.loc[0, "heading_diff"] < 5       # eastbound arc agrees with travel
    assert mid.loc[3, "heading_diff"] > 175     # its reverse twin points the other way
    assert (df.low_speed == 0).all() and (df.speed_mps > 5).all()
    assert (df.has_twin == 1).all()
    assert df[df.point_idx == 0].connects_prev_best.eq(0).all()
    assert df[(df.point_idx == 5) & (df.arc_id == 0)].connects_prev_best.iloc[0] == 1


def test_low_speed_gives_neutral_heading(tmp_path):
    d = _make_trip(tmp_path, np.full(10, 8.0003), np.full(10, 49.00001))
    df = build_features(d)
    assert (df.low_speed == 1).all() and (df.heading_diff == 90).all()


def test_attach_labels_one_positive_and_feature_columns():
    df = pd.DataFrame({"point_idx": [0, 0, 1, 1], "arc_id": [0, 3, 0, 3]})
    out = attach_labels(df, pd.DataFrame({"point_idx": [0, 1], "arc_id": [0, 3]}))
    assert out.label.tolist() == [1, 0, 0, 1]
    for c in ("trip_id", "lon", "lat", "t", "point_idx", "arc_id", "label"):
        assert c not in FEATURE_COLUMNS


def test_real_trip_miss_rate_is_low():
    trips = list_trips("student_test")
    if not trips:
        pytest.skip("student_test data not available")
    from data_processing.label_points import label_trip
    lab = label_trip(trips[0])
    df = attach_labels(build_features(trips[0]), lab)
    hit = df.groupby("point_idx").label.max().mean()
    assert hit > 0.97
    assert twin_ids(*(load_arcs(trips[0])[c] for c in ("from_node", "to_node"))).max() >= 0
