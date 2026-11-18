import numpy as np

from data_processing.label_points import label_trip as label_points

STEP = 0.001  # ~73 m of longitude at lat 49


def _make_trip(tmp_path, points):
    """Straight east-west road of 3 arcs (+ 3 reverse arcs that are not on the route)."""
    d = tmp_path / "t"
    d.mkdir()
    (d / "t.nodes").write_text("\n".join(f"{8.0 + k * STEP} 49.0" for k in range(4)))
    (d / "t.arcs").write_text("0 1\n1 2\n2 3\n1 0\n2 1\n3 2\n")
    (d / "t.route").write_text("0\n1\n2\n")
    (d / "t.track").write_text(
        "\n".join(f"{lon} {lat} {i + 1}.0" for i, (lon, lat) in enumerate(points)))
    return d


def test_labels_monotonic_and_correct(tmp_path):
    rng = np.random.default_rng(42)
    # 10 points, a few per arc, ~2 m north/south noise
    lons = [8.0 + s * STEP for s in (0.1, 0.4, 0.8, 1.2, 1.4, 1.7, 2.1, 2.4, 2.7, 2.9)]
    pts = [(lon, 49.0 + rng.normal(0, 0.00002)) for lon in lons]
    lab = label_points(_make_trip(tmp_path, pts))
    assert len(lab) == 10
    assert (np.diff(lab.route_idx) >= 0).all()
    assert lab.arc_id.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]


def test_outlier_does_not_jump_ahead(tmp_path):
    # 8 points on arc 0; point 4 is an outlier sitting on arc 2 (but 20 m off the road)
    lons = [8.0 + s * STEP for s in (0.1, 0.2, 0.3, 0.4, 2.5, 0.6, 0.7, 0.8, 0.85)]
    pts = [(lon, 49.0 + (0.00018 if i == 4 else 0.0)) for i, lon in enumerate(lons)]
    lab = label_points(_make_trip(tmp_path, pts))
    assert (np.diff(lab.route_idx) >= 0).all()
    assert (lab.arc_id == 0).all()  # jumping to arc 2 would force all later points onto it
