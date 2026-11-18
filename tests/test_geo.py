import numpy as np

from common.geo import (angle_diff_deg, bearing_deg, haversine_m, make_projector,
                        point_segment_distance, to_metres)


def test_latitude_degree_scale():
    assert abs(haversine_m(8.0, 49.0, 8.0, 49.001) - 111.2) < 1.0
    tr = make_projector([8.0, 8.0], [49.0, 49.001])
    x, y = to_metres(tr, [8.0, 8.0], [49.0, 49.001])
    assert abs(np.hypot(x[1] - x[0], y[1] - y[0]) - 111.2) < 1.0


def test_projector_hemispheres():
    assert make_projector([8.6], [49.8]).target_crs.to_epsg() == 32632
    assert make_projector([77.1], [-10.0]).target_crs.to_epsg() == 32743


def test_point_on_segment_is_zero():
    d, t = point_segment_distance(5, 0, 0, 0, 10, 0)
    assert d == 0 and t == 0.5


def test_perpendicular_and_clamping():
    d, t = point_segment_distance(3, 4, 0, 0, 10, 0)
    assert (d, t) == (4.0, 0.3)
    d, t = point_segment_distance(-3, 4, 0, 0, 10, 0)  # beyond A: clamps to A
    assert (d, t) == (5.0, 0.0)


def test_vectorised():
    d, t = point_segment_distance(np.array([5, 3]), np.array([0, 4]), 0, 0, 10, 0)
    assert np.allclose(d, [0, 4]) and np.allclose(t, [0.5, 0.3])


def test_zero_length_segment():
    d, t = point_segment_distance(3, 4, 0, 0, 0, 0)
    assert d == 5.0 and t == 0.0


def test_bearing():
    assert bearing_deg(0, 0, 1, 0) == 90          # due east
    assert bearing_deg(0, 0, 0, 1) == 0           # north
    assert bearing_deg(0, 0, 0, -1) == 180        # south
    assert bearing_deg(0, 0, -1, 0) == 270        # west


def test_angle_diff():
    assert angle_diff_deg(350, 10) == 20
    assert angle_diff_deg(0, 180) == 180
    assert angle_diff_deg(90, 90) == 0
