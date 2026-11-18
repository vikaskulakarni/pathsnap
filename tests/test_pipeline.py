import shutil

import pytest

from common.config import ARTIFACTS
from common.io import list_trips, load_track

needs = pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists() or not list_trips("student_test"),
                           reason="model artifacts or student_test data not available")


@needs
def test_predict_trip_works_without_route_file(tmp_path):
    from hmm.pipeline import Predictor
    src = list_trips("student_test")[0]
    dst = tmp_path / src.name
    dst.mkdir()
    for f in src.iterdir():
        if f.suffix != ".route":  # the reference route must not be needed
            shutil.copy(f, dst / f.name)
    assert not list(dst.glob("*.route"))
    res = Predictor().predict_trip(dst)
    n = len(load_track(dst))
    assert len(res["points"]) == n == len(res["classifier_arcs"])
    assert all(isinstance(p["arc_id"], int) for p in res["points"])
    assert len(res["route_coords"]) >= 2
