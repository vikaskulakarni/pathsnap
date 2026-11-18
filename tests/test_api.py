import pytest
from fastapi.testclient import TestClient

from api.main import app
from common.config import ARTIFACTS
from common.io import list_trips, load_track

needs_model = pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists(),
                                 reason="model artifacts not available")


@needs_model
def test_health_trips_and_404(monkeypatch, capsys):
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "no_such_split")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/trips").json() == []
        r = client.post("/predict", json={"trip_id": "nope"})
    assert r.status_code == 404
    assert "| nope | 404 |" in capsys.readouterr().out  # log line carries trip id and status


@needs_model
def test_predict_real_pipeline_and_no_labels_in_response(monkeypatch, capsys):
    trips = list_trips("student_test")
    if not trips:
        pytest.skip("student_test data not available")
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "student_test")
    trip = trips[0]
    with TestClient(app) as client:
        assert trip.name in client.get("/trips").json()
        r = client.post("/predict", json={"trip_id": trip.name})
    assert r.status_code == 200
    body = r.json()
    n = len(load_track(trip))
    assert body["n_points"] == n == len(body["points"])
    assert 0.0 <= body["accuracy"] <= 1.0 and 0.0 <= body["route_continuity"] <= 1.0
    # only predictions and two numbers: no reference arcs or route anywhere in the response
    assert set(body) == {"trip_id", "n_points", "points", "route_coords", "accuracy",
                         "route_continuity", "latency_ms", "model_version"}
    assert set(body["points"][0]) == {"point_idx", "lon", "lat", "arc_id"}
    log = capsys.readouterr().out
    assert f"| {trip.name} | 200 | n={n} | acc=" in log
