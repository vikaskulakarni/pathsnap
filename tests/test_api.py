import pytest
from fastapi.testclient import TestClient

from api.main import app
from common.io import list_trips, load_track

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_trips_empty_when_folder_missing(monkeypatch):
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "no_such_split")
    assert client.get("/trips").json() == []


def test_unknown_trip_is_404(monkeypatch, capsys):
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "no_such_split")
    r = client.post("/predict", json={"trip_id": "nope"})
    assert r.status_code == 404
    assert "| nope | 404 |" in capsys.readouterr().out  # log line carries trip id and status


def test_predict_one_arc_per_point_and_no_labels_in_response(monkeypatch):
    trips = list_trips("student_test")
    if not trips:
        pytest.skip("student_test data not available")
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "student_test")
    trip = trips[0]
    assert trip.name in client.get("/trips").json()
    r = client.post("/predict", json={"trip_id": trip.name})
    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] == len(load_track(trip)) == len(body["points"])
    assert body["accuracy"] is None and body["model_version"] == "dummy-nearest"
    assert set(body) == {"trip_id", "n_points", "points", "route_coords", "accuracy",
                         "route_continuity", "latency_ms", "model_version"}
    assert len(body["route_coords"]) >= 2
