"""The evaluator UI (ui/app.js) rejects any response that does not match this contract,
so the API shape is locked here: /health, /trips and /predict as the UI validates them."""
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from common.config import ARTIFACTS
from common.io import list_trips

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists() or not list_trips("student_test"),
                                reason="model artifacts or student_test data not available")


def _coord(p):
    return (isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
            and abs(p[0]) <= 180 and abs(p[1]) <= 90)


def test_ui_contract(monkeypatch):
    monkeypatch.setenv("PATHSNAP_EVAL_SPLIT", "student_test")
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        trips = client.get("/trips").json()
        assert isinstance(trips, list) and trips and all(isinstance(t, str) and t for t in trips)
        r = client.post("/predict", json={"trip_id": trips[0]}).json()
    assert r["trip_id"] == trips[0]
    assert isinstance(r["points"], list) and r["n_points"] == len(r["points"])
    assert all(_coord([p["lon"], p["lat"]]) for p in r["points"])
    assert isinstance(r["route_coords"], list) and all(_coord(c) for c in r["route_coords"])
    assert isinstance(r["model_version"], str)
    assert math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0
    for v in (r["accuracy"], r["route_continuity"]):
        assert v is None or (math.isfinite(v) and 0 <= v <= 1)
