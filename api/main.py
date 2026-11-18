"""PathSnap API (stage 5: dummy nearest-arc predictor; the real pipeline comes later).

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
This module never reads a .route file.
"""
import os
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PointPrediction, PredictRequest, PredictResponse
from common.config import DATA_RAW
from common.geo import make_projector, to_metres
from common.io import load_arcs, load_nodes, load_track
from features.candidates import build_arc_index, get_candidates

MODEL_VERSION = "dummy-nearest"

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def eval_dir():
    """Folder with the evaluation trips. PATHSNAP_EVAL_SPLIT lets you rehearse locally
    (e.g. student_test) while the real eval_test folder is not available."""
    return DATA_RAW / os.environ.get("PATHSNAP_EVAL_SPLIT", "eval_test")


def list_eval_trips() -> list[str]:
    base = eval_dir()
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - start) * 1000
        trip = getattr(request.state, "trip_id", "-")
        client = request.client.host if request.client else "-"
        print(f"{datetime.now():%H:%M:%S} | {client} | {request.method} {request.url.path} "
              f"| {trip} | {status} | {ms:.0f} ms", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return list_eval_trips()


def _dummy_predict(trip_dir):
    """Nearest arc per GPS point; route line from those arcs' node coordinates."""
    track, nodes, arcs = load_track(trip_dir), load_nodes(trip_dir), load_arcs(trip_dir)
    tr = make_projector(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx, ny = to_metres(tr, nodes.lon, nodes.lat)
    px, py = to_metres(tr, track.lon, track.lat)
    index = build_arc_index(np.column_stack([nx, ny]), arcs)
    cand = get_candidates(np.column_stack([px, py]), index, k=1)
    nearest = cand.drop_duplicates("point_idx").set_index("point_idx").arc_id.reindex(
        np.arange(len(track))).to_numpy()
    points = [PointPrediction(point_idx=i, lon=float(track.lon[i]), lat=float(track.lat[i]),
                              arc_id=int(a)) for i, a in enumerate(nearest)]
    keep = np.concatenate([[True], nearest[1:] != nearest[:-1]])  # collapse repeated arcs
    seq = nearest[keep]
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    coords = [[float(nodes.lon[fn[a]]), float(nodes.lat[fn[a]])] for a in seq]
    coords.append([float(nodes.lon[tn[seq[-1]]]), float(nodes.lat[tn[seq[-1]]])])
    return points, coords


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    request.state.trip_id = req.trip_id
    if req.trip_id not in list_eval_trips():
        raise HTTPException(status_code=404, detail=f"unknown trip_id: {req.trip_id}")
    start = time.perf_counter()
    points, coords = _dummy_predict(eval_dir() / req.trip_id)
    return PredictResponse(trip_id=req.trip_id, n_points=len(points), points=points,
                           route_coords=coords, accuracy=None, route_continuity=None,
                           latency_ms=(time.perf_counter() - start) * 1000,
                           model_version=MODEL_VERSION)
