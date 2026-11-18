"""PathSnap API: real pipeline (classifier + HMM/Viterbi) with server-side accuracy.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
Prediction never reads a .route file. Only api/evaluation.py reads the reference route,
after the prediction exists, and only two numbers are returned.
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.evaluation import evaluate
from api.schemas import PointPrediction, PredictRequest, PredictResponse
from common.config import DATA_RAW
from hmm.pipeline import Predictor

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = Predictor()  # model, feature list and decoder config loaded once
    yield
    STATE.clear()


app = FastAPI(title="PathSnap API", lifespan=lifespan)
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
        s = request.state
        n = getattr(s, "n_points", "-")
        acc = getattr(s, "accuracy", None)
        acc = f"{acc:.3f}" if isinstance(acc, float) else "-"
        client = request.client.host if request.client else "-"
        print(f"{datetime.now():%H:%M:%S} | {client} | {request.method} {request.url.path} "
              f"| {getattr(s, 'trip_id', '-')} | {status} | n={n} | acc={acc} | {ms:.0f} ms", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return list_eval_trips()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    request.state.trip_id = req.trip_id
    if req.trip_id not in list_eval_trips():
        raise HTTPException(status_code=404, detail=f"unknown trip_id: {req.trip_id}")
    trip_dir = eval_dir() / req.trip_id
    predictor: Predictor = STATE["predictor"]
    start = time.perf_counter()
    result = predictor.predict_trip(trip_dir)          # 1. predict (no labels involved)
    latency_ms = (time.perf_counter() - start) * 1000
    scores = evaluate(trip_dir, [p["arc_id"] for p in result["points"]])  # 2. score afterwards
    request.state.n_points = len(result["points"])
    request.state.accuracy = scores["accuracy"]
    return PredictResponse(trip_id=req.trip_id, n_points=len(result["points"]),
                           points=[PointPrediction(**p) for p in result["points"]],
                           route_coords=result["route_coords"], accuracy=scores["accuracy"],
                           route_continuity=scores["route_continuity"], latency_ms=latency_ms,
                           model_version=predictor.version)
