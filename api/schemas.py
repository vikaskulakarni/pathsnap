from pydantic import BaseModel


class PredictRequest(BaseModel):
    trip_id: str


class PointPrediction(BaseModel):
    point_idx: int
    lon: float
    lat: float
    arc_id: int


class PredictResponse(BaseModel):
    trip_id: str
    n_points: int
    points: list[PointPrediction]
    route_coords: list[list[float]]
    accuracy: float | None = None
    route_continuity: float | None = None
    latency_ms: float
    model_version: str
