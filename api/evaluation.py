"""Server-side scoring against the held-out reference route.

This is the ONLY place where an evaluation trip's .route is read, and only AFTER the
prediction exists. It returns two numbers; reference arcs never leave this function.
"""
import numpy as np

from common.io import load_arcs
from data_processing.label_points import label_trip


def evaluate(trip_dir, predicted_arcs) -> dict:
    """{"accuracy": share of points whose arc equals the reference arc,
        "route_continuity": share of consecutive predicted arcs that are equal or connected}"""
    pred = np.asarray(predicted_arcs, dtype=int)
    ref = label_trip(trip_dir).arc_id.to_numpy()  # reads .route, computes per-point labels
    arcs = load_arcs(trip_dir)
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    connected = (pred[:-1] == pred[1:]) | (tn[pred[:-1]] == fn[pred[1:]])
    return {"accuracy": float((pred == ref).mean()),
            "route_continuity": float(connected.mean()) if len(pred) > 1 else 1.0}
