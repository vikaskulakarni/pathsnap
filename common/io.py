from pathlib import Path

import pandas as pd

from common.config import DATA_RAW


def list_trips(split):
    """Sorted trip directories of a split (empty list if the split folder is absent)."""
    base = DATA_RAW / split
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def trip_files(trip_dir):
    """Map extension (without dot, e.g. 'track') -> file path, found by extension."""
    return {p.suffix.lstrip("."): p for p in Path(trip_dir).iterdir() if p.is_file()}


def _read(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_track(trip_dir):
    return _read(trip_files(trip_dir)["track"], ["lon", "lat", "t"])


def load_nodes(trip_dir):
    df = _read(trip_files(trip_dir)["nodes"], ["lon", "lat"])
    df.insert(0, "node_id", range(len(df)))
    return df


def load_arcs(trip_dir):
    df = _read(trip_files(trip_dir)["arcs"], ["from_node", "to_node"])
    df.insert(0, "arc_id", range(len(df)))
    return df


def load_route(trip_dir):
    p = trip_files(trip_dir).get("route")
    if p is None:
        return None
    return pd.read_csv(p, sep=r"\s+", header=None)[0].astype(int).tolist()
