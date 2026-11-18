"""Geometry helpers. All distances are in metres, never raw degrees."""
import numpy as np
from pyproj import Transformer


def make_projector(lon, lat) -> Transformer:
    """Transformer WGS84 -> UTM zone of the mean lon/lat (EPSG 326xx north, 327xx south).

    One zone per trip keeps distortion tiny, and metres make radius/K logic meaningful.
    """
    mlon = float(np.mean(lon))
    mlat = float(np.mean(lat))
    zone = int((mlon + 180.0) // 6.0) % 60 + 1
    epsg = (32600 if mlat >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def to_metres(transformer: Transformer, lon, lat):
    """Project lon/lat arrays to (x, y) metre arrays."""
    x, y = transformer.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x, float), np.asarray(y, float)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres; vectorised, projection-free sanity check."""
    r = 6371008.8
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, float)) for v in (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from P to segment AB and t in [0, 1], the projection's position along AB.

    A zero-length segment (A == B) is treated as the point A with t = 0, so it never
    divides by zero.
    """
    px, py, ax, ay, bx, by = (np.asarray(v, float) for v in (px, py, ax, ay, bx, by))
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    safe = np.where(len2 > 0, len2, 1.0)
    t = np.where(len2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    dist = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return dist, t


def bearing_deg(ax, ay, bx, by):
    """Bearing A->B in projected coords: 0 = north, clockwise, in [0, 360)."""
    ang = np.degrees(np.arctan2(np.asarray(bx, float) - ax, np.asarray(by, float) - ay))
    return np.mod(ang, 360.0)


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.mod(np.asarray(a, float) - np.asarray(b, float), 360.0))
    return np.minimum(d, 360.0 - d)
