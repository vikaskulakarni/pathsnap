"""Interactive Folium map of one development trip: GPS points, nearby road network and the
reference route as separate toggleable layers.

Run: python -m visualisations.eda_map --split train --trip 00000001
"""
import argparse

import folium
import numpy as np

from common.config import ROOT
from common.io import list_trips, load_arcs, load_nodes, load_route, load_track

OUT_DIR = ROOT / "visualisations" / "maps"


def make_eda_map(split: str, trip_id: str, max_gps: int = 1500, margin_deg: float = 0.004):
    """Return (map, path). Only train/student_test trips are allowed here (they carry a route)."""
    if split not in ("train", "student_test"):
        raise ValueError("EDA maps are for train/student_test only")
    trip = next(p for p in list_trips(split) if p.name == trip_id)
    track, nodes, arcs, route = load_track(trip), load_nodes(trip), load_arcs(trip), load_route(trip)
    step = max(1, len(track) // max_gps)
    gps = track.iloc[::step]
    center = [float(track.lat.mean()), float(track.lon.mean())]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    a, b = nodes[["lon", "lat"]].to_numpy()[fn], nodes[["lon", "lat"]].to_numpy()[tn]
    # corridor around the track: keep arcs whose start node lies within one grid cell of a GPS point
    cell = margin_deg / 2
    ci = np.floor(track[["lon", "lat"]].to_numpy() / cell).astype(int)
    near = {(x + dx, y + dy) for x, y in map(tuple, np.unique(ci, axis=0)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    acell = np.floor(a / cell).astype(int)
    inside = np.fromiter(((x, y) in near for x, y in acell), dtype=bool, count=len(acell)) & (fn < tn)
    inside &= np.cumsum(inside) <= 12000  # hard cap so the HTML stays small
    net = folium.FeatureGroup(name="Road network (one direction drawn)", show=True)
    for p, q in zip(a[inside], b[inside]):
        folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color="#7f7f7f", weight=1, opacity=0.6).add_to(net)
    net.add_to(fmap)

    if route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        pts = []
        for arc in route:
            pts.extend([[float(nodes.lat[fn[arc]]), float(nodes.lon[fn[arc]])],
                        [float(nodes.lat[tn[arc]]), float(nodes.lon[tn[arc]])]])
        folium.PolyLine(pts, color="#2ca02c", weight=5, opacity=0.85, tooltip="Reference route").add_to(ref)
        ref.add_to(fmap)

    obs = folium.FeatureGroup(name=f"GPS observations (every {step}th)", show=True)
    for lon, lat, t in gps[["lon", "lat", "t"]].itertuples(index=False):
        folium.CircleMarker([lat, lon], radius=2.5, color="#1f77b4", fill=True, fill_opacity=0.8,
                            tooltip=f"t = {t:.0f} s").add_to(obs)
    obs.add_to(fmap)
    legend = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
              'border:1px solid #888;font-size:12px"><b>{}</b><br>'
              '<span style="color:#1f77b4">&#9679;</span> GPS observations<br>'
              '<span style="color:#2ca02c">&#9644;</span> Reference route<br>'
              '<span style="color:#7f7f7f">&#9644;</span> Road network</div>').format(f"{split} trip {trip_id}")
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(track.lat.min()), float(track.lon.min())], [float(track.lat.max()), float(track.lon.max())]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"eda_{split}_{trip_id}.html"
    fmap.save(str(path))
    return fmap, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "student_test"])
    ap.add_argument("--trip", default=None)
    args = ap.parse_args()
    trip = args.trip or list_trips(args.split)[0].name
    _, path = make_eda_map(args.split, trip)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
