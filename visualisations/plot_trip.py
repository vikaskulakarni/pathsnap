"""Build local Folium maps from permitted development trips."""

import argparse
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd

from common.io import load_arcs, load_nodes, load_route, load_track
from visualisations.eda_plots import ROOT, SPLITS, development_trips, project_trip


def make_map(
    trip_dir: Path, predictions: pd.DataFrame | None = None,
    show_reference: bool = True, out_path: Path | None = None,
) -> folium.Map:
    """Draw each road arc separately so disconnected predictions stay disconnected."""
    trip_dir = Path(trip_dir).resolve()
    if trip_dir.parent.name not in SPLITS:
        raise ValueError("This map tool accepts train and student_test trips only.")

    track = load_track(trip_dir)
    nodes = load_nodes(trip_dir)
    arcs = load_arcs(trip_dir).set_index("arc_id")
    valid = track["lon"].between(-180, 180) & track["lat"].between(-90, 90)
    track = track.loc[valid]
    if track.empty:
        raise ValueError(f"No valid GPS positions in {trip_dir.name}.")

    gps_xy, nodes_xy = project_trip(track["lon"].to_numpy(), track["lat"].to_numpy(), nodes)
    bounds_lo = gps_xy.min(axis=0) - 200
    bounds_hi = gps_xy.max(axis=0) + 200
    segments = nodes_xy[arcs[["from_node", "to_node"]].to_numpy(dtype=int)]
    nearby = ((segments.max(axis=1) >= bounds_lo) & (segments.min(axis=1) <= bounds_hi)).all(axis=1)
    map_view = folium.Map(location=[track["lat"].median(), track["lon"].median()], prefer_canvas=True)
    node_positions = nodes.set_index("node_id")[["lat", "lon"]]
    layers: list[tuple[str, str]] = []

    def road_layer(name: str, arc_ids: list[int], color: str, **style: Any) -> None:
        if not arc_ids:
            return
        rows = arcs.loc[list(dict.fromkeys(arc_ids)), ["from_node", "to_node"]]
        coordinates = node_positions.loc[rows.to_numpy().ravel()].to_numpy().reshape(-1, 2, 2)
        layer = folium.FeatureGroup(name=name).add_to(map_view)
        folium.PolyLine(coordinates.tolist(), color=color, **style).add_to(layer)
        layers.append((name, color))

    road_layer("Road network", arcs.index[nearby].tolist(), "#737b87", weight=1, opacity=0.45)
    if show_reference:
        road_layer("Reference route", load_route(trip_dir) or [], "#198245", weight=4)
    if predictions is not None:
        if predictions["point_idx"].duplicated().any():
            raise ValueError("Expected one prediction row per GPS observation.")
        road_layer("Classifier route", predictions["classifier_arc"].dropna().astype(int).tolist(),
                   "#c36511", weight=4, dash_array="7 5")
        road_layer("HMM route", predictions["hmm_arc"].dropna().astype(int).tolist(),
                   "#c92f38", weight=4)
    gps = folium.FeatureGroup(name="GPS observations").add_to(map_view)
    for point in track.itertuples():
        folium.CircleMarker(
            (point.lat, point.lon), radius=3, color="#1c60bd", fill=True, fill_opacity=0.8,
            popup=f"Observation {int(point.Index)} · {float(point.t):g} s",
        ).add_to(gps)
    layers.append(("GPS observations", "#1c60bd"))
    folium.LayerControl().add_to(map_view)
    map_view.fit_bounds(
        [[track["lat"].min(), track["lon"].min()], [track["lat"].max(), track["lon"].max()]],
        padding=(20, 20), max_zoom=17,
    )
    legend_rows = "".join(f'<div><span style="color:{color}">&#9679;</span> {name}</div>'
                          for name, color in layers)
    map_view.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;background:white;'
        'z-index:1000;padding:8px 12px;border:1px solid #aeb6c2;border-radius:4px">'
        + legend_rows + "</div>"
    ))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        map_view.save(str(out_path))
    return map_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()
    trips = {trip.name: trip for trip in development_trips(args.split, args.data_root)}
    if args.trip not in trips:
        parser.error("Trip ID was not found in that development split.")
    out = args.out or ROOT / "visualisations" / "maps" / f"{args.split}_{args.trip}.html"
    make_map(trips[args.trip], show_reference=not args.no_reference, out_path=out)
    print(out)


if __name__ == "__main__":
    main()
