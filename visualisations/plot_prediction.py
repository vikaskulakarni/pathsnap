"""Create student-test comparison maps from Laptop 1's saved prediction files."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from common.io import load_arcs, load_nodes, load_route, load_track
from visualisations.eda_plots import ROOT, development_trips, project_trip
from visualisations.plot_trip import make_map


def render_predictions(data_root: Path | None = None) -> None:
    trips = {p.name: p for p in development_trips("student_test", data_root)}
    folder = ROOT / "results" / "predictions"
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Waiting for student-test predictions in {folder}")
    figure_dir = ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.stem not in trips:
            continue
        trip = trips[file.stem]
        predictions = pd.read_csv(file).sort_values("point_idx")
        track, nodes, arcs = load_track(trip), load_nodes(trip), load_arcs(trip)
        ids = predictions["point_idx"]
        if ids.duplicated().any() or not ids.isin(track.index).all():
            raise ValueError(f"Unexpected observation IDs in {file.name}")
        html = ROOT / "visualisations" / "maps" / f"pred_{trip.name}.html"
        make_map(trip, predictions, out_path=html)
        gps, node_xy = project_trip(track.lon.to_numpy(), track.lat.to_numpy(), nodes)
        segments = node_xy[arcs[["from_node", "to_node"]].to_numpy(int)]
        fig, ax = plt.subplots(figsize=(8, 6))
        lo, hi = gps.min(axis=0) - 150, gps.max(axis=0) + 150
        visible = ((segments.max(axis=1) >= lo) & (segments.min(axis=1) <= hi)).all(axis=1)
        ax.add_collection(LineCollection(segments[visible], colors="#b6b8ba", linewidths=.4))
        layers = [
            ("Reference", load_route(trip) or [], "#008458", "solid"),
            ("Classifier", predictions.classifier_arc.dropna(), "#e98425", "dashed"),
            ("HMM", predictions.hmm_arc.dropna(), "#c83850", "solid"),
        ]
        for label, arc_ids, color, style in layers:
            selected = np.unique(np.asarray(arc_ids, dtype=int))
            if np.any((selected < 0) | (selected >= len(arcs))):
                raise ValueError(f"Unknown {label} road segment in {file.name}")
            ax.add_collection(LineCollection(segments[selected], colors=color, linewidths=2,
                                             linestyles=style, label=label))
        ax.scatter(*gps.T, s=2, color="#175bb4", label="GPS")
        different = predictions.loc[predictions.classifier_arc.ne(predictions.hmm_arc), "point_idx"]
        if len(different):
            ax.scatter(*gps[different.to_numpy(int)].T, s=20, facecolors="none",
                       edgecolors="#462060", label="Disagreement")
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="Easting (m)", ylabel="Northing (m)",
               title=f"Student-test trip {trip.name}", aspect="equal")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"pred_{trip.name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    render_predictions(parser.parse_args().data_root)
