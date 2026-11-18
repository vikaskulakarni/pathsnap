"""Measure data quality and road-network coverage on development trips."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection
from pyproj import Geod, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree

from common.io import list_trips, load_arcs, load_nodes, load_route, load_track, trip_files

SPLITS = ("train", "student_test")
GEOD = Geod(ellps="WGS84")
ROOT = Path(__file__).resolve().parents[1]


def development_trips(split: str, data_root: Path | None) -> list[Path]:
    """Select only the requested development partition."""
    if split not in SPLITS:
        raise ValueError("EDA accepts only train and student_test.")
    if data_root is None:
        return list_trips(split)
    folder = data_root / split
    return sorted(path for path in folder.iterdir() if path.is_dir())


def project_trip(
    lon: np.ndarray, lat: np.ndarray, nodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Use one local UTM projection for points and road segments in the trip."""
    center_lon, center_lat = float(np.median(lon)), float(np.median(lat))
    zone = min(60, int((center_lon + 180) // 6) + 1)
    epsg = (32600 if center_lat >= 0 else 32700) + zone
    transform = Transformer.from_crs(4326, epsg, always_xy=True)
    track_xy = np.column_stack(transform.transform(lon, lat))
    node_xy = np.column_stack(transform.transform(nodes["lon"], nodes["lat"]))
    return track_xy, node_xy


def _quality(track: pd.DataFrame) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    values = track[["lon", "lat", "t"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    lon, lat, times = values.T
    valid_coords = (
        np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    )
    valid_times = np.isfinite(times)
    dt = np.diff(times)
    finite_dt = dt[np.isfinite(dt)]
    pairs = valid_coords[:-1] & valid_coords[1:] & np.isfinite(dt) & (dt > 0)
    _, _, distance = GEOD.inv(lon[:-1][pairs], lat[:-1][pairs], lon[1:][pairs], lat[1:][pairs])
    result = {
        "missing_values": int(np.isnan(values).sum()),
        "invalid_coordinates": int((~valid_coords).sum()),
        "invalid_timestamps": int((~valid_times).sum()),
        "duplicate_timestamps": int(pd.Series(times[valid_times]).duplicated().sum()),
        "non_increasing_steps": int((finite_dt <= 0).sum()),
        "steps": int(finite_dt.size),
        "one_second_steps": int(np.isclose(finite_dt, 1, rtol=0, atol=1e-6).sum()),
        "gaps_over_5s": int((finite_dt > 5).sum()),
        "largest_gap_s": float(finite_dt.max()) if finite_dt.size else np.nan,
        "duration_s": float(times[valid_times][-1] - times[valid_times][0]) if valid_times.any() else np.nan,
        "length_m": float(distance.sum()),
    }
    result["share_one_second"] = result["one_second_steps"] / finite_dt.size if finite_dt.size else np.nan
    return result, valid_coords, finite_dt, distance / dt[pairs]


def _network(
    n_nodes: int, edges: np.ndarray, reference: list[int] | None,
) -> tuple[dict[str, float | int], np.ndarray]:
    starts, ends = edges.T
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (starts, ends)), shape=(n_nodes, n_nodes))
    degree = np.bincount(edges.ravel(), minlength=n_nodes)
    stats = {
        "nodes": n_nodes,
        "arcs": len(edges),
        "weak_components": connected_components(graph, directed=True, connection="weak", return_labels=False),
        "isolated_nodes": int((degree == 0).sum()),
        "reference_arcs": len(reference) if reference is not None else 0,
        "reference_connected_share": np.nan,
    }
    if reference and len(reference) > 1:
        ids = np.asarray(reference)
        if np.any(ids < 0) or np.any(ids >= len(edges)):
            raise ValueError("Reference route contains an unknown arc.")
        stats["reference_connected_share"] = float(
            ((ids[:-1] == ids[1:]) | (ends[ids[:-1]] == starts[ids[1:]])).mean()
        )
    return stats, degree


def _noise(track_xy: np.ndarray, node_xy: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Query the nearest full road segment, including segments with distant endpoints."""
    segments = shapely.linestrings(node_xy[edges])
    _, distances = STRtree(segments).query_nearest(
        shapely.points(track_xy), return_distance=True, all_matches=False,
    )
    return distances


def _save_dashboard(
    panels: list[tuple[str, str, np.ndarray, bool]], name: str, figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9), squeeze=False)
    for ax, (title, unit, values, log_scale) in zip(axes[0], panels):
        values = values[np.isfinite(values)]
        if log_scale:
            values = values[values > 0]
        if log_scale and values.size and values.max() > values.min():
            bins = np.geomspace(values.min(), values.max(), 55)
            ax.set_xscale("log")
        else:
            bins = 36
        ax.hist(values, bins=bins, color="#1d6e8c")
        ax.set(title=title, xlabel=unit, ylabel="Count")
    fig.tight_layout()
    fig.savefig(figure_dir / name, dpi=150)
    plt.close(fig)


def _save_coordinates(
    coords: np.ndarray, example: tuple[str, np.ndarray, np.ndarray],
    split: str, figure_dir: Path,
) -> None:
    trip, points, segments = example
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(*coords.T, s=1, alpha=0.25, color="#1d6e8c")
    axes[0].set(title=f"{split} GPS coverage", xlabel="Longitude (°)", ylabel="Latitude (°)")
    low, high = points.min(axis=0) - 200, points.max(axis=0) + 200
    in_view = ((segments.max(axis=1) >= low) & (segments.min(axis=1) <= high)).all(axis=1)
    axes[1].add_collection(LineCollection(segments[in_view], colors="#9ca3af", linewidths=0.45))
    axes[1].plot(*points.T, linewidth=0.6, color="#1d6e8c")
    axes[1].scatter(*points.T, s=3, color="#1d6e8c")
    axes[1].set(
        title=f"{split} road network and trip {trip}", xlabel="Easting (m)", ylabel="Northing (m)",
        xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / f"eda_coordinates_{split}.png", dpi=150)
    plt.close(fig)


def analyse_split(
    split: str, data_root: Path | None, figure_dir: Path,
) -> dict[str, list[dict]]:
    """Scan each trip once so the figures and tables use the same inputs."""
    tables = {name: [] for name in ("inventory", "quality", "network_stats", "gps_noise")}
    trips = development_trips(split, data_root)
    if not trips:
        raise FileNotFoundError(f"No {split} trips found. Supply --data-root to locate the student dataset.")
    gaps, speeds, coords, degrees, distances = [], [], [], [], []
    example = None
    for trip in trips:
        keys = {"split": split, "trip_id": trip.name}
        files = trip_files(trip)
        missing = [kind for kind in ("track", "nodes", "arcs", "route") if kind not in files]
        track = load_track(trip) if "track" in files else None
        tables["inventory"].append({
            **keys, "gps_points": len(track) if track is not None else 0,
            "missing_files": ";".join(missing),
            **{f"{kind}_bytes": files[kind].stat().st_size if kind in files else 0
               for kind in ("track", "nodes", "arcs", "route")},
        })
        if track is None:
            continue
        quality, valid, dt, speed = _quality(track)
        tables["quality"].append({**keys, **quality})
        gaps.append(dt)
        speeds.append(speed)
        points = track.loc[valid, ["lon", "lat"]].astype(float).to_numpy()
        if points.size:
            coords.append(points)
        if "nodes" not in files or "arcs" not in files:
            continue
        nodes, arcs = load_nodes(trip), load_arcs(trip)
        edges = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
        network, degree = _network(len(nodes), edges, load_route(trip))
        tables["network_stats"].append({**keys, **network})
        degrees.append(degree)
        if points.size and edges.size:
            gps_xy, nodes_xy = project_trip(*points.T, nodes)
            distance = _noise(gps_xy, nodes_xy, edges)
            distances.append(distance)
            tables["gps_noise"].append({
                **keys, "n_points": len(distance), "median_m": float(np.median(distance)),
                "p95_m": float(np.quantile(distance, 0.95)),
            })
            if example is None:
                example = trip.name, gps_xy, nodes_xy[edges]

    if distances:
        all_distances = np.concatenate(distances)
        median, p95 = np.quantile(all_distances, [0.5, 0.95])
        tables["gps_noise"].append({
            "split": split, "trip_id": "all", "n_points": len(all_distances),
            "median_m": float(median), "p95_m": float(p95),
        })
        _save_dashboard([(f"{split} nearest road", "Distance (m)", all_distances, False)],
                        f"eda_gps_noise_{split}.png", figure_dir)
    if gaps:
        _save_dashboard([(f"{split} time gaps", "Time gap (s)", np.concatenate(gaps), True)],
                        f"eda_time_gaps_{split}.png", figure_dir)
        quality = pd.DataFrame(tables["quality"])
        _save_dashboard([
            (f"{split} duration", "Duration (s)", quality["duration_s"].to_numpy(), True),
            (f"{split} length", "Track length (m)", quality["length_m"].to_numpy(), True),
            (f"{split} speed", "Speed (m/s)", np.concatenate(speeds), True),
        ], f"eda_trip_stats_{split}.png", figure_dir)
    if degrees:
        network = pd.DataFrame(tables["network_stats"])
        _save_dashboard([
            (f"{split} node degree", "Incident arcs", np.concatenate(degrees), False),
            (f"{split} components", "Weak components", network["weak_components"].to_numpy(), False),
        ], f"eda_network_{split}.png", figure_dir)
    if example is not None:
        _save_coordinates(np.concatenate(coords), example, split, figure_dir)
    return tables


def run_eda(data_root: Path | None = None, results_dir: Path = ROOT / "results") -> None:
    figure_dir = results_dir / "eda" / "figures"
    table_dir = results_dir / "eda"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    results = [analyse_split(split, data_root, figure_dir) for split in SPLITS]
    for name in results[0]:
        table = pd.DataFrame([row for split_tables in results for row in split_tables[name]])
        table.to_csv(table_dir / f"{name}.csv", index=False)
    inventory = pd.read_csv(table_dir / "inventory.csv")
    summary = inventory.groupby("split")["gps_points"].agg(["size", "min", "median", "max", "sum"])
    summary.to_csv(table_dir / "inventory_summary.csv")
    print(summary.to_string())
    noise = pd.read_csv(table_dir / "gps_noise.csv")
    if not noise.empty:
        print(noise.loc[noise["trip_id"] == "all"].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Parent directory of train and student_test.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_eda(args.data_root, args.results_dir)


if __name__ == "__main__":
    main()
