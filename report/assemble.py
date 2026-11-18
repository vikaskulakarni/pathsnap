"""Assemble measured results and figures into a printable technical report draft."""
import argparse
import base64
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def render(root: Path, output: Path) -> None:
    results = root / "results"
    parts = ["<h1>PathSnap</h1><p>Distributed map matching · Technical report draft</p>"]
    def heading(title: str) -> None:
        parts.append(f"<h2>{escape(title)}</h2>")

    def table(name: str, columns: list[str] | None = None) -> None:
        path = results / name
        if path.exists():
            data = pd.read_csv(path)
            if name.endswith("gps_noise.csv"):
                data = data.loc[data.trip_id == "all"]
            if columns:
                data = data[columns]
            parts.append(data.to_html(index=False, border=0, float_format=lambda n: f"{n:.4g}"))
            parts.append(f"<small>Source · results/{escape(name)}</small>")
        else:
            parts.append(f"<p>Pending experiment output · {escape(name)}</p>")

    def image(path: Path, source: str) -> None:
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append(f'<figure><img alt="{escape(path.stem)}" src="data:image/png;base64,{encoded}">'
                         f'<figcaption>{escape(path.name)} · {escape(source)}</figcaption></figure>')

    heading("2 · Dataset and exploratory analysis")
    table("eda/inventory_summary.csv")
    table("eda/gps_noise.csv")
    notes = results / "eda" / "data_quality.md"
    if notes.exists():
        parts.append("<ul>" + "".join("<li>" + escape(line[2:]) + "</li>" for line in
                                     notes.read_text(encoding="utf-8-sig").splitlines() if line.startswith("- ")) + "</ul>")
    for path in sorted((results / "figures").glob("eda_*.png")):
        image(path, "visualisations/eda_plots.py")
    heading("3 · Model experiments and comparison")
    table("model_comparison.csv", ["model", "family", "cv_macro_f1_mean", "test_top1", "test_macro_f1",
                                   "test_route_continuity", "test_logloss", "infer_ms_per_point"])
    for name in ["cv_macro_f1", "test_top1_f1", "logloss", "infer_time", "overfit_gap"]:
        image(results / "figures" / f"{name}.png", "models/compare_models.py")
    heading("4 · Bayesian hyperparameter optimisation")
    table("tuned_vs_untuned.csv", ["model", "cv_macro_f1_mean", "test_top1", "test_macro_f1",
                                   "test_logloss", "infer_ms_per_point"])
    for name in ["optuna_history", "optuna_param_importance"]:
        image(results / "figures" / f"{name}.png", "optimisation/finish_models.py")
    heading("5 · HMM and Viterbi decoding")
    table("hmm_results.csv")
    heading("6 · Geographical visualisation")
    for path in sorted((results / "figures").glob("pred_*.png")):
        image(path, "visualisations/plot_prediction.py")
    image(root / "report/screenshots/folium.png", "Development trip Folium map")
    heading("7 · API, UI and deployment")
    parts.append("<p>Laptop 2 retrieves the trip catalog, sends the selected ID to Laptop 1, "
                 "and displays returned positions, predicted geometry and aggregate scores.</p>")
    for name in ["ui", "api"]:
        image(root / "report/screenshots" / f"{name}.png", "Two-laptop evaluation")
    css = ("body{max-width:880px;margin:40px auto;font:15px/1.5 Arial;color:#252337}"
           "h1{font-size:40px}h2{margin-top:32px}table{border-collapse:collapse;font-size:12px}"
           "td,th{padding:6px;border:1px solid #ddd}figure{break-inside:avoid;margin:16px 0}"
           "img{max-width:100%;max-height:380px}small,figcaption{color:#555}h2{break-after:avoid}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('<!doctype html><meta charset="utf-8"><title>PathSnap report draft</title>'
                      + "<style>" + css + "</style>" + "".join(parts), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "draft.html")
    args = parser.parse_args()
    render(ROOT, args.output)
