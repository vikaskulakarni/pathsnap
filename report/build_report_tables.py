"""Generate report/PathSnap_Report.md: the report SKELETON in the order required by the Deliverables guide.

Only tables and figure references are filled in, and only from results/*.csv, artifacts/*.json and
results/figures/. Every explanatory paragraph and every answer is a "TODO (team)" placeholder that the
team writes. Re-run after any experiment so the tables stay traceable:

    python report/build_report_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "PathSnap_Report.md"
FIG = "../results/figures"
EDA_FIG = "../results/eda/figures"


def table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def todo(what: str, evidence: str) -> str:
    return f"> **TODO (team):** {what}  \n> *Evidence:* {evidence}"


def fig(path: str, caption: str) -> str:
    return f"![{caption}]({path})\n\n*{caption}*"


def f3(x) -> str:
    return f"{x:.3f}"


def f4(x) -> str:
    return f"{x:.4f}"


def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RES / name, **kw)


def eda_section() -> str:
    inv = csv("eda/inventory_summary.csv")
    inv = pd.DataFrame({"Partition": inv.split, "Trips": inv["size"], "GPS points (min)": inv["min"],
                        "median": inv["median"], "max": inv["max"], "total": inv["sum"]})
    q = csv("eda/quality.csv")
    g = q.groupby("split").agg(trips=("trip_id", "size"), missing=("missing_values", "sum"),
                               invalid_coord=("invalid_coordinates", "sum"), invalid_time=("invalid_timestamps", "sum"),
                               duplicate_time=("duplicate_timestamps", "sum"),
                               non_increasing=("non_increasing_steps", "sum"), steps=("steps", "sum"),
                               one_second=("one_second_steps", "sum"), gaps_gt_5s=("gaps_over_5s", "sum"),
                               largest_gap_s=("largest_gap_s", "max")).reset_index()
    g["share_one_second_steps"] = (g.one_second / g.steps).map(f4)
    miss = g.rename(columns={"split": "Partition"})
    n = csv("eda/network_stats.csv")
    ns = n.groupby("split").agg(trips=("trip_id", "size"), mean_nodes=("nodes", "mean"), mean_arcs=("arcs", "mean"),
                                trips_multi_component=("weak_components", lambda s: int((s > 1).sum())),
                                isolated_nodes=("isolated_nodes", "sum"),
                                mean_reference_connected_share=("reference_connected_share", "mean")).reset_index()
    ns[["mean_nodes", "mean_arcs"]] = ns[["mean_nodes", "mean_arcs"]].round(0).astype(int)
    ns["mean_reference_connected_share"] = ns.mean_reference_connected_share.map(f4)
    noise = csv("eda/gps_noise.csv", dtype={"trip_id": str})
    noise = noise[noise.trip_id == "all"][["split", "n_points", "median_m", "p95_m"]].round(2)
    lq = csv("eda/label_quality.csv", dtype={"trip": str})
    flagged = lq[lq.flag_p95_gt_30][["split", "trip", "n_points", "mean_dist_m", "p95_dist_m"]].round(2)
    cm = csv("eda/candidate_miss_by_trip.csv", dtype={"trip_id": str})
    pooled = lambda d: (d.miss_rate * d.points).sum() / d.points.sum()  # noqa: E731
    tr = cm[cm.split == "train"]
    miss_tbl = pd.DataFrame({"Partition": ["train, all 80 trips", "train, trips without flagged labels", "train, flagged trips",
                                           "student_test"],
                             "Candidate miss rate": [f4(pooled(tr)), f4(pooled(tr[~tr.label_flagged])),
                                                     f4(pooled(tr[tr.label_flagged])), f4(pooled(cm[cm.split == "student_test"]))]})
    sw = csv("eda/candidate_sweep.csv").pivot(index="k", columns="radius_m", values="miss_rate").round(4)
    sw.columns = [f"radius {c} m" for c in sw.columns]
    sw = sw.reset_index().rename(columns={"k": "K"})
    return "\n\n".join([
        "## 2.2 Dataset Understanding and Exploratory Data Analysis",
        "*Partitions: train (80 trips) and student_test (10 trips). eval_test is never used in the EDA.*",
        "### Question 1. How did you preprocess the data?",
        todo("describe projection to metres (UTM), per-trip road networks, per-point reference labels (monotone DP), candidate "
             "generation (K nearest within RADIUS_M plus reverse twins) and the feature builder; name the partition used for each step.",
             "docs/METHODOLOGY.md sections 1-4; notebooks/02_labels_and_features.ipynb"),
        "### Question 2. What patterns did you observe, and which features did you extract and why?",
        todo("explain directed twin arcs, heading as the direction cue, GPS noise, time gaps, and the reason for each feature.",
             "docs/METHODOLOGY.md section 4; results/feature_importance.csv; notebooks/01_eda.ipynb"),
        "### Dataset inventory (train, student_test)", table(inv),
        "### Missing-value and quality analysis (per partition, summed over trips)", table(miss),
        "### Road-network size and connectivity (per partition)", table(ns),
        "### GPS-to-nearest-road distance (all points in the partition)", table(noise),
        "### GPS trajectory plots",
        fig(f"{EDA_FIG}/eda_coordinates_train.png", "GPS coordinates, train partition"),
        fig(f"{EDA_FIG}/eda_coordinates_student_test.png", "GPS coordinates, student_test partition"),
        fig(f"{EDA_FIG}/eda_trip_stats_train.png", "Trip statistics, train partition"),
        fig(f"{EDA_FIG}/eda_time_gaps_train.png", "Time gaps, train partition"),
        fig(f"{EDA_FIG}/eda_gps_noise_train.png", "GPS noise, train partition"),
        "### Road-network visualisations",
        fig(f"{EDA_FIG}/eda_network_train.png", "Road-network size and connectivity, train partition"),
        fig(f"{EDA_FIG}/eda_network_student_test.png", "Road-network size and connectivity, student_test partition"),
        "### Folium map (GPS, road network, reference route)",
        "Interactive maps: `visualisations/maps/eda_train_00000001.html` and `visualisations/maps/eda_student_test_00000007.html` "
        "(generated by `python -m visualisations.eda_map`). Insert a screenshot: `![](screenshots/folium_eda.png)`",
        todo("comment on what the map shows (noise vs road, parallel streets, reference route).", "visualisations/maps/eda_*.html"),
        "### Reference-label quality, candidate coverage and data-quality decisions",
        table(flagged.rename(columns={"trip": "trip_id"})), table(miss_tbl), table(sw),
        todo("list the data-quality issues, assumptions and decisions (flagged trips, K and radius choice, treatment of the reference route as "
             "reference information, not ground truth).", "results/eda/data_quality.md; results/eda/label_quality.csv; results/ablation_flagged_trips.csv"),
    ])


def models_section() -> str:
    mc = csv("model_comparison.csv")
    comp = pd.DataFrame({"Model": mc.model, "Algorithm Family": mc.algorithm_family, "Key Configuration": mc.configuration,
                         "CV Macro F1": mc.cv_macro_f1.map(f4), "CV Top-1": mc.cv_top1.map(f4),
                         "Top-1 Accuracy (student_test)": mc.test_top1.map(f4),
                         "Route continuity (student_test)": mc.test_route_continuity.map(f4),
                         "Student-Test Macro F1": mc.test_macro_f1.map(f4), "Log Loss (student_test)": mc.test_log_loss.map(f4),
                         "Inference Time (ms per 1000 rows)": mc.infer_ms_per_1k_rows.map("{:.2f}".format)})
    parts = ["## 3. Model Experimentation and Comparison",
             "### 3.1 Overall Model Comparison", "*CV = 5-fold GroupKFold by trip on the 80 train trips; student-test columns use the 10 student_test trips.*",
             table(comp),
             todo("state the primary selection metric (observation-level Top-1) and why, and which model was chosen.",
                  "results/model_comparison.csv; docs/METHODOLOGY.md section 5"),
             "### 3.2 Compulsory Questions for EVERY Model"]
    for i, r in enumerate(mc.itertuples(), 1):
        res = pd.DataFrame({"Metric": ["CV Top-1 (mean)", "CV Top-1 (std over folds)", "CV Macro F1", "Student Top-1", "Student Macro F1",
                                       "Student precision", "Student recall", "Student log loss", "Student route continuity",
                                       "Fit time (s)", "CV time (s)"],
                            "Value": [f4(r.cv_top1), f4(r.cv_top1_std), f4(r.cv_macro_f1), f4(r.test_top1), f4(r.test_macro_f1),
                                      f4(r.test_precision), f4(r.test_recall), f4(r.test_log_loss), f4(r.test_route_continuity),
                                      f"{r.fit_seconds:.1f}", f"{r.cv_seconds:.1f}"]})
        parts += [f"#### Model {i}: `{r.model}` ({r.algorithm_family})", f"*Configuration:* {r.configuration}", table(res),
                  "**A. Model Understanding**",
                  todo("1. why this algorithm suits PathSnap (per-candidate probabilities for the decoder).", "docs/METHODOLOGY.md section 5"),
                  todo("2. strengths and limitations for this task, connected to the results above.", "table above; results/figures/model_confusion_all.png"),
                  "**B. Training and Configuration**",
                  todo("5. important hyperparameters and their initial values.", f"models/train_models.py (model_zoo, {r.model})"),
                  todo("6. loss function / optimisation objective.", "models/train_models.py; scikit-learn documentation for this estimator"),
                  "**C. Experimental Results**",
                  todo("7. performance on the student-testing dataset.", "table above"),
                  todo("8. observation-level Top-1, macro F1 and log loss on student_test.", "table above (student rows)"),
                  todo("9. overfitting or underfitting, with evidence (compare CV and student_test, fold std).",
                       "CV vs student rows above; results/model_comparison.csv")]
    parts += ["### 3.3 Model Visualisations",
              fig(f"{FIG}/model_cv_macro_f1.png", "Cross-validation macro F1 across models"),
              fig(f"{FIG}/model_cv_top1.png", "Cross-validation Top-1 across models"),
              fig(f"{FIG}/model_student_top1.png", "student_test Top-1 across models"),
              fig(f"{FIG}/model_student_log_loss.png", "student_test log loss across models"),
              fig(f"{FIG}/model_confusion_all.png", "Confusion matrices on student_test candidate rows"),
              fig(f"{FIG}/model_calibration.png", "Reliability diagram on student_test"),
              fig(f"{FIG}/model_feature_importance.png", "Permutation importance of the selected model"),
              todo("explain why training/validation loss curves are N/A for these models (no epoch-wise history), and use the plots to justify the model selection.",
                   "results/figures/model_*.png"),
              "Folium map of a selected trip: see section 6."]
    return "\n\n".join(parts)


def bayes_section() -> str:
    bp = json.loads((ROOT / "artifacts" / "best_params.json").read_text())
    tc = csv("tuning_comparison.csv")
    t = pd.DataFrame({"Configuration": tc.configuration, "CV Macro F1": tc.cv_macro_f1.map(f4),
                      "CV Top-1": tc.cv_top1.map(f4), "Student-Test Top-1": tc.test_top1.map(f4),
                      "Student-Test Macro F1": tc.test_macro_f1.map(f4), "Log Loss (student_test)": tc.test_log_loss.map(f4),
                      "Inference Time (ms per 1000 rows)": tc.infer_ms_per_1k_rows.map("{:.2f}".format)})
    tr = csv("optuna_trials.csv")
    pc = [c for c in tr.columns if c.startswith("params_")]
    top = tr.sort_values("value", ascending=False).head(5)[["number", "value"] + pc].round(4)
    best = pd.DataFrame({"Hyperparameter": list(bp["best_params"]), "Best value": [bp["best_params"][k] for k in bp["best_params"]]})
    s = bp["search"]
    meta = pd.DataFrame({"Item": ["Model tuned", "Sampler", "Trials", "Folds per trial", "Trips in the search sample", "Objective",
                                  "Best objective (search sample)", "Search time (s)", "Seed", "Selected for deployment"],
                         "Value": [bp["model"], s["sampler"], s["n_trials"], s["folds"], s["trips"], s["objective"],
                                   f4(s["best_objective"]), f"{s['seconds']:.0f}", s["seed"], bp["selected_file"]]})
    return "\n\n".join([
        "## 4. Bayesian Hyperparameter Optimisation (for the chosen model)",
        table(meta),
        todo("1. which hyperparameters were searched.", "optimisation/bayes_tune.py (suggest)"),
        todo("2. ranges / distributions considered and why.", "optimisation/bayes_tune.py (suggest)"),
        todo("3. objective function optimised and why.", "artifacts/best_params.json; docs/METHODOLOGY.md section 5"),
        todo("4. number of trials performed.", "table above; results/optuna_trials.csv"),
        todo("5. best hyperparameter values obtained.", "table below; artifacts/best_params.json"),
        table(best), "Top 5 trials:", table(top), fig(f"{FIG}/optuna_history.png", "Optuna search history"),
        todo("6. did the tuned model outperform the untuned model? (same metrics, same partitions)", "tuned-vs-untuned table below"),
        "### Tuned vs untuned (same folds and partitions)", table(t),
        todo("discuss the size of the difference relative to fold-to-fold variation, and the computational cost.",
             "results/tuning_comparison.csv; results/model_comparison.csv (cv_top1_std)"),
    ])


def hmm_section() -> str:
    h = csv("hmm_results.csv")
    cmp = pd.DataFrame({"Method": h.method, "Top-1": h.top1.map(f4), "Macro F1": h.macro_f1.map(f4),
                        "Route continuity": h.route_continuity.map(f4), "Wrong-direction rate": h.wrong_direction_rate.map(f4),
                        "Trips >= 90% correct": (h.trips_ge90_share * 100).map("{:.0f}%".format),
                        "Epsilon-transition share": h.eps_transition_share.map(lambda v: "-" if pd.isna(v) else f"{v:.4f}"),
                        "Decode s/trip": h.decode_s_per_trip.map("{:.2f}".format)})
    pt = csv("hmm_per_trip.csv", dtype={"trip": str}).pivot(index="trip", columns="method", values="top1").round(3).reset_index()
    fail = csv("hmm_failures.csv", dtype={"trip": str})
    cfg = json.loads((ROOT / "artifacts" / "hmm_config.json").read_text())
    base = json.loads((ROOT / "artifacts" / "hmm_config_distance.json").read_text())
    params = pd.DataFrame({"Parameter": list(cfg), "Classifier + HMM": [cfg[k] for k in cfg],
                           "Distance-only HMM": [base.get(k, "-") for k in cfg]})
    tun = csv("hmm_tuning.csv")
    best = tun.loc[tun.groupby("param").top1.idxmax()][["param", "value", "top1", "continuity"]].round(4)
    return "\n\n".join([
        "## 5. HMM and Viterbi Route Reconstruction",
        todo("1. what are the hidden states (candidate arcs per GPS point).", "docs/METHODOLOGY.md section 6"),
        todo("2. how emission probabilities are calculated (classifier probability x distance Gaussian, weight w).", "hmm/emission.py; artifacts/hmm_config.json"),
        todo("3. how transition probabilities are calculated (directed road-graph hops, U-turn, epsilon).", "hmm/transition.py; artifacts/hmm_config.json"),
        todo("4. how missing candidates and large GPS gaps are handled.", "hmm/decode.py; hmm/transition.py (gap_s, gap_extra_hops)"),
        todo("5. what happens when consecutive observations suggest disconnected segments (epsilon floor; report its usage share).", "table below (epsilon-transition share)"),
        todo("write the Viterbi recurrence and explain why log space and the T x K^2 cost.", "hmm/viterbi.py; tests/test_viterbi.py (hand-checked example)"),
        "### Decoder parameters (artifacts)", table(params),
        "### Decoder tuning: best value per parameter (train out-of-fold sample)", table(best),
        "### Classifier-only vs classifier + HMM (student_test)", table(cmp),
        fig(f"{FIG}/hmm_comparison.png", "Classifier-only vs HMM decoding on student_test"),
        todo("6. did HMM decoding improve the final route? support with the table (Top-1, continuity, trips >= 90%, wrong-direction rate).",
             "results/hmm_results.csv; results/hmm_per_trip.csv"),
        "### Per-trip Top-1", table(pt),
        "### Failure cases (trips where the HMM improved Top-1 least)", table(fail),
        todo("discuss cases where HMM decoding did not help (open the trips on the map; compare with the distance-only HMM).",
             "results/hmm_failures.csv; visualisations/maps/pred_<trip>.html"),
    ])


def folium_section() -> str:
    pngs = sorted((RES / "figures").glob("pred_*.png"))
    maps = sorted((ROOT / "visualisations" / "maps").glob("pred_*.html"))
    listing = pd.DataFrame({"Trip": [p.stem.replace("pred_", "") for p in maps],
                            "Interactive map": [f"visualisations/maps/{p.name}" for p in maps]})
    figs = [fig(f"{FIG}/{p.name}", f"Reference (green), classifier-only (orange), HMM (red), GPS (blue): trip {p.stem.replace('pred_', '')}")
            for p in pngs[:2]]
    return "\n\n".join([
        "## 6. Geographical Visualisation Using Folium",
        "Maps are generated by `python -m visualisations.plot_prediction` (predictions) and `python -m visualisations.eda_map` (EDA). "
        "Layers: GPS observations, road network, reference route, predicted route, with a legend and layer control.",
        table(listing),
        "Insert at least one exported view of an interactive map: `![](screenshots/folium_prediction.png)` (save it in `report/screenshots/`).",
        *figs,
        todo("explain how to read the map and what it shows about where the HMM corrects the classifier.", "visualisations/maps/pred_<trip>.html"),
    ])


def api_section() -> str:
    ep = pd.DataFrame({"Endpoint": ["GET /health", "GET /trips", "POST /predict"],
                       "Purpose": ["liveness", "ids of the evaluation trips (data/eval_test/)",
                                   "predict one trip; returns points, route_coords, accuracy, route_continuity, latency_ms, model_version"]})
    return "\n\n".join([
        "## 7. API, UI and Deployment",
        "### Endpoints", table(ep),
        todo("explain the prediction flow (features, classifier, Viterbi) and how model, features and decoder config are loaded once at start-up.",
             "api/main.py; hmm/pipeline.py"),
        todo("explain how ground truth stays server-side (evaluation after prediction, only two numbers returned) and how this is tested.",
             "api/evaluation.py; tests/test_pipeline.py; tests/test_api.py; tests/test_ui_contract.py"),
        "### Screenshots (captured on the real two-laptop run, saved in `report/screenshots/`)",
        "1. Working UI after a prediction: `![](screenshots/ui_prediction.png)`",
        "2. API request log on the receiving laptop (Laptop 1): `![](screenshots/api_log.png)`",
        todo("describe the deployment (which laptop runs what, addresses, firewall) and the commands used.", "README.md quick start; scripts/start_api.ps1; scripts/start_ui.ps1"),
        todo("state that the submitted code matches the implementation demonstrated.", "repository at the submitted commit"),
    ])


def ai_section() -> str:
    return "\n\n".join(["## 8. AI Usage",
                        todo("disclose AI assistance as required by the event policy: what tools were used for what, and what the team verified.",
                             "AI_USAGE.md")])


def main():
    head = ("# PathSnap: Technical Report\n\n"
            "| # | SRN | Name |\n|---|-----|------|\n| 1 | `<SRN>` | `<NAME>` |\n| 2 | `<SRN>` | `<NAME>` |\n"
            "| 3 | `<SRN>` | `<NAME>` |\n| 4 | `<SRN>` | `<NAME>` |\n\n"
            "> Skeleton generated by `report/build_report_tables.py`. Tables and figure references come from `results/` and are regenerated by the script; "
            "every **TODO (team)** is an explanation or answer that the team writes in its own words.\n")
    parts = [head, eda_section(), models_section(), bayes_section(), hmm_section(), folium_section(), api_section(), ai_section()]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {sum(p.count('TODO (team)') for p in parts)} TODO placeholders)")


if __name__ == "__main__":
    main()
