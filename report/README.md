# Technical report: what to include and where it comes from

The report must be written by the team and every member must be able to explain it. This page maps each required section to the
repository files that back it, so that every number in the PDF can be verified from the repository.

| Report section | Source in this repository |
|---|---|
| 2.2 Dataset understanding and EDA (partition named per analysis) | `notebooks/01_eda.ipynb`, `results/eda/` (tables, `figures/`), `visualisations/maps/eda_*.html` (Folium) |
| Preprocessing, labels, candidates, features and why | `docs/METHODOLOGY.md` sections 1-4, `notebooks/02_labels_and_features.ipynb`, `results/eda/label_quality.csv`, `candidate_sweep.csv`, `candidate_miss_by_trip.csv`, `results/feature_importance.csv` |
| 3.1 Overall model comparison table | `results/model_comparison.csv` |
| 3.2 Per-model questions | `models/train_models.py` (configurations), `results/model_comparison.csv`, `notebooks/03_model_comparison.ipynb` |
| 3.3 Visualisations | `results/figures/model_*.png` (CV F1, student Top-1, log loss, confusion matrices, calibration) |
| 4 Bayesian optimisation | `optimisation/bayes_tune.py`, `results/optuna_trials.csv`, `results/tuning_comparison.csv`, `artifacts/best_params.json`, `notebooks/04_bayesian_optimisation.ipynb` |
| 5 HMM and Viterbi (six questions, classifier vs classifier+HMM, failure cases) | `docs/METHODOLOGY.md` section 6, `results/hmm_results.csv`, `hmm_per_trip.csv`, `hmm_failures.csv`, `hmm_tuning*.csv`, `notebooks/05_hmm_route_reconstruction.ipynb` |
| 6 Folium visualisation (GPS, road network, reference route, predicted route) | `visualisations/maps/pred_<trip>.html` and `eda_*.html`; code in `visualisations/` |
| 7 API, UI and deployment | `api/`, `ui/`, `scripts/`, `notebooks/06_api_and_ui.ipynb`; screenshots below |

## Screenshots to capture (save them in `report/screenshots/`)

1. The evaluator UI after **Reconstruct route** on a trip (map, accuracy, route continuity, latency).
2. The **API terminal on Laptop 1** showing the request log lines (`time | client | METHOD path | trip | status | n | acc | latency`) while the UI on Laptop 2 is used.
3. `http://<laptop-1-ip>:8000/docs` or a `/predict` response, if useful.
4. One Folium map from `visualisations/maps/` (export a view of the layers: GPS, reference route, predicted route).

Take these during the real two-laptop run so that the code shown matches the implementation demonstrated.
