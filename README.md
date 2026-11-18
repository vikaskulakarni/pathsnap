# PathSnap: map-matching and route reconstruction

Noisy GPS points are matched to a directed road network and turned into a connected route.
Pipeline: **candidate arcs per point -> classifier score per (point, arc) -> HMM/Viterbi -> FastAPI (Laptop 1) -> evaluator UI (Laptop 2)**.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                              # unit + API/UI-contract tests

# Laptop 1: API (use -Rehearse until data/eval_test/ exists; it serves the student_test trips)
.\scripts\start_api.ps1 -Rehearse               # http://<laptop-1-ip>:8000  (docs at /docs)

# Laptop 2: UI, pointing at Laptop 1
.\scripts\start_ui.ps1 -Api http://<laptop-1-ip>:8000    # serves on http://127.0.0.1:8501
```
Open the UI, press **Connect & load trips**, pick a trip, press **Reconstruct route**. Every request prints one log line on
Laptop 1: `time | client | METHOD path | trip | status | n=<points> | acc=<accuracy> | latency`.

The API predicts first (no labels involved); only afterwards `api/evaluation.py` reads the held-out route and returns two
numbers (accuracy and route continuity). Reference arcs never appear in a response.

## Repository layout

| Folder | What it holds |
|---|---|
| `common/` | config (paths, `SEED`, `K`, `RADIUS_M`), file loaders, geometry (UTM projection, distances, bearing) |
| `data_processing/` | network check, per-point reference labels (monotone DP) |
| `features/` | candidate generation (STRtree), feature builder, labelled feature tables, K x radius sweep |
| `models/` | five-model training and comparison (trip-grouped CV, out-of-fold probabilities) |
| `optimisation/` | Bayesian hyperparameter search (Optuna TPE) |
| `hmm/` | road graph, Viterbi, emissions/transitions, decoder tuning, evaluation, `Predictor` (`predict_trip`) |
| `api/` | FastAPI service, server-side evaluation, smoke test |
| `ui/` | evaluator interface (Laptop 2) |
| `visualisations/`, `notebooks/` | EDA plots, Folium maps, exploration notebook |
| `artifacts/` | deployed model (`model.joblib`, `model_alt.joblib`), `feature_list.json`, `hmm_config.json`, `best_params.json` |
| `results/` | every reported number and figure, produced by code (`eda/`, `figures/`, `predictions/`, CSV tables) |
| `tests/`, `scripts/` | tests; demo start scripts |
| `legacy/` | superseded phase-1 pipeline, kept for reference only (see `legacy/README.md`) |

Raw data lives in `data/{train,student_test,eval_test}/<trip>/` (`.track`, `.nodes`, `.arcs`, `.route`) and is not committed.

## Reproduce the results

```powershell
python -m data_processing.inspect_network                          # results/eda/network_check.txt
python -m data_processing.label_points --splits train student_test # results/eda/label_quality.csv
python -m features.candidate_sweep                                 # results/eda/candidate_sweep.csv
python -m features.make_tables --splits train student_test         # data/processed/*_features.parquet
python -m models.train_models                                      # results/model_comparison.csv, figures, results/oof/
python -m optimisation.bayes_tune                                  # artifacts/model*.joblib, results/optuna_trials.csv
python -m hmm.tune                                                 # decoder tuning on train OOF -> artifacts/hmm_config.json
python -m hmm.evaluate                                             # results/hmm_results.csv, hmm_failures.csv, figure
python -m hmm.pipeline --split student_test --save                 # results/predictions/
python -m visualisations.plot_prediction                           # Folium comparison maps in visualisations/maps/
python -m api.smoke_test --url http://localhost:8000               # crash test over all served trips
```

## Results (student_test, 10 trips; produced by the commands above)

Route decoding (`results/hmm_results.csv`, reference labels from `label_points`):

| Method | Top-1 | Macro F1 | Route continuity | Trips >= 90% correct |
|---|---|---|---|---|
| nearest_arc | 0.540 | 0.723 | 0.820 | 0% |
| classifier_only(model) | 0.856 | 0.916 | 0.934 | 40% |
| classifier_only(alt) | 0.854 | 0.914 | 0.930 | 40% |
| hmm_distance | 0.886 | 0.935 | 0.960 | 40% |
| hmm_classifier(model) | 0.939 | 0.965 | 0.980 | 80% |
| hmm_classifier(alt) | 0.921 | 0.953 | 0.980 | 70% |

Model comparison (`results/model_comparison.csv`; CV = 5-fold GroupKFold by trip on 73 train trips):

| Model | Family | CV Top-1 | CV macro F1 | Student Top-1 | Student log loss |
|---|---|---|---|---|---|
| hist_gradient_boosting | Boosting | 0.872 | 0.924 | 0.856 | 0.111 |
| random_forest | Ensemble (bagging) | 0.872 | 0.923 | 0.854 | 0.113 |
| decision_tree | Tree-based | 0.867 | 0.920 | 0.854 | 0.117 |
| logistic_regression | Linear / probabilistic | 0.857 | 0.902 | 0.842 | 0.157 |
| gaussian_naive_bayes | Probabilistic (generative) | 0.846 | 0.817 | 0.826 | 0.541 |

Notes: the 7 train trips whose reference labels fail the quality check (p95 distance > 30 m, `results/eda/label_quality.csv`)
are excluded from fitting, CV and decoder tuning. Reference routes are treated as reference information, not guaranteed ground truth.

## Rules this code follows
`eval_test` data is only read by the prediction pipeline (`.track/.nodes/.arcs`); its `.route` is read only in `api/evaluation.py`
after prediction. Learned steps are fitted on train trips only, CV is grouped by trip, distances are in metres, seed 42.
