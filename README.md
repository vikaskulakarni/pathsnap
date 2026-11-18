# PathSnap

PathSnap is a machine learning hackathon project for matching noisy GPS observations to a road network and reconstructing a vehicle's route.

## Two-laptop deployment

- **Laptop 1** runs FastAPI, loads the trained classifier and road network, reconstructs the route with HMM/Viterbi, and computes evaluation scores on the server.
- **Laptop 2** runs the evaluator UI. It lists the ten reserved trips at evaluation time, sends the selected trip ID to Laptop 1, and displays the predicted route and results.

Evaluator reference routes stay on Laptop 1 with the scorer. The prediction pipeline receives GPS observations and road-network data. The UI receives predictions and aggregate results.

## Repository layout

- `common/`, `data_processing/`, `features/`, `hmm/`, and `api/` are Laptop 1's shared utilities, trip processing, candidate features, route decoding, and FastAPI service.
- `models/`, `optimisation/`, `ui/`, `notebooks/`, and `visualisations/` are Laptop 2's training, tuning, Streamlit UI, exploration, and Folium map code.
- `artifacts/` holds the final model and inference configuration. `results/` holds reproducible metrics, EDA outputs, and figures. `report/` holds the technical report and screenshots.
- `tests/` holds checks for both laptops.

## Branches

- `main` is the shared integration branch.
- Laptop 1 starts on `feat/setup`.
- Laptop 2 starts on `feat/train-utils`.

## Local data and dependencies

The local data layout is `data/{train,student_test,eval_test}/<trip_id>/` (each trip folder holds `.track`, `.nodes`, `.arcs` and, where supplied, `.route`). Generated candidate tables go in `data/processed/`. Keep these data files local. Use the 80 training trips for fitting and trip-grouped cross-validation, the 10 student-test trips for development comparisons, and the reserved evaluator partition for final evaluation.

Use Python 3.10 or later and install the shared dependencies in a virtual environment with `python -m pip install -r requirements.txt`. Run instructions for the API, UI, and model pipeline will be added as those components land.

### Data pipeline (run from the repo root)

```
python -m pytest -q                                              # unit tests
python -m data_processing.inspect_network                        # network sharing/direction check
python -m data_processing.label_points --splits train student_test   # per-point reference labels + quality CSV
python -m features.candidate_sweep                               # candidate miss rate for K x radius (train)
python -m features.make_tables --splits train student_test       # labelled feature tables in data/processed/
```

Model inputs are `features.build_features.FEATURE_COLUMNS`; the identifier and location columns in `METADATA_COLUMNS` must not be used as model inputs.

### Models, decoder and API (run after the data pipeline)

```
python -m models.train_models          # 5 model families, GroupKFold by trip, OOF probabilities, comparison table/figures
python -m optimisation.bayes_tune      # Optuna TPE on the best model; writes artifacts/model.joblib, model_alt.joblib, best_params.json
python -m hmm.tune                     # decoder tuning on train out-of-fold probabilities, confirmed on student_test
python -m hmm.evaluate                 # classifier-only vs HMM comparison -> results/hmm_results.csv, hmm_failures.csv, figure
python -m hmm.pipeline --split student_test --save   # predictions for the maps -> results/predictions/
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
python -m api.smoke_test --url http://localhost:8000
```

Set `PATHSNAP_EVAL_SPLIT=student_test` to rehearse the API while `data/eval_test/` is not present. The API predicts first (no labels
involved) and only afterwards `api/evaluation.py` reads the held-out route and returns two numbers (accuracy, route continuity).
Large reproducible outputs (`results/oof/`, `artifacts/models/`) are git-ignored; regenerate them with the commands above.

## Submission checklist

- [ ] Commit the code and dependencies for preprocessing, candidate generation, feature engineering, training, evaluation, FastAPI, the UI, and deployment.
- [ ] Train and compare at least five distinct models. Retain configurations, evaluation outputs, and timings.
- [ ] Record the Bayesian hyperparameter search and compare the tuned and untuned model.
- [ ] Implement HMM/Viterbi and report a comparison against classifier-only predictions.
- [ ] Include the final model weights and required inference configuration, subject to repository size constraints.
- [ ] Commit Folium map-generation code and supporting experiment results.
- [ ] Document installation, execution, and the two-laptop deployment. Verify live requests and capture UI screenshots and Laptop 1 API responses.
- [ ] Submit the public GitHub repository and one consolidated technical PDF with results from the team's actual experiments.
