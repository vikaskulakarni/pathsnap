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

The shared loaders default to `data/{train,student_test,eval_test}/<trip_id>/`. Generated candidate tables go in `data/processed/`. Keep these data files local. Use the 80 training trips for fitting and trip-grouped cross-validation, the 10 student-test trips for development comparisons, and the reserved evaluator partition for final evaluation.

Use Python 3.10 or later and install the shared dependencies in a virtual environment with `python -m pip install -r requirements.txt`.

## Run the UI on Laptop 2

From the repository root, run:

```text
python -m streamlit run ui/app.py
```

In the sidebar, enter Laptop 1's API address in the form `http://<laptop-1-ip>:8000` and click **Check connection**. The default is `http://localhost:8000`, or the value of `PATHSNAP_API_URL` if set. Both laptops need to be on the same reachable network and Laptop 1 must be serving FastAPI.

Select a trip from `GET /trips` and click **Predict**. The UI sends its ID to `POST /predict`, then draws the GPS observations and predicted route and shows the server's scores. Use **Refresh trips** if the server's trip list changes.

## Reproduce the Laptop 2 analysis

Use the student dataset directory as `--data-root` if it differs from the shared loader default:

```text
python -m pytest -q
python -m visualisations.eda_plots --data-root "<student-data-root>"
python -m visualisations.plot_trip --split train --trip "<trip-id>" --data-root "<student-data-root>"
```

EDA writes inventory, quality, network and GPS-distance tables to `results/eda/` and PNGs to `results/figures/`. The notebook `notebooks/01_eda.ipynb` uses the same analysis. The Folium command writes a local HTML map to `visualisations/maps/`.

## Train after the Laptop 1 feature handoff

Place the Laptop 1 outputs `train_features.parquet` and `student_test_features.parquet` under `data/processed/`. The runner also needs the development road graph for route continuity.

```text
python -m models.train_logreg --data-root "<student-data-root>"
python -m models.train_baselines --data-root "<student-data-root>"
```

The second command runs all five baselines, including logistic regression. Use `--model` with `logreg`, `knn`, `rf`, `lgbm`, or `mlp` to run a single model. Each run saves candidate weights and feature order under `artifacts/candidates/`, out-of-fold probabilities under `results/oof/`, and metrics in `results/experiments.csv`. Final model selection and Bayesian tuning follow the real development runs.

See [AI_USAGE.md](AI_USAGE.md) for the Phase 1 assistance record.

## Submission checklist

- [ ] Commit the code and dependencies for preprocessing, candidate generation, feature engineering, training, evaluation, FastAPI, the UI, and deployment.
- [ ] Train and compare at least five distinct models. Retain configurations, evaluation outputs, and timings.
- [ ] Record the Bayesian hyperparameter search and compare the tuned and untuned model.
- [ ] Implement HMM/Viterbi and report a comparison against classifier-only predictions.
- [ ] Include the final model weights and required inference configuration, subject to repository size constraints.
- [ ] Commit Folium map-generation code and supporting experiment results.
- [ ] Document installation, execution, and the two-laptop deployment. Verify live requests and capture UI screenshots and Laptop 1 API responses.
- [ ] Submit the public GitHub repository and one consolidated technical PDF with results from the team's actual experiments.
