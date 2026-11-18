# PathSnap

PathSnap is a machine learning hackathon project for matching noisy GPS observations to a road network and reconstructing a vehicle's route.

## Two-laptop deployment

- **Laptop 1** runs FastAPI, loads the trained classifier and road network, reconstructs the route with HMM/Viterbi, and computes evaluation scores on the server.
- **Laptop 2** runs the evaluator UI. It lists the ten reserved trips at evaluation time, sends the selected trip ID to Laptop 1, and displays the predicted route and results.

Evaluator reference routes stay on Laptop 1 with the scorer. The prediction pipeline receives GPS observations and road-network data. The UI receives predictions and aggregate results.

## Repository layout

- `common/`, `data_processing/`, `features/`, `hmm/`, and `api/` are Laptop 1's shared utilities, trip processing, candidate features, route decoding, and FastAPI service.
- `models/`, `optimisation/`, `ui/`, `notebooks/`, and `visualisations/` are Laptop 2's training, tuning, web UI, exploration, and Folium map code.
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

From the repository root, run `python -m ui.app` and open `http://127.0.0.1:8501`. The UI uses the Python standard library to serve its HTML, CSS, and JavaScript. It works without an internet connection or a map-service key. The Folium tools remain available for the report maps.

Enter Laptop 1's FastAPI address (for example `http://<laptop-1-ip>:8000`) and choose **Connect & load trips**. The default address can be set with `PATHSNAP_API_URL`. Laptop 1 must allow browser requests from the UI origin, including the POST preflight. Both laptops must be on a reachable network.

Select an API-provided trip and choose **Reconstruct route**. Use the layer controls, drag to pan, or use the zoom and fit buttons to inspect GPS observations and the predicted route. Scores are displayed as percentages and missing scores appear as unavailable. The API response is available below the results.

### Hosting options

Use the local server for the two-laptop evaluation. For a separate hosted preview, the three static assets in `ui/` can also be served by [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) or [Cloudflare Pages](https://developers.cloudflare.com/pages/framework-guides/deploy-anything/). A hosted interface needs an API reachable from the evaluator's browser, an allowed origin, and a compatible HTTPS setup. Public hosting is optional.

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

## Complete the Laptop 2 model and report workflow

After the real feature handoff, run `python -m optimisation.finish_models --data-root "<student-data-root>"`. This command trains all five baselines, compares observation-level Top-1 (ties use CV macro F1, then log loss), exports the two leading candidates, and tunes the leader on training trips with grouped CV macro F1 as the objective. It records Bayesian trials and before/after results, saves the final out-of-fold probabilities, and exports the tuned model, feature order, and best parameters. The defaults are 40 trials and a 30-minute search timeout.

Once Laptop 1 supplies student-test per-observation classifier and HMM predictions under `results/predictions/`, run `python -m visualisations.plot_prediction --data-root "<student-data-root>"` to render comparison maps and static figures.

Run `python -m report.assemble` to generate a printable HTML report draft from available results. It includes source captions and marks missing experiments. Add the real Folium view, evaluator UI screenshot, and receiving-terminal screenshot as `report/screenshots/folium.png`, `ui.png`, and `api.png`. The final PDF needs the team's interpretations for each model, tuning, and decoding, plus the measured results from both laptops.

## Submission checklist

- [ ] Commit the code and dependencies for preprocessing, candidate generation, feature engineering, training, evaluation, FastAPI, the UI, and deployment.
- [ ] Train and compare at least five distinct models. Retain configurations, evaluation outputs, and timings.
- [ ] Record the Bayesian hyperparameter search and compare the tuned and untuned model.
- [ ] Implement HMM/Viterbi and report a comparison against classifier-only predictions.
- [ ] Include the final model weights and required inference configuration, subject to repository size constraints.
- [ ] Commit Folium map-generation code and supporting experiment results.
- [ ] Document installation, execution, and the two-laptop deployment. Verify live requests and capture UI screenshots and Laptop 1 API responses.
- [ ] Submit the public GitHub repository and one consolidated technical PDF with results from the team's actual experiments.
