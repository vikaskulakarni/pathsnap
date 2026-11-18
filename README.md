# PathSnap

PathSnap is a machine learning hackathon project for matching noisy GPS observations to a road network and reconstructing a vehicle's route.

## Two-laptop deployment

- **Laptop 1** runs FastAPI, loads the trained classifier and road network, reconstructs the route with HMM/Viterbi, and computes evaluation scores on the server.
- **Laptop 2** runs the evaluator UI. It lists the ten reserved trips at evaluation time, sends the selected trip ID to Laptop 1, and displays the predicted route and results.

Evaluator reference routes stay on Laptop 1 with the scorer. The prediction pipeline receives GPS observations and road-network data. The UI receives predictions and aggregate results.

## Planned repository layout

- `api/` for FastAPI endpoints, model loading, and live scoring.
- `ui/` for the evaluator trip picker, map, and results.
- `ml/` for data inspection, preprocessing, features, training, evaluation, Bayesian search, and route decoding.
- `artifacts/` for the final model weights and inference configuration.
- `visualizations/` for Folium map generation and experiment plots.
- `results/` for reproducible experiment logs, configurations, and summaries.
- `docs/` for the API contract and deployment details.

## Setup status

This is the initial repository. Installation, execution, and deployment commands will be added as the implementation lands. Keep the supplied raw trip files local. Use the 80 training trips for fitting and trip-grouped cross-validation, the 10 student-test trips for development comparisons, and the reserved evaluator partition only for final evaluation.

## Submission checklist

- [ ] Commit the code and dependencies for preprocessing, candidate generation, feature engineering, training, evaluation, FastAPI, the UI, and deployment.
- [ ] Train and compare at least five distinct models. Retain configurations, evaluation outputs, and timings.
- [ ] Record the Bayesian hyperparameter search and compare the tuned and untuned model.
- [ ] Implement HMM/Viterbi and report a comparison against classifier-only predictions.
- [ ] Include the final model weights and required inference configuration, subject to repository size constraints.
- [ ] Commit Folium map-generation code and supporting experiment results.
- [ ] Document installation, execution, and the two-laptop deployment. Verify live requests and capture UI screenshots and Laptop 1 API responses.
- [ ] Submit the public GitHub repository and one consolidated technical PDF with results from the team's actual experiments.
