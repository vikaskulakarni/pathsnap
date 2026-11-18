# Laptop 1 integration handoff

The `feat/train-utils` branch combines the Laptop 1 feature pipeline from `b61f86a` with Laptop 2's evaluator, EDA and training code. Both laptops can work from this revision. The first model for decoder development is a logistic-regression baseline fitted on all 80 training trips. Final model selection and tuning are still open.

## Run a real inference check

Install the repository requirements. The starter model uses scikit-learn 1.7.0. From the repository root, pass a local development trip directory to the smoke check:

```text
python -m models.check_handoff data/student_test/<trip-directory>
```

The check loads the published model, builds candidates directly from the track and road network, and computes one positive-class probability per candidate. Use `artifacts/starter/logreg.joblib` and `artifacts/starter/feature_list.json` together. The fitted pipeline includes scaling. Preserve the correspondence between probability and `(trip_id, point_idx, arc_id)` when grouping for Viterbi. To regenerate training-only out-of-fold probabilities locally, run `python -m features.make_tables --splits train student_test` and `python -m models.train_logreg` with the development data installed as described in the README. The output `results/oof/logreg.parquet` contains those three keys and `proba`.

## Connect the evaluator

- Return `{"status":"ok"}` from `GET /health` and an array of trip-ID strings from `GET /trips`. Preserve leading zeros.
- Accept `{"trip_id":"<selected id>"}` at `POST /predict`. Return the same `trip_id`, `points` as objects with numeric `lon` and `lat`, `route_coords` as ordered longitude/latitude pairs, integer `n_points` matching the point count, string `model_version`, and nonnegative `latency_ms`. Return `accuracy` and `route_continuity` as fractions in [0,1] or JSON null when unavailable. Score on Laptop 1.
- Allow the Laptop 2 browser origin (default `http://127.0.0.1:8501`), GET, POST and the JSON Content-Type preflight. Run `python -m ui.app` on Laptop 2 and enter Laptop 1's LAN address.
- For the report, export student-test predictions to `results/predictions/<trip_id>.csv` with columns `point_idx,classifier_arc,hmm_arc`. Laptop 2's plotting command joins these to the local trip by zero-based observation and arc identifiers.

## Available evidence

See [baseline metrics](results/experiments.csv), [feature build statistics](results/eda/feature_build_stats.csv), [EDA figures](results/figures), [report generator](report/assemble.py), and [UI preview](report/screenshots/ui-preview.png). Generate interactive training-trip maps locally with the README commands. The final report still needs the five-model comparison, tuning results, HMM comparison and live two-laptop screenshots.
