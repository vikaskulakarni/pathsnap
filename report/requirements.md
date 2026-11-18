# Submission evidence

The organiser's **Deliverables.pdf** calls for a public GitHub repository and one consolidated technical PDF. The accompanying **Models and Training Guidelines.pdf**, **Dataset Exploration.pdf**, **Main Guidelines.pdf** and **Assessment and Plagiarism Policy.pdf** specify how to produce and defend the evidence. The laptop planning documents describe the team's implementation order.

| Requirement | Evidence and reproduction |
|---|---|
| EDA and partitions | `visualisations/eda_plots.py`, `notebooks/01_eda.ipynb` and `results/eda/` cover training and student-test inventory, time gaps, road connectivity, missing values and nearest-road distance. |
| Feature engineering and labels | `data_processing/label_points.py` aligns training references. `features/build_features.py` constructs label-free candidate features used for inference. |
| Five distinct models; fitting and grouped CV on 80 trips | `models/train_models.py --all-trips` (run as a Python module) and `results/all80/model_comparison.csv`. The full run includes the seven quality-flagged trips. The optional `-AllTrips` API switch serves the full-partition checkpoint. Omitting it preserves the previously demonstrated release. |
| Model interpretation and diagnostics | The report gives configurations, objectives, validation and student-test comparisons, fit times, and limitations. `models/fit_diagnostics.py` supplies real fit/OOF diagnostics and student-test confusion counts for all five estimators. |
| Bayesian tuning on training trips | `optimisation/bayes_tune.py --all-trips`, `results/all80/optuna_trials.csv`, `results/all80/run.json` and `results/all80/tuning_comparison.csv` give trials, objective, timings, selected hyperparameters and before/after results. |
| HMM, Viterbi and failure analysis | `hmm/`, `artifacts/hmm_config.json`, `results/hmm_per_trip.csv`, `results/decoder_changes.csv` and `results/all80/hmm_results.csv`. Classifier and sequence decoder are compared on the same student-test observations. |
| Geographical views | `visualisations/plot_prediction.py` and the generated interactive maps compare GPS, road geometry, reference, classifier and HMM layers. The local illustrated submission PDF includes the actual Folium screenshot. |
| Model operations | `artifacts/release.json` fingerprints the deployed weights, feature order and configuration. `scripts.verify_release` verifies them and the rehearsal model version. Experiment tables and trial logs are versioned in Git. The full training run records its Python and library versions, seed, partition counts and input-table hashes in `results/all80/environment.json`. |
| API and UI | `api/main.py`, `api/evaluation.py`, `ui/`, the PowerShell launchers, API and UI-contract tests, and the ten-trip local rehearsal. The API predicts before reading the reference for scoring. |
| Final physical demo | **Pending team evidence.** Photograph or capture the working Laptop 2 UI and the successful request on Laptop 1's console for the same trip. The report builder can include both files. |
| Academic integrity and submission | `AI_USAGE.md` records assistance. Each member must check the report, understand the experiments and be ready to explain the decisions. The team submits the repository URL and the illustrated PDF. |

## Report evidence directory

To reproduce the illustrated PDF, the evidence directory should contain `folium-submission.png` and `ui-submission.png` from the student-test rehearsal. For the official two-device evidence, add `laptop2-ui.png` and `laptop1-terminal.png` together. Rebuild using the command in the README. Keep the development rehearsal and official capture identifiable.
