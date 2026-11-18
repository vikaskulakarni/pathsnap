# Legacy phase-1 pipeline (superseded)

An earlier, separate implementation (`src/`, `run_pipeline.py`, its models, plots and report script).
It is kept for reference only and is **not** part of the submitted system:

- its API (`src/api.py`) loads the reference `.route` while predicting and draws it in the map returned to the UI,
  so it must not be used for the evaluation (ground truth has to stay server-side);
- it trained on 25 of the 80 train trips with different features and a different labelling rule, so its numbers are not
  comparable with `results/`;
- `generate_pdf_report.py` contains hard-typed figures. The report must be regenerated from the CSVs in `results/`.

The maintained code is in `common/`, `data_processing/`, `features/`, `models/`, `optimisation/`, `hmm/`, `api/` and `ui/`.
Run the legacy code only from inside `legacy/phase1_pipeline/` (it uses paths relative to that folder).
