# PathSnap — rules for coding agents

## Project
PES UE24CS352A map-matching hackathon. Noisy GPS points are matched to road arcs.
Pipeline: candidate arcs per GPS point -> classifier scores each (point, arc) pair
-> HMM/Viterbi picks a connected route -> FastAPI on laptop 1 serves predictions
to the browser UI on laptop 2.

## Data (data/{train,student_test,eval_test}/<trip>/, NOT data/raw/)
- train = 80 trips, student_test = 10 trips, eval_test = 10 trips (final evaluation only)
- .track : lon lat t   (whitespace, no header, t in seconds, ~1 Hz)
- .nodes : lon lat     (row index, 0-based = node_id)
- .arcs  : from_node to_node (row index, 0-based = arc_id)
- .route : one arc_id per line, in travel order (reference route)
- Row i of .track has NO relation to row i of any other file.

## Hard rules
1. eval_test: only .track/.nodes/.arcs may be read, and only by build_features /
   predict_trip. Its .route may be read ONLY in api/evaluation.py, AFTER prediction.
2. Never hardcode predictions, trip IDs, thresholds tuned on eval_test, or results.
3. Never invent or type metrics. Every number is computed by code and saved under results/.
4. Fit learned preprocessing on train trips only. CV = GroupKFold grouped by trip_id.
5. Distances in metres (pyproj UTM or haversine), never raw degrees.
6. Use SEED from common/config.py everywhere.
7. Do not run git commit, push, merge or rebase. The user handles git.
8. Only edit files in the folders the task names. common/ belongs to laptop 1: ask first.

## Style
Python 3.10+, type hints, short docstrings that explain WHY, pathlib, reusable logic
in .py modules (notebooks only call them), every script runnable as
`python -m package.module`. Figures: PNG in results/figures/, dpi 150, title + axis labels.

## After every task
List the files changed, how to run them, what the user should check, and explain the
design decisions in plain English. The user will be questioned on them.
