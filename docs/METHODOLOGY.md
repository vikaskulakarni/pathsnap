# Methodology, design decisions and limitations

This document explains *why* the pipeline looks the way it does. Numbers are not repeated here; they live in `results/` (see the README tables and the notebooks), so they cannot drift from the code.

## 1. Data and preprocessing

| File | Content | Link to the others |
|---|---|---|
| `.track` | `lon lat t` per GPS fix, about 1 Hz | none by row index: a track row has no relation to a row of any other file |
| `.nodes` | `lon lat`, row index = `node_id` | `.arcs` index into it |
| `.arcs` | `from_node to_node`, row index = `arc_id` (directed) | geometry = straight line between its two nodes |
| `.route` | reference `arc_id`s in travel order | used for labels and server-side scoring only |

* **Projection.** Longitude/latitude are angles, so distances are computed in a per-trip UTM zone (zone from the mean longitude, EPSG 326xx/327xx). A trip spans a few kilometres, where UTM distortion is a few centimetres, far below GPS noise.
* **Per-trip networks.** Every trip has its own `.nodes`/`.arcs` (checked in `results/eda/network_check.txt`), so everything that depends on the network (index, graph) is built per trip and `arc_id` is never used as a model input.
* **Directed twins.** Most arcs have a reverse arc with identical geometry (share in `results/eda/network_check.txt`). Distance cannot separate them, so a directed heading feature is required and twins are always kept together as candidates.
* **Quality checks.** `results/eda/data_quality.md` and `quality.csv` record missing/invalid values, time-step irregularities (some very large gaps), network components and GPS-to-road distances.

## 2. Reference labels (from `.route` to one arc per point)

The route says which arcs were travelled, not which arc each point lies on. For point *i* we pick a route index `j_i` that never decreases and minimises the total point-to-arc distance:

```
C[0][j] = d[0][j]
C[i][j] = d[i][j] + min_{k <= j} C[i-1][k]        (running prefix-minimum, O(N*R))
```

Back-pointers recover the path. A greedy nearest-arc rule would let one noisy point pull the index forward and force every later point onto that arc.

**Where it can go wrong:** the track continues after the route ends (the remaining points are pinned to the last arc), very long gaps, roads travelled twice, curved roads approximated by straight arcs. Trips whose labelled arcs are often far from the GPS points are flagged (p95 distance above `P95_FLAG_M` in `label_points.py`) in `results/eda/label_quality.csv`. The reference route is reference information, not verified ground truth.

## 3. Candidate generation

An STRtree over arc LineStrings (metres) returns arcs within `RADIUS_M` of each point; the nearest `K` are kept (both in `common/config.py`), and the reverse twin of every kept arc is added. If nothing is in range the radius is doubled until `K` arcs are found and `fallback=True` is set. The **candidate miss rate** (reference arc absent from the candidates) caps any model's Top-1. `features/candidate_sweep.py` measures it over K x radius; `features/miss_report.py` shows it per trip, which separates real candidate failures from unreliable reference tails.

## 4. Features and why

| Feature | Why |
|---|---|
| `dist_m`, `dist_rank`, `dist_minus_min` | proximity, and how ambiguous the point is relative to the nearest candidate |
| `heading_diff` (0 to 180 deg) | GPS heading (previous to next point) vs the *directed* arc bearing; the only feature that separates an arc from its reverse twin. It is not folded to 0-90 for that reason |
| `low_speed` | below `LOW_SPEED_MPS` (`features/build_features.py`) the heading is noise; then `heading_diff` is set to a neutral 90 and the flag is raised |
| `speed_mps`, `dt_s` | movement and gap context |
| `arc_len_m`, `proj_t` | arc length and where along the arc the point projects |
| `has_twin`, `fallback` | structural flags |
| `connects_prev_best` | the arc equals or directly follows the previous point's nearest correctly-headed arc: connectivity information for the classifier |

Identifiers and location (`trip_id`, `point_idx`, `t`, `lon`, `lat`, `arc_id`) are kept for joins and maps and are never model inputs, so a model cannot memorise places or trips. The feature builder never reads labels; labels are attached by a separate function.

## 5. Models, selection and tuning

Five families are fitted on all 80 train trips with `GroupKFold(N_FOLDS)` grouped by trip (rows of one trip are almost identical, so a random split would leak): logistic regression, Gaussian naive Bayes, decision tree, random forest, histogram gradient boosting. Metrics: observation-level Top-1 (selection metric: it measures the matching objective), macro F1, precision, recall, log loss, route continuity, training and inference time. Out-of-fold probabilities are stored so the decoder can be tuned on probabilities that are not over-confident.

Bayesian optimisation (Optuna TPE, `SEED` from config) tunes the best model on a seeded sample of train trips (`SEARCH_TRIPS` in `optimisation/bayes_tune.py`) to keep trials cheap; the final tuned-vs-untuned comparison uses the full protocol (all 80 trips, the same folds as `models.train_models`); the tuned and untuned models are compared on the same folds and on student_test, and the deployed model is chosen by cross-validation, not by student_test. A second model (`model_alt`) is kept for the decoder comparison.

Training/validation loss curves are not drawn: none of these models, as configured, produces an epoch-wise history.

## 6. HMM and Viterbi (the six report questions)

1. **Hidden states.** At GPS point *t* the state is one of its candidate arcs (K_t states per step), so the candidate-to-arc correspondence is preserved.
2. **Emission.** `w * log(clip(p_classifier, floor)) + (1 - w) * log N(dist | 0, sigma)`: the classifier probability mixed with a Gaussian on the GPS-to-arc distance; `w = 0` is the distance-only HMM.
3. **Transition.** From the previous arc to the current one on the *directed* road graph: same arc, successor (1 hop), 2 hops, 3 hops, U-turn to the reverse twin (rare), otherwise epsilon. The values are probabilities stored in `artifacts/hmm_config.json` and tuned on train out-of-fold probabilities.
4. **Missing candidates and large gaps.** A point with no candidates takes the previous decoded arc. If the time step exceeds `gap_s`, `gap_extra_hops` more hops are accepted (scored like 3 hops).
5. **Disconnected segments.** Unconnected pairs get a small but finite epsilon, never zero, so a path always exists and Viterbi cannot die. The share of chosen transitions that fell back to epsilon is reported (`eps_transition_share` in `results/hmm_results.csv`).
6. **Did the HMM help?** See `results/hmm_results.csv`, `hmm_per_trip.csv`, `hmm_failures.csv` and notebook 05: classifier-only, a tuned distance-only HMM and classifier + HMM are compared with Top-1, macro F1, route continuity, wrong-direction rate and the share of trips with at least 90% of points correct.

Viterbi recurrence (log space, O(T x K^2)):

```
delta_0(j) = e_0(j)
delta_t(j) = e_t(j) + max_i [ delta_{t-1}(i) + A_t(i, j) ]      back-pointer: argmax_i
path       = back-track from argmax_j delta_T(j)
```

Log space turns products of thousands of small probabilities into sums and avoids underflow.

## 7. API and leakage controls

* Model, feature list and decoder config are loaded once at start-up.
* `POST /predict` builds features from `.track/.nodes/.arcs` only, scores candidates, decodes, and returns one arc per GPS point. Only afterwards does `api/evaluation.py` read the held-out route (through the labelling module) and return two numbers.
* `tests/test_pipeline.py` proves prediction works on a trip folder without a `.route` file; `tests/test_api.py` and `tests/test_ui_contract.py` lock the response shape.

## 8. Decisions backed by experiments

| Question | Evidence |
|---|---|
| Exclude the unreliable-label trips from training? | `results/ablation_flagged_trips.csv` compares training with and without them on student_test; the protocol uses all 80 trips and the flagged trips are only reported as this ablation |
| Is a heading term useful in the HMM emission? | `results/hmm_tuning_distance.csv`: tuned on train; see the heading rows |
| K and radius | `results/eda/candidate_sweep.csv` and `candidate_miss_by_trip.csv` |
| Which features matter | `results/feature_importance.csv` (permutation importance) |

## 9. Known limitations

* Reference tails: when the track outlives the route, labels for the tail are unreliable (flagged trips).
* Straight-line arcs approximate curved roads.
* Transition probabilities are tuned, not learned from data.
* `eval_test` was never available during development; the final run on it is the first genuinely unseen check.
