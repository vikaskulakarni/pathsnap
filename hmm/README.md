# HMM route decoding (distance-only emissions, stage 6)

Hidden state at GPS point t = one candidate arc of that point (so K_t states per step; the
candidate-to-arc correspondence is kept). Each trip has its own directed road network, and
two-way roads already exist as twin arcs, so `RoadGraph` is used exactly as given.

Viterbi (`hmm/viterbi.py`, log space, back-pointers, O(T*K^2)):
`delta_0(j) = e_0(j)`, `delta_t(j) = e_t(j) + max_i [delta_{t-1}(i) + A_t(i,j)]`, then back-track.
Log space avoids underflow on thousands of points.

Emission (`emission.py`): Gaussian on the GPS-to-arc distance. The `distance+heading` variant adds a
Gaussian penalty on `heading_diff` (GPS heading vs the directed arc bearing) so the correctly directed twin can win.

Transition (`transition.py`): by hops from the previous arc to the current arc: same, U-turn to the twin,
successor (1 hop), 2 hops, 3 hops, otherwise `eps` (never -inf, so the path always survives).
After a time gap above `gap_s` the vehicle may have moved further, so `gap_extra_hops` more hops are accepted.
A point with no candidates takes the previous decoded arc, so every GPS point gets exactly one arc.

## Initial parameters (`artifacts/hmm_config.json`)
These are reasoned starting values, NOT tuned (tuning on train, then confirmed on student_test, is the next stage).

| value | meaning / why |
|---|---|
| `sigma_m` 8.0 | emission std in metres; about the median GPS-to-reference-arc distance seen in `results/eda/label_quality.csv` |
| `heading_sigma_deg` 45 | std of the heading mismatch (heading variant only); tolerant of heading noise and curved roads |
| `heading_weight` 1.0 | weight of the heading term (heading variant only) |
| `same` 0.60 | staying on the same arc is the most likely step at 1 Hz (arcs are tens to hundreds of metres) |
| `succ` 0.30 | moving to a directed successor arc |
| `hop2` 0.05, `hop3` 0.02 | skipping one or two arcs between consecutive points (short arcs, faster driving) |
| `uturn` 0.001 | jumping to the reverse twin: possible but rare, penalised |
| `eps` 1e-6 | anything else (disconnected or far arc): tiny but finite, so a path always exists |
| `gap_s` 10 | a time step above 10 s counts as a gap |
| `gap_extra_hops` 3 | extra hops (up to 6) accepted after a gap; scored like 3 hops |

## Evaluate
`python -m hmm.evaluate` compares nearest-arc, the distance-only HMM and the distance+heading HMM on student_test
against `label_points` labels, and appends rows to `results/hmm_results.csv`.
