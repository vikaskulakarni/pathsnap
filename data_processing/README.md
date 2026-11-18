# Data processing notes

## Data layout
Trips live at `data/<split>/<trip_id>/` (no `data/raw/` folder). Splits: `train` (80 trips),
`student_test` (10); `eval_test` is held out and not present locally.
Each trip folder has `<id>.track` (lon, lat, t), `<id>.nodes` (lon, lat; node_id = row index),
`<id>.arcs` (from_node, to_node; arc_id = row index) and `<id>.route` (arc_id per line).
All files are whitespace-separated with no header. Raw data is git-ignored.

## Network check (L1-3)
Run: `python -m data_processing.inspect_network` (train + student_test only).
Output: `results/eda/network_check.txt`.

- **Shared or per-trip?** Per-trip. 90 trips give 90 distinct `.nodes` files and 90 distinct
  `.arcs` files. `arc_id` and `node_id` are only meaningful within one trip, so the road graph
  and candidate index must be built per trip.
- **One-way or two-way?** Mostly two-way. 92.3% of arcs (3,747,676 of 4,060,042) have a reverse
  arc (to -> from) already present in `.arcs`. The remaining ~8% are one-way, so the reverse arc
  does not have to be added artificially.
- **Route validity:** all 34,813 route arc ids exist in `.arcs`.
- **Route continuity:** 99.96% of consecutive route arcs connect (to_node of one equals
  from_node of the next): 34,709 of 34,723. The 14 breaks mean the HMM must allow
  disconnected transitions (epsilon) rather than forbid them.

## Per-point labels (`label_points.py`)
Rule: only arcs of the trip's own `.route` are allowed, and the route index j assigned to
point i+1 must be >= the one for point i (stay or move forward, skipping is allowed, never
go back). Arcs are straight segments between their two nodes, distances are in UTM metres.
DP: let d[i][j] = distance from point i to route arc j, and C[i][j] the cheapest total cost
of labelling points 0..i with point i on route index j. Then
`C[0][j] = d[0][j]` and `C[i][j] = d[i][j] + min_{k<=j} C[i-1][k]`.
The inner min is a running prefix-min (`np.minimum.accumulate`) with its argmin kept as a
backpointer, so each step is O(R) and the whole trip is O(T*R) time and memory
(T points, R route arcs). The label path is read back from the best C[T-1][j].
Why not greedy nearest arc: one noisy point near a later arc would drag the index forward
and every later point would be forced onto it, which costs far more than keeping it behind.
Quality: `python -m data_processing.label_points --splits train student_test` writes
`results/eda/label_quality.csv` (mean and p95 distance per trip) and flags p95 > 30 m.
