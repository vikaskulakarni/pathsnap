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
