"""Network check on train + student_test only (never eval_test).
Run: python -m data_processing.inspect_network"""
import hashlib

from common.config import RESULTS
from common.io import list_trips, load_arcs, load_route, trip_files


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    lines = []
    out = lines.append
    nodes_h, arcs_h = {}, {}
    rev_num = rev_den = 0
    missing = total_route = conn = pairs = 0
    n_trips = 0
    for split in ("train", "student_test"):
        for trip in list_trips(split):
            n_trips += 1
            f = trip_files(trip)
            nodes_h.setdefault(_md5(f["nodes"]), []).append(trip.name)
            arcs_h.setdefault(_md5(f["arcs"]), []).append(trip.name)
            arcs = load_arcs(trip)
            pairs_set = set(zip(arcs.from_node, arcs.to_node))
            rev_num += sum((b, a) in pairs_set for a, b in pairs_set)
            rev_den += len(pairs_set)
            route = load_route(trip)
            if route is None:
                continue
            n = len(arcs)
            total_route += len(route)
            missing += sum(not (0 <= a < n) for a in route)
            valid = [a for a in route if 0 <= a < n]
            fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
            for a, b in zip(valid, valid[1:]):
                pairs += 1
                conn += tn[a] == fn[b]
    out(f"trips checked (train + student_test): {n_trips}")
    out(f"distinct .nodes files: {len(nodes_h)}  distinct .arcs files: {len(arcs_h)}")
    out("road network: " + ("SHARED across trips" if len(arcs_h) == 1 and len(nodes_h) == 1
                             else "PER-TRIP (differs between trips)"))
    out(f"arcs whose reverse (to->from) also exists: {rev_num}/{rev_den} = {rev_num / max(rev_den, 1):.1%}"
        f"  (high => mostly two-way; low => one-way)")
    out(f"route arc_ids not present in .arcs: {missing}/{total_route}")
    out(f"consecutive route arcs that connect (to_node == next from_node): "
        f"{conn}/{pairs} = {conn / max(pairs, 1):.1%}")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "eda").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eda" / "network_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
