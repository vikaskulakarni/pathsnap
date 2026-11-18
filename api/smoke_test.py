"""Crash test of /predict on every evaluation trip: status 200 and one arc per GPS point.
Prints latency only. Accuracies are deliberately NOT printed (never tune on eval trips).

Run: python -m api.smoke_test --url http://localhost:8000
"""
import argparse
import sys

import requests

from api.main import eval_dir
from common.io import load_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    url = ap.parse_args().url.rstrip("/")
    trips = requests.get(f"{url}/trips", timeout=30).json()
    if not trips:
        sys.exit("no trips listed by /trips (is the evaluation folder present on the server?)")
    failures = 0
    for tid in trips:
        r = requests.post(f"{url}/predict", json={"trip_id": tid}, timeout=300)
        ok = r.status_code == 200
        if ok:
            body = r.json()
            ok = len(body["points"]) == body["n_points"] == len(load_track(eval_dir() / tid))
        failures += not ok
        ms = r.json().get("latency_ms") if r.status_code == 200 else None
        print(f"{tid}: {'OK' if ok else 'FAIL'} status={r.status_code} "
              f"latency={'-' if ms is None else f'{ms:.0f} ms'}")
    print(f"\n{len(trips) - failures}/{len(trips)} trips passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
