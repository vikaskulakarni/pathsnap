"""Record an API rehearsal over the student_test trips: report/rehearsal.csv and report/rehearsal.log.

Starts the API itself (student_test served in place of eval_test), sends the requests a UI makes
(health, trips, CORS preflight with a UI origin, then POST /predict per trip) over HTTP, writes the
server log and the per-trip scores, and stops the server it started. It is a client on one host, not
a browser: browser screenshots and the physical two-laptop run are captured separately.

Run: python -m report.rehearse
"""
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UI_ORIGIN = "http://127.0.0.1:8501"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = ROOT / "report" / "rehearsal.log"
    env = {**os.environ, "PATHSNAP_EVAL_SPLIT": "student_test", "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                                   "--port", str(port), "--log-level", "warning"],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                try:
                    if requests.get(f"{base}/health", timeout=2).json().get("status") == "ok":
                        break
                except requests.RequestException:
                    time.sleep(1)
            else:
                raise SystemExit("API did not start")
            assert requests.get(f"{base}/trips", timeout=10).status_code == 200
            rows = []
            for trip in requests.get(f"{base}/trips", timeout=10).json():
                pre = requests.options(f"{base}/predict", headers={"Origin": UI_ORIGIN, "Access-Control-Request-Method": "POST",
                                                                   "Access-Control-Request-Headers": "content-type"}, timeout=10)
                assert pre.status_code == 200, pre.status_code
                r = requests.post(f"{base}/predict", json={"trip_id": trip}, headers={"Origin": UI_ORIGIN}, timeout=300)
                r.raise_for_status()
                b = r.json()
                assert len(b["points"]) == b["n_points"]
                rows.append({"trip_id": trip, "n_points": b["n_points"], "accuracy": b["accuracy"],
                             "route_continuity": b["route_continuity"], "latency_ms": b["latency_ms"],
                             "model_version": b["model_version"]})
        finally:
            server.terminate()
            server.wait(timeout=20)
    with (ROOT / "report" / "rehearsal.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trips recorded; model version {rows[0]['model_version']}")


if __name__ == "__main__":
    main()
