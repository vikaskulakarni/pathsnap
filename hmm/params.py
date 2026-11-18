"""Decoder parameters. The defaults are reasoned starting values (see hmm/README.md),
NOT tuned on any evaluation data; tuning (train, then student_test) comes later."""
import json
from pathlib import Path

from common.config import ARTIFACTS

CONFIG_PATH = ARTIFACTS / "hmm_config.json"
BASELINE_CONFIG_PATH = ARTIFACTS / "hmm_config_distance.json"  # distance-only baseline (w = 0)

DEFAULT_PARAMS = {
    "w": 0.5,                  # emission: weight of the classifier log-prob vs the distance term
    "sigma_m": 8.0,            # emission: GPS noise std in metres
    "heading_sigma_deg": 45.0,  # emission: std of the heading mismatch (heading variant only)
    "heading_weight": 0.0,      # emission: weight of the heading term (distance+heading variant only; 0 = off)
    "same": 0.60,               # transition: stay on the same arc
    "succ": 0.30,               # transition: move to a directed successor arc (1 hop)
    "hop2": 0.05,               # transition: 2 hops ahead
    "hop3": 0.02,               # transition: 3 hops ahead
    "uturn": 0.001,             # transition: jump to the reverse twin arc
    "eps": 1e-6,                # transition: anything else (keeps the path alive)
    "gap_s": 10.0,              # time gap (s) above which more hops are allowed
    "gap_extra_hops": 3,        # extra hops allowed after such a gap
}


def load_params(path: Path = CONFIG_PATH) -> dict:
    return {**DEFAULT_PARAMS, **json.loads(Path(path).read_text())}


def save_params(params: dict, path: Path = CONFIG_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(params, indent=2) + "\n")


if __name__ == "__main__":
    save_params(DEFAULT_PARAMS)
    print(f"saved {CONFIG_PATH}")
