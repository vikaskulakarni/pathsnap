from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Raw trips live directly under data/<split>/<trip_id>/ (there is no data/raw/ folder).
DATA_RAW = ROOT / "data"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts"

SPLITS = ("train", "student_test", "eval_test")

SEED = 42
K = 8
RADIUS_M = 60.0
