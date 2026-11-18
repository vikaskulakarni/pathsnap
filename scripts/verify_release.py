"""Check the deployed artifact set and its recorded student-test rehearsal without raw GPS."""
import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FILES = ("model.joblib", "model_alt.joblib", "feature_list.json", "best_params.json", "hmm_config.json")


def fingerprint(directory=ARTIFACTS):
    hashes = {}
    for name in FILES:
        source = ARTIFACTS if name == "hmm_config.json" else directory
        with (source / name).open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            hashes[name] = digest.hexdigest()
    return hashes


def record():
    """Record a deliberate release after checking that both exported estimators load."""
    from features.build_features import FEATURE_COLUMNS
    from hmm.classifier import inference_directory, load_artifacts
    directory = inference_directory()
    for alt in (False, True):
        model, columns, _ = load_artifacts(alt=alt, directory=directory)
        if list(columns) != list(FEATURE_COLUMNS) or model.n_features_in_ != len(columns):
            raise ValueError("An exported model disagrees with the feature builder.")
    manifest = {"scikit_learn": version("scikit-learn"), "files_sha256": fingerprint(directory)}
    (directory / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate(*, runtime_only=False):
    from hmm.classifier import inference_directory
    directory = inference_directory()
    manifest = json.loads((directory / "release.json").read_text())
    if version("scikit-learn") != manifest["scikit_learn"]:
        raise ValueError(f"Install scikit-learn {manifest['scikit_learn']} before starting the API.")
    if fingerprint(directory) != manifest["files_sha256"]:
        raise ValueError("Deployed model or configuration differs from the recorded release.")
    from hmm.pipeline import Predictor
    from features.build_features import FEATURE_COLUMNS
    predictor = Predictor()
    if list(predictor.features) != list(FEATURE_COLUMNS) or predictor.model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("Saved feature schema disagrees with the deployed estimator.")
    if runtime_only:
        print(f"Verified deployed model {predictor.version} and its feature schema.")
        return
    rehearsal = ROOT / ("results/all80/rehearsal.csv" if directory != ARTIFACTS else "report/rehearsal.csv")
    with rehearsal.open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    if len(runs) != 10 or len({r["trip_id"] for r in runs}) != 10:
        raise ValueError("Expected ten distinct student-test predictions.")
    with (ROOT / "results/eda/inventory_summary.csv").open(newline="") as stream:
        student = next(row for row in csv.DictReader(stream) if row["split"] == "student_test")
    if sum(int(r["n_points"]) for r in runs) != int(student["sum"]):
        raise ValueError('The rehearsal does not cover all student-test observations.')
    if any(r["model_version"] != predictor.version for r in runs):
        raise ValueError("The recorded rehearsal used a different model version.")
    print(f"Verified {len(FILES)} artifact hashes, {len(FEATURE_COLUMNS)} features, and ten rehearsal trips for {predictor.version}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-only', action='store_true')
    parser.add_argument('--record', action='store_true', help='Record hashes after an intentional model export.')
    args = parser.parse_args()
    if args.record:
        record()
    else:
        validate(runtime_only=args.runtime_only)
