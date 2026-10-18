import argparse
import math
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score

from src.common import parser, sha256, write_json


def threshold_arg(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("F1 threshold must be finite and between 0 and 1")
    return number


def quality_passes(score, threshold):
    return math.isfinite(score) and math.isfinite(threshold) and 0 <= threshold <= score <= 1


def main():
    cli = parser("Evaluate held-out data and fail the pipeline if quality is too low")
    cli.add_argument("--min-f1", type=threshold_arg, default=0.85)
    args = cli.parse_args()
    run = args.output_dir / args.version
    # A re-evaluation failure must not leave a stale approval manifest behind.
    (run / "manifest.json").unlink(missing_ok=True)
    metrics = {}
    with np.load(run / "heldout.npz") as data:
        for name in ("model", "calibrated_model"):
            model = joblib.load(run / f"{name}.joblib")
            prediction = model.predict(data["X_test"])
            probability = model.predict_proba(data["X_test"])[:, 1]
            metrics[name] = {
                "f1": float(f1_score(data["y_test"], prediction)),
                "accuracy": float(accuracy_score(data["y_test"], prediction)),
                "brier_score": float(brier_score_loss(data["y_test"], probability)),
            }
    score = metrics["calibrated_model"]["f1"]
    passed = quality_passes(score, args.min_f1)
    report = {"version": args.version, "test_rows": 400, "metrics": metrics,
              "gate": {"metric": "calibrated_model.f1", "minimum": args.min_f1, "passed": passed}}
    write_json(run / "metrics.json", report)
    status = "PASS" if passed else "FAIL"
    summary = f"{status}: calibrated F1={score:.4f}, required >= {args.min_f1:.4f}"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(f"### Lab 2 — {args.version}\n\n{summary}\n\n")
            stream.write("| Model | F1 | Accuracy | Brier score (lower is better) |\n|---|---:|---:|---:|\n")
            for name, values in metrics.items():
                stream.write(f"| {name} | {values['f1']:.4f} | {values['accuracy']:.4f} | {values['brier_score']:.4f} |\n")
    if not passed:
        raise SystemExit(1)
    files = ["model.joblib", "calibrated_model.joblib", "training.json", "metrics.json"]
    write_json(run / "manifest.json", {
        "version": args.version, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA", "local-uncommitted"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
        "approved": True, "sha256": {name: sha256(run / name) for name in files},
    })


if __name__ == "__main__":
    main()
