import argparse
import json
import math
import os

import joblib
import numpy as np
from sklearn.metrics import f1_score

from src.common import parser, write_json


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
    metrics = {}
    with np.load(run / "heldout.npz") as data:
        test_rows = len(data["y_test"])
        for name in ("model", "calibrated_model"):
            model = joblib.load(run / f"{name}.joblib")
            prediction = model.predict(data["X_test"])
            metrics[name] = {
                "f1": float(f1_score(data["y_test"], prediction, average="macro", zero_division=0)),
            }
    score = metrics["calibrated_model"]["f1"]
    passed = quality_passes(score, args.min_f1)
    report = {"version": args.version, "test_rows": test_rows, "average": "macro", "metrics": metrics,
              "gate": {"metric": "calibrated_model.f1", "minimum": args.min_f1, "passed": passed}}
    write_json(run / "metrics.json", report)
    experiment = json.loads((run / "experiment.json").read_text())
    experiment.update({
        "status": "passed" if passed else "rejected",
        "evaluation": report,
    })
    write_json(run / "experiment.json", experiment)
    status = "PASS" if passed else "FAIL"
    summary = f"{status}: calibrated macro F1={score:.4f}, required >= {args.min_f1:.4f}"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(f"### Lab 2 — {args.version}\n\n{summary}\n\n")
            stream.write("| Model | Macro F1 |\n|---|---:|\n")
            for name, values in metrics.items():
                stream.write(f"| {name} | {values['f1']:.4f} |\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
