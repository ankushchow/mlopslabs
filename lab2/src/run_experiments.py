"""Run the three predefined experiments and keep a log for every attempt."""
import argparse
import json
import subprocess
import sys

from src.common import parser, version_arg, write_json
from src.evaluate_model import threshold_arg
from src.train_model import EXPERIMENTS


def main():
    cli = parser("Compare three Random Forest depths on the same Wine splits")
    cli.add_argument("--min-f1", type=threshold_arg, default=0.85)
    args = cli.parse_args()
    try:
        versions = {name: version_arg(f"{args.version}-{name}") for name in EXPERIMENTS}
    except argparse.ArgumentTypeError as error:
        cli.error(str(error))
    batch = args.output_dir / args.version
    batch.mkdir(parents=True, exist_ok=False)
    summary = {
        "batch": args.version,
        "comparison": "Only max_depth changes; data, splits, tree count, seed, and calibration stay fixed.",
        "minimum_macro_f1": args.min_f1,
        "experiments": [],
    }
    failed = False
    for name, config in EXPERIMENTS.items():
        version = versions[name]
        run = batch / version
        log_path = batch / f"{name}.log"
        stages = [
            ("train_model", ["--experiment", name]),
            ("calibrate_model", []),
            ("evaluate_model", ["--min-f1", str(args.min_f1)]),
        ]
        with log_path.open("w") as log:
            log.write(config["description"] + "\n")
            for stage, extra in stages:
                command = [sys.executable, "-m", f"src.{stage}", "--version", version,
                           "--output-dir", str(batch), *extra]
                result = subprocess.run(command, capture_output=True, text=True)
                log.write(f"\nStage: {stage}\nArguments: {extra}\n")
                log.write(result.stdout + result.stderr)
                log.write(f"Exit code: {result.returncode}\n")
                print(result.stdout, end="")
                if result.returncode != 0:
                    failed = True
                    record_path = run / "experiment.json"
                    record = json.loads(record_path.read_text()) if record_path.exists() else {
                        "version": version, "experiment": name,
                        "description": config["description"],
                    }
                    if record.get("status") != "rejected":
                        record["status"] = "error"
                    record["failed_stage"] = stage
                    record["error"] = result.stderr.strip() or result.stdout.strip()
                    run.mkdir(parents=True, exist_ok=True)
                    write_json(record_path, record)
                    print(f"{name}: {record['status']} (see {log_path})")
                    break
        record = json.loads((run / "experiment.json").read_text())
        summary["experiments"].append(record)
        # Save after each experiment, so earlier results survive a later failure.
        write_json(batch / "experiments.json", summary)
    print(f"Experiment descriptions and results: {batch / 'experiments.json'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
