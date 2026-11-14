"""Load the selected MLflow model for CSV inference or local HTTP serving."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd

from .common import configure_tracking


def predict(summary_path, input_path, output_path):
    configure_tracking()
    summary = json.loads(summary_path.read_text())
    data = pd.read_csv(input_path)
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Input must contain finite numeric rows")
    model = mlflow.pyfunc.load_model(summary["selected"]["model_uri"])
    data["prediction"] = model.predict(data)
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    if args.serve:
        configure_tracking()
        summary = json.loads(args.summary.read_text())
        raise SystemExit(subprocess.call([
            sys.executable, "-m", "mlflow", "models", "serve", "-m", summary["selected"]["model_uri"],
            "--env-manager", "local", "--host", "127.0.0.1", "--port", str(args.port),
        ], env={**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}))
    if args.input is None or args.output is None:
        parser.error("CSV prediction requires --input and --output")
    predict(args.summary, args.input, args.output)


if __name__ == "__main__":
    main()
