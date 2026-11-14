"""Compare three models, track them in MLflow, and evaluate the selected model."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .common import LAB_DIR, configure_tracking, load_data, load_splits, regression_metrics, write_json


def select_best(candidates):
    if not candidates or any(not np.isfinite(c["validation"]["rmse"]) for c in candidates):
        raise ValueError("Selection requires candidates with finite validation RMSE")
    # Prefer the shallower model if validation scores tie.
    return min(candidates, key=lambda c: (c["validation"]["rmse"], c["max_depth"]))


def run_experiments(batch, root=LAB_DIR):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch):
        raise ValueError("Batch must be 1-80 letters, digits, underscores or hyphens")
    root = Path(root).resolve()
    output = root / "outputs" / batch
    output.mkdir(parents=True, exist_ok=False)
    configure_tracking(root)
    splits, indices = load_splits()
    _, _, data_report = load_data()
    write_json(output / "data_quality.json", data_report)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]
    test_x, test_y = splits["test"]
    write_json(output / "split.json", indices)
    versions = {name: version(name) for name in ("mlflow", "scikit-learn", "numpy", "pandas")}
    candidates, fitted = [], {}
    # Autolog parameters/training metrics; explicitly name validation/test metrics
    # and log one model with its input example and signature per run.
    mlflow.sklearn.autolog(log_models=False, log_datasets=False, log_post_training_metrics=False)
    try:
        for depth in (1, 2, 3):
            description = f"Compare depth {depth}; all other settings and data stay fixed."
            with mlflow.start_run(run_name=f"{batch}-depth-{depth}", tags={
                "batch": batch, "dataset": "uci-auto-mpg", "selected": "false",
                "mlflow.note.content": description,
            }) as run:
                model = GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=depth, random_state=42,
                )
                model.fit(train_x, train_y)
                metrics = regression_metrics(val_y, model.predict(val_x))
                mlflow.log_params({"train_rows": len(train_x), "validation_rows": len(val_x),
                                   "test_rows": len(test_x), "split_seeds": "42,43",
                                   "preprocessing": "drop missing required values; six raw numeric features; no scaling"})
                mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
                mlflow.log_artifact(str(output / "split.json"))
                mlflow.log_artifact(str(output / "data_quality.json"))
                mlflow.log_dict(versions, "versions.json")
                example = train_x.head(5)
                info = mlflow.sklearn.log_model(
                    model, name="model", input_example=example,
                    signature=infer_signature(example, model.predict(example)),
                    pip_requirements=[f"{name}=={versions[name]}" for name in versions],
                )
                candidate = {"name": f"depth-{depth}", "description": description,
                             "max_depth": depth, "run_id": run.info.run_id,
                             "model_uri": info.model_uri, "validation": metrics}
                candidates.append(candidate)
                fitted[depth] = model
                print(f"depth-{depth}: validation RMSE={metrics['rmse']:.4f}")

        selected = select_best(candidates)
        model = fitted[selected["max_depth"]]
        predictions = model.predict(test_x)
        test_metrics = regression_metrics(test_y, predictions)
        baseline = regression_metrics(test_y, np.full(len(test_y), train_y.mean()))
        reloaded = mlflow.pyfunc.load_model(selected["model_uri"])
        np.testing.assert_allclose(reloaded.predict(test_x), predictions, rtol=1e-12, atol=1e-12)
        summary = {
            "batch": batch, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "UCI Auto MPG", "data_quality": data_report, "versions": versions,
            "split_sizes": {key: len(value) for key, value in indices.items()},
            "settings": {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42},
            "selection_metric": "lowest validation RMSE; shallower model breaks ties",
            "candidates": candidates, "selected": selected, "test": test_metrics,
            "mean_baseline_test": baseline, "reload_matches": True,
        }
        frame = pd.DataFrame({"row_id": test_x.index, "actual": test_y.to_numpy(),
                              "prediction": predictions, "residual": test_y.to_numpy() - predictions})
        frame.to_csv(output / "test_predictions.csv", index=False)
        test_x.head(5).to_csv(output / "sample_input.csv", index=False)
        write_json(output / "serving_input.json", {"dataframe_split": {
            "columns": list(test_x.columns), "data": test_x.head(5).values.tolist()}})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(test_y, predictions, alpha=0.7, s=22)
        low, high = min(test_y.min(), predictions.min()), max(test_y.max(), predictions.max())
        ax.plot([low, high], [low, high], "--", color="gray")
        ax.set(xlabel="Actual fuel economy (mpg)", ylabel="Predicted fuel economy (mpg)", title=f"Selected {selected['name']}: held-out test set")
        fig.tight_layout()
        fig.savefig(output / "actual_vs_predicted.png", dpi=160)
        plt.close(fig)
        write_json(output / "experiments.json", summary)
        with mlflow.start_run(run_id=selected["run_id"]):
            mlflow.set_tag("selected", "true")
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            mlflow.log_metrics({f"mean_baseline_test_{key}": value for key, value in baseline.items()})
            mlflow.log_metric("reload_matches", 1.0)
            for name in ("experiments.json", "test_predictions.csv", "actual_vs_predicted.png",
                         "sample_input.csv", "serving_input.json"):
                mlflow.log_artifact(str(output / name))
        print(f"Selected {selected['name']}; test RMSE={test_metrics['rmse']:.4f}; "
              f"mean baseline RMSE={baseline['rmse']:.4f}; reload matched.")
        print(f"Results: {output}")
        return summary
    finally:
        mlflow.sklearn.autolog(disable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="New output name, e.g. auto-mpg-comparison")
    args = parser.parse_args()
    run_experiments(args.batch)
