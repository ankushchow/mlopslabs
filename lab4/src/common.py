"""Shared data, metrics, and local tracking configuration."""

import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT = "lab4-auto-mpg-gradient-boosting"


def configure_tracking(root=LAB_DIR):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return root


FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
DATA_FILE = LAB_DIR / "data" / "auto-mpg.data"


def load_data(path=DATA_FILE):
    columns = ["mpg", *FEATURES, "origin", "car_name"]
    raw = pd.read_csv(path, sep=r"\s+", names=columns, na_values="?")
    clean = raw.dropna(subset=["mpg", *FEATURES])
    X, y = clean[FEATURES].astype(float), clean["mpg"].astype(float)
    if len(clean) < 100 or not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("Dataset needs at least 100 complete, finite numeric rows")
    report = {
        "dataset": "UCI Auto MPG", "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "source_rows": len(raw), "usable_rows": len(clean), "dropped_rows": len(raw) - len(clean),
        "dropped_row_ids": raw.index.difference(clean.index).tolist(),
        "features": FEATURES, "target": "mpg", "excluded_columns": ["origin", "car_name"],
    }
    return X, y, report


def load_splits():
    X, y, _ = load_data()
    train_val, test = train_test_split(X.index.to_numpy(), test_size=0.2, random_state=42)
    train, validation = train_test_split(train_val, test_size=0.25, random_state=43)
    indices = {"train": train, "validation": validation, "test": test}
    return {name: (X.loc[idx], y.loc[idx]) for name, idx in indices.items()}, {
        name: idx.tolist() for name, idx in indices.items()
    }


def regression_metrics(actual, predicted):
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
