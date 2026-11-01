"""File-backed clustering stages; XCom contains paths and small JSON values only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SEED = 42
CANDIDATES = list(range(2, 9))


class QualityGateError(ValueError):
    """A model is not approved for publication."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_directory(root, run_id):
    # Airflow run IDs contain punctuation; hash the entire ID to avoid path traversal.
    directory = Path(root) / hashlib.sha256(run_id.encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_data(source, root, run_id):
    directory = run_directory(root, run_id)
    frame = pd.read_csv(source)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
    clean = numeric.loc[valid].drop_duplicates()
    report = {
        "run_id": run_id, "source_sha256": checksum(source),
        "features": FEATURES, "input_rows": len(frame),
        "invalid_rows": int((~valid).sum()),
        "duplicate_rows": int(valid.sum() - len(clean)),
        "clean_rows": len(clean),
        "invalid_fraction": float((~valid).mean()) if len(frame) else 1.0,
    }
    report["passed"] = bool(len(clean) >= 100 and report["invalid_fraction"] <= 0.05)
    write_json(directory / "data_quality.json", report)
    if not report["passed"]:
        raise ValueError("Data gate failed: need >=100 unique rows and <=5% invalid feature rows")
    clean.to_csv(directory / "clean.csv", index=False)
    return str(directory)


def preprocess(directory):
    directory = Path(directory)
    data = pd.read_csv(directory / "clean.csv")
    train, validation = train_test_split(data, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(train)
    np.savez_compressed(directory / "split.npz", train=scaler.transform(train),
                        validation=scaler.transform(validation),
                        train_indices=train.index.to_numpy(), validation_indices=validation.index.to_numpy())
    joblib.dump(scaler, directory / "scaler.joblib")
    write_json(directory / "split.json", {"seed": SEED, "train_rows": len(train),
               "validation_rows": len(validation), "scaler_fit": "training rows only"})
    return str(directory)


def train_candidate(directory, k):
    directory = Path(directory)
    if k not in CANDIDATES:
        raise ValueError("k must be in 2..8")
    with np.load(directory / "split.npz") as split:
        train, validation = split["train"], split["validation"]
    estimator = GaussianMixture(n_components=k, covariance_type="diag", n_init=10,
                                max_iter=300, reg_covar=1e-6, random_state=SEED).fit(train)
    labels = estimator.predict(validation)
    unique = len(np.unique(labels))
    score = float(silhouette_score(validation, labels, sample_size=min(1500, len(validation)),
                                  random_state=SEED)) if 1 < unique < len(validation) else -1.0
    counts = np.bincount(estimator.predict(train), minlength=k)
    model = Pipeline([("scaler", joblib.load(directory / "scaler.joblib")), ("gmm", estimator)])
    joblib.dump(model, directory / f"candidate_{k}.joblib")
    metrics = {"k": k, "silhouette": score, "training_bic": float(estimator.bic(train)),
               "validation_log_likelihood": float(estimator.score(validation)),
               "converged": bool(estimator.converged_),
               "cluster_sizes": counts.tolist(), "min_cluster_fraction": float(counts.min() / counts.sum())}
    write_json(directory / f"candidate_{k}.json", metrics)
    return metrics


def select_model(directory, candidates):
    candidates = list(candidates)
    if sorted(item["k"] for item in candidates) != CANDIDATES:
        raise ValueError("All seven candidate results are required")
    # Exclude collapsed/tiny clusters before ranking. Threshold is checked downstream.
    eligible = [c for c in candidates if c["converged"] and c["min_cluster_fraction"] >= 0.01
                and np.isfinite(c["silhouette"]) and c["silhouette"] > -1]
    best = max(eligible, key=lambda c: (c["silhouette"], -c["k"])) if eligible else None
    result = {"candidates": candidates, "selected": best,
              "selection_metric": "validation silhouette; ties prefer smaller k",
              "min_cluster_fraction": 0.01}
    write_json(Path(directory) / "selection.json", result)
    return str(directory)


def quality_gate(directory, min_silhouette=0.35):
    directory = Path(directory)
    # Clearing/re-running a task must not leave an old approval after a rejection.
    for name in ["manifest.json", "model.joblib", "predictions.csv"]:
        (directory / name).unlink(missing_ok=True)
    if not np.isfinite(min_silhouette) or not 0 <= min_silhouette <= 1:
        raise ValueError("min_silhouette must be between 0 and 1")
    best = json.loads((directory / "selection.json").read_text())["selected"]
    passed = bool(best and best["silhouette"] >= min_silhouette)
    write_json(directory / "quality_gate.json", {"passed": passed,
               "min_silhouette": min_silhouette, "selected": best})
    if not passed:
        raise QualityGateError(f"No eligible model meets silhouette >= {min_silhouette}")
    return str(directory)


def publish_model(directory):
    directory = Path(directory)
    gate = json.loads((directory / "quality_gate.json").read_text())
    if not gate["passed"]:
        raise QualityGateError("Cannot publish an unapproved model")
    k = gate["selected"]["k"]
    model = joblib.load(directory / f"candidate_{k}.joblib")
    joblib.dump(model, directory / "model.joblib")
    quality = json.loads((directory / "data_quality.json").read_text())
    write_json(directory / "manifest.json", {
        "approved": True, "run_id": quality["run_id"], "selected_k": k,
        "source_sha256": quality["source_sha256"], "features": FEATURES,
        "seed": SEED, "quality_gate": gate, "dataset": "Palmer Penguins",
        "model_type": "GaussianMixture", "covariance_type": "diag", "scaler": "StandardScaler",
        "sha256": {name: checksum(directory / name) for name in
                   ["model.joblib", "selection.json", "data_quality.json", "split.json", "quality_gate.json"]},
    })
    return str(directory)


def predict(directory, source):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["approved"] or checksum(directory / "model.joblib") != manifest["sha256"]["model.joblib"]:
        raise QualityGateError("Model integrity check failed")
    frame = pd.read_csv(source)[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame).all().all() or (frame < 0).any().any():
        raise ValueError("Prediction inputs must be finite nonnegative numbers")
    model = joblib.load(directory / "model.joblib")
    probabilities = model.predict_proba(frame)
    frame["cluster"] = probabilities.argmax(axis=1)
    frame["membership_probability"] = probabilities.max(axis=1)
    frame.to_csv(directory / "predictions.csv", index=False)
    return {"prediction_rows": len(frame), "selected_k": manifest["selected_k"], "directory": str(directory)}
