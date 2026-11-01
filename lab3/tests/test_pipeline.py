import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from clustering import pipeline as p


@pytest.fixture
def prepared(tmp_path):
    values, _ = make_blobs(n_samples=300, centers=3, n_features=len(p.FEATURES), cluster_std=0.1, random_state=42)
    values -= values.min(axis=0)
    source = tmp_path / "data.csv"
    pd.DataFrame(values, columns=p.FEATURES).to_csv(source, index=False)
    directory = p.validate_data(source, tmp_path / "runs", "unit-test")
    return Path(p.preprocess(directory)), source


def test_reject_missing_columns(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text(f"{p.FEATURES[0]}\n1\n")
    with pytest.raises(ValueError, match="Missing required"):
        p.validate_data(source, tmp_path, "bad")


def test_reject_excess_invalid_values(tmp_path):
    source = tmp_path / "bad.csv"
    frame = pd.DataFrame(np.arange(200 * len(p.FEATURES), dtype=float).reshape(200, len(p.FEATURES)), columns=p.FEATURES)
    frame.loc[:20, p.FEATURES[0]] = np.inf
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Data gate failed"):
        p.validate_data(source, tmp_path, "bad")
    report = json.loads((p.run_directory(tmp_path, "bad") / "data_quality.json").read_text())
    assert report["invalid_rows"] == 21 and not report["passed"]


def test_only_required_features_control_row_filter(tmp_path):
    source = tmp_path / "data.csv"
    frame = pd.DataFrame(np.arange(200 * len(p.FEATURES), dtype=float).reshape(200, len(p.FEATURES)), columns=p.FEATURES)
    frame["UNUSED"] = np.nan
    frame.loc[0, p.FEATURES[0]] = -1
    frame.to_csv(source, index=False)
    directory = Path(p.validate_data(source, tmp_path, "test"))
    report = json.loads((directory / "data_quality.json").read_text())
    assert report["clean_rows"] == 199


def test_disjoint_split_and_training_only_scaler(prepared):
    directory, _ = prepared
    data = pd.read_csv(directory / "clean.csv")
    with np.load(directory / "split.npz") as split:
        assert not set(split["train_indices"]) & set(split["validation_indices"])
        assert len(split["train_indices"]) + len(split["validation_indices"]) == len(data)
        scaler = joblib.load(directory / "scaler.joblib")
        np.testing.assert_allclose(scaler.mean_, data.iloc[split["train_indices"]].mean())
        first = split["train"].copy()
    p.preprocess(directory)
    with np.load(directory / "split.npz") as split:
        np.testing.assert_array_equal(first, split["train"])


def test_selected_model_prediction_and_rejection(prepared):
    directory, source = prepared
    candidates = [p.train_candidate(directory, k) for k in p.CANDIDATES]
    p.select_model(directory, candidates)
    p.quality_gate(directory, 0.35)
    p.publish_model(directory)
    result = p.predict(directory, source)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert result["prediction_rows"] == 300
    assert manifest["selected_k"] == 3
    for name, expected in manifest["sha256"].items():
        assert p.checksum(directory / name) == expected
    model = joblib.load(directory / "model.joblib")
    frame = pd.read_csv(source)
    np.testing.assert_array_equal(model.predict(frame), model["gmm"].predict(model["scaler"].transform(frame)))
    assert model["gmm"].n_components == manifest["selected_k"]
    probabilities = model.predict_proba(frame)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    predictions = pd.read_csv(directory / "predictions.csv")
    np.testing.assert_allclose(predictions["membership_probability"], probabilities.max(axis=1))
    assert manifest["model_type"] == "GaussianMixture"
    with pytest.raises(p.QualityGateError):
        p.quality_gate(directory, 1.0)
    assert not (directory / "manifest.json").exists()
    assert not (directory / "model.joblib").exists()
    assert not (directory / "predictions.csv").exists()
    with pytest.raises(p.QualityGateError):
        p.publish_model(directory)


def test_tampered_model_rejected(prepared):
    directory, source = prepared
    p.select_model(directory, [p.train_candidate(directory, k) for k in p.CANDIDATES])
    p.publish_model(p.quality_gate(directory, 0.35))
    with (directory / "model.joblib").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(p.QualityGateError, match="integrity"):
        p.predict(directory, source)


def test_missing_candidate_blocks_selection(prepared):
    with pytest.raises(ValueError, match="seven"):
        p.select_model(prepared[0], [])


def test_run_paths_are_isolated(tmp_path):
    first = p.run_directory(tmp_path, "../../escape")
    second = p.run_directory(tmp_path, "other")
    assert first.parent == tmp_path and first != second


def test_penguin_partition_is_disjoint_and_label_free():
    root = Path(__file__).resolve().parents[1] / "data"
    original = pd.read_csv(root / "penguins_original.csv")
    partition = json.loads((root / "partition.json").read_text())
    training = pd.read_csv(root / "penguins_training.csv")
    prediction = pd.read_csv(root / "penguins_predict.csv")
    train_rows = partition["training_source_rows"]
    predict_rows = partition["prediction_source_rows"]
    assert set(train_rows).isdisjoint(predict_rows)
    assert set(train_rows) | set(predict_rows) == set(range(len(original)))
    assert len(training) == 310 and len(prediction) == 34
    assert list(training.columns) == list(prediction.columns) == p.FEATURES
    assert partition["source_sha256"] == p.checksum(root / "penguins_original.csv")
    pd.testing.assert_frame_equal(training, original.loc[train_rows, p.FEATURES].reset_index(drop=True))
    pd.testing.assert_frame_equal(prediction, original.loc[predict_rows, p.FEATURES].reset_index(drop=True))
    assert np.isfinite(prediction).all().all()


def test_unconverged_candidate_cannot_win(prepared):
    candidates = [{"k": k, "silhouette": 0.5, "converged": False,
                   "min_cluster_fraction": 0.2} for k in p.CANDIDATES]
    candidates[1]["converged"] = True
    candidates[1]["silhouette"] = 0.4
    directory = prepared[0]
    p.select_model(directory, candidates)
    assert json.loads((directory / "selection.json").read_text())["selected"]["k"] == 3
    candidates[1]["converged"] = False
    p.select_model(directory, candidates)
    with pytest.raises(p.QualityGateError):
        p.quality_gate(directory)
