import argparse
import json
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.common import dataset, version_arg
from src.evaluate_model import quality_passes


def test_splits_are_disjoint_and_reproducible():
    X, y, rows = dataset()
    repeated_X, repeated_y, repeated = dataset()
    np.testing.assert_array_equal(X, repeated_X)
    np.testing.assert_array_equal(y, repeated_y)
    assert {k: len(v) for k, v in rows.items()} == {"train": 106, "calibration": 36, "test": 36}
    sets = [set(v) for v in rows.values()]
    assert len(set.union(*sets)) == len(X) == len(y) == 178
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for name in rows:
        assert (rows[name] == repeated[name]).all()


@pytest.mark.parametrize("score,threshold,expected", [
    (0.85, 0.85, True), (0.849, 0.85, False), (float("nan"), 0.85, False),
    (0.9, float("nan"), False), (0.9, -1, False), (1.1, 0.85, False),
])
def test_gate(score, threshold, expected):
    assert quality_passes(score, threshold) is expected


def test_version_cannot_escape_output_folder():
    with pytest.raises(argparse.ArgumentTypeError, match="letters"):
        version_arg("../escape")


def test_pipeline_pass_failure_and_version_collision(tmp_path):
    def run(module, *extra):
        return subprocess.run(
            [sys.executable, "-m", f"src.{module}", "--version", "test-run",
             "--output-dir", str(tmp_path), *extra], capture_output=True, text=True,
        )
    for module in ("train_model", "calibrate_model", "evaluate_model"):
        result = run(module)
        assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "test-run"
    report = json.loads((output / "metrics.json").read_text())
    assert report["gate"]["passed"]
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["status"] == "passed"
    assert experiment["description"]
    assert experiment["preprocessing"]["scaling"]
    assert experiment["evaluation"] == report
    assert report["metrics"]["calibrated_model"]["f1"] >= 0.85
    assert (output / "model.joblib").is_file()
    assert (output / "calibrated_model.joblib").is_file()
    assert report["test_rows"] == 36
    assert report["average"] == "macro"
    with np.load(output / "heldout.npz") as data:
        for name in ("model", "calibrated_model"):
            model = joblib.load(output / f"{name}.joblib")
            prediction = model.predict(data["X_test"])
            probability = model.predict_proba(data["X_test"])
            assert probability.shape == (36, 3)
            assert np.isfinite(probability).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1)
            expected_f1 = f1_score(data["y_test"], prediction, average="macro")
            assert report["metrics"][name]["f1"] == pytest.approx(expected_f1)
    assert run("train_model").returncode != 0
    rejected = run("evaluate_model", "--min-f1", "1.0")
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "FAIL" in rejected.stdout
    assert not json.loads((output / "metrics.json").read_text())["gate"]["passed"]
    assert json.loads((output / "experiment.json").read_text())["status"] == "rejected"


def test_wine_data_has_expected_features_and_labels():
    X, y, _ = dataset()
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert np.isfinite(X).all()
    assert set(y) == {0, 1, 2}
    np.testing.assert_array_equal(np.bincount(y), [59, 71, 48])


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_every_split_contains_each_wine_class(split):
    _, y, rows = dataset()
    assert set(y[rows[split]]) == {0, 1, 2}
    overall = np.bincount(y) / len(y)
    partition = np.bincount(y[rows[split]]) / len(rows[split])
    np.testing.assert_allclose(partition, overall, atol=0.03, rtol=0)


@pytest.mark.parametrize("threshold", ["nan", "inf", "-0.1", "1.1"])
def test_cli_rejects_invalid_threshold_before_loading_files(tmp_path, threshold):
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluate_model", "--version", "missing",
         "--output-dir", str(tmp_path), "--min-f1", threshold],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "F1 threshold must be finite and between 0 and 1" in result.stderr
    assert not (tmp_path / "missing" / "metrics.json").exists()


@pytest.mark.parametrize("module", ["calibrate_model", "evaluate_model"])
def test_missing_training_outputs_fail_without_metrics(tmp_path, module):
    result = subprocess.run(
        [sys.executable, "-m", f"src.{module}", "--version", "missing",
         "--output-dir", str(tmp_path)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not (tmp_path / "missing" / "metrics.json").exists()


@pytest.mark.parametrize("threshold", ["0.85", "1.0"])
def test_three_experiments_are_distinct_and_keep_logs_on_rejection(tmp_path, threshold):
    result = subprocess.run(
        [sys.executable, "-m", "src.run_experiments", "--version", "comparison",
         "--output-dir", str(tmp_path), "--min-f1", threshold],
        capture_output=True, text=True,
    )
    assert result.returncode == (0 if threshold == "0.85" else 1), result.stdout + result.stderr
    batch = tmp_path / "comparison"
    summary = json.loads((batch / "experiments.json").read_text())
    assert len(summary["experiments"]) == 3
    expected_depths = {"baseline": None, "depth-2": 2, "depth-5": 5}
    reference = None
    for record in summary["experiments"]:
        name = record["experiment"]
        run = batch / record["version"]
        model = joblib.load(run / "model.joblib")
        assert model.max_depth == record["parameters"]["max_depth"] == expected_depths[name]
        assert record["description"] and record["preprocessing"]["split_seeds"] == [42, 43]
        assert json.loads((run / "experiment.json").read_text()) == record
        evaluation = record["evaluation"]
        assert evaluation["gate"]["minimum"] == float(threshold)
        assert record["status"] == ("passed" if evaluation["gate"]["passed"] else "rejected")
        log = (batch / f"{name}.log").read_text()
        assert record["description"] in log
        assert all(f"Stage: {stage}" in log for stage in ("train_model", "calibrate_model", "evaluate_model"))
        with np.load(run / "heldout.npz") as data:
            current = {key: data[key].copy() for key in data.files}
        if reference is not None:
            for key in reference:
                np.testing.assert_array_equal(reference[key], current[key])
        reference = current
    if threshold == "1.0":
        assert any(record["status"] == "rejected" for record in summary["experiments"])
