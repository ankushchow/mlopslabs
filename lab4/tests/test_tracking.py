import json

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from src.common import FEATURES, load_data, load_splits, regression_metrics
from src.run_experiments import run_experiments, select_best


def test_splits_are_disjoint_complete_and_repeatable():
    splits, indices = load_splits()
    assert [len(indices[name]) for name in ("train", "validation", "test")] == [234, 79, 79]
    groups = [set(ids) for ids in indices.values()]
    assert len(set.union(*groups)) == 392
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert indices == load_splits()[1]
    assert splits["train"][0].shape[1] == 6
    assert "mpg" not in splits["train"][0].columns


def test_metrics_have_known_values():
    metrics = regression_metrics([1, 2, 3], [2, 2, 2])
    assert metrics == pytest.approx({"rmse": np.sqrt(2 / 3), "mae": 2 / 3, "r2": 0})


def test_selection_uses_validation_and_breaks_ties_by_depth():
    candidates = [
        {"max_depth": 3, "validation": {"rmse": 5}, "test": {"rmse": 1}},
        {"max_depth": 2, "validation": {"rmse": 4}, "test": {"rmse": 20}},
        {"max_depth": 1, "validation": {"rmse": 4}, "test": {"rmse": 30}},
    ]
    assert select_best(candidates)["max_depth"] == 1


@pytest.mark.parametrize("candidates", [[], [{"max_depth": 1, "validation": {"rmse": float("nan")}}]])
def test_invalid_selection_rejected(candidates):
    with pytest.raises(ValueError):
        select_best(candidates)


@pytest.mark.parametrize("batch", ["../escape", "", "with space"])
def test_invalid_batch_rejected(tmp_path, batch):
    with pytest.raises(ValueError):
        run_experiments(batch, root=tmp_path)


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    root = tmp_path_factory.mktemp("tracking")
    summary = run_experiments("test-comparison", root=root)
    return root, summary


def test_real_mlflow_runs_models_metrics_and_artifacts(tracked):
    root, summary = tracked
    client = MlflowClient()
    assert len(summary["candidates"]) == 3
    assert summary["reload_matches"] is True
    assert summary["selected"] == select_best(summary["candidates"])
    for candidate in summary["candidates"]:
        run = client.get_run(candidate["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.params["max_depth"] == str(candidate["max_depth"])
        assert run.data.params["n_estimators"] == "100"
        assert run.data.metrics["validation_rmse"] == candidate["validation"]["rmse"]
        assert "training_mean_squared_error" in run.data.metrics
        selected = candidate["run_id"] == summary["selected"]["run_id"]
        assert ("test_rmse" in run.data.metrics) == selected
        assert run.data.tags["selected"] == str(selected).lower()
        assert "split.json" in {a.path for a in client.list_artifacts(candidate["run_id"])}
        model = mlflow.pyfunc.load_model(candidate["model_uri"])
        assert len(model.metadata.signature.inputs.inputs) == 6
    output = root / "outputs" / "test-comparison"
    assert json.loads((output / "experiments.json").read_text()) == summary
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert len(predictions) == 79
    assert regression_metrics(predictions.actual, predictions.prediction) == pytest.approx(summary["test"])


def test_reloaded_model_matches_and_rejects_missing_feature(tracked):
    _, summary = tracked
    X, _ = load_splits()[0]["test"]
    model = mlflow.pyfunc.load_model(summary["selected"]["model_uri"])
    predictions = model.predict(X)
    assert predictions.shape == (79,)
    assert np.isfinite(predictions).all()
    with pytest.raises(MlflowException, match="missing inputs"):
        model.predict(X.drop(columns=["horsepower"]))


def test_duplicate_batch_preserves_existing_results(tracked):
    root, _ = tracked
    path = root / "outputs" / "test-comparison" / "experiments.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run_experiments("test-comparison", root=root)
    assert path.read_bytes() == before


def test_auto_mpg_cleaning_and_feature_selection():
    X, y, report = load_data()
    assert report["source_rows"] == 398
    assert report["usable_rows"] == 392
    assert report["dropped_rows"] == 6
    assert len(report["dropped_row_ids"]) == 6
    assert list(X.columns) == FEATURES
    assert not {"mpg", "origin", "car_name"} & set(X.columns)
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(y).all()
    assert not set(report["dropped_row_ids"]) & set(X.index)
    assert set().union(*map(set, load_splits()[1].values())) == set(X.index)
