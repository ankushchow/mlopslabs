# MLOps Lab 4 — MLflow experiment tracking

This submission adapts [Ramin Mohammadi's MLflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1). Its core concept is tracking experiments, comparing parameters and metrics, saving models, and loading them for prediction. This is **Lab 4** in this repository; the source tutorial is **MLflow Lab 1**.

## Changes from the starter

- **Data:** Replaced the starter dataset with **UCI Auto MPG**, predicting vehicle fuel economy. Bundled the source data, added missing-value checks, and recorded cleaning results and a source checksum. Training runs work offline.
- **Model:** Replaced the notebook's Random Forest with Gradient Boosting, comparing tree depths 1, 2, and 3 while keeping other settings fixed.
- **Evaluation:** Added reproducible training, validation, and test partitions. Select by validation RMSE, then evaluate only the selected model on the test set. Also report a training-mean baseline.
- **Tracking:** Combined sklearn autologging for parameters/training metrics with explicitly named validation/test metrics, readable run names, batch tags, and experiment descriptions.
- **Storage:** Use a local SQLite database for tracking metadata and a separate artifact directory. No remote tracking service is required.
- **Models:** Save each candidate with an input example, named feature signature, and package requirements. Load the selected model using its recorded MLflow URI instead of a hard-coded run directory.
- **Artifacts:** Save split row IDs, experiment summaries, test predictions, an actual-versus-predicted plot, and sample inference inputs.
- **Tests:** Check split isolation, metric calculations, model selection, actual MLflow records, saved-model inference, schema enforcement, and duplicate-batch protection.

The implementation retains the course notes' tracking, comparison, model loading, and local serving concepts. The model settings and split choices are decisions for this submission.

## Dataset: UCI Auto MPG

Source: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg). The target is city-cycle fuel economy in **miles per gallon (mpg)**.

The bundled source contains 398 vehicles. Six rows have missing horsepower and are removed, leaving **392 complete rows**. We use six numeric features: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, and `model_year`. Vehicle names and the categorical origin code are excluded. No scaling or imputation is used.

Seeds 42 and 43 produce **234 training / 79 validation / 79 test rows**, approximately 60/20/20. All candidates use the same split. Original source row IDs are preserved after cleaning. Only training rows fit models; validation rows determine selection; test rows are evaluated afterward. Dropping incomplete records limits these results to complete cases.

The source file is included in `data/`; attribution, licensing, and its checksum are in [data/SOURCE.md](data/SOURCE.md). Runs record cleaning counts, excluded row IDs, and selected features in `data_quality.json`.

## Install prerequisites

Use **Python 3.11** (validated with 3.11.14). Internet access is needed to install dependencies. No Docker, GPU, cloud account, or notebook is required.

Open this repository in PyCharm or a terminal, then create a separate environment for Lab 4:

```sh
cd lab4
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.11 -m venv .venv` and `.\.venv\Scripts\Activate.ps1` instead. In PyCharm, select `lab4/.venv` as the interpreter and set the working directory to `lab4`. All following commands run from `lab4` with this environment active.

## Run the three experiments locally

```sh
python -m pytest -q
python -m src.run_experiments --batch auto-mpg-comparison
```

The runner compares these configurations:

- **depth-1:** shallow trees to establish a simple reference model.
- **depth-2:** allows two levels of splits per tree.
- **depth-3:** allows more complex interactions.

All use 100 estimators, learning rate 0.05, squared-error loss, and seed 42. Only maximum tree depth changes. RMSE and MAE are better when smaller; R² is better when larger and can be negative.

The workflow is:

```text
Load and split → Train/log 3 candidates → Select by validation RMSE
              → Evaluate selected model on test rows → Reload and verify predictions
```

Success prints three validation scores, the selected depth, test RMSE, baseline RMSE, and `reload matched`. The selected model is not refitted after selection, so its saved artifact is exactly the one evaluated. Ties choose the shallower tree.

Use a fresh batch name, such as `auto-mpg-comparison-2`, for another run. Existing batch directories are rejected so earlier evidence remains intact. If a run fails midway, its output and MLflow records remain available for inspection; rerun with a new batch name.

## Compare runs in the MLflow UI

Start the local UI from `lab4`:

```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open [MLflow](http://127.0.0.1:5001), then choose **lab4-auto-mpg-gradient-boosting**.

1. Filter runs with `tags.batch = 'auto-mpg-comparison'` and select the three matching runs.
2. Compare `max_depth`, `validation_rmse`, `validation_mae`, and `validation_r2`. Sort validation RMSE ascending.
3. Open the run tagged `selected = true`. It has test metrics, baseline test metrics, and the reload check; the other candidates intentionally have no test scores.
4. Inspect the run artifacts for split IDs, versions, summary, predictions, and plot. Open its logged model to inspect the input signature, example, and requirements. MLflow 3 shows logged models separately from ordinary run artifacts.

For review screenshots, capture the three-run comparison, the selected model's test metrics, and its signature/artifacts. Stop the UI with **Ctrl+C**. SQLite and artifacts stay on disk for the next session.

## Load the saved model and predict

In another terminal with the same environment active:

```sh
python -m src.predict --summary outputs/auto-mpg-comparison/experiments.json --input outputs/auto-mpg-comparison/sample_input.csv --output outputs/auto-mpg-comparison/reloaded_predictions.csv
```

This loads the selected model through `mlflow.pyfunc`, checks its named input schema, and writes five predictions. These sample rows come from the held-out test set. For your own CSV, supply all six raw feature columns with finite numeric values. The target column is not an input.

## Serve locally over HTTP

Start the selected model on a different port from the tracking UI:

```sh
python -m src.predict --summary outputs/auto-mpg-comparison/experiments.json --serve --port 5002
```

Wait for the server startup message. From another terminal in `lab4`:

```sh
curl -X POST http://127.0.0.1:5002/invocations -H 'Content-Type: application/json' --data-binary @outputs/auto-mpg-comparison/serving_input.json
```

The response contains a `predictions` array with five values. The helper uses MLflow's `models serve` with the active local environment and binds to localhost. Stop it with **Ctrl+C**. This demonstrates local serving, not a cloud deployment.

## Tests

```sh
python -m pytest -q --junitxml=test-results.xml
python -m pip check
```

Tests create an isolated temporary SQLite store, fit the three real models, inspect the MLflow records, and load saved artifacts. They verify the source row counts, removal of incomplete records, and feature selection. They also check that only the validation winner receives test metrics, missing features are rejected, and reusing a batch name does not overwrite results. They do not change the demonstration runs in `mlflow.db`.

## Output files

Generated tracking state and new runs are ignored by Git:

| Path | Contents |
|---|---|
| `mlflow.db` | Experiments, run IDs, parameters, metrics, and tags |
| `mlartifacts/` | MLflow models, signatures, input examples, requirements, and run artifacts |
| `outputs/<batch>/experiments.json` | Candidate descriptions/results, selected model URI, test metrics, baseline, and versions |
| `outputs/<batch>/data_quality.json` | Source checksum, cleaning counts, excluded rows, and feature list |
| `outputs/<batch>/split.json` | Exact row IDs for all three partitions |
| `outputs/<batch>/test_predictions.csv` | 79 test targets, predictions, and residuals |
| `outputs/<batch>/actual_vs_predicted.png` | Selected model's held-out predictions |
| `outputs/<batch>/sample_input.csv`, `serving_input.json` | Five-row examples for CSV and HTTP inference |

The checked-in `experiments/auto-mpg-comparison/` directory contains a compact snapshot for review. Its run IDs and model URI refer to the original local store; run the experiment yourself to create a usable model on another machine. Keep `mlflow.db` together with `mlartifacts/` in this checkout. Copying summary JSON alone does not copy the models.

## Run on GitHub Actions

The locally prepared workflow `.github/workflows/lab4.yml` runs tests, the three-experiment comparison, and saved-model CSV prediction. It uploads JSON/CSV/PNG evidence and test results for 30 days. It uses read-only repository permissions and does not commit models.

After you review and push the files, changes to Lab 4 trigger this workflow on `main` and pull requests. It can also be started manually from **Actions → Lab 4 - Validate MLflow tracking → Run workflow**. Hosted execution has not been verified; local results are recorded below. The uploaded evidence is a report, not a portable tracking database or deployed service.

## Results

All candidates used the same fixed split:

| Configuration | Validation RMSE (mpg) | Validation MAE (mpg) | Validation R² |
|---|---:|---:|---:|
| depth-1 | 3.1029 | 2.1540 | 0.8179 |
| depth-2 | 2.9361 | 2.1214 | 0.8369 |
| depth-3 | 2.8745 | 2.0723 | 0.8437 |

- **Selected:** depth 3, with the lowest validation RMSE.
- **Held-out test:** RMSE 2.6818 mpg, MAE 2.0223 mpg, R² 0.8591 on 79 rows.
- **Training-mean baseline test RMSE:** 7.2188 mpg.
- **Validation:** 12 tests passed; saved-model reload and CSV/HTTP predictions matched; missing HTTP features returned 400.

The saved [experiment record](experiments/auto-mpg-comparison/experiments.json) includes the settings, data-cleaning report, and measured scores. The compact evidence folder also contains [split row IDs](experiments/auto-mpg-comparison/split.json), [test predictions](experiments/auto-mpg-comparison/test_predictions.csv), the [prediction plot](experiments/auto-mpg-comparison/actual_vs_predicted.png), and [HTTP validation results](experiments/auto-mpg-comparison/http_validation.json).

Validation RMSE determines selection. The test set is small, and a single fixed split does not establish performance on other vehicle populations or later model years. The training-mean baseline provides context; it does not influence selection. This lab has no publication quality gate: its purpose is to preserve and compare tracked experiments, including poorer results.

## Common issues

- **`No module named src`:** run from `lab4` with its environment activated.
- **`FileExistsError`:** choose a new batch name; keep earlier outputs for comparison.
- **UI is empty:** start it from `lab4` with the SQLite URI above. A default `mlflow ui` can open a different store.
- **Model cannot be found:** use a summary generated in this checkout and keep the database and artifact directory. The checked-in evidence snapshot is not a model bundle.
- **Input schema error:** provide all six named raw numeric features, including `horsepower`; use the generated sample CSV as the format reference.
- **Port occupied:** choose another UI port or pass a different `--port` to the serving command.
- **Dependency error:** recreate this lab's environment using Python 3.11 and the supplied requirements. Other labs have separate dependency sets.

## Source references

- [Original MLflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1)
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow sklearn logging and model APIs](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
