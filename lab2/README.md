# MLOps Lab 2 — GitHub Actions model pipeline

This submission adapts [Ramin Mohammadi's Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2). Its core concept is automating model training, evaluation, and versioning with GitHub Actions. The meaningful changes are in `.github/workflows/lab2.yml`: tests must pass before training, a quality gate must pass before publication, and each successful run publishes a versioned model bundle. This is model artifact publication, not deployment to a prediction service.

## Changes from the starter

- **Data:** Replaced synthetic data with UCI Wine: 178 samples, 13 features, and 3 classes. Added reproducible, stratified splits of 106 training, 36 calibration, and 36 test rows.
- **Model:** Added sigmoid probability calibration using separate calibration data while keeping the Random Forest classifier.
- **Evaluation:** Added held-out macro F1 evaluation for both models and a minimum calibrated macro F1 of 0.85. Evaluation fails below this threshold.
- **Workflow:** Added two dependent jobs: tests followed by training, calibration, and evaluation. The final step publishes the model only after the quality gate passes on `main`.
- **Workflow triggers:** Added path-filtered push and pull-request checks, plus manual runs with a configurable F1 threshold; removed scheduled execution.
- **Model versioning:** Replaced bot commits with downloadable model artifacts identified by run ID and attempt. Artifacts are retained for 30 days, and repository permissions are read-only.
- **Failure reports:** Added JUnit test reports and saved metrics that remain available when checks fail.
- **Tests:** Added 19 tests covering the dataset, split separation, class representation, model probabilities, macro F1, quality-gate behavior, invalid inputs, and missing files.

We retain the random forest algorithm and replace the synthetic data with UCI Wine. JSON reports replace the starter's local MLflow logging; there is no tracking server to install. We report macro F1 for both models: the mean of the F1 scores for the three classes, giving each class equal weight. Calibration adjusts predicted probabilities and is not guaranteed to improve F1. The gate checks the calibrated model's macro F1 only.

## Dataset: UCI Wine

Source: [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine). The dataset contains 178 wines from three cultivars, with 13 chemical measurements such as alcohol, malic acid, color intensity, and proline. We predict the cultivar class, not a quality rating.

The data is loaded from scikit-learn using `load_wine()`.

Stratified splits use fixed seeds 42 and 43: **106 training / 36 calibration / 36 test rows**, approximately 60/20/20 after rounding. Only training rows fit the forest; only calibration rows fit sigmoid calibration. Test rows are used for evaluation.

## Install prerequisites from a clean machine

You need internet access to install tools/dependencies. No GPU, Docker, AWS account, manual dataset download, notebook, or MLflow server is required.

1. Obtain the submission repository: use GitHub's **Code → Download ZIP**, extract it, and open a terminal in the extracted repository root (the folder containing `.github`, `lab1`, and `lab2`). Alternatively, install Git 2.x from [git-scm.com](https://git-scm.com/downloads), then use `git clone https://github.com/ankushchow/mlopslabs.git` followed by `cd mlopslabs`.
2. Install **uv 0.9.5**, which can install Python and create a virtual environment. On macOS/Linux:

   ```sh
   curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
   ```

   On Windows PowerShell:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.9.5/install.ps1 | iex"
   ```

   If Linux does not have `curl`, install it using the distribution's package manager (Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y curl`). Restart the terminal after installation, return to the repository root, and verify `uv --version` prints `uv 0.9.5`.
3. At the repository root, install Python and dependencies:

   ```sh
   uv python install 3.11.14
   uv venv --python 3.11.14 .venv
   uv pip install --python .venv -r lab2/requirements.txt
   ```

4. Activate the environment. On macOS/Linux:

   ```sh
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

5. Verify `python --version` prints `Python 3.11.14`, then enter the lab directory:

   ```sh
   cd lab2
   ```

All commands in the next two sections run from that directory with the environment activated. If Python 3.11.14 is already installed, standard `python -m venv .venv` and `python -m pip install -r lab2/requirements.txt` from the root after activation are an alternative to uv.

## Run locally

Run each command in order; stop and inspect the error if any command fails:

```sh
python -m pytest -q --junitxml=test-results.xml
python -m src.train_model --version wine-demo
python -m src.calibrate_model --version wine-demo
python -m src.evaluate_model --version wine-demo --min-f1 0.85
```

Success means all 19 tests pass, each script exits with code 0, and the final line starts with `PASS: calibrated macro F1=`. The measured calibrated macro F1 is 0.9752. Outputs are under `artifacts/wine-demo/`:

| File | Purpose |
|---|---|
| `model.joblib` | Original fitted random forest |
| `calibrated_model.joblib` | Forest with calibrated probabilities |
| `heldout.npz` | Calibration/test data for this version; kept locally, excluded from published model bundle |
| `training.json` | Dataset source, feature/class names, seed, split sizes, model parameters, and environment versions |
| `metrics.json` | Both models' held-out metrics and gate decision |

Use another version, such as `wine-demo-2`, for another training run. Existing version directories are intentionally rejected. GitHub versions use the run ID and attempt, for example `123456789-1`. Generated outputs and environments are ignored by Git.

## Demonstrate rejection

To check that the quality gate works, raise the required F1 score to 1.0:

```sh
python -m src.evaluate_model --version wine-demo --min-f1 1.0
```

The model scores below 1.0, so this prints `FAIL`, exits with code 1, and saves `passed: false` in `metrics.json`. This is the expected result. You can check the exit code with `echo $?` on macOS/Linux or `$LASTEXITCODE` in PowerShell.

Run it again with the normal threshold to get a passing result:

```sh
python -m src.evaluate_model --version wine-demo --min-f1 0.85
```

## Tests

The 19 tests check the Wine data, train/calibration/test splits, saved model predictions, and F1 calculations.
They also check invalid inputs, missing files, duplicate versions, and rejection when the model falls below the required score. All 19 passed locally.

## Run on GitHub Actions

The workflow is in `.github/workflows/lab2.yml`. It runs when changes to Lab 2 or the workflow are pushed to `main`, and on pull requests that change those files.

From the repository root, commit and push your changes:

```sh
git add .gitignore .github/workflows/lab2.yml lab2
git commit -m "Update Lab 2 model pipeline"
git push origin main
```

Open the repository's **Actions** tab and select **Lab 2 - Train, calibrate, and version**. You can also start it using **Run workflow** on `main`, with `min_f1` set to `0.85`. If Actions is disabled in your fork, enable it first.

GitHub installs Python and the dependencies automatically. A successful run shows two green jobs: `test` and `train-evaluate`. The second job trains the model, calibrates it, checks its F1 score, and uploads the model if it passes. Pull requests run the checks but skip the model upload.

Download `lab2-model-VERSION` from the run's **Artifacts** section. It contains the original model, calibrated model, training details, and F1 results. Test reports and metrics are also available separately. These downloads are kept for 30 days.

To try the failure case on GitHub, start a manual run with `min_f1` set to `1.0`. The evaluation should fail and the model upload should be skipped, while the metrics remain available.

## Results

Tested locally with Python 3.11.14. Both models scored **0.9752 macro F1** on the 36 test rows, above the required **0.85**. Calibration changed the predicted probabilities but did not change F1 for this split. The workflow also passed actionlint.

Both jobs passed in GitHub Actions using commit `dd0d353`. The model was saved as `lab2-model-39173683615-1`.

## Common issues

- **`No module named src`:** activate the virtual environment and run the commands from the `lab2` folder.
- **`FileExistsError`:** use a new version name, such as `wine-demo-2`.
- **Missing model or data:** run training, calibration, and evaluation in that order, using the same version name.
- **Workflow not showing:** check that the workflow file is on the default branch and Actions is enabled.
- **Import errors:** check that the environment uses Python 3.11.14 and the packages in `requirements.txt`.