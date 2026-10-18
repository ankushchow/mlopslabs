# MLOps Lab 2 — GitHub Actions model pipeline

This submission adapts [Ramin Mohammadi's Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2). Its core concept is automating model training, evaluation, and versioning with GitHub Actions. The meaningful changes are in `.github/workflows/lab2.yml`: tests must pass before training, a quality gate must pass before publication, and each successful run publishes a versioned model bundle. This is model artifact publication, not deployment to a prediction service.

## Changes from the starter

- **Data:** Replaced synthetic data with UCI Wine: 178 samples, 13 features, and 3 classes. Added reproducible, stratified splits of 106 training, 36 calibration, and 36 test rows.
- **Model:** Compared three Random Forest configurations: unrestricted depth, depth 2, and depth 5. All use 100 trees and sigmoid calibration on separate data.
- **Evaluation:** Added held-out macro F1 evaluation for both models and a minimum calibrated macro F1 of 0.85. Evaluation fails below this threshold.
- **Workflow:** Added two dependent jobs: tests followed by training, calibration, and evaluation. The final step publishes the model only after the quality gate passes on `main`.
- **Workflow triggers:** Added path-filtered push and pull-request checks, plus manual runs with a configurable F1 threshold; removed scheduled execution.
- **Model versioning:** Replaced bot commits with downloadable model artifacts identified by run ID and attempt. Artifacts are retained for 30 days, and repository permissions are read-only.
- **Failure reports:** Added JUnit test reports and saved metrics that remain available when checks fail.
- **Tests:** Added 21 tests covering the data, model outputs, quality gate, and experiment logs, including failed evaluations.
- **Experiment logs:** Each run records what was tried, preprocessing choices, model settings, results, and pass/fail status. Text logs also capture the output of each stage.

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

## Run the three experiments locally

From `lab2`, with the virtual environment active:

```sh
python -m pytest -q --junitxml=test-results.xml
python -m src.run_experiments --version wine-comparison --min-f1 0.85
```

The runner trains, calibrates, and evaluates these three configurations:

- **baseline:** 100 trees with no depth limit, to establish a reference score.
- **depth-2:** 100 trees limited to depth 2, to see how a simpler model performs.
- **depth-5:** 100 trees limited to depth 5, to compare a moderate depth limit.

Only tree depth changes. Every experiment uses the same 106 training, 36 calibration, and 36 test rows, with the same seeds and calibration method. These configurations were chosen before comparing their results; the test scores are not used to select or tune a model.

There are no missing values to fill, and Random Forest does not need feature scaling. All 13 original features are used. These preprocessing decisions are written into every experiment record.

A successful run prints three `PASS` messages and exits with code 0. Results are saved in `artifacts/wine-comparison/`:

- `experiments.json` contains all three descriptions, settings, preprocessing choices, scores, and outcomes.
- `baseline.log`, `depth-2.log`, and `depth-5.log` contain the output and exit code from each stage.
- Each experiment has its own folder, such as `wine-comparison-baseline/`, containing `experiment.json`, `training.json`, `metrics.json`, both saved models, and the held-out data.

The runner keeps going if one experiment fails, so the other experiments still get recorded. It exits with code 1 if any experiment fails. Existing batch names cannot be overwritten; use `wine-comparison-2` for another run. GitHub uses its run ID and attempt as the batch name.

To run just one configuration:

```sh
python -m src.train_model --version depth2-demo --experiment depth-2
python -m src.calibrate_model --version depth2-demo
python -m src.evaluate_model --version depth2-demo --min-f1 0.85
```

Use a fresh version name for a new experiment. The default training configuration is `baseline`.

## Demonstrate rejection

Run a separate batch requiring perfect F1:

```sh
python -m src.run_experiments --version wine-rejection --min-f1 1.0
```

The measured scores are below 1.0, so this batch exits with code 1. All three experiments still run, and their failed evaluations are saved in `artifacts/wine-rejection/`. The earlier successful batch stays unchanged. Check the exit code with `echo $?` on macOS/Linux or `$LASTEXITCODE` in PowerShell.

## Tests

The 21 tests check the data splits, model predictions, F1 calculations, invalid inputs, and missing files.
They also check that all three configurations use the same data, save their settings and results correctly, and keep their logs when the quality gate fails.

## Run on GitHub Actions

The workflow is in `.github/workflows/lab2.yml`. It runs when changes to Lab 2 or the workflow are pushed to `main`, and on pull requests that change those files.

From the repository root, commit and push your changes:

```sh
git add .gitignore .github/workflows/lab2.yml lab2
git commit -m "Update Lab 2 model pipeline"
git push origin main
```

Open the repository's **Actions** tab and select **Lab 2 - Train, calibrate, and version**. You can also start it using **Run workflow** on `main`, with `min_f1` set to `0.85`. If Actions is disabled in your fork, enable it first.

GitHub installs Python and the dependencies automatically. A successful run shows two green jobs: `test` and `train-evaluate`. The second job runs all three experiments and uploads the models only if all three pass the quality gate. Pull requests run the checks but skip the model upload.

Download `lab2-models-VERSION` from the run's **Artifacts** section for the three model pairs and their results. `lab2-experiments-VERSION` contains the experiment descriptions, metrics, and text logs, even if an experiment fails. Test reports are available separately. These downloads are kept for 30 days.

To try the failure case on GitHub, start a manual run with `min_f1` set to `1.0`. The experiment job should fail and the model upload should be skipped, while all experiment logs remain available.

## Results

All three experiments passed the 0.85 threshold on the same 36 test rows:

- **baseline:** base macro F1 0.9752; calibrated macro F1 0.9752; passed.
- **depth-2:** base macro F1 0.9453; calibrated macro F1 0.9752; passed.
- **depth-5:** base macro F1 0.9752; calibrated macro F1 0.9752; passed.

The depth-2 model scored lower before calibration, but calibration brought its macro F1 up to the other two models. All three calibrated models tied on this split. This is a small test set, so equal scores do not show that the models would perform equally on new data.

The saved [experiment records](experiments/wine-comparison/experiments.json) and [text logs](experiments/wine-comparison/) are included in the repository. New runs write to `artifacts/`, which is ignored by Git; copy logs you want to submit into `experiments/` under a new batch name.

All 21 tests passed locally, and the workflow passed actionlint. The earlier single-experiment workflow passed on GitHub Actions. This three-experiment version has only been tested locally and still needs an Actions run after it is pushed.

## Common issues

- **`No module named src`:** activate the virtual environment and run the commands from the `lab2` folder.
- **`FileExistsError`:** use a new version name, such as `wine-comparison-2`.
- **Missing model or data:** run training, calibration, and evaluation in that order, using the same version name.
- **Workflow not showing:** check that the workflow file is on the default branch and Actions is enabled.
- **Import errors:** check that the environment uses Python 3.11.14 and the packages in `requirements.txt`.