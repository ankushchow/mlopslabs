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
- **Environment:** Pinned Python to 3.11.14 and fixed dependency versions for reproducible runs.
- **Documentation:** Added installation instructions, PyCharm setup, exact run commands, expected outputs, dataset attribution, and validation results.

We retain the random forest algorithm and replace the synthetic data with UCI Wine. JSON reports replace the starter's local MLflow logging; there is no tracking server to install. We report macro F1 for both models: the mean of the F1 scores for the three classes, giving each class equal weight. Calibration adjusts predicted probabilities and is not guaranteed to improve F1. The gate checks the calibrated model's macro F1 only; 0.85 is this submission's chosen threshold, not an instructor-specified value.

## Dataset: UCI Wine

Source: [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), Aeberhard, S. & Forina, M. (1992), [DOI: 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), licensed CC BY 4.0. The dataset contains 178 wines from three cultivars, with 13 chemical measurements such as alcohol, malic acid, color intensity, and proline. We predict the cultivar class, not a quality rating.

The code uses [scikit-learn's bundled copy](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) through `load_wine()`. No data download is needed after installing the pinned dependencies. Class labels are encoded as 0, 1, and 2; features are kept in their original order and scale. Random forests do not require feature scaling here.

Stratified splits use fixed seeds 42 and 43: **106 training / 36 calibration / 36 test rows**, approximately 60/20/20 after rounding. Only training rows fit the forest; only calibration rows fit sigmoid calibration. Test rows are used for evaluation. The small test set makes this an instructional example with limited evidence of generalization.

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

## Use PyCharm

Open the `mlopslabs` repository folder as the project. In **Settings → Project → Python Interpreter**, select the existing interpreter at `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows) after following the setup steps above. Set the terminal working directory to the repository root, activate the virtual environment, then enter `lab2` before running the commands below. If using a Python run configuration, select a **module name** such as `src.train_model`, set its working directory to `lab2`, and supply `--version wine-demo`. Run training, calibration, and evaluation in order using the same version.

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

After the successful run, execute:

```sh
python -m src.evaluate_model --version wine-demo --min-f1 1.0
```

This deliberately requires perfect F1. Expected result: `FAIL`, exit code **1**, `metrics.json` records `passed: false`. Inspect the exit code immediately with `echo $?` (macOS/Linux) or `$LASTEXITCODE` (PowerShell). This failure is expected and proves the gate is effective.

Restore the normal successful result:

```sh
python -m src.evaluate_model --version wine-demo --min-f1 0.85
```

The 19 tests cover:

- Dataset dimensions, finite feature values, expected class labels, and class counts.
- Reproducible, disjoint splits with all three classes and similar class proportions in each split.
- Quality-gate boundaries, invalid scores, and invalid CLI thresholds (`nan`, `inf`, negative, and above 1).
- Version path validation and refusal to overwrite an existing training version.
- A complete pipeline, saved models' three-class probability outputs, and independently recalculated macro F1.
- Missing training outputs and an actual evaluation failure when the threshold is 1.0.

## Run on GitHub Actions

This submission lives at [ankushchow/mlopslabs](https://github.com/ankushchow/mlopslabs). Lab 2 code is in `lab2/`; its active workflow is `.github/workflows/lab2.yml` at the repository root. You need a GitHub account and write access to push changes or manually trigger runs. A reviewer can inspect the public runs without installing anything.

1. Commit and push these source files to `main` in your repository. From the repository root after editing Lab 2:

   ```sh
   git add .gitignore .github/workflows/lab2.yml lab2
   git commit -m "Add gated Lab 2 training and versioning pipeline"
   git push origin main
   ```

   If Git requests your identity, configure your own name/email. Authenticate to GitHub using your chosen Git credential manager or SSH setup; no credentials belong in source files. If your fork already has other active workflows, inspect those separately; this submission's workflow has a distinct name.
2. Open **Actions**. If GitHub says workflows are disabled in the fork, enable them. Choose **Lab 2 - Train, calibrate, and version**.
3. Open the run triggered by your source push, or choose **Run workflow**, select `main`, keep `min_f1` at `0.85`, and run it manually. Manual dispatch requires the workflow to exist on the default branch. PR runs execute tests and evaluation, but the final model-upload step is skipped.
4. A successful main-branch run has two green jobs: `test` and `train-evaluate`, including a successful final model-upload step. Open the run summary to see the quality decision and metric table.
5. Download the `lab2-model-VERSION` artifact from the run's **Artifacts** section. It contains both models, training metadata, and F1 metrics. Test reports and metrics are also available as separate diagnostic artifacts. Artifacts expire after **30 days**; download and retain submission evidence before then. A new run creates a separate version; it does not overwrite or deploy an existing model.
6. Optional failure demonstration: manually run with `min_f1` set to `1.0`. The training/evaluation job must fail and the model-upload step must be skipped. Metrics remain downloadable; that failed run must have no `lab2-model-...` artifact. Then run again with `0.85` so you have a successful final run to submit.

The workflow installs Python and packages itself on GitHub-hosted Ubuntu runners. It needs no custom secrets and uses `contents: read`. Caching speeds dependency installation; concurrency cancels older runs on the same ref. Timeouts stop stuck jobs. The last step uploads the model bundle only when all preceding steps succeed and the run is on `main` rather than a pull request.

## Submission and troubleshooting

Submit your repository URL and the URL of a successful GitHub Actions run. Include the README, source, tests, pinned requirements, and root workflow in Git. See the validation section below for the current status. Local tests alone do not prove GitHub Actions completed.

- **`No module named src`**: run from `lab2` using `python -m src...` with the virtual environment active.
- **`FileExistsError`**: pick a new version for training; this prevents accidental overwrites.
- **Missing model/data file**: use the same version in train, calibrate, and evaluate, in that order.
- **Workflow not visible**: confirm `.github/workflows/lab2.yml` exists on the remote default branch and Actions is enabled. Changes under `lab2/` or to this workflow trigger a run; a root-README-only push does not. Use manual dispatch if needed.
- **Quality failure**: inspect the downloadable metrics and threshold. Do not use `continue-on-error` to bypass the gate.
- **Dependency/import errors**: recreate the environment using Python 3.11.14 and the pinned requirements. `FrozenEstimator` needs the supplied sklearn version.

## Validation

Local validation completed on October 18, 2026, using Python 3.11.14:

| Check | Result |
|---|---|
| Automated tests | 19 passed, including the complete pipeline and rejection at macro F1 threshold 1.0 |
| Separate training/calibration/evaluation run | Passed on 36 test rows; base and calibrated macro F1 both 0.9752 |
| GitHub workflow syntax | actionlint 1.7.7 passed |

The quality gate still requires calibrated macro F1 ≥ 0.85. The report contains macro F1 and the gate decision. Calibration changed probabilities but did not change F1 on this test split.

The [earlier GitHub Actions run](https://github.com/ankushchow/mlopslabs/actions/runs/39170436589) passed for the previous three-job version. The current two-job workflow with Wine data and 19 tests has not been pushed or run on GitHub. After pushing it yourself, record the new successful run URL here for submission.

## Source references

- [Original Lab 2 instructions and starter](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
- [GitHub artifact upload action](https://github.com/actions/upload-artifact)
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

The upstream repository's root license is CC0 1.0; its Lab 2 README refers to MIT, so consult the actual upstream root license for upstream material. This submission retains the lab concept and a matching lab folder structure while replacing the training/evaluation implementation and workflow with the behavior described above.
