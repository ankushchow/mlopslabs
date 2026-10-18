# MLOps Lab 2 — GitHub Actions model pipeline

This submission adapts [Ramin Mohammadi's Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2). Its core concept is automating model training, evaluation, and versioning with GitHub Actions. The meaningful changes are in `.github/workflows/lab2.yml`: tests must pass before training, a quality gate must pass before publication, and versioned artifacts carry provenance and checksums. This is model artifact publication, not deployment to a prediction service.

## Changes from the starter

| Area | Starter | This submission |
|---|---|---|
| Job ordering | One retraining job | `test → train-evaluate → publish`, enforced with `needs` |
| Quality control | Writes an F1 score | Exits with code 1 below calibrated F1 0.85; publication is skipped |
| Triggers | Push to main and separate scheduled example | Path-filtered push/PR checks and manual runs with configurable F1 threshold |
| Model versioning | Timestamp filenames committed by a bot | UTC timestamp + commit SHA + run ID + attempt; downloadable artifacts and SHA-256 manifest |
| Permissions | Pushes generated models to Git | Read-only repository permissions; artifacts retained for 30 days |
| Failure evidence | Logs | JUnit test report and metrics retained even when their checks fail |
| Data | Random sample count; evaluation regenerates data | Fixed 2,000 rows; disjoint 1,200/400/400 train/calibration/test split |
| Calibration | Named in the workflow but not implemented in the supplied scripts | Sigmoid calibration of a frozen forest on separate calibration rows |
| Dependencies | Unpinned packages | Python 3.11.14 and pinned Python packages |

The random forest and synthetic classification task preserve the starter's approach. JSON reports replace the starter's local MLflow logging; there is no tracking server to install. We report F1, accuracy, and Brier score for both models. Calibration is not guaranteed to improve every metric. The gate checks the calibrated model's F1 only; 0.85 is this submission's chosen threshold, not an instructor-specified value.

## Install prerequisites from a clean machine

You need internet access to install tools/dependencies. No GPU, Docker, AWS account, external dataset, notebook, or MLflow server is required.

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
python -m src.train_model --version local-demo
python -m src.calibrate_model --version local-demo
python -m src.evaluate_model --version local-demo --min-f1 0.85
```

Success means all 9 tests pass, each script exits with code 0, and the final line starts with `PASS: calibrated F1=`. The measured calibrated F1 is 0.9827 (see `lab2/VALIDATION.md`). Outputs are under `artifacts/local-demo/`:

| File | Purpose |
|---|---|
| `model.joblib` | Original fitted random forest |
| `calibrated_model.joblib` | Forest with calibrated probabilities |
| `heldout.npz` | Calibration/test data for this version; kept locally, excluded from published model bundle |
| `training.json` | Seed, split sizes, parameters, Python and sklearn versions |
| `metrics.json` | Both models' held-out metrics and gate decision |
| `manifest.json` | Successful approval, provenance, and SHA-256 hashes |

Use another version, such as `local-demo-2`, for another training run. Existing version directories are intentionally rejected. Local manifests say `local-uncommitted`; GitHub runs populate the actual source commit and run identity. Generated outputs and environments are ignored by Git.

## Demonstrate rejection

After the successful run, execute:

```sh
python -m src.evaluate_model --version local-demo --min-f1 1.0
```

This deliberately requires perfect F1. Expected result: `FAIL`, exit code **1**, `metrics.json` records `passed: false`, and the previous approval manifest is removed. Inspect the exit code immediately with `echo $?` (macOS/Linux) or `$LASTEXITCODE` (PowerShell). This failure is expected and proves the gate is effective.

Restore the normal successful result:

```sh
python -m src.evaluate_model --version local-demo --min-f1 0.85
```

Tests cover disjoint deterministic splits, threshold boundary/invalid scores, version path validation, a complete successful pipeline, checksums, duplicate-version rejection, and an actual failing evaluation process that removes stale approval.

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
3. Open the run triggered by your source push, or choose **Run workflow**, select `main`, keep `min_f1` at `0.85`, and run it manually. Manual dispatch requires the workflow to exist on the default branch. PR runs execute tests and evaluation, but the final `publish` job is skipped.
4. A successful main-branch run has three green jobs: `test`, `train-evaluate`, and `publish`. Open the run summary to see the quality decision and metric table.
5. Download the `lab2-model-VERSION` artifact from the run's **Artifacts** section. It contains both models, training metadata, metrics, and the manifest. Test reports, metrics, and an approved transfer bundle are also listed separately. Artifacts expire after **30 days**; download and retain submission evidence before then. A new run creates a separate version; it does not overwrite or deploy an existing model.
6. Optional failure demonstration: manually run with `min_f1` set to `1.0`. The training/evaluation job must fail and `publish` must be skipped. Metrics remain downloadable; that failed run must have no `lab2-model-...` artifact. Then run again with `0.85` so you have a successful final run to submit.

The workflow installs Python and packages itself on GitHub-hosted Ubuntu runners. It needs no custom secrets and uses `contents: read`. Caching speeds dependency installation; concurrency cancels older runs on the same ref. Timeouts stop stuck jobs. The separate publication job downloads the approved bundle and verifies its hashes before publishing the final artifact.

## Submission and troubleshooting

Submit your repository URL and the URL of a successful GitHub Actions run. Include the README, source, tests, pinned requirements, and root workflow in Git. See `lab2/VALIDATION.md` for the local validation record and hosted-run status. Local tests alone do not prove GitHub Actions completed.

- **`No module named src`**: run from `lab2` using `python -m src...` with the virtual environment active.
- **`FileExistsError`**: pick a new version for training; this prevents accidental overwrites.
- **Missing model/data file**: use the same version in train, calibrate, and evaluate, in that order.
- **Workflow not visible**: confirm `.github/workflows/lab2.yml` exists on the remote default branch and Actions is enabled. A README-only push is intentionally not a trigger; use manual dispatch.
- **Quality failure**: inspect the downloadable metrics and threshold. Do not use `continue-on-error` to bypass the gate.
- **Dependency/import errors**: recreate the environment using Python 3.11.14 and the pinned requirements. `FrozenEstimator` needs the supplied sklearn version.

## Source references

- [Original Lab 2 instructions and starter](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
- [GitHub artifact upload action](https://github.com/actions/upload-artifact)
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

The upstream repository's root license is CC0 1.0; its Lab 2 README refers to MIT, so consult the actual upstream root license for upstream material. This submission retains the lab concept and a matching lab folder structure while replacing the training/evaluation implementation and workflow with the behavior described above.
