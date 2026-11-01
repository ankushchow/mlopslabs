# MLOps Lab 3 — Airflow penguin clustering pipeline

This submission adapts [Ramin Mohammadi's Airflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1). Its core concept is using Airflow to load data, preprocess it, train a clustering model, save it, and make predictions. The workflow compares seven models and publishes the selected model only after it passes a quality gate.

## Changes from the starter

- **Data:** Replaced the starter's credit-card data with Palmer Penguins. Uses four physical measurements and excludes species labels.
- **Model:** Replaced K-Means with Gaussian Mixture Models (GMMs), comparing 2 through 8 components on the same data.
- **Preprocessing:** Replaced MinMaxScaler with StandardScaler fitted only on training rows. Added checks for missing columns, invalid values, duplicate rows, and minimum usable data.
- **Evaluation:** Added separate training and validation sets. Selects the eligible model with the highest validation silhouette score instead of using the elbow method.
- **Quality gate:** Requires convergence, at least 1% of training rows in each cluster, and a silhouette score of at least 0.35. Publication is blocked when the gate fails.
- **Airflow workflow:** Expanded four sequential tasks into seven task definitions, with seven mapped training instances and up to four concurrent tasks. Added retries and task timeouts.
- **Prediction:** Saves the selected model and fitted scaler together. This fixes the starter's use of raw prediction inputs after training on scaled data and its saving of the last fitted model rather than the elbow-selected model.
- **Artifacts:** Each run has a separate output directory containing metrics, the approved model, checksums, and predictions. XCom carries small metrics and file paths instead of serialized datasets.
- **Tests:** Added 11 tests covering preprocessing, data separation, model selection, convergence, prediction, artifact integrity, quality-gate failure, and DAG structure.

We retain the original clustering workflow and Airflow orchestration. A GMM assigns each row to a component and also provides membership probabilities. The component IDs are arbitrary groups, not predicted species. The 0.35 threshold and candidate range are choices for this submission.

## Dataset: Palmer Penguins

Source: [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/). The dataset contains 344 penguins. We use `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g`; species, island, sex, and year are excluded from modeling.

A fixed seed of 42 reserves **34 complete rows for prediction**. The remaining 310 rows enter the DAG. Validation removes two rows with missing measurements, leaving **246 training / 62 validation rows** after an 80/20 split. Prediction rows are never used to fit or select models.

The source CSV and prepared partitions are included in `data/`, so DAG runs do not download data. Attribution and checksums are in [data/SOURCE.md](data/SOURCE.md).

## Install prerequisites

Install and start [Docker Desktop](https://docs.docker.com/desktop/) with Docker Compose v2. Allocate at least 4 GB of memory, ideally 8 GB. Docker supplies Python 3.11, Airflow 2.10.5, and the pinned dependencies; no host Python environment is required. The first image build needs internet access.

Clone the repository or download and extract its ZIP:

```sh
git clone https://github.com/ankushchow/mlopslabs.git
cd mlopslabs/lab3
```

If you already have the repository, enter its `lab3` folder. On Windows, use WSL2 with Docker Desktop integration. All following commands run from `lab3`.

## Run locally

```sh
bash setup.sh
docker compose ps
```

The setup script builds the image, initializes the database and user, and starts the webserver and scheduler. Wait until both Airflow services show **healthy**.

Open [Airflow](http://localhost:8080) and log in with username **airflow** and password **airflow**. Find **airflow_lab1_penguin_clustering**, enable it, and click **Trigger DAG** with `min_silhouette` set to **0.35**.

The workflow is:

```text
validate_data → preprocess → train_candidate [7] → select_model → quality_gate → publish_model → predict
```

Open the Grid or Graph view and expand `train_candidate` to see the seven model fits. A successful run has all 13 task instances green. The DAG has a daily 06:00 UTC schedule, with catchup disabled and one active run at a time. Enabling it can also create the latest eligible scheduled run.

You can also trigger it from the terminal after the scheduler discovers the DAG:

```sh
docker compose exec airflow-scheduler airflow dags unpause airflow_lab1_penguin_clustering
docker compose exec airflow-scheduler airflow dags trigger airflow_lab1_penguin_clustering --conf '{"min_silhouette":0.35}'
```

The seven candidates use the same split and preprocessing. Only the number of components changes. All use diagonal covariance, 10 initializations, a maximum of 300 iterations, covariance regularization of 1e-6, and seed 42. Training BIC and validation log-likelihood are also logged, but silhouette determines selection among eligible models.

## Demonstrate rejection

Trigger a separate run with a silhouette threshold of 1.0:

```sh
docker compose exec airflow-scheduler airflow dags trigger airflow_lab1_penguin_clustering --conf '{"min_silhouette":1.0}'
```

Training and selection finish, then `quality_gate` fails. Both `publish_model` and `predict` become **upstream_failed**, and no approved model or manifest is produced for that run. The intentional gate failure is not retried; other tasks have one retry after 15 seconds and a five-minute execution timeout.

## Tests

```sh
docker compose run --rm airflow-scheduler python -m pytest -q /opt/airflow/tests -p no:cacheprovider
```

Expected result: **11 passed**. To execute both complete DAG outcomes and verify their task states:

```sh
docker compose run --rm airflow-scheduler python /opt/airflow/scripts/verify_dag.py
```

This uses Airflow's `dag.test()` and saves `working_data/penguins_gmm/dag_validation.json`. The rejection case prints an expected error traceback. The script exits successfully only if the normal run passes and the strict run blocks publication and prediction.

## Output files

Each run writes to `working_data/penguins_gmm/<hash-of-run-id>/`. These generated files are ignored by Git.

- `data_quality.json` records the source checksum, run ID, row counts, and data checks.
- `split.json`, `split.npz`, and `clean.csv` record the prepared data and split.
- `candidate_2..8.json` and `selection.json` record candidate settings/results and the selected model.
- `quality_gate.json` records the threshold and pass/fail decision.
- `model.joblib` contains the approved scaler and GMM; `manifest.json` records its metadata and artifact checksums.
- `predictions.csv` contains the 34 reserved rows, assigned clusters, and membership probabilities.

Candidate pipelines and the fitted scaler are also retained for inspection. Prediction verifies the approved model's checksum before loading it. Use only trusted joblib files.

## Run on GitHub Actions

The workflow `.github/workflows/lab3.yml` runs on pushes to `main` and pull requests that change Lab 3 or its workflow. It can also be started manually from **Actions → Lab 3 - Validate Airflow clustering DAG → Run workflow** once the workflow is on the default branch.

It builds the Docker image, initializes Airflow, runs the 11 tests, and verifies both DAG outcomes. JSON reports and predictions are uploaded as `lab3-airflow-validation` and retained for 30 days. Hosted execution has not yet been verified; the results below are from local Docker runs.

## Results

- **Selected model:** 2-component Gaussian Mixture; convergence successful.
- **Validation silhouette:** 0.5477, passing the 0.35 threshold.
- **Training cluster sizes:** 161 and 85 rows.
- **Predictions:** 34 reserved rows, each with a cluster and membership probability.
- **Rejection check:** threshold 1.0 blocked publication and prediction.
- **Tests:** all 11 passed.

The saved [validation record](VALIDATION.md) and [run evidence](evidence/) are included in the repository. Silhouette describes separation on the validation set used for model selection; it is not an independent test score or species accuracy. Membership probabilities are not calibrated species confidence scores.

## Stop and common issues

```sh
docker compose down
```

This stops the containers while retaining the database volume and generated files. Run `bash setup.sh` to start them again. The Docker configuration and default login are intended for a local lab; the webserver binds to localhost.

- **Docker connection error:** start Docker Desktop and wait for the engine.
- **DAG missing:** allow time for discovery, then check `docker compose exec airflow-scheduler airflow dags list-import-errors`.
- **Port 8080 occupied:** change the host port in Compose to 8081, restart, and open port 8081.
- **Task failed:** inspect its Airflow log and the data-quality or gate report.
- **Linux permission errors:** confirm `.env` has your `AIRFLOW_UID` from `id -u` and that you own `logs` and `working_data`.

To reproduce the prepared CSV partitions from the included source, run `docker compose run --rm airflow-scheduler python /opt/airflow/scripts/prepare_penguins.py`. The files are written to `working_data/penguins_export/`.
