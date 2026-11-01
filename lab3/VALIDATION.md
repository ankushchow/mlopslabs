# Lab 3 — Palmer Penguins / Gaussian Mixture validation

Validated locally on macOS ARM64 with Linux ARM64 Docker containers, Airflow 2.10.5 and Python 3.11. This record covers the Palmer Penguins clustering pipeline.

| Check | Observed result |
|---|---|
| Pipeline, dataset partition, and DagBag tests | 11 passed |
| Original dataset | 344 Palmer Penguins, four numeric modeling features; no species labels used |
| Prediction partition | 34 complete rows withheld before training/validation splitting |
| Data-quality stage | 310 input rows; 2 missing-measurement rows removed; 308 retained; no duplicates |
| Disjoint fitting/selection split | 246 training / 62 validation rows |
| Fitted model | StandardScaler plus GaussianMixture, diagonal covariance, 10 initializations, seed 42 |
| Candidate comparison | Seven candidates, 2..8 components; only converged models with adequate cluster sizes eligible |
| Selected model | 2 components; convergence successful |
| Validation silhouette | 0.5477115849434546 across all 62 validation rows |
| Training BIC / validation average log-likelihood | 2218.5975511001234 / -4.4093332568324435 |
| Training cluster sizes | 161 / 85 |
| Quality threshold 0.35 | Passed; model, manifest, and 34 predictions with membership probabilities produced |
| Quality threshold 1.0 | Gate failed; publish and predict marked `upstream_failed`; no approved model or manifest |
| Model integrity | Reloaded pipeline preserves scaling; tampered model rejected |
| Convergence guard | Unconverged candidates excluded even if their silhouette score is higher |
| Workflow lint | actionlint passed |

[evidence/dag_validation.json](evidence/dag_validation.json) records real Airflow task-instance states from `dag.test()`. Successful execution completed all 13 task instances, including seven mapped fits. Reports under `evidence/success/` and `evidence/failed/` preserve the measured results. Model binaries are intentionally omitted from evidence; use the README commands to regenerate them. Recorded container paths are not portable download links.

Two selected components do not imply two biological species. This unsupervised workflow chooses the best silhouette among its candidates; species labels are never used to fit or select the model. Silhouette is measured on the selection partition, so it is not an independent final test score. The 34 reserved prediction rows demonstrate inference, not species accuracy. Membership probabilities are not calibrated species confidence scores.

## Scheduler execution

Verified with the real scheduler and LocalExecutor:

- `penguins-success`: **success**, all 13 task instances completed.
- `penguins-rejection`: **failed** at the quality gate; publication and prediction were **upstream_failed**.
- A daily scheduled run also completed successfully.

[evidence/scheduler_validation.json](evidence/scheduler_validation.json) records task states and execution timestamps from the metadata database. The DAG is `airflow_lab1_penguin_clustering`. Enable it in Airflow before triggering a run.

## Hosted CI status

The workflow `.github/workflows/lab3.yml` runs the eleven tests and both DAG outcomes in Docker and retains JSON reports/predictions for 30 days. Hosted execution is not yet verified; local validation does not establish that GitHub Actions passed.
