# Validation record

Local validation completed on October 18, 2026, on macOS ARM64 with Python 3.11.14 and the pinned requirements.

| Check | Observed result |
|---|---|
| `python -m pytest -q --junitxml=test-results.xml` | 9 passed |
| Training, version `local-demo` | 1,200 training rows; success |
| Sigmoid calibration | 400 separate calibration rows; success |
| Held-out evaluation | 400 test rows; quality gate passed at 0.85 |
| Base model F1 / accuracy / Brier | 0.977887 / 0.977500 / 0.018366 |
| Calibrated model F1 / accuracy / Brier | 0.982716 / 0.982500 / 0.014123 |
| Rejection test (`--min-f1 1.0`) | Subprocess returned code 1, metrics recorded failure, approval manifest removed |
| Duplicate version test | Training refused to overwrite the prior version |
| Successful bundle checksums | All manifest SHA-256 values matched output files |
| GitHub workflow syntax | actionlint 1.7.7 passed without diagnostics |

The rejection and checksum checks run inside the end-to-end pytest test using a temporary output directory. A separate `local-demo` run also passed during initial preparation. Generated local outputs and the XML test report are ignored by Git and can be reproduced with the README commands.

## GitHub Actions status

**Pending hosted verification in [ankushchow/mlopslabs](https://github.com/ankushchow/mlopslabs).** Local checks passed; the successful Actions run URL will be recorded here after the workflow completes. Local tests and workflow linting do not verify GitHub runner execution, artifact transfer, or hosted publication.
