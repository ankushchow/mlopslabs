import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.common import dataset, parser, write_json


def main():
    args = parser("Train a reproducible random forest and persist held-out data").parse_args()
    run = args.output_dir / args.version
    # Never silently overwrite an existing version.
    run.mkdir(parents=True, exist_ok=False)
    X, y, rows = dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X[rows["train"]], y[rows["train"]])
    joblib.dump(model, run / "model.joblib")
    np.savez_compressed(run / "heldout.npz",
                        X_cal=X[rows["calibration"]], y_cal=y[rows["calibration"]],
                        X_test=X[rows["test"]], y_test=y[rows["test"]])
    write_json(run / "training.json", {
        "version": args.version, "python": platform.python_version(),
        "scikit_learn": sklearn.__version__, "seed": 42,
        "split_sizes": {key: len(value) for key, value in rows.items()},
        "parameters": model.get_params(),
    })
    print(f"Trained {args.version}: train=1200, calibration=400, test=400")


if __name__ == "__main__":
    main()
