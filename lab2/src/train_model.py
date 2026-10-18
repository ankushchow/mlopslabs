import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

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
        "dataset": "UCI Wine",
        "dataset_source": "https://archive.ics.uci.edu/dataset/109/wine",
        "feature_names": load_wine().feature_names,
        "class_names": load_wine().target_names.tolist(),
        "split_sizes": {key: len(value) for key, value in rows.items()},
        "parameters": model.get_params(),
    })
    sizes = ", ".join(f"{name}={len(ids)}" for name, ids in rows.items())
    print(f"Trained {args.version} on UCI Wine: {sizes}")


if __name__ == "__main__":
    main()
