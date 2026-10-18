import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

from src.common import dataset, parser, write_json


EXPERIMENTS = {
    "baseline": {
        "max_depth": None,
        "description": "Baseline: 100-tree Random Forest with unrestricted depth, followed by sigmoid calibration.",
    },
    "depth-2": {
        "max_depth": 2,
        "description": "Limit each tree to depth 2 to compare a much simpler forest against the baseline.",
    },
    "depth-5": {
        "max_depth": 5,
        "description": "Limit each tree to depth 5 to compare a moderate depth limit against the baseline and depth 2.",
    },
}


def main():
    cli = parser("Train a reproducible Random Forest experiment")
    cli.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    args = cli.parse_args()
    run = args.output_dir / args.version
    # Never silently overwrite an existing version.
    run.mkdir(parents=True, exist_ok=False)
    X, y, rows = dataset()
    config = EXPERIMENTS[args.experiment]
    model = RandomForestClassifier(
        n_estimators=100, max_depth=config["max_depth"], random_state=42, n_jobs=1,
    )
    experiment = {
        "version": args.version,
        "experiment": args.experiment,
        "description": config["description"],
        "dataset": "UCI Wine",
        "dataset_source": "https://archive.ics.uci.edu/dataset/109/wine",
        "preprocessing": {
            "missing_values": "No missing values in Wine; no imputation needed.",
            "scaling": "No scaling; Random Forest uses feature thresholds.",
            "features": "All 13 original measurements, in sklearn feature order.",
            "target": "Three cultivar classes encoded as 0, 1, and 2.",
            "split": "Stratified, disjoint training/calibration/test partitions.",
            "split_seeds": [42, 43],
        },
        "split_sizes": {key: len(value) for key, value in rows.items()},
        "model": "RandomForestClassifier",
        "parameters": model.get_params(),
        "calibration": "Sigmoid calibration on separate rows; trained forest is frozen.",
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "status": "training",
    }
    write_json(run / "experiment.json", experiment)
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
    experiment["status"] = "trained"
    write_json(run / "experiment.json", experiment)
    sizes = ", ".join(f"{name}={len(ids)}" for name, ids in rows.items())
    print(f"Trained {args.version} on UCI Wine: {sizes}")


if __name__ == "__main__":
    main()
