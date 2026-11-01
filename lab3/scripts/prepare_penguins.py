"""Reproduce feature-only training and separate prediction CSVs without species labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from clustering.pipeline import FEATURES, SEED, checksum, write_json


def prepare(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(source)[FEATURES]
    valid_indices = features.index[np.isfinite(features).all(axis=1)]
    predict_indices = np.sort(np.random.default_rng(SEED).choice(valid_indices, 34, replace=False))
    train_indices = features.index.difference(predict_indices).to_numpy()
    # Leave the original missing measurements in training so the DAG validates them.
    features.loc[train_indices].to_csv(destination / "penguins_training.csv", index=False)
    features.loc[predict_indices].to_csv(destination / "penguins_predict.csv", index=False)
    write_json(destination / "partition.json", {
        "source_sha256": checksum(source), "seed": SEED,
        "training_source_rows": train_indices.tolist(),
        "prediction_source_rows": predict_indices.tolist(),
        "features": FEATURES, "labels_used": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/opt/airflow/data/penguins_original.csv")
    parser.add_argument("--output-dir", default="/opt/airflow/working_data/penguins_export")
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.output_dir)
