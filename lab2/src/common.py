import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def version_arg(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise argparse.ArgumentTypeError("Use 1-100 letters, digits, dots, underscores or hyphens; start with a letter/digit")
    return value


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--version", required=True, type=version_arg)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset():
    """Generate once, then split row IDs into disjoint 60/20/20 partitions."""
    X, y = make_classification(
        n_samples=2000, n_features=6, n_informative=3, n_redundant=0,
        n_classes=2, random_state=42,
    )
    train, remaining = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=42,
    )
    calibration, test = train_test_split(
        remaining, test_size=0.5, stratify=y[remaining], random_state=43,
    )
    return X, y, {"train": train, "calibration": calibration, "test": test}
