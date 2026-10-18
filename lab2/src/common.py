import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
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


def dataset():
    """Load UCI Wine from sklearn; split into disjoint 106/36/36 partitions."""
    X, y = load_wine(return_X_y=True)
    train, remaining = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=42,
    )
    calibration, test = train_test_split(
        remaining, test_size=0.5, stratify=y[remaining], random_state=43,
    )
    return X, y, {"train": train, "calibration": calibration, "test": test}
