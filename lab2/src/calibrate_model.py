import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.common import parser, write_json


def main():
    args = parser("Calibrate a frozen trained forest using only calibration rows").parse_args()
    run = args.output_dir / args.version
    model = joblib.load(run / "model.joblib")
    with np.load(run / "heldout.npz") as data:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(data["X_cal"], data["y_cal"])
        calibration_rows = len(data["y_cal"])
    joblib.dump(calibrated, run / "calibrated_model.joblib")
    experiment = json.loads((run / "experiment.json").read_text())
    experiment["status"] = "calibrated"
    write_json(run / "experiment.json", experiment)
    print(f"Calibrated {args.version} using {calibration_rows} separate rows")


if __name__ == "__main__":
    main()
