import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.common import parser


def main():
    args = parser("Calibrate a frozen trained forest using only calibration rows").parse_args()
    run = args.output_dir / args.version
    model = joblib.load(run / "model.joblib")
    with np.load(run / "heldout.npz") as data:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(data["X_cal"], data["y_cal"])
        calibration_rows = len(data["y_cal"])
    joblib.dump(calibrated, run / "calibrated_model.joblib")
    print(f"Calibrated {args.version} using {calibration_rows} separate rows")


if __name__ == "__main__":
    main()
