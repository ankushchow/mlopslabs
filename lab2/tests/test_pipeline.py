import json
import subprocess
import sys

import pytest

from src.common import dataset, sha256, version_arg
from src.evaluate_model import quality_passes


def test_splits_are_disjoint_and_reproducible():
    X, y, rows = dataset()
    _, _, repeated = dataset()
    assert {k: len(v) for k, v in rows.items()} == {"train": 1200, "calibration": 400, "test": 400}
    sets = [set(v) for v in rows.values()]
    assert len(set.union(*sets)) == len(X) == len(y) == 2000
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for name in rows:
        assert (rows[name] == repeated[name]).all()


@pytest.mark.parametrize("score,threshold,expected", [
    (0.85, 0.85, True), (0.849, 0.85, False), (float("nan"), 0.85, False),
    (0.9, float("nan"), False), (0.9, -1, False), (1.1, 0.85, False),
])
def test_gate(score, threshold, expected):
    assert quality_passes(score, threshold) is expected


def test_version_cannot_escape_output_folder():
    with pytest.raises(Exception, match="letters"):
        version_arg("../escape")


def test_pipeline_pass_failure_and_version_collision(tmp_path):
    def run(module, *extra):
        return subprocess.run(
            [sys.executable, "-m", f"src.{module}", "--version", "test-run",
             "--output-dir", str(tmp_path), *extra], capture_output=True, text=True,
        )
    for module in ("train_model", "calibrate_model", "evaluate_model"):
        result = run(module)
        assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "test-run"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["approved"]
    for name, digest in manifest["sha256"].items():
        assert sha256(output / name) == digest
    assert run("train_model").returncode != 0
    rejected = run("evaluate_model", "--min-f1", "1.0")
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "FAIL" in rejected.stdout
    assert not (output / "manifest.json").exists()
    assert not json.loads((output / "metrics.json").read_text())["gate"]["passed"]
