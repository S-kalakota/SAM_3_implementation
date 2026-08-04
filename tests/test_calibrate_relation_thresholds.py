import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import calibrate_relation_thresholds as calibration


def test_calibration_manifest_rejects_truthy_string_labels(tmp_path):
    manifest = {
        "schema_version": 1,
        "threshold_grid": {"direction_margin_fraction": [0.03]},
        "cases": [{
            "case_id": "case-1",
            "relationship": "left_of",
            "target_mask": "target.png",
            "anchor_mask": "anchor.png",
            "depth": None,
            "xyz": None,
            "expected_pass": "false",
            "safety_case": True,
        }],
    }
    with pytest.raises(ValueError, match="must be booleans"):
        calibration.load_cases(manifest, tmp_path)


def test_calibration_manifest_rejects_unknown_threshold(tmp_path):
    manifest = {
        "schema_version": 1,
        "threshold_grid": {"unsafe_magic": [1.0]},
        "cases": [],
    }
    with pytest.raises(ValueError, match="unknown calibration thresholds"):
        calibration.load_cases(manifest, tmp_path)
