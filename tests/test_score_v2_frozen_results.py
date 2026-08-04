#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_v2_frozen_results as scoring  # noqa: E402


class FrozenV2ScoreTests(unittest.TestCase):
    def _write_mask(self, path: Path, mask: np.ndarray) -> None:
        self.assertTrue(cv2.imwrite(str(path), mask.astype(np.uint8) * 255))

    def test_scores_candidate_relation_final_masks_and_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = np.zeros((10, 12), dtype=bool)
            target[4:7, 5:8] = True
            anchor = np.zeros_like(target)
            anchor[2:9, 3:10] = True
            target_path = root / "target.png"
            anchor_path = root / "anchor.png"
            candidate_target = root / "candidate_target.png"
            candidate_anchor = root / "candidate_anchor.png"
            final_target = root / "final_target.png"
            final_anchor = root / "final_anchor.png"
            for path, mask in (
                (target_path, target),
                (anchor_path, anchor),
                (candidate_target, target),
                (candidate_anchor, anchor),
                (final_target, target),
                (final_anchor, anchor),
            ):
                self._write_mask(path, mask)
            accepted_response = {
                "accepted": True,
                "elapsed_s": 2.0,
                "raw_candidate_pools": {
                    "target": [{"label": "T1", "mask_path_crop": str(candidate_target)}],
                    "anchor_1": [{"label": "A1", "mask_path_crop": str(candidate_anchor)}],
                },
                "relationship_measurements": [{
                    "relationship_index": 0,
                    "relationship": "inside",
                    "anchor_id": "anchor_1",
                    "target_candidate": "T1",
                    "anchor_candidate": "A1",
                    "status": "pass",
                }],
                "target_mask": {"mask_full": str(final_target)},
                "anchor_masks": {"anchor_1": {"mask_full": str(final_anchor)}},
            }
            accepted_path = root / "accepted.json"
            accepted_path.write_text(json.dumps(accepted_response), encoding="utf-8")
            rejected_path = root / "rejected.json"
            rejected_path.write_text(
                json.dumps({
                    **accepted_response,
                    "accepted": False,
                    "elapsed_s": 4.0,
                    "relationship_measurements": [{
                        "relationship_index": 0,
                        "relationship": "inside",
                        "anchor_id": "anchor_1",
                        "target_candidate": "T1",
                        "anchor_candidate": "A1",
                        "status": "fail",
                    }],
                    "target_mask": None,
                    "anchor_masks": {},
                }),
                encoding="utf-8",
            )
            base = {
                "reviewed": True,
                "scenario": "valid",
                "expected_accept": True,
                "safety_case": False,
                "target_mask": str(target_path),
                "anchor_masks": {"anchor_1": str(anchor_path)},
                "relationships": [{
                    "type": "inside",
                    "anchor_id": "anchor_1",
                    "expected_status": "pass",
                }],
            }
            valid = scoring.score_case(
                {"id": "valid", "result_json": str(accepted_path), **base},
                match_iou=0.5,
            )
            safety = scoring.score_case(
                {
                    "id": "safety",
                    "result_json": str(rejected_path),
                    **base,
                    "scenario": "relationship_false",
                    "expected_accept": False,
                    "safety_case": True,
                    "relationships": [{
                        "type": "inside",
                        "anchor_id": "anchor_1",
                        "expected_status": "fail",
                    }],
                },
                match_iou=0.5,
            )
            metrics = scoring.aggregate_scores([valid, safety])
            self.assertEqual(metrics["target_candidate_recall"], 1.0)
            self.assertEqual(metrics["anchor_candidate_recall"], 1.0)
            self.assertEqual(metrics["relationship_pair_accuracy"], 1.0)
            self.assertEqual(metrics["valid_final_selection_accuracy"], 1.0)
            self.assertEqual(metrics["wrong_object_acceptances_safety"], 0)
            self.assertEqual(metrics["mean_final_iou"], 1.0)
            self.assertAlmostEqual(metrics["p95_latency_s"], 3.9)

    def test_manifest_requires_human_review_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            result.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "schema_version": 2,
                    "name": "review",
                    "cases": [{
                        "id": "one",
                        "reviewed": False,
                        "scenario": "missing_target",
                        "result_json": "result.json",
                        "expected_accept": False,
                        "safety_case": True,
                        "target_mask": None,
                        "anchor_masks": {},
                        "relationships": [],
                    }],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(scoring.FrozenScoreError, "reviewed must be true"):
                scoring.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
