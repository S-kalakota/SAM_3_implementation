#!/usr/bin/env python3
"""Metric and manifest tests for frozen-frame A/B evaluation."""

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

import evaluate_segmentation as evaluation  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_scores_candidate_recall_selection_and_absence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = np.zeros((8, 10), dtype=np.uint8)
            expected[3:6, 4:8] = 255
            expected_path = root / "expected.png"
            final_path = root / "final.png"
            crop_path = root / "crop.png"
            cv2.imwrite(str(expected_path), expected)
            cv2.imwrite(str(final_path), expected)
            cv2.imwrite(str(crop_path), expected[2:7, 3:9])
            response = {
                "num_kept": 1,
                "elapsed_s": 1.5,
                "zed_frame": {
                    "crop": {
                        "enabled": True,
                        "full_width": 10,
                        "full_height": 8,
                        "applied_xyxy": [3, 2, 9, 7],
                    }
                },
                "candidate_generation": {
                    "merged_candidate_count": 1,
                    "candidate_manifest": [{"mask_path_crop": str(crop_path)}],
                },
                "final_masks": [{"mask_full": str(final_path)}],
                "verification": {"status": "selected", "attempts": []},
                "overlay": {"output": "/tmp/overlay.png"},
            }
            case = {
                "id": "present",
                "expected_present": True,
                "expected_mask": str(expected_path),
            }
            score = evaluation.score_case(case, response, match_iou=0.5)
            self.assertTrue(score["raw_candidate_recalled"])
            self.assertTrue(score["correct_selection"])
            self.assertEqual(score["max_final_iou"], 1.0)

            absent = evaluation.score_case(
                {"id": "absent", "expected_present": False, "expected_mask": None},
                {**response, "num_kept": 1, "final_masks": []},
                match_iou=0.5,
            )
            self.assertTrue(absent["absent_false_positive"])

    def test_manifest_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            mask = root / "mask.png"
            cv2.imwrite(str(image), np.zeros((4, 4, 3), dtype=np.uint8))
            cv2.imwrite(str(mask), np.zeros((4, 4), dtype=np.uint8))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "schema_version": 1,
                    "name": "test",
                    "cases": [{
                        "id": "one",
                        "image": "frame.png",
                        "source_phrase": "orange box",
                        "expected_present": True,
                        "expected_mask": "mask.png",
                    }],
                }),
                encoding="utf-8",
            )
            loaded = evaluation.load_manifest(manifest)
            self.assertEqual(loaded["cases"][0]["image"], str(image.resolve()))

    def test_aggregate_reports_required_metrics(self) -> None:
        cases = [
            {
                "expected_present": True,
                "raw_candidate_recalled": True,
                "correct_selection": True,
                "absent_false_positive": False,
                "max_final_iou": 0.8,
                "max_final_dice": 0.9,
                "malformed_verifier_attempts": 1,
                "malformed_intent_attempts": 0,
                "malformed_final_response": False,
                "elapsed_s": 2.0,
            },
            {
                "expected_present": False,
                "raw_candidate_recalled": False,
                "correct_selection": False,
                "absent_false_positive": True,
                "max_final_iou": 0.0,
                "max_final_dice": 0.0,
                "malformed_verifier_attempts": 0,
                "malformed_intent_attempts": 0,
                "malformed_final_response": False,
                "elapsed_s": 4.0,
            },
        ]
        metrics = evaluation.aggregate_scores(cases)
        self.assertEqual(metrics["raw_candidate_recall"], 1.0)
        self.assertEqual(metrics["correct_object_selection_accuracy"], 0.5)
        self.assertEqual(metrics["no_object_false_positive_rate"], 1.0)
        self.assertEqual(metrics["mean_latency_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
