#!/usr/bin/env python3
"""Tests for no-motion soak acceptance accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import segmentation_soak as soak  # noqa: E402
import grounding_v2  # noqa: E402


class SoakSummaryTests(unittest.TestCase):
    def test_nineteen_of_twenty_passes(self) -> None:
        rows = [
            {
                "passed": index < 19,
                "identity_ok": True,
                "selected_count": 1,
                "malformed_verifier_attempts": 0,
            }
            for index in range(20)
        ]
        summary = soak.summarize(rows, "present")
        self.assertTrue(summary["acceptance_19_of_20"])
        self.assertEqual(summary["passed_rounds"], 19)

    def test_absent_wrong_acceptance_is_counted(self) -> None:
        rows = [{
            "passed": False,
            "identity_ok": True,
            "selected_count": 1,
            "malformed_verifier_attempts": 0,
        }]
        summary = soak.summarize(rows, "absent")
        self.assertEqual(summary["wrong_acceptances_when_absent"], 1)

    def test_v2_round_authenticates_envelope_and_target_mask(self) -> None:
        base = {
            "schema_version": 2,
            "raw_command": "find red box",
            "visual_source_phrase": "red box",
            "action": {"type": "identify", "evidence": "find"},
            "destination": None,
            "target": {
                "id": "target",
                "mention": "red box",
                "head_noun": "box",
                "noun_modifiers": [],
                "attributes": [
                    {"type": "color", "value": "red", "evidence": "red"}
                ],
                "selector": None,
            },
            "anchors": [],
            "relationships": [],
        }
        envelope = grounding_v2.seal_command_envelope(base)
        response = {
            "schema_version": 2,
            "command_envelope": envelope,
            "num_kept": 1,
            "target_mask": {"mask_full": "/unused.png"},
            "verification": {},
        }
        row = soak.score_round(
            response,
            expected_intent=None,
            expected_hash=envelope["envelope_hash"],
            expected="present",
            reference_mask=None,
            min_iou=0.5,
            raw_command="find red box",
        )
        self.assertTrue(row["passed"])
        self.assertTrue(row["identity_ok"])

        wrong_relationship = soak.score_round(
            response,
            expected_intent=None,
            expected_hash=envelope["envelope_hash"],
            expected="present",
            reference_mask=None,
            min_iou=0.5,
            raw_command="find red box",
            expected_relationship="inside",
        )
        self.assertFalse(wrong_relationship["passed"])
        self.assertFalse(wrong_relationship["relationship_ok"])


if __name__ == "__main__":
    unittest.main()
