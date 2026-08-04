#!/usr/bin/env python3
"""Focused tests for the fail-closed Qwen candidate verifier."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import qwen_candidate_verifier as verifier  # noqa: E402


def verifier_json(
    *,
    decision: str = "select",
    selected: list[int] | None = None,
    confidence: float = 0.9,
    reason: str = "candidate matches",
) -> str:
    if selected is None:
        selected = [1] if decision == "select" else []
    return json.dumps(
        {
            "decision": decision,
            "selected_candidate_ids": selected,
            "confidence": confidence,
            "reason": reason,
        }
    )


class ParseVerifierResponseTests(unittest.TestCase):
    def test_accepts_strict_select(self) -> None:
        parsed = verifier.parse_verifier_response(
            verifier_json(selected=[2, 1]),
            candidate_count=2,
        )
        self.assertEqual(parsed["decision"], "select")
        self.assertEqual(parsed["selected_candidate_ids"], [1, 2])
        self.assertEqual(parsed["confidence"], 0.9)

    def test_accepts_strict_no_match(self) -> None:
        parsed = verifier.parse_verifier_response(
            verifier_json(decision="no_match", reason="wrong object"),
            candidate_count=2,
        )
        self.assertEqual(parsed["selected_candidate_ids"], [])

    def test_rejects_markdown_wrapped_json(self) -> None:
        with self.assertRaises(verifier.VerifierResponseError):
            verifier.parse_verifier_response(
                f"```json\n{verifier_json()}\n```",
                candidate_count=1,
            )

    def test_rejects_extra_keys(self) -> None:
        value = json.loads(verifier_json())
        value["extra"] = True
        with self.assertRaises(verifier.VerifierResponseError):
            verifier.parse_verifier_response(json.dumps(value), candidate_count=1)

    def test_rejects_invalid_candidate_ids(self) -> None:
        invalid_lists = ([0], [3], [1, 1], [True])
        for selected in invalid_lists:
            with self.subTest(selected=selected):
                with self.assertRaises(verifier.VerifierResponseError):
                    verifier.parse_verifier_response(
                        verifier_json(selected=list(selected)),
                        candidate_count=2,
                    )

    def test_rejects_inconsistent_decision(self) -> None:
        with self.assertRaises(verifier.VerifierResponseError):
            verifier.parse_verifier_response(
                verifier_json(decision="no_match", selected=[1]),
                candidate_count=1,
            )


class RunVisualVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "request": "pick the topmost box",
            "target_phrase": "box",
            "selector": "topmost",
            "frame_path": Path("/tmp/raw.png"),
            "candidate_overlay_path": Path("/tmp/candidates.png"),
            "candidate_zoom_path": Path("/tmp/candidate_zooms.png"),
            "candidate_records": [{"candidate_id": 1, "sam_index": 4}],
            "min_select_confidence": 0.7,
        }

    def test_retries_invalid_format_once_then_accepts(self) -> None:
        responses = iter(["not json", verifier_json()])
        seen_messages = []

        def sender(messages):
            seen_messages.append(list(messages))
            return next(responses)

        result = verifier.run_visual_verifier(
            **self.common,
            send_generate_request=sender,
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_candidate_ids"], [1])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(seen_messages), 2)

    def test_fails_closed_after_two_invalid_responses(self) -> None:
        result = verifier.run_visual_verifier(
            **self.common,
            send_generate_request=lambda _messages: "not json",
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["selected_candidate_ids"], [])
        self.assertEqual(len(result["attempts"]), 2)

    def test_fails_closed_below_confidence_threshold(self) -> None:
        result = verifier.run_visual_verifier(
            **self.common,
            send_generate_request=lambda _messages: verifier_json(confidence=0.69),
        )
        self.assertEqual(result["status"], "low_confidence")
        self.assertEqual(result["decision"], "select")
        self.assertEqual(result["selected_candidate_ids"], [])

    def test_no_match_is_not_selected(self) -> None:
        result = verifier.run_visual_verifier(
            **self.common,
            send_generate_request=lambda _messages: verifier_json(
                decision="no_match",
                confidence=0.95,
                reason="mask is the storage bin",
            ),
        )
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["selected_candidate_ids"], [])


class ApplySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask_a = np.zeros((5, 6), dtype=bool)
        self.mask_a[1:3, 1:3] = True
        self.mask_b = np.zeros((5, 6), dtype=bool)
        self.mask_b[2:5, 3:6] = True
        self.kept = [(self.mask_a, 0.8), (self.mask_b, 0.9)]
        self.candidates = [
            {
                "index": 0,
                "score": 0.1,
                "area_pixels": 1,
                "kept": False,
                "reject_reasons": ["low_score"],
            },
            {
                "index": 1,
                "score": 0.8,
                "area_pixels": 4,
                "kept": True,
                "reject_reasons": [],
            },
            {
                "index": 2,
                "score": 0.9,
                "area_pixels": 9,
                "kept": True,
                "reject_reasons": [],
            },
        ]

    def test_preserves_sam_index_mapping(self) -> None:
        kept, candidates, selected_indices = verifier.apply_verifier_selection(
            self.kept,
            self.candidates,
            {"status": "selected", "selected_candidate_ids": [2]},
        )
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0][0], self.mask_b)
        self.assertEqual(selected_indices, [2])
        self.assertIn("qwen_verifier_not_selected", candidates[1]["reject_reasons"])
        self.assertTrue(candidates[2]["kept"])

    def test_rejects_every_candidate_on_verifier_error(self) -> None:
        kept, candidates, selected_indices = verifier.apply_verifier_selection(
            self.kept,
            self.candidates,
            {"status": "error", "selected_candidate_ids": []},
        )
        self.assertEqual(kept, [])
        self.assertEqual(selected_indices, [])
        self.assertTrue(
            all(
                "qwen_verifier_error" in candidate["reject_reasons"]
                for candidate in candidates
                if candidate["index"] in {1, 2}
            )
        )

    def test_records_oversized_policy_rejection_on_sam_candidate(self) -> None:
        policy_result = verifier.apply_max_area_fraction_policy(
            {
                "status": "selected",
                "decision": "select",
                "selected_candidate_ids": [1],
                "reason": "model selected it",
            },
            [{"candidate_id": 1, "area_fraction": 0.40}],
            max_area_fraction=0.25,
        )
        kept, candidates, selected_indices = verifier.apply_verifier_selection(
            self.kept,
            self.candidates,
            policy_result,
        )
        self.assertEqual(kept, [])
        self.assertEqual(selected_indices, [])
        self.assertIn(
            "qwen_verifier_oversized_candidate",
            candidates[1]["reject_reasons"],
        )
        self.assertIn(
            "qwen_verifier_not_selected",
            candidates[2]["reject_reasons"],
        )
        self.assertNotIn(
            "qwen_verifier_oversized_candidate",
            candidates[2]["reject_reasons"],
        )


class AreaFractionPolicyTests(unittest.TestCase):
    def test_retains_small_qwen_selection(self) -> None:
        result = verifier.apply_max_area_fraction_policy(
            {
                "status": "selected",
                "decision": "select",
                "selected_candidate_ids": [1],
                "reason": "tight mask",
            },
            [{"candidate_id": 1, "area_fraction": 0.021}],
            max_area_fraction=0.25,
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_candidate_ids"], [1])
        self.assertEqual(result["policy_rejections"], [])

    def test_rejects_large_qwen_selection(self) -> None:
        result = verifier.apply_max_area_fraction_policy(
            {
                "status": "selected",
                "decision": "select",
                "selected_candidate_ids": [1],
                "reason": "model selected it",
            },
            [{"candidate_id": 1, "area_fraction": 0.331}],
            max_area_fraction=0.25,
        )
        self.assertEqual(result["status"], "policy_rejected")
        self.assertEqual(result["qwen_decision"], "select")
        self.assertEqual(result["qwen_selected_candidate_ids"], [1])
        self.assertEqual(result["selected_candidate_ids"], [])
        self.assertEqual(result["final_decision"], "no_match")

    def test_keeps_small_and_rejects_large_from_mixed_selection(self) -> None:
        result = verifier.apply_max_area_fraction_policy(
            {
                "status": "selected",
                "decision": "select",
                "selected_candidate_ids": [1, 2],
                "reason": "two candidates",
            },
            [
                {"candidate_id": 1, "area_fraction": 0.02},
                {"candidate_id": 2, "area_fraction": 0.40},
            ],
            max_area_fraction=0.25,
        )
        self.assertEqual(result["status"], "selected_with_policy_rejections")
        self.assertEqual(result["selected_candidate_ids"], [1])
        self.assertEqual(result["policy_rejections"][0]["candidate_id"], 2)


class RenderCandidatesTests(unittest.TestCase):
    def test_writes_numbered_overlay(self) -> None:
        rgb = np.full((40, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:25, 10:30] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.png"
            result = verifier.render_numbered_candidates(
                rgb,
                [(mask, 0.9)],
                output,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(result["num_masks_drawn"], 1)

    def test_writes_candidate_zoom_sheet(self) -> None:
        rgb = np.full((40, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((40, 50), dtype=bool)
        mask[15:20, 20:35] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate_zooms.png"
            result = verifier.render_candidate_zooms(
                rgb,
                [(mask, 0.9)],
                output,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(result["num_candidates_drawn"], 1)


if __name__ == "__main__":
    unittest.main()
