#!/usr/bin/env python3
"""Unit tests for bounded Grounding DINO proposal generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_dino  # noqa: E402


class PhraseFamilyTests(unittest.TestCase):
    def test_box_family_is_exact_head_and_controlled_bounded_synonyms(self) -> None:
        self.assertEqual(
            grounding_dino.build_phrase_family("Small orange box"),
            [
                "small orange box",
                "box",
                "package",
                "carton",
                "flat rectangular item",
            ],
        )

    def test_phrase_family_never_exceeds_five(self) -> None:
        phrases = grounding_dino.build_phrase_family("orange cardboard box")
        self.assertLessEqual(len(phrases), 5)
        self.assertEqual(phrases[0], "orange cardboard box")
        self.assertEqual(phrases[1], "cardboard box")


class ProposalGeometryTests(unittest.TestCase):
    def test_threshold_clamp_padding_nms_and_top_three(self) -> None:
        detections = [
            {
                "dino_index": 0,
                "phrase": "box",
                "text_label": "box",
                "dino_score": 0.95,
                "box_xyxy_crop_pixels": [-10, 10, 110, 90],
            },
            {
                "dino_index": 1,
                "phrase": "package",
                "text_label": "package",
                "dino_score": 0.90,
                "box_xyxy_crop_pixels": [0, 10, 105, 90],
            },
            {
                "dino_index": 2,
                "phrase": "carton",
                "text_label": "carton",
                "dino_score": 0.80,
                "box_xyxy_crop_pixels": [120, 0, 160, 30],
            },
            {
                "dino_index": 3,
                "phrase": "box",
                "text_label": "box",
                "dino_score": 0.70,
                "box_xyxy_crop_pixels": [120, 35, 160, 65],
            },
            {
                "dino_index": 4,
                "phrase": "box",
                "text_label": "box",
                "dino_score": 0.60,
                "box_xyxy_crop_pixels": [120, 70, 160, 100],
            },
            {
                "dino_index": 5,
                "phrase": "box",
                "text_label": "box",
                "dino_score": 0.24,
                "box_xyxy_crop_pixels": [170, 70, 190, 95],
            },
        ]
        proposals, rejected = grounding_dino.prepare_proposals(
            detections,
            image_width=200,
            image_height=100,
            box_threshold=0.25,
            nms_iou=0.50,
            max_proposals=3,
            padding_fraction=0.05,
        )

        self.assertEqual([item["dino_index"] for item in proposals], [0, 2, 3])
        self.assertEqual(
            proposals[0]["original_box_xyxy_crop_pixels"],
            [0.0, 10.0, 110.0, 90.0],
        )
        self.assertEqual(
            proposals[0]["padded_box_xyxy_crop_pixels"],
            [0.0, 6.0, 115.5, 94.0],
        )
        reasons = {item["dino_index"]: item["reject_reason"] for item in rejected}
        self.assertEqual(reasons[1], "cross_phrase_nms")
        self.assertEqual(reasons[4], "proposal_limit")
        self.assertEqual(reasons[5], "below_box_threshold")

    def test_pixel_xyxy_to_normalized_sam_xywh(self) -> None:
        self.assertEqual(
            grounding_dino.pixel_xyxy_to_normalized_sam_xywh(
                [10, 20, 110, 220],
                200,
                400,
            ),
            [0.05, 0.05, 0.5, 0.5],
        )

    def test_historical_crop_translation(self) -> None:
        self.assertEqual(
            grounding_dino.crop_point_to_full(
                [12, 34],
                [448, 360, 384, 360],
            ),
            [460.0, 394.0],
        )

    def test_mask_box_is_derived_from_nonzero_pixels(self) -> None:
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:40, 10:60] = True
        self.assertEqual(
            grounding_dino.mask_bbox_xywh_normalized(mask),
            [0.05, 0.2, 0.25, 0.2],
        )


class AdapterBoundaryTests(unittest.TestCase):
    def test_malformed_dino_output_fails_closed(self) -> None:
        with self.assertRaises(grounding_dino.GroundingDinoOutputError):
            grounding_dino.decode_dino_result(
                {
                    "scores": [0.9, 0.8],
                    "boxes": [[1, 2, 3, 4]],
                    "text_labels": ["box"],
                },
                ["box"],
            )

    def test_missing_cached_snapshot_has_explicit_error(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class FakeTorch:
            cuda = FakeCuda()
            bfloat16 = "bf16"

        class MissingProcessor:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise OSError("snapshot not found")

        class UnusedModel:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise AssertionError("processor should fail first")

        with self.assertRaisesRegex(
            grounding_dino.GroundingDinoCacheError,
            "cache_grounding_dino.py",
        ):
            grounding_dino.GroundingDinoAdapter(
                "missing/model",
                processor_class=MissingProcessor,
                model_class=UnusedModel,
                torch_module=FakeTorch,
            )


if __name__ == "__main__":
    unittest.main()
