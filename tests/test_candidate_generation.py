#!/usr/bin/env python3
"""Tests for canonical-mask projection and prompt/view deduplication."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import candidate_generation as generation  # noqa: E402


class CandidateProjectionTests(unittest.TestCase):
    def test_region_roi_is_padded_and_clamped_to_image(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:12, 0:10] = True
        self.assertEqual(
            generation.padded_mask_roi(mask, padding_fraction=0.15),
            (0, 0, 12, 14),
        )

    def test_upper_region_selection_beats_a_higher_scoring_lower_region(self) -> None:
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:7, 5:20] = True
        masks[1, 13:18, 5:20] = True
        selected = generation.select_source_region_candidate(
            masks,
            np.asarray([0.60, 0.95]),
            source_region="upper bin",
            conf_threshold=0.10,
            min_area=5,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(generation.source_region_prompt("upper bin"), "bin")

    def test_unqualified_region_uses_highest_sam_score(self) -> None:
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:7, 5:20] = True
        masks[1, 13:18, 5:20] = True
        selected = generation.select_source_region_candidate(
            masks,
            np.asarray([0.60, 0.95]),
            source_region="bin",
            conf_threshold=0.10,
            min_area=5,
        )
        self.assertEqual(selected, 1)

    def test_tile_mask_projects_into_canonical_crop(self) -> None:
        tile_mask = np.zeros((4, 5), dtype=bool)
        tile_mask[1:3, 2:5] = True
        projected = generation.project_mask_to_canonical(
            tile_mask,
            source_roi_xyxy=(0, 0, 5, 4),
            canonical_roi_xyxy=(3, 2, 8, 6),
            canonical_shape_hw=(8, 10),
        )
        expected = np.zeros((8, 10), dtype=bool)
        expected[3:5, 5:8] = True
        np.testing.assert_array_equal(projected, expected)

    def test_full_frame_mask_is_cropped_to_workspace(self) -> None:
        full = np.zeros((8, 12), dtype=bool)
        full[3:6, 5:9] = True
        projected = generation.project_mask_to_canonical(
            full,
            source_roi_xyxy=(4, 2, 10, 7),
            canonical_roi_xyxy=(0, 0, 6, 5),
            canonical_shape_hw=(5, 6),
        )
        self.assertEqual(projected.shape, (5, 6))
        self.assertEqual(int(projected.sum()), 12)

    def test_expands_crop_mask_without_coordinate_shift(self) -> None:
        crop = np.zeros((3, 4), dtype=bool)
        crop[1, 2] = True
        full = generation.expand_crop_mask_to_full(
            crop,
            {
                "enabled": True,
                "full_width": 10,
                "full_height": 8,
                "applied_xyxy": [3, 4, 7, 7],
            },
        )
        self.assertTrue(full[5, 5])
        self.assertEqual(int(full.sum()), 1)

    def test_historical_crop_translation_preserves_target_pixel(self) -> None:
        crop = np.zeros((360, 384), dtype=bool)
        crop[50, 100] = True
        crop_info = {
            "enabled": True,
            "full_width": 1280,
            "full_height": 720,
            "applied_xyxy": [448, 360, 832, 720],
        }
        full = generation.expand_crop_mask_to_full(crop, crop_info)
        self.assertTrue(full[410, 548])
        self.assertEqual(int(full.sum()), 1)
        self.assertEqual(
            generation.crop_bbox_to_full([100, 50, 1, 1], crop_info),
            [548, 410, 1, 1],
        )

    def test_overlapping_tiles_cover_every_pixel(self) -> None:
        rois = generation.overlapping_tile_rois(100, 80, scale=0.7)
        coverage = np.zeros((80, 100), dtype=bool)
        for x0, y0, x1, y1 in rois:
            coverage[y0:y1, x0:x1] = True
        self.assertEqual(len(rois), 4)
        self.assertTrue(coverage.all())


class CandidateDeduplicationTests(unittest.TestCase):
    @staticmethod
    def raw(mask: np.ndarray, score: float, prompt: str) -> dict:
        return {
            "mask": mask,
            "score": score,
            "prompt": prompt,
            "view_id": "workspace",
            "view_kind": "workspace",
            "source_candidate_index": 0,
            "source_json": "/tmp/result.json",
        }

    def test_near_identical_masks_merge_and_keep_best_score(self) -> None:
        first = np.zeros((20, 20), dtype=bool)
        first[2:12, 2:12] = True
        second = first.copy()
        second[2, 2] = False
        merged = generation.deduplicate_candidates(
            [self.raw(first, 0.7, "box"), self.raw(second, 0.9, "package")],
            iou_threshold=0.8,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["score"], 0.9)
        self.assertEqual(merged[0]["duplicate_count"], 2)
        self.assertEqual(
            {item["prompt"] for item in merged[0]["provenance"]},
            {"box", "package"},
        )

    def test_nested_object_and_container_masks_do_not_merge(self) -> None:
        container = np.ones((20, 20), dtype=bool)
        target = np.zeros((20, 20), dtype=bool)
        target[8:12, 7:13] = True
        merged = generation.deduplicate_candidates(
            [self.raw(container, 0.8, "bin"), self.raw(target, 0.7, "box")],
            iou_threshold=0.8,
        )
        self.assertEqual(len(merged), 2)

    def test_gate_uses_new_point_one_threshold_and_candidate_cap(self) -> None:
        masks = []
        for index, score in enumerate((0.09, 0.7, 0.6)):
            mask = np.zeros((20, 20), dtype=bool)
            mask[index : index + 5, index : index + 5] = True
            masks.append(self.raw(mask, score, str(index)))
        merged = generation.deduplicate_candidates(masks, iou_threshold=0.95)
        kept, candidates = generation.gate_merged_candidates(
            merged,
            conf_threshold=0.10,
            min_area=5,
            max_candidates=1,
        )
        self.assertEqual(len(kept), 1)
        reasons = [reason for item in candidates for reason in item["reject_reasons"]]
        self.assertIn("low_score", reasons)
        self.assertIn("candidate_limit", reasons)

    def test_gate_rejects_mask_mostly_outside_workspace(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        raw = self.raw(mask, 0.9, "box")
        raw["workspace_retained_fraction"] = 0.4
        merged = generation.deduplicate_candidates([raw])
        kept, candidates = generation.gate_merged_candidates(
            merged,
            conf_threshold=0.1,
            min_area=5,
            max_candidates=2,
            min_workspace_retained_fraction=0.9,
        )
        self.assertEqual(kept, [])
        self.assertIn(
            "mask_extends_outside_workspace",
            candidates[0]["reject_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
