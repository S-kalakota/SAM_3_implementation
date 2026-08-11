#!/usr/bin/env python3
"""Synthetic tests for DINO-anchored ZED depth mask refinement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mask_depth  # noqa: E402


class DepthDiscontinuityRefinementTests(unittest.TestCase):
    def test_removes_farther_lobe_and_preserves_attached_unknown_depth(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[20:60, 20:100] = True
        depth = np.full(mask.shape, 1.0, dtype=np.float32)
        depth[20:60, 70:100] = 1.5
        depth[30:35, 40:45] = np.nan

        refined, report = mask_depth.refine_mask_at_depth_discontinuities(
            mask,
            depth,
            anchor_box_xyxy=[20, 20, 100, 60],
        )

        self.assertEqual(report["status"], "applied")
        self.assertTrue(report["applied"])
        self.assertGreater(report["removed_farther_pixels"], 0)
        self.assertFalse(refined[20:60, 70:100].any())
        self.assertTrue(refined[25, 40])
        self.assertTrue(refined[32, 42])
        self.assertFalse(np.any(refined & ~mask))

    def test_center_patch_wins_when_mask_median_is_background(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[20:60, 10:110] = True
        depth = np.full(mask.shape, 1.42, dtype=np.float32)
        depth[20:60, 40:80] = 1.15

        refined, report = mask_depth.refine_mask_at_depth_discontinuities(
            mask,
            depth,
            anchor_box_xyxy=[20, 20, 100, 60],
        )

        self.assertEqual(report["status"], "applied")
        self.assertAlmostEqual(report["anchor_reference_depth_m"], 1.15, places=4)
        self.assertAlmostEqual(report["before_depth_stats_m"]["median"], 1.42, places=4)
        self.assertAlmostEqual(report["after_depth_stats_m"]["median"], 1.15, places=4)
        self.assertTrue(refined[30, 50])
        self.assertFalse(refined[30, 20])
        self.assertFalse(refined[30, 90])

    def test_gradual_depth_slope_is_not_cut(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[20:60, 20:100] = True
        depth = np.ones(mask.shape, dtype=np.float32)
        slope = np.linspace(1.0, 1.08, 80, dtype=np.float32)
        depth[20:60, 20:100] = slope[None, :]

        refined, report = mask_depth.refine_mask_at_depth_discontinuities(
            mask,
            depth,
            anchor_box_xyxy=[20, 20, 100, 60],
        )

        self.assertEqual(report["status"], "no_depth_discontinuity")
        np.testing.assert_array_equal(refined, mask)

    def test_missing_center_depth_returns_original_mask(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[20:60, 20:100] = True
        depth = np.ones(mask.shape, dtype=np.float32)
        depth[30:50, 50:70] = np.nan

        refined, report = mask_depth.refine_mask_at_depth_discontinuities(
            mask,
            depth,
            anchor_box_xyxy=[20, 20, 100, 60],
        )

        self.assertEqual(report["status"], "skipped_insufficient_anchor_depth")
        np.testing.assert_array_equal(refined, mask)

    def test_retention_guardrail_rejects_destructive_cut(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[10:70, 10:110] = True
        depth = np.full(mask.shape, 1.5, dtype=np.float32)
        depth[34:46, 54:66] = 1.0

        refined, report = mask_depth.refine_mask_at_depth_discontinuities(
            mask,
            depth,
            anchor_box_xyxy=[10, 10, 110, 70],
        )

        self.assertEqual(report["status"], "skipped_retention_guardrail")
        self.assertLess(report["proposed_retained_mask_fraction"], 0.20)
        np.testing.assert_array_equal(refined, mask)


if __name__ == "__main__":
    unittest.main()
