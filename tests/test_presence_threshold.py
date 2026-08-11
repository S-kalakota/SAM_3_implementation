#!/usr/bin/env python3
"""Regression tests for the shared SAM mask-presence threshold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import task2_sam31_image_prompt as task2  # noqa: E402
import task5_zed_live_prompt as task5  # noqa: E402


class PresenceThresholdTests(unittest.TestCase):
    def test_live_paths_share_point_one_default(self) -> None:
        self.assertEqual(task2.DEFAULT_PRESENCE_CONF_THRESH, 0.10)
        self.assertEqual(
            task5.DEFAULT_PRESENCE_CONF_THRESH,
            task2.DEFAULT_PRESENCE_CONF_THRESH,
        )

    def test_default_gate_accepts_score_above_point_one_only(self) -> None:
        masks = np.ones((2, 20, 20), dtype=bool)
        kept, candidates = task2.gate_masks(
            masks,
            np.asarray([0.09, 0.11]),
            min_area=0,
        )

        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0][1], 0.11, places=6)
        self.assertEqual(candidates[0]["reject_reasons"], ["low_score"])
        self.assertTrue(candidates[1]["kept"])


if __name__ == "__main__":
    unittest.main()
