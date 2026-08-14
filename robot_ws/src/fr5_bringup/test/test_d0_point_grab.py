"""Tests for D0 retry depths and fixed downward fingertip correction."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'd0_point_grab.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('d0_point_grab', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PointGrabGeometryTests(unittest.TestCase):
    """Ensure the extra correction is vertical in the base frame."""

    def test_default_attempts_and_contact_offset(self):
        args = MODULE.parse_args([])

        self.assertEqual(args.grasp_retries, 2)
        self.assertEqual(args.retry_step_mm, 10.0)
        self.assertEqual(args.fingertip_down_offset_mm, 47.0)
        self.assertEqual(
            MODULE.grasp_depths(
                args.grasp_depth_mm, args.grasp_retries, args.retry_step_mm),
            [5.0, 15.0, 25.0],
        )

    def test_down_offset_never_changes_xy(self):
        target = np.asarray([0.4, -0.2, 0.3])
        corrected = MODULE.apply_fingertip_down_offset(target, 47.0)

        np.testing.assert_array_equal(corrected[:2], target[:2])
        np.testing.assert_allclose(corrected, [0.4, -0.2, 0.253])

    def test_rejects_excessive_combined_tip_extension(self):
        with self.assertRaises(SystemExit):
            MODULE.parse_args(['--fingertip-down-offset-mm', '80'])


if __name__ == '__main__':
    unittest.main()
