"""Tests for the no-motion VLA-to-FR5 target bridge."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'vla_pick_target.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('vla_pick_target', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VlaPickTargetTests(unittest.TestCase):
    """Exercise intent, box, depth, and target-contract boundaries."""

    def test_default_minimum_score_matches_sam_presence_gate(self):
        """The bridge must not silently restore the old 0.25 cutoff."""

        self.assertEqual(MODULE.DEFAULT_MIN_SCORE, 0.10)

    def test_dino_is_default_and_agent_fallback_is_opt_in(self):
        """Robot targets use the bounded DINO architecture unless requested."""

        args = MODULE.parse_args(['--text', 'pick up the orange and grey box'])
        self.assertTrue((args.sam_project / 'sam3-dino').is_file())
        self.assertEqual(
            args.vla_project,
            args.sam_project / 'VLA_project',
        )
        self.assertFalse(args.agent_fallback)

        diagnostic = MODULE.parse_args([
            '--text', 'pick up the orange and grey box', '--agent-fallback',
        ])
        self.assertTrue(diagnostic.agent_fallback)

    def test_splits_supported_spatial_qualifier(self):
        """A supported qualifier is canonicalized and removed."""

        self.assertEqual(
            MODULE.split_spatial_qualifier('the rightmost yellow box'),
            ('the yellow box', 'rightmost'),
        )
        self.assertEqual(
            MODULE.split_spatial_qualifier('closest red cup'),
            ('red cup', 'nearest'),
        )

    def test_refuses_multiple_spatial_qualifiers(self):
        """Two competing selectors are ambiguous and refused."""

        with self.assertRaises(MODULE.IntegrationError):
            MODULE.split_spatial_qualifier('leftmost nearest yellow box')

    def test_extracts_normalized_sam_box_center_with_crop_offset(self):
        """A normalized local box center maps into full-frame pixels."""

        with tempfile.TemporaryDirectory() as directory:
            sam_json = Path(directory) / 'sam.json'
            sam_json.write_text(json.dumps({
                'orig_img_w': 640,
                'orig_img_h': 480,
                'pred_boxes': [[0.25, 0.25, 0.5, 0.5]],
            }), encoding='utf-8')
            response = {
                'num_kept': 1,
                'presence_gate': {'kept_indices': [0]},
                'sam_json': str(sam_json),
                'zed_frame': {
                    'crop': {
                        'enabled': True,
                        'applied_xyxy': [10, 20, 650, 500],
                    },
                },
            }

            bbox, local, full, width, height = MODULE.selected_box(response)

        self.assertEqual(bbox, [160, 120, 480, 360])
        self.assertEqual(local, [320, 240])
        self.assertEqual(full, [330, 260])
        self.assertEqual((width, height), (640, 480))

    def test_prefers_versioned_selected_mask_coordinates(self):
        response = {
            'num_kept': 1,
            'selected_mask': {
                'bbox_xywh_crop_pixels': [10, 20, 30, 40],
                'center_xy_crop_pixels': [25.0, 40.0],
                'center_xy_full_pixels': [473.0, 400.0],
                'center_method': 'sam_mask_centroid',
            },
            'zed_frame': {
                'crop': {'output_width': 384, 'output_height': 360},
            },
        }

        bbox, local, full, width, height = MODULE.selected_box(response)

        self.assertEqual(bbox, [10, 20, 40, 60])
        self.assertEqual(local, [25, 40])
        self.assertEqual(full, [473, 400])
        self.assertEqual((width, height), (384, 360))

    def test_refuses_stale_bbox_center_service_contract(self):
        response = {
            'num_kept': 1,
            'selected_mask': {
                'bbox_xywh_crop_pixels': [10, 20, 30, 40],
                'center_xy_crop_pixels': [25.0, 40.0],
                'center_xy_full_pixels': [473.0, 400.0],
            },
            'zed_frame': {
                'crop': {'output_width': 384, 'output_height': 360},
            },
        }

        with self.assertRaisesRegex(MODULE.IntegrationError, 'restart'):
            MODULE.selected_box(response)

    def test_refuses_ambiguous_sam_result(self):
        """An unresolved multiple-mask result cannot become a target."""

        with self.assertRaisesRegex(MODULE.IntegrationError, 'exactly one mask'):
            MODULE.selected_box({'num_kept': 2})

    def test_structured_intent_identity_round_trip_is_fail_closed(self):
        grounding = MODULE.load_grounding_contract(MODULE.DEFAULT_SAM_PROJECT)
        structured = grounding.parse_grounding_intent(
            'small orange box in the bin'
        )
        structured_hash = grounding.intent_hash(structured)
        intent = MODULE.PickIntent(
            transcript='pick up the small orange box in the bin',
            transcript_source='text',
            object_name='box',
            qualifier=None,
            destination='drop zone',
            source_phrase=structured['source_phrase'],
            grounding_intent=structured,
            intent_hash=structured_hash,
        )
        response = {
            'schema_version': 1,
            'grounding_intent': structured,
            'intent_hash': structured_hash,
            'selection': None,
        }
        MODULE.validate_response_identity(response, intent)

        with self.assertRaisesRegex(MODULE.IntegrationError, 'changed'):
            MODULE.validate_response_identity(
                {**response, 'grounding_intent': {**structured, 'category': 'bin'}},
                intent,
            )

        with self.assertRaisesRegex(MODULE.IntegrationError, 'inconsistent'):
            MODULE.validate_response_identity(
                {
                    **response,
                    'selection': {
                        'selector': 'largest',
                    },
                },
                intent,
            )

    def test_depth_quality_gate_and_back_projection(self):
        """Good mask depth back-projects to the expected camera XYZ."""

        response = {
            'object_depth': {
                'objects': [{
                    'score': 0.9,
                    'depth_stats_m': {
                        'valid_fraction': 0.95,
                        'valid_depth_pixels': 100,
                        'median': 2.0,
                        'p10': 1.99,
                        'p90': 2.01,
                    },
                    'xyz_centroid_m': [0.2, 0.4, 2.0],
                }],
            },
        }
        score, depth, _stats, xyz = MODULE.depth_evidence(
            response,
            min_score=0.25,
            min_valid_fraction=0.8,
            min_valid_pixels=20,
        )
        projected = MODULE.backproject_pixel(
            [110, 220], depth,
            {'fx': 1000.0, 'fy': 1000.0, 'cx': 10.0, 'cy': 20.0},
        )

        self.assertEqual(score, 0.9)
        np.testing.assert_allclose(projected, xyz)

    def test_depth_spread_does_not_reject_center_target(self):
        """Object relief is recorded but does not block center targeting."""

        response = {
            'object_depth': {
                'objects': [{
                    'score': 0.9,
                    'depth_stats_m': {
                        'valid_fraction': 0.95,
                        'valid_depth_pixels': 100,
                        'median': 1.0,
                        'p10': 0.95,
                        'p90': 1.10,
                    },
                    'xyz_centroid_m': [0.1, 0.2, 1.0],
                }],
            },
        }

        _score, depth, stats, _xyz = MODULE.depth_evidence(
            response,
            min_score=0.25,
            min_valid_fraction=0.8,
            min_valid_pixels=20,
        )

        self.assertEqual(depth, 1.0)
        self.assertAlmostEqual(stats['p90'] - stats['p10'], 0.15)

    def test_builds_existing_robot_target_contract(self):
        """The bridge output retains the fields B3 and D0 consume."""

        calibration = MODULE.Calibration(
            path=Path('/tmp/calibration.json'),
            raw={'created': 'now', 'source_sha256': 'abc'},
            rotation=np.eye(3),
            translation=np.asarray([0.1, -0.1, 0.0]),
            intrinsics={'fx': 1000.0, 'fy': 1000.0, 'cx': 0.0, 'cy': 0.0},
            camera_points=np.asarray([
                [-1.0, -1.0, 0.5], [1.0, 1.0, 2.0],
                [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
            ]),
            base_points=np.asarray([
                [-1.0, -1.0, 0.5], [1.1, 0.9, 2.0],
                [-0.9, 0.9, 1.0], [1.1, -1.1, 1.0],
            ]),
            resolution='hd720',
        )
        grounding = MODULE.load_grounding_contract(MODULE.DEFAULT_SAM_PROJECT)
        structured = grounding.parse_grounding_intent('box')
        intent = MODULE.PickIntent(
            transcript='pick up the box',
            transcript_source='text',
            object_name='box',
            qualifier=None,
            destination='drop zone',
            source_phrase='box',
            grounding_intent=structured,
            intent_hash=grounding.intent_hash(structured),
        )
        stats = {
            'valid_fraction': 1.0,
            'valid_depth_pixels': 100,
            'median': 1.0,
            'p10': 0.99,
            'p90': 1.01,
        }
        record = MODULE.build_target_record(
            intent=intent,
            response={'path': 'direct', 'sam_prompt': 'box'},
            calibration=calibration,
            captured=datetime(2026, 1, 1, tzinfo=timezone.utc),
            frame_age_s=1.0,
            frame_path=Path('/tmp/frame.png'),
            bbox_local=[90, 190, 110, 210],
            center_local=[100, 200],
            center_full=[100, 200],
            score=0.9,
            depth_m=1.0,
            stats=stats,
            service_xyz=np.asarray([0.1, 0.2, 1.0]),
            max_xyz_disagreement_m=0.001,
            min_object_extent_m=0.005,
            max_object_extent_m=0.6,
            surface_z_margin_m=0.075,
        )

        self.assertEqual(record['schema_version'], 1)
        self.assertEqual(record['base_frame'], 'base_link')
        self.assertEqual(record['camera_frame'], 'zed_left_optical')
        np.testing.assert_allclose(
            record['base_surface_xyz_m'], [0.2, 0.1, 1.0])
        np.testing.assert_allclose(
            record['base_hover_xyz_m'], [0.2, 0.1, 1.1])
        self.assertEqual(record['intent']['grounding'], structured)
        self.assertAlmostEqual(
            record['physical_size_evidence']['largest_extent_m'], 0.02)


if __name__ == '__main__':
    unittest.main()
