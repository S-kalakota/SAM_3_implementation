#!/usr/bin/env python3
"""Offline tests for the versioned resident-service orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_intent  # noqa: E402
import mask_service  # noqa: E402


def decode_test_masks(output):
    masks = output.get("test_masks", [])
    if masks:
        return np.asarray(masks, dtype=bool)
    return np.zeros(
        (0, int(output["orig_img_h"]), int(output["orig_img_w"])),
        dtype=bool,
    )


def service_args(**overrides):
    values = {
        "candidate_max_prompts": 5,
        "intent_parser_mode": "deterministic",
        "intent_max_new_tokens": 128,
        "qwen_model": "test",
        "allow_qwen_downloads": False,
        "qwen_device_map": "cpu",
        "candidate_multiscale": False,
        "candidate_tile_scale": 0.72,
        "candidate_region_padding_fraction": 0.15,
        "candidate_full_frame": False,
        "candidate_dedup_iou": 0.8,
        "candidate_max_count": 12,
        "candidate_min_workspace_retained_fraction": 0.9,
        "presence_conf_threshold": 0.10,
        "min_area": 5,
        "selection_roi": None,
        "selection_min_valid_depth_fraction": 0.8,
        "candidate_min_valid_depth_pixels": 20,
        "candidate_max_depth_spread_mm": 75.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class VersionedRequestTests(unittest.TestCase):
    def test_multiscale_defaults_keep_full_frame_candidates_disabled(self) -> None:
        with patch.object(sys, "argv", ["mask_service.py"]):
            args = mask_service.parse_args()
        self.assertTrue(args.candidate_multiscale)
        self.assertFalse(args.candidate_full_frame)
        self.assertEqual(args.candidate_region_padding_fraction, 0.15)

    def test_accepts_exact_hashed_request(self) -> None:
        intent = grounding_intent.parse_grounding_intent(
            "small orange box in the bin"
        )
        request = mask_service.validate_v1_segment_request(
            {
                "schema_version": 1,
                "source_phrase": intent["source_phrase"],
                "grounding_intent": intent,
                "intent_hash": grounding_intent.intent_hash(intent),
                "use_agent_fallback": True,
            }
        )
        self.assertEqual(request["grounding_intent"], intent)

    def test_refuses_changed_hash_and_extra_task_field(self) -> None:
        intent = grounding_intent.parse_grounding_intent("orange box")
        base = {
            "schema_version": 1,
            "source_phrase": "orange box",
            "grounding_intent": intent,
            "intent_hash": "0" * 64,
            "use_agent_fallback": True,
        }
        with self.assertRaises(grounding_intent.GroundingIntentError):
            mask_service.validate_v1_segment_request(base)
        with self.assertRaises(grounding_intent.GroundingIntentError):
            mask_service.validate_v1_segment_request(
                {**base, "destination": "drop zone"}
            )

    def test_supplied_intent_is_not_reparsed(self) -> None:
        intent = grounding_intent.parse_grounding_intent("topmost orange box")
        bundle = mask_service.parse_request_intent(
            intent["source_phrase"],
            service_args(intent_parser_mode="deterministic"),
            supplied_intent=intent,
        )
        self.assertEqual(bundle["grounding_intent"], intent)
        self.assertEqual(bundle["parser"]["mode"], "supplied_v1")
        self.assertEqual(bundle["prompt_family"][0], "orange box")


class CandidatePipelineTests(unittest.TestCase):
    def test_prompt_runs_merge_into_one_canonical_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_path = root / "frame.png"
            image = np.zeros((40, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), image[:, :, ::-1])
            intent = grounding_intent.parse_grounding_intent(
                "small orange and white box"
            )
            bundle = mask_service.parse_request_intent(
                intent["source_phrase"],
                service_args(),
                supplied_intent=intent,
            )
            call_count = 0

            def fake_sam(*, image_path, text_prompt, output_folder_path):
                nonlocal call_count
                call_count += 1
                path = root / f"sam_{call_count}.json"
                mask = np.zeros((40, 50), dtype=np.uint8)
                mask[10:25, 15:35] = 1
                path.write_text(
                    json.dumps({"pred_scores": [0.8], "test_masks": [mask.tolist()]}),
                    encoding="utf-8",
                )
                return str(path)

            previous_sam = mask_service.STATE.get("sam")
            mask_service.STATE["sam"] = fake_sam
            try:
                with patch.object(
                    mask_service.task6,
                    "decode_agent_masks",
                    side_effect=decode_test_masks,
                ):
                    kept, candidates, merged_json, generation = (
                        mask_service.generate_direct_candidates(
                            rgb_np=image,
                            full_rgb_np=image,
                            frame_info={
                                "crop": {
                                    "enabled": False,
                                    "full_width": 50,
                                    "full_height": 40,
                                    "output_width": 50,
                                    "output_height": 40,
                                },
                                "saved_full_frame": str(frame_path),
                            },
                            frame_path=frame_path,
                            req_dir=root,
                            intent_bundle=bundle,
                            args=service_args(),
                        )
                    )
            finally:
                if previous_sam is None:
                    mask_service.STATE.pop("sam", None)
                else:
                    mask_service.STATE["sam"] = previous_sam

            self.assertEqual(call_count, len(bundle["prompt_family"]))
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(
                candidates[0]["duplicate_count"],
                len(bundle["prompt_family"]),
            )
            self.assertTrue(merged_json.is_file())
            self.assertEqual(generation["merged_candidate_count"], 1)

    def test_source_region_adds_localized_view_and_all_requested_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_path = root / "frame.png"
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), image[:, :, ::-1])
            intent = grounding_intent.parse_grounding_intent(
                "small orange and white box in the upper bin"
            )
            args = service_args(candidate_multiscale=True)
            bundle = mask_service.parse_request_intent(
                intent["source_phrase"],
                args,
                supplied_intent=intent,
            )
            calls = []

            def fake_sam(*, image_path, text_prompt, output_folder_path):
                source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                height, width = source.shape[:2]
                calls.append((text_prompt, Path(output_folder_path).name, width, height))
                path = root / f"sam_{len(calls)}.json"
                masks = []
                scores = []
                if text_prompt == "bin":
                    upper = np.zeros((height, width), dtype=np.uint8)
                    lower = np.zeros((height, width), dtype=np.uint8)
                    upper[5:25, 10:70] = 1
                    lower[50:75, 10:70] = 1
                    masks = [upper.tolist(), lower.tolist()]
                    scores = [0.60, 0.95]
                elif (
                    "localized_region" in Path(output_folder_path).parts
                    and text_prompt == bundle["primary_sam_phrase"]
                ):
                    target = np.zeros((height, width), dtype=np.uint8)
                    target[5:10, 7:12] = 1
                    masks = [target.tolist()]
                    scores = [0.80]
                path.write_text(
                    json.dumps(
                        {
                            "orig_img_h": height,
                            "orig_img_w": width,
                            "pred_scores": scores,
                            "test_masks": masks,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(path)

            previous_sam = mask_service.STATE.get("sam")
            mask_service.STATE["sam"] = fake_sam
            try:
                with patch.object(
                    mask_service.task6,
                    "decode_agent_masks",
                    side_effect=decode_test_masks,
                ):
                    kept, candidates, _merged_json, generation = (
                        mask_service.generate_direct_candidates(
                            rgb_np=image,
                            full_rgb_np=image,
                            frame_info={
                                "crop": {
                                    "enabled": False,
                                    "full_width": 100,
                                    "full_height": 80,
                                    "output_width": 100,
                                    "output_height": 80,
                                },
                                "saved_full_frame": str(frame_path),
                            },
                            frame_path=frame_path,
                            req_dir=root,
                            intent_bundle=bundle,
                            args=args,
                        )
                    )
            finally:
                if previous_sam is None:
                    mask_service.STATE.pop("sam", None)
                else:
                    mask_service.STATE["sam"] = previous_sam

            expected_target_runs = len(bundle["prompt_family"]) + 2 + (4 * 2)
            self.assertEqual(len(calls), expected_target_runs + 1)
            self.assertEqual(len(generation["source_runs"]), expected_target_runs)
            self.assertEqual(generation["successful_source_run_count"], expected_target_runs)
            self.assertEqual(
                [view["view_kind"] for view in generation["views"]],
                [
                    "workspace_crop",
                    "localized_region",
                    "overlapping_tile",
                    "overlapping_tile",
                    "overlapping_tile",
                    "overlapping_tile",
                ],
            )
            localization = generation["region_localization"]
            self.assertEqual(localization["status"], "localized")
            self.assertEqual(localization["prompt"], "bin")
            self.assertEqual(localization["selected_source_candidate_index"], 0)
            self.assertEqual(
                localization["source_bbox_xywh_crop_pixels"],
                [10, 5, 60, 20],
            )
            self.assertEqual(
                localization["padded_roi_xyxy_crop_pixels"],
                [1, 2, 79, 28],
            )
            self.assertEqual(len(kept), 1)
            self.assertEqual(
                candidates[0]["bbox_xywh_crop_pixels"],
                [8, 7, 5, 5],
            )

    def test_all_target_sam_run_errors_fail_after_writing_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_path = root / "frame.png"
            image = np.zeros((40, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), image[:, :, ::-1])
            intent = grounding_intent.parse_grounding_intent("orange box")
            args = service_args()
            bundle = mask_service.parse_request_intent(
                intent["source_phrase"],
                args,
                supplied_intent=intent,
            )

            def failing_sam(**_kwargs):
                raise RuntimeError("synthetic SAM failure")

            previous_sam = mask_service.STATE.get("sam")
            mask_service.STATE["sam"] = failing_sam
            try:
                with self.assertRaisesRegex(RuntimeError, "all target candidate"):
                    mask_service.generate_direct_candidates(
                        rgb_np=image,
                        full_rgb_np=image,
                        frame_info={
                            "crop": {
                                "enabled": False,
                                "full_width": 50,
                                "full_height": 40,
                                "output_width": 50,
                                "output_height": 40,
                            },
                            "saved_full_frame": str(frame_path),
                        },
                        frame_path=frame_path,
                        req_dir=root,
                        intent_bundle=bundle,
                        args=args,
                    )
            finally:
                if previous_sam is None:
                    mask_service.STATE.pop("sam", None)
                else:
                    mask_service.STATE["sam"] = previous_sam

            audit = json.loads((root / "candidate_generation.json").read_text())
            self.assertTrue(audit["all_target_runs_failed"])
            self.assertEqual(audit["successful_source_run_count"], 0)
            self.assertEqual(
                audit["failed_source_run_count"],
                len(bundle["prompt_family"]),
            )

    def test_direct_path_calls_qwen_once_with_merged_candidates(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        depth = np.ones((20, 30), dtype=np.float32)
        first = np.zeros((20, 30), dtype=bool)
        second = np.zeros((20, 30), dtype=bool)
        first[2:8, 3:9] = True
        second[10:16, 18:24] = True
        initial_kept = [(first, 0.9), (second, 0.8)]
        initial_candidates = [
            {
                "index": index,
                "score": score,
                "area_pixels": int(mask.sum()),
                "kept": True,
                "reject_reasons": [],
                "provenance": [{"prompt": "box", "view_id": "workspace"}],
            }
            for index, (mask, score) in enumerate(initial_kept)
        ]
        intent = grounding_intent.parse_grounding_intent("orange box")
        bundle = mask_service.parse_request_intent(
            intent["source_phrase"],
            service_args(),
            supplied_intent=intent,
        )
        verified_kept = [initial_kept[1]]
        verified_candidates = [dict(item) for item in initial_candidates]
        verified_candidates[0]["kept"] = False
        verified_candidates[0]["reject_reasons"] = ["qwen_not_selected"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_path = root / "frame.png"
            cv2.imwrite(str(frame_path), image[:, :, ::-1])
            with (
                patch.object(
                    mask_service,
                    "generate_direct_candidates",
                    return_value=(
                        initial_kept,
                        initial_candidates,
                        root / "merged_candidates.json",
                        {"merged_candidate_count": 2},
                    ),
                ),
                patch.object(
                    mask_service,
                    "verify_candidates_with_qwen",
                    return_value=(
                        verified_kept,
                        verified_candidates,
                        {"status": "accepted", "candidate_overlay": None,
                         "candidate_zoom": None},
                    ),
                ) as verifier,
                patch.object(
                    mask_service,
                    "apply_geometry_safety_gates",
                    return_value=(verified_kept, verified_candidates, {}),
                ),
                patch.object(
                    mask_service,
                    "select_spatial_mask",
                    return_value=(verified_kept, verified_candidates, None),
                ),
                patch.object(
                    mask_service.task2,
                    "overlay_masks",
                    return_value={"output": str(root / "overlay.png")},
                ),
            ):
                mask_service.direct_segment(
                    request=intent["source_phrase"],
                    rgb_np=image,
                    full_rgb_np=image,
                    depth_np=depth,
                    frame_path=frame_path,
                    frame_info={"crop": {"enabled": False}},
                    req_dir=root,
                    args=service_args(),
                    intent_bundle=bundle,
                )

        verifier.assert_called_once()
        self.assertIs(verifier.call_args.kwargs["kept"], initial_kept)
        self.assertIs(verifier.call_args.kwargs["candidates"], initial_candidates)

    def test_geometry_gate_rejects_depth_spread(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        depth = np.ones((10, 10), dtype=np.float32)
        depth[:, 5:] = 1.3
        kept, candidates, report = mask_service.apply_geometry_safety_gates(
            kept=[(mask, 0.9)],
            candidates=[{
                "index": 0,
                "score": 0.9,
                "area_pixels": 100,
                "kept": True,
                "reject_reasons": [],
            }],
            depth_np=depth,
            args=service_args(candidate_max_depth_spread_mm=75.0),
        )
        self.assertEqual(kept, [])
        self.assertIn("excessive_depth_spread", candidates[0]["reject_reasons"])
        self.assertEqual(report["retained_candidate_count"], 0)

    def test_all_deterministic_selectors_choose_expected_mask(self) -> None:
        masks = []
        for x, y, size, depth in ((1, 1, 2, 1.0), (6, 5, 3, 2.0)):
            mask = np.zeros((10, 10), dtype=bool)
            mask[y : y + size, x : x + size] = True
            depth_map = np.full((10, 10), np.nan, dtype=np.float32)
            depth_map[mask] = depth
            masks.append((mask, depth_map))
        combined_depth = np.where(
            np.isfinite(masks[0][1]), masks[0][1], masks[1][1]
        )
        candidates = [
            {"index": index, "score": 0.9, "area_pixels": int(mask.sum()),
             "kept": True, "reject_reasons": []}
            for index, (mask, _depth) in enumerate(masks)
        ]
        expectations = {
            "leftmost": 0,
            "topmost": 0,
            "nearest": 0,
            "smallest": 0,
            "rightmost": 1,
            "bottommost": 1,
            "farthest": 1,
            "largest": 1,
        }
        for selector, expected_index in expectations.items():
            with self.subTest(selector=selector):
                _selected, _updated, selection = mask_service.select_spatial_mask(
                    [(mask, 0.9) for mask, _depth in masks],
                    candidates,
                    selector,
                    depth_np=combined_depth,
                )
                self.assertEqual(selection["selected_candidate_index"], expected_index)


class FinalMaskArtifactTests(unittest.TestCase):
    def test_historical_workspace_crop_keeps_depth_and_full_pixel_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((360, 384), dtype=bool)
            mask[50, 100] = True
            depth = np.full((360, 384), np.nan, dtype=np.float32)
            depth[50, 100] = 0.73
            records = mask_service.write_final_mask_artifacts(
                kept=[(mask, 0.8)],
                kept_indices=[0],
                crop_info={
                    "enabled": True,
                    "full_width": 1280,
                    "full_height": 720,
                    "output_width": 384,
                    "output_height": 360,
                    "applied_xyxy": [448, 360, 832, 720],
                },
                req_dir=Path(directory),
            )
            stats = mask_service.mask_depth.mask_depth_stats(mask, depth)

        self.assertAlmostEqual(stats["median"], 0.73, places=5)
        self.assertEqual(records[0]["center_xy_crop_pixels"], [100.0, 50.0])
        self.assertEqual(records[0]["center_xy_full_pixels"], [548.0, 410.0])

    def test_records_crop_and_full_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:5] = True
            records = mask_service.write_final_mask_artifacts(
                kept=[(mask, 0.8)],
                kept_indices=[3],
                crop_info={
                    "enabled": True,
                    "full_width": 20,
                    "full_height": 15,
                    "output_width": 5,
                    "output_height": 4,
                    "applied_xyxy": [7, 8, 12, 12],
                },
                req_dir=Path(directory),
            )
        self.assertEqual(records[0]["bbox_xywh_crop_pixels"], [2, 1, 3, 2])
        self.assertEqual(records[0]["bbox_xywh_full_pixels"], [9, 9, 3, 2])
        self.assertEqual(records[0]["center_xy_full_pixels"], [10.0, 9.5])


if __name__ == "__main__":
    unittest.main()
