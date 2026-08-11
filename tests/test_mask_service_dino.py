#!/usr/bin/env python3
"""Pipeline-contract tests for DINO -> SAM -> Qwen integration."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mask_service_dino as mask_service  # noqa: E402
import task6_sam31_agent as task6  # noqa: E402


def make_args(**updates):
    values = {
        "pipeline_mode": "dino",
        "dino_model": "IDEA-Research/grounding-dino-base",
        "dino_box_threshold": 0.25,
        "dino_text_threshold": 0.20,
        "dino_nms_iou": 0.50,
        "dino_max_proposals": 3,
        "dino_box_padding": 0.05,
        "presence_conf_threshold": 0.25,
        "min_area": 20,
        "selection_min_valid_depth_fraction": 0.8,
        "selection_roi": None,
        "depth_refinement": True,
        "qwen_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen_device_map": "auto",
        "allow_qwen_downloads": False,
        "verifier_max_new_tokens": 256,
        "verifier_min_confidence": 0.70,
        "verifier_max_area_fraction": 0.25,
        "view": "LEFT",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def one_mask(x: int = 10) -> np.ndarray:
    mask = np.zeros((360, 384), dtype=bool)
    mask[20:40, x : x + 30] = True
    return mask


def candidate(index: int, mask: np.ndarray, *, kept: bool = True) -> dict:
    return {
        "index": index,
        "score": 0.9,
        "area_pixels": int(mask.sum()),
        "kept": kept,
        "reject_reasons": [] if kept else ["qwen_verifier_no_match"],
    }


def accepted_verification(kept, candidates):
    return (
        kept,
        candidates,
        {
            "status": "selected",
            "selected_candidate_ids": list(range(1, len(kept) + 1)),
            "selected_sam_indices": [item["index"] for item in candidates if item["kept"]],
            "candidate_overlay": None,
            "candidate_zoom": None,
            "inference_s": 0.1,
            "elapsed_s": 0.2,
        },
    )


class QwenVerificationInputTests(unittest.TestCase):
    def test_dino_candidates_use_clean_identity_crops(self) -> None:
        mask = one_mask()
        kept = [(mask, 0.9)]
        candidate_record = candidate(0, mask)
        candidate_record["proposal_provenance"] = {
            "proposal_id": 1,
            "dino_phrase": "orange and grey box",
            "dino_score": 0.8,
            "sam_prompt_box_xyxy_crop_pixels": [8.0, 18.0, 52.0, 42.0],
            "dino_original_box_xyxy_crop_pixels": [10.0, 20.0, 50.0, 40.0],
        }
        identity_result = {
            "status": "selected",
            "decision": "select",
            "selected_candidate_ids": [1],
            "model_selected_candidate_ids": [1],
            "candidate_assessments": [
                {
                    "candidate_id": 1,
                    "most_likely_object": "orange and grey box",
                    "matches_target": True,
                }
            ],
            "confidence": 0.95,
            "reason": "candidate 1 is the requested box",
            "attempts": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            request_dir = Path(directory)
            frame_path = request_dir / "frame.png"
            Image.fromarray(np.zeros((360, 384, 3), dtype=np.uint8)).save(
                frame_path
            )
            with (
                mock.patch.object(
                    mask_service.candidate_verifier,
                    "run_identity_verifier",
                    return_value=identity_result,
                ) as identity_call,
                mock.patch.object(
                    mask_service.candidate_verifier,
                    "run_visual_verifier",
                ) as legacy_call,
            ):
                selected, updated, verification = (
                    mask_service.verify_candidates_with_qwen(
                        request="pick up the orange and grey box",
                        target_phrase="orange and grey box",
                        selector=None,
                        rgb_np=np.zeros((360, 384, 3), dtype=np.uint8),
                        frame_path=frame_path,
                        req_dir=request_dir,
                        artifact_stem="dino",
                        kept=kept,
                        candidates=[candidate_record],
                        args=make_args(),
                    )
                )

        identity_call.assert_called_once()
        legacy_call.assert_not_called()
        self.assertEqual(len(selected), 1)
        self.assertTrue(updated[0]["kept"])
        self.assertEqual(
            verification["verifier_input_mode"],
            "clean_dino_crops_identity_only",
        )
        self.assertTrue(
            verification["candidate_clean_crops"].endswith(
                "dino_qwen_clean_dino_crops.png"
            )
        )


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_state = dict(mask_service.STATE)
        mask_service.STATE.clear()

    def tearDown(self) -> None:
        mask_service.STATE.clear()
        mask_service.STATE.update(self.saved_state)

    def test_one_dino_call_and_no_more_than_three_sam_box_calls(self) -> None:
        class FakeDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, _rgb, phrases, **_kwargs):
                self.calls += 1
                detections = []
                for index, x1 in enumerate((5, 100, 195, 290)):
                    detections.append(
                        {
                            "dino_index": index,
                            "phrase": phrases[min(index, len(phrases) - 1)],
                            "text_label": "box",
                            "dino_score": 0.9 - index * 0.05,
                            "box_xyxy_crop_pixels": [x1, 10, x1 + 60, 90],
                        }
                    )
                return {
                    "phrases": list(phrases),
                    "detections": detections,
                    "inference_s": 0.01,
                    "total_s": 0.02,
                }

        class FakeSam:
            def __init__(self):
                self.proposal_count = None

            def segment_boxes(self, *, proposals, output_folder_path, **_kwargs):
                self.proposal_count = len(proposals)
                output = Path(output_folder_path) / "combined_candidates.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                provenance = []
                for index, proposal in enumerate(proposals):
                    provenance.append(
                        {
                            "candidate_index": index,
                            "proposal_id": proposal["proposal_id"],
                            "dino_phrase": proposal["phrase"],
                            "dino_score": proposal["dino_score"],
                            "dino_original_box_xyxy_crop_pixels": proposal[
                                "original_box_xyxy_crop_pixels"
                            ],
                            "sam_prompt_box_xyxy_crop_pixels": proposal[
                                "padded_box_xyxy_crop_pixels"
                            ],
                        }
                    )
                output.write_text(
                    json.dumps(
                        {
                            "orig_img_w": 384,
                            "orig_img_h": 360,
                            "pred_scores": [0.9] * len(proposals),
                            "pred_boxes": [[0.0, 0.0, 0.1, 0.1]] * len(proposals),
                            "pred_masks": ["fake"] * len(proposals),
                            "proposal_provenance": provenance,
                            "sam_box_prompt_timings": [
                                {"proposal_id": item["proposal_id"], "elapsed_s": 0.01}
                                for item in proposals
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return str(output)

        detector = FakeDetector()
        sam = FakeSam()
        mask_service.STATE.update({"dino": detector, "sam": sam})
        decoded_masks = np.stack([one_mask(10), one_mask(120), one_mask(230)])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            mask_service.task6,
            "decode_agent_masks",
            return_value=decoded_masks,
        ):
            result = mask_service.dino_candidate_generation(
                target_phrase="small orange box",
                rgb_np=np.zeros((360, 384, 3), dtype=np.uint8),
                frame_path=Path(directory) / "frame.png",
                req_dir=Path(directory),
                args=make_args(),
            )

        self.assertEqual(detector.calls, 1)
        self.assertEqual(sam.proposal_count, 3)
        self.assertEqual(len(result["kept"]), 3)
        self.assertEqual(
            [item["index"] for item in result["candidates"]],
            [0, 1, 2],
        )
        self.assertEqual(
            [item["proposal_provenance"]["candidate_index"] for item in result["candidates"]],
            [0, 1, 2],
        )


class SamBoxPromptTests(unittest.TestCase):
    def test_reuses_one_image_state_and_keeps_highest_center_valid_mask(self) -> None:
        class FakeModel:
            def __init__(self):
                self.init_calls = 0
                self.prompt_calls = []

            def init_state(self, **_kwargs):
                self.init_calls += 1
                return {"state": True}

            def add_prompt(self, **kwargs):
                self.prompt_calls.append(kwargs)
                normalized = kwargs["boxes_xywh"][0]
                x1 = int(round(normalized[0] * 384))
                inside = np.zeros((360, 384), dtype=bool)
                inside[20:35, x1 + 5 : x1 + 20] = True
                outside = np.zeros((360, 384), dtype=bool)
                outside[320:340, 5:20] = True
                return 0, {
                    "out_binary_masks": np.stack([outside, inside]),
                    "out_probs": np.asarray([0.99, 0.80], dtype=np.float32),
                }

        class FakeRender:
            def save(self, output_path):
                Image.new("RGB", (384, 360)).save(output_path)

        proposals = []
        for index, x1 in enumerate((10, 110, 210), start=1):
            proposals.append(
                {
                    "proposal_id": index,
                    "dino_index": index - 1,
                    "phrase": "box",
                    "text_label": "box",
                    "dino_score": 0.9,
                    "original_box_xyxy_crop_pixels": [x1, 0, x1 + 80, 100],
                    "padded_box_xyxy_crop_pixels": [x1, 0, x1 + 80, 100],
                }
            )

        service = task6.MultiplexSam3AgentService.__new__(
            task6.MultiplexSam3AgentService
        )
        service.model = FakeModel()
        service.threshold = 0.05
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            Image.new("RGB", (384, 360)).save(frame)
            with (
                mock.patch.object(task6.torch.cuda, "is_available", return_value=False),
                mock.patch.object(
                    task6.torch,
                    "autocast",
                    return_value=contextlib.nullcontext(),
                ),
                mock.patch.object(
                    task6,
                    "rle_encode",
                    side_effect=lambda masks: [
                        {"counts": f"rle-{index}"} for index in range(len(masks))
                    ],
                ),
                mock.patch.object(task6, "visualize", return_value=FakeRender()),
            ):
                output_path = service.segment_boxes(
                    image_path=str(frame),
                    proposals=proposals,
                    output_folder_path=str(Path(directory) / "sam"),
                )
            output = json.loads(Path(output_path).read_text(encoding="utf-8"))

        self.assertEqual(service.model.init_calls, 1)
        self.assertEqual(len(service.model.prompt_calls), 3)
        np.testing.assert_allclose(output["pred_scores"], [0.8, 0.8, 0.8])
        self.assertEqual([item["sam_raw_index"] for item in output["proposal_provenance"]], [1, 1, 1])
        for call in service.model.prompt_calls:
            self.assertEqual(call["text_str"], "visual")
            self.assertEqual(call["box_labels"], [1])
            self.assertEqual(len(call["boxes_xywh"]), 1)

    def test_prefers_geometry_and_saves_raw_and_refined_masks(self) -> None:
        class FakeModel:
            def init_state(self, **_kwargs):
                return {"state": True}

            def add_prompt(self, **_kwargs):
                tiny = np.zeros((100, 120), dtype=bool)
                tiny[47:53, 57:63] = True
                aligned = np.zeros((100, 120), dtype=bool)
                aligned[35:65, 45:95] = True
                aligned[5:10, 5:10] = True
                return 0, {
                    "out_binary_masks": np.stack([tiny, aligned]),
                    "out_probs": np.asarray([0.99, 0.75], dtype=np.float32),
                }

        class FakeRender:
            def save(self, output_path):
                Image.new("RGB", (120, 100)).save(output_path)

        service = task6.MultiplexSam3AgentService.__new__(
            task6.MultiplexSam3AgentService
        )
        service.model = FakeModel()
        service.threshold = 0.05
        proposal = {
            "proposal_id": 1,
            "dino_index": 0,
            "phrase": "box",
            "text_label": "box",
            "dino_score": 0.9,
            "original_box_xyxy_crop_pixels": [45, 35, 95, 65],
            "padded_box_xyxy_crop_pixels": [40, 30, 100, 70],
        }

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            output_dir = Path(directory) / "sam"
            Image.new("RGB", (120, 100)).save(frame)
            with (
                mock.patch.object(task6.torch.cuda, "is_available", return_value=False),
                mock.patch.object(
                    task6.torch,
                    "autocast",
                    return_value=contextlib.nullcontext(),
                ),
                mock.patch.object(
                    task6,
                    "rle_encode",
                    return_value=[{"counts": "refined-rle"}],
                ),
                mock.patch.object(task6, "visualize", return_value=FakeRender()),
            ):
                output_path = service.segment_boxes(
                    image_path=str(frame),
                    proposals=[proposal],
                    output_folder_path=str(output_dir),
                )
            output = json.loads(Path(output_path).read_text(encoding="utf-8"))
            provenance = output["proposal_provenance"][0]
            raw_mask = np.asarray(Image.open(provenance["raw_mask_artifact"])) > 0
            refined_mask = np.asarray(Image.open(provenance["mask_artifact"])) > 0

        self.assertEqual(provenance["sam_raw_index"], 1)
        self.assertAlmostEqual(provenance["sam_score"], 0.75)
        self.assertEqual(provenance["raw_mask_area_pixels"], 1525)
        self.assertEqual(provenance["mask_area_pixels"], 1500)
        self.assertTrue(raw_mask[7, 7])
        self.assertFalse(refined_mask[7, 7])
        self.assertTrue(refined_mask[50, 60])
        self.assertGreater(provenance["mask_geometry_score"], 0.9)


class DepthRefinementIntegrationTests(unittest.TestCase):
    def test_verified_dino_mask_is_cut_and_artifacts_are_auditable(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[20:60, 20:100] = True
        depth = np.full(mask.shape, 1.0, dtype=np.float32)
        depth[20:60, 70:100] = 1.5
        candidate_record = candidate(0, mask)
        candidate_record["proposal_provenance"] = {
            "candidate_index": 0,
            "dino_original_box_xyxy_crop_pixels": [20, 20, 100, 60],
            "sam_prompt_box_xyxy_crop_pixels": [20, 20, 100, 60],
            "mask_artifact": "/tmp/pre_depth_mask.png",
        }

        with tempfile.TemporaryDirectory() as directory:
            refined, updated, manifest = (
                mask_service.refine_verified_dino_masks_with_depth(
                    kept=[(mask, 0.9)],
                    candidates=[candidate_record],
                    rgb_np=np.zeros((80, 120, 3), dtype=np.uint8),
                    depth_np=depth,
                    req_dir=Path(directory),
                    args=make_args(min_area=20),
                    enabled=True,
                )
            )
            final_mask = refined[0][0]
            provenance = updated[0]["proposal_provenance"]
            self.assertTrue(Path(manifest["artifact"]).is_file())
            self.assertTrue(Path(manifest["depth_artifact"]).is_file())
            self.assertTrue(Path(provenance["mask_artifact"]).is_file())
            self.assertTrue(
                Path(provenance["depth_refinement_overlay_artifact"]).is_file()
            )

        self.assertEqual(manifest["applied_count"], 1)
        self.assertTrue(final_mask[30, 40])
        self.assertFalse(final_mask[30, 80])
        self.assertEqual(updated[0]["area_pixels"], int(final_mask.sum()))
        self.assertEqual(provenance["geometry_mask_artifact"], "/tmp/pre_depth_mask.png")
        self.assertEqual(provenance["depth_refinement"]["status"], "applied")


class DirectPipelineTests(unittest.TestCase):
    def _dino_generation(self, masks):
        candidates = [candidate(index, mask) for index, mask in enumerate(masks)]
        provenance = [
            {
                "candidate_index": index,
                "proposal_id": index + 1,
                "dino_phrase": "box",
                "dino_score": 0.9 - index * 0.1,
            }
            for index in range(len(masks))
        ]
        return {
            "kept": [(mask, 0.9) for mask in masks],
            "candidates": candidates,
            "sam_json": "/tmp/combined_candidates.json",
            "combined_candidates": "/tmp/combined_candidates.json",
            "outputs": {"proposal_provenance": provenance},
            "phrases": ["orange box", "box", "package", "carton"],
            "proposals": [{"proposal_id": index + 1} for index in range(len(masks))],
            "proposal_manifest": "/tmp/dino_proposals.json",
            "dino_elapsed_s": 0.01,
            "sam_box_prompt_timings": [],
            "sam_box_elapsed_s": 0.02,
        }

    def test_one_qwen_verification_and_stable_selected_index(self) -> None:
        masks = [one_mask(10), one_mask(100)]
        generation = self._dino_generation(masks)
        calls = []

        def select_second(**kwargs):
            calls.append(kwargs)
            updated = [
                {**kwargs["candidates"][0], "kept": False, "reject_reasons": ["qwen"]},
                kwargs["candidates"][1],
            ]
            return (
                [kwargs["kept"][1]],
                updated,
                {
                    "status": "selected",
                    "selected_candidate_ids": [2],
                    "selected_sam_indices": [1],
                    "candidate_overlay": None,
                    "candidate_zoom": None,
                    "inference_s": 0.1,
                    "elapsed_s": 0.2,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(mask_service, "dino_candidate_generation", return_value=generation),
                mock.patch.object(mask_service, "verify_candidates_with_qwen", side_effect=select_second),
                mock.patch.object(mask_service.task2, "overlay_masks", return_value={"output": "overlay.png"}),
            ):
                result = mask_service.direct_segment(
                    request="pick the orange box",
                    intent={"target_phrase": "orange box", "selector": None},
                    rgb_np=np.zeros((360, 384, 3), dtype=np.uint8),
                    depth_np=np.ones((360, 384), dtype=np.float32),
                    frame_path=Path(directory) / "frame.png",
                    req_dir=Path(directory),
                    args=make_args(),
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["presence_gate"]["kept_indices"], [1])
        self.assertEqual(result["proposal_provenance"][1]["candidate_index"], 1)

    def test_dino_empty_runs_one_legacy_pass_before_verification(self) -> None:
        empty = self._dino_generation([])
        empty.update({"sam_json": None, "combined_candidates": None, "outputs": None})
        mask = one_mask()
        legacy = {
            "kept": [(mask, 0.9)],
            "candidates": [candidate(0, mask)],
            "sam_json": "/tmp/legacy.json",
            "outputs": {},
            "elapsed_s": 0.03,
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(mask_service, "dino_candidate_generation", return_value=empty) as dino_call,
                mock.patch.object(mask_service, "legacy_candidate_generation", return_value=legacy) as legacy_call,
                mock.patch.object(mask_service, "verify_candidates_with_qwen", side_effect=lambda **kwargs: accepted_verification(kwargs["kept"], kwargs["candidates"])) as qwen_call,
                mock.patch.object(mask_service.task2, "overlay_masks", return_value={"output": "overlay.png"}),
            ):
                result = mask_service.direct_segment(
                    request="box",
                    intent={"target_phrase": "box", "selector": None},
                    rgb_np=np.zeros((360, 384, 3), dtype=np.uint8),
                    depth_np=np.ones((360, 384), dtype=np.float32),
                    frame_path=Path(directory) / "frame.png",
                    req_dir=Path(directory),
                    args=make_args(),
                )

        self.assertEqual(dino_call.call_count, 1)
        self.assertEqual(legacy_call.call_count, 1)
        self.assertEqual(qwen_call.call_count, 1)
        self.assertEqual(result["candidate_generation"]["source_used"], "legacy_text_fallback")
        self.assertEqual(result["candidate_generation"]["legacy_fallback_reason"], "no_valid_dino_proposals")

    def test_qwen_rejection_produces_no_target_and_no_fallback(self) -> None:
        mask = one_mask()
        generation = self._dino_generation([mask])

        def reject(**kwargs):
            rejected = [
                {**kwargs["candidates"][0], "kept": False, "reject_reasons": ["qwen_verifier_no_match"]}
            ]
            return (
                [],
                rejected,
                {
                    "status": "no_match",
                    "selected_candidate_ids": [],
                    "selected_sam_indices": [],
                    "candidate_overlay": None,
                    "candidate_zoom": None,
                    "inference_s": 0.1,
                    "elapsed_s": 0.2,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(mask_service, "dino_candidate_generation", return_value=generation),
                mock.patch.object(mask_service, "legacy_candidate_generation") as legacy_call,
                mock.patch.object(mask_service, "verify_candidates_with_qwen", side_effect=reject),
                mock.patch.object(mask_service.task2, "overlay_masks", return_value={"output": "overlay.png"}),
            ):
                result = mask_service.direct_segment(
                    request="box",
                    intent={"target_phrase": "box", "selector": None},
                    rgb_np=np.zeros((360, 384, 3), dtype=np.uint8),
                    depth_np=np.ones((360, 384), dtype=np.float32),
                    frame_path=Path(directory) / "frame.png",
                    req_dir=Path(directory),
                    args=make_args(),
                )

        legacy_call.assert_not_called()
        self.assertEqual(result["presence_gate"]["num_kept"], 0)
        self.assertFalse(result["fallback_eligible"])
        self.assertIsNone(
            mask_service.build_selected_mask_record(
                [], result["presence_gate"], {"crop": {}}, []
            )
        )

    def test_ordinary_command_parser_does_not_call_qwen(self) -> None:
        with mock.patch.object(mask_service.local_qwen, "qwen_generate") as qwen:
            intent = mask_service.parse_request_intent(
                "please pick the rightmost small orange box",
                make_args(),
            )
        qwen.assert_not_called()
        self.assertEqual(intent["target_phrase"], "small orange box")
        self.assertEqual(intent["selector"], "rightmost")

    def test_relation_anchor_is_not_mistaken_for_target_category(self) -> None:
        with mock.patch.object(mask_service.local_qwen, "qwen_generate") as qwen:
            intent = mask_service.parse_request_intent(
                "pick the small orange package inside the storage bin",
                make_args(),
            )
        qwen.assert_not_called()
        self.assertEqual(intent["target_phrase"], "small orange package")
        self.assertEqual(intent["relation_context"], "inside the storage bin")

    def test_ambiguous_reference_uses_existing_qwen_text_parser(self) -> None:
        with mock.patch.object(
            mask_service.local_qwen,
            "qwen_generate",
            return_value='{"target_phrase":"orange box","selector":null}',
        ) as qwen:
            intent = mask_service.parse_request_intent("pick it", make_args())
        self.assertEqual(qwen.call_count, 1)
        self.assertEqual(intent["target_phrase"], "orange box")
        self.assertEqual(intent["parser"], "qwen_json_ambiguity_fallback")

    def test_selected_mask_center_preserves_historical_crop_translation(self) -> None:
        mask = one_mask(10)
        record = mask_service.build_selected_mask_record(
            [(mask, 0.9)],
            {"kept_indices": [0]},
            {
                "crop": {
                    "enabled": True,
                    "applied_xyxy": [448, 360, 832, 720],
                }
            },
            [],
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["center_xy_crop_pixels"], [25.0, 30.0])
        self.assertEqual(record["center_xy_full_pixels"], [473.0, 390.0])


if __name__ == "__main__":
    unittest.main()
