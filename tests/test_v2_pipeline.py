import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_v2
import v2_pipeline


def relational_envelope(relationship="inside", evidence="inside"):
    raw = f"find the red block {evidence} the blue bin"
    value = {
        "schema_version": 2,
        "raw_command": raw,
        "visual_source_phrase": f"red block {evidence} the blue bin",
        "action": {"type": "identify", "evidence": "find"},
        "destination": None,
        "target": {
            "id": "target",
            "mention": "red block",
            "head_noun": "block",
            "noun_modifiers": [],
            "attributes": [
                {"type": "color", "value": "red", "evidence": "red"}
            ],
            "selector": None,
        },
        "anchors": [
            {
                "id": "anchor_1",
                "mention": "blue bin",
                "head_noun": "bin",
                "noun_modifiers": [],
                "attributes": [
                    {"type": "color", "value": "blue", "evidence": "blue"}
                ],
                "selector": None,
            }
        ],
        "relationships": [
            {
                "type": relationship,
                "target_id": "target",
                "anchor_id": "anchor_1",
                "evidence": evidence,
            }
        ],
    }
    return grounding_v2.seal_command_envelope(value)


def mask(shape, x0, y0, x1, y1):
    value = np.zeros(shape, dtype=bool)
    value[y0:y1, x0:x1] = True
    return value


def test_visual_grounding_response_requires_exact_entity_order_and_bounds():
    value = {
        "entities": [
            {"entity_id": "target", "boxes": [[100, 100, 200, 200]]},
            {"entity_id": "anchor_1", "boxes": []},
        ]
    }
    assert v2_pipeline.validate_visual_grounding_response(
        value, expected_entity_ids=["target", "anchor_1"]
    )["target"] == [[100, 100, 200, 200]]
    value["entities"].reverse()
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        v2_pipeline.validate_visual_grounding_response(
            value, expected_entity_ids=["target", "anchor_1"]
        )
    assert caught.value.code == "grounding_identity_mismatch"


def test_candidate_deduplication_never_crosses_entity_or_role():
    target = {
        "entity_id": "target",
        "role": "target",
        "mask": mask((20, 20), 2, 2, 8, 8),
        "score": 0.9,
        "provenance": [],
    }
    anchor = {**target, "entity_id": "anchor_1", "role": "anchor"}
    with pytest.raises(ValueError, match="cannot cross"):
        v2_pipeline.deduplicate_entity_candidates(
            [target, anchor],
            iou_threshold=0.8,
            conf_threshold=0.1,
            min_area=1,
            max_candidates=4,
        )
    assert len(
        v2_pipeline.deduplicate_entity_candidates(
            [target, {**target, "score": 0.8}],
            iou_threshold=0.8,
            conf_threshold=0.1,
            min_area=1,
            max_candidates=4,
        )
    ) == 1


def test_full_context_box_records_workspace_retained_fraction():
    converted = v2_pipeline._normalized_box_to_workspace(
        [0, 0, 500, 500],
        full_shape_hw=(100, 100),
        crop_info={"enabled": True, "applied_xyxy": [25, 25, 75, 75]},
    )
    assert converted is not None
    box, retained = converted
    assert box == [0.0, 0.0, 25.0, 25.0]
    assert retained == pytest.approx(0.25)


def test_entity_scoped_selector_returns_only_its_pool_winner():
    left = {
        "label": "A1",
        "mask": mask((20, 20), 1, 5, 4, 8),
        "area_pixels": 9,
    }
    right = {
        "label": "A2",
        "mask": mask((20, 20), 15, 5, 19, 9),
        "area_pixels": 16,
    }
    assert (
        v2_pipeline.selector_winner(
            [left, right],
            {"type": "rightmost", "evidence": "rightmost"},
            depth=None,
            min_valid_depth_pixels=2,
        )
        == "A2"
    )
    assert (
        v2_pipeline.selector_winner(
            [left, right],
            {"type": "smallest", "evidence": "smallest"},
            depth=None,
            min_valid_depth_pixels=2,
        )
        == "A1"
    )


def test_relationship_matrix_records_every_target_anchor_pair():
    envelope = relational_envelope()
    targets = [
        {"label": "T1", "mask": mask((30, 30), 10, 10, 15, 15)},
        {"label": "T2", "mask": mask((30, 30), 0, 0, 3, 3)},
    ]
    anchors = [
        {"label": "A1", "mask": mask((30, 30), 5, 5, 20, 20)},
        {"label": "A2", "mask": mask((30, 30), 20, 20, 29, 29)},
    ]
    matrix = v2_pipeline.compute_relationship_matrix(
        envelope,
        {"target": targets, "anchor_1": anchors},
        depth=None,
        xyz=None,
    )
    assert len(matrix) == 4
    assert {
        (item["target_candidate"], item["anchor_candidate"]) for item in matrix
    } == {("T1", "A1"), ("T1", "A2"), ("T2", "A1"), ("T2", "A2")}


class FakeSamImageService:
    def predict_text(self, image_path, prompt):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        if "bin" in prompt:
            result = mask((height, width), int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8))
            score = 0.94
        else:
            result = mask((height, width), int(width * 0.45), int(height * 0.45), int(width * 0.55), int(height * 0.55))
            score = 0.95
        return {
            "masks": np.stack([result]),
            "scores": np.asarray([score], dtype=np.float32),
            "boxes_xyxy": np.zeros((1, 4), dtype=np.float32),
        }

    def predict_boxes(self, image_path, boxes_xyxy):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        records = []
        for box in boxes_xyxy:
            x0, y0, x1, y1 = [int(round(item)) for item in box]
            records.append(
                {
                    "box_xyxy": list(box),
                    "mask": mask(
                        (height, width),
                        max(0, x0),
                        max(0, y0),
                        min(width, x1),
                        min(height, y1),
                    ),
                    "score": 0.90,
                }
            )
        return records


def fake_qwen(messages, images=None, **kwargs):
    del messages, kwargs
    if len(images) == 1:
        return json.dumps(
            {
                "entities": [
                    {"entity_id": "target", "boxes": [[400, 400, 500, 500]]},
                    {"entity_id": "anchor_1", "boxes": [[200, 200, 800, 800]]},
                ]
            }
        )
    return json.dumps(
        {
            "decision": "select",
            "target": "T1",
            "anchors": {"anchor_1": "A1"},
            "confidence": 0.99,
            "reason": "identity and deterministic containment agree",
        }
    )


def pipeline_args():
    return argparse.Namespace(
        qwen_model="fake-qwen",
        qwen_device_map="cpu",
        allow_qwen_downloads=False,
        v2_visual_max_new_tokens=128,
        candidate_multiscale=False,
        candidate_tile_scale=0.72,
        candidate_full_frame=False,
        candidate_dedup_iou=0.80,
        presence_conf_threshold=0.10,
        min_area=4,
        candidate_max_count=8,
        selection_roi=None,
        verifier_max_area_fraction=0.25,
        selection_min_valid_depth_fraction=0.8,
        candidate_min_valid_depth_pixels=5,
        candidate_max_depth_spread_mm=75.0,
        verifier_max_new_tokens=128,
        verifier_min_confidence=0.7,
        v2_relation_thresholds=None,
    )


def test_segment_frame_requires_qwen_sam_relations_and_safety_gates(tmp_path):
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), rgb)
    frame_info = {
        "saved_frame": str(frame_path),
        "saved_full_frame": str(frame_path),
        "crop": {
            "enabled": False,
            "full_width": 100,
            "full_height": 100,
            "output_width": 100,
            "output_height": 100,
        },
    }
    depth = np.ones((100, 100), dtype=np.float32)
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 0.1
    result = v2_pipeline.segment_frame(
        relational_envelope(),
        rgb=rgb,
        full_rgb=rgb,
        depth=depth,
        xyz=xyz,
        frame_info=frame_info,
        frame_path=frame_path,
        req_dir=tmp_path / "run",
        sam_service=FakeSamImageService(),
        args=pipeline_args(),
        generate=fake_qwen,
    )
    assert result["status"] == "accepted"
    assert result["accepted"] is True
    assert result["selected_entities"] == {"target": "T1", "anchor_1": "A1"}
    assert result["selected_relationships"][0]["status"] == "pass"
    assert result["robot_target"] is None
    assert result["motion_permitted"] is False
    assert result["_target_mask"].dtype == bool


def test_verifier_cannot_select_anchor_as_target():
    pools = {
        "target": [{"label": "T1"}],
        "anchor_1": [{"label": "A1"}],
    }
    value = {
        "decision": "select",
        "target": "A1",
        "anchors": {"anchor_1": "A1"},
        "confidence": 0.9,
        "reason": "wrong role",
    }
    with pytest.raises(grounding_v2.GroundingV2Error):
        v2_pipeline.validate_verification_response(value, pools=pools)
