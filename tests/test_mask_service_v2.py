import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_v2
import mask_service


def interpretation_value():
    return {
        "schema_version": 2,
        "raw_command": "find the red block beside the blue bin",
        "visual_source_phrase": "red block beside the blue bin",
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
                "type": "next_to",
                "target_id": "target",
                "anchor_id": "anchor_1",
                "evidence": "beside",
            }
        ],
    }


def interpretation_semantics():
    return {
        "action": {"type": "identify", "evidence": "find"},
        "destination": None,
        "target": {
            "mention": "red block",
            "head_noun": "block",
            "noun_modifiers": [],
            "attributes": [{"type": "color", "evidence": "red"}],
            "selector": None,
        },
        "anchors": [
            {
                "mention": "blue bin",
                "head_noun": "bin",
                "noun_modifiers": [],
                "attributes": [{"type": "color", "evidence": "blue"}],
                "selector": None,
            }
        ],
        "relationships": [
            {"type": "next_to", "anchor_index": 0, "evidence": "beside"}
        ],
    }


def args():
    return argparse.Namespace(
        qwen_model="fake",
        qwen_device_map="cpu",
        allow_qwen_downloads=False,
        v2_interpret_max_new_tokens=512,
    )


def test_qwen_interpretation_repairs_format_then_evidence_with_fresh_prompts():
    invalid_evidence = interpretation_semantics()
    invalid_evidence["target"]["mention"] = "green block"
    valid = json.dumps(interpretation_semantics())
    with mock.patch.object(
        mask_service.local_qwen,
        "qwen_generate",
        side_effect=["not-json", json.dumps(invalid_evidence), valid],
    ) as generate:
        envelope, record = mask_service.qwen_command_envelope(
            interpretation_value()["raw_command"], args()
        )
    assert generate.call_count == 3
    assert record["status"] == "accepted"
    assert record["schema_constrained_generation"] is True
    assert envelope["relationships"][0]["type"] == "next_to"
    for call in generate.call_args_list:
        assert call.kwargs["do_sample"] is False
        assert call.kwargs["repetition_penalty"] == 1.0
        assert call.kwargs["json_schema"] == grounding_v2.qwen_semantic_json_schema()
        assert "response_prefix" not in call.kwargs
        assert len(call.args[0]) == 2
    second_user = generate.call_args_list[1].args[0][1]["content"]
    third_user = generate.call_args_list[2].args[0][1]["content"]
    assert "invalid_interpretation_json" in second_user
    assert "missing_source_evidence" in third_user
    assert "not-json" not in second_user


def test_semantic_evidence_retry_exhaustion_preserves_validation_error():
    invalid = interpretation_semantics()
    invalid["target"]["mention"] = "green block"
    with mock.patch.object(
        mask_service.local_qwen,
        "qwen_generate",
        return_value=json.dumps(invalid),
    ) as generate:
        with pytest.raises(grounding_v2.GroundingV2Error) as caught:
            mask_service.qwen_command_envelope(
                interpretation_value()["raw_command"],
                args(),
            )
    assert caught.value.code == "missing_source_evidence"
    assert caught.value.details["attempt_count"] == 3
    assert generate.call_count == 3


def test_unsafe_unresolved_reference_is_not_retried():
    semantics = {
        "action": {"type": "pick", "evidence": "pick"},
        "destination": None,
        "target": {
            "mention": "it",
            "head_noun": "it",
            "noun_modifiers": [],
            "attributes": [],
            "selector": None,
        },
        "anchors": [],
        "relationships": [],
    }
    with mock.patch.object(
        mask_service.local_qwen,
        "qwen_generate",
        return_value=json.dumps(semantics),
    ) as generate:
        with pytest.raises(grounding_v2.GroundingV2Error) as caught:
            mask_service.qwen_command_envelope("pick it", args())
    assert caught.value.code == "unsupported_reference"
    assert generate.call_count == 1


def test_runtime_failure_does_not_invoke_semantic_fallback():
    with mock.patch.object(
        mask_service.local_qwen,
        "qwen_generate",
        side_effect=RuntimeError("model unavailable"),
    ) as generate:
        with pytest.raises(grounding_v2.GroundingV2Error) as caught:
            mask_service.qwen_command_envelope(
                interpretation_value()["raw_command"], args()
            )
    assert caught.value.code == "grounding_parser_unavailable"
    assert generate.call_count == 1


def test_v2_segment_validator_rejects_hash_or_extra_key_tampering():
    sealed = grounding_v2.seal_command_envelope(interpretation_value())
    assert mask_service.validate_v2_segment_request(sealed) == sealed
    changed = dict(sealed)
    changed["envelope_hash"] = "0" * 64
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        mask_service.validate_v2_segment_request(changed)
    assert caught.value.code == "grounding_identity_mismatch"
    extra = dict(sealed, source_phrase=sealed["visual_source_phrase"])
    with pytest.raises(grounding_v2.GroundingV2Error):
        mask_service.validate_v2_segment_request(extra)


def test_v2_segment_requires_matching_qwen_attestation():
    sealed = grounding_v2.seal_command_envelope(interpretation_value())
    previous = mask_service.STATE.get("v2_interpretations")
    try:
        mask_service.STATE["v2_interpretations"] = {}
        with pytest.raises(grounding_v2.GroundingV2Error) as caught:
            mask_service.require_v2_interpretation_attestation(sealed)
        assert caught.value.code == "interpretation_attestation_missing"

        record = {"status": "accepted", "model": "fake"}
        mask_service.cache_v2_interpretation(sealed, record)
        attestation = mask_service.require_v2_interpretation_attestation(sealed)
        assert attestation["qwen_interpretation"] == record
        assert attestation["command_envelope"] == sealed
    finally:
        if previous is None:
            mask_service.STATE.pop("v2_interpretations", None)
        else:
            mask_service.STATE["v2_interpretations"] = previous


def test_image_backend_requires_explicit_approved_parity_report(tmp_path):
    with pytest.raises(RuntimeError, match="requires"):
        mask_service.load_image_parity_approval(None)
    rejected = tmp_path / "rejected.json"
    rejected.write_text(
        json.dumps({
            "schema_version": 1,
            "approved": False,
            "cases": 60,
            "min_iou": 0.95,
        })
    )
    with pytest.raises(RuntimeError, match="not approved"):
        mask_service.load_image_parity_approval(rejected)
    approved = tmp_path / "approved.json"
    approved.write_text(
        json.dumps({
            "schema_version": 1,
            "approved": True,
            "cases": 60,
            "min_iou": 0.95,
        })
    )
    assert mask_service.load_image_parity_approval(approved)["cases"] == 60


def test_v2_requires_safety_constrained_relation_calibration(tmp_path):
    with pytest.raises(RuntimeError, match="requires"):
        mask_service.load_relation_threshold_calibration(None)
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": 60,
                "safety_false_accepts": 1,
                "thresholds": {"direction_margin_fraction": 0.03},
            }
        )
    )
    with pytest.raises(RuntimeError, match="false accepts"):
        mask_service.load_relation_threshold_calibration(unsafe)
    approved = tmp_path / "relations.json"
    approved.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": 60,
                "safety_false_accepts": 0,
                "thresholds": {"direction_margin_fraction": 0.04},
                "relationship_case_counts": {
                    relationship: 6
                    for relationship in mask_service.relation_geometry.SUPPORTED_RELATIONSHIPS
                },
                "relationship_safety_case_counts": {
                    relationship: 1
                    for relationship in mask_service.relation_geometry.SUPPORTED_RELATIONSHIPS
                },
            }
        )
    )
    report = mask_service.load_relation_threshold_calibration(approved)
    assert report["thresholds"]["direction_margin_fraction"] == 0.04


def test_v2_enabled_startup_requires_fully_approved_release_report(tmp_path):
    with pytest.raises(RuntimeError, match="requires"):
        mask_service.load_v2_release_approval(None)
    rejected = tmp_path / "release_rejected.json"
    rejected.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "approved": False,
                "gates": {"language": False},
            }
        )
    )
    with pytest.raises(RuntimeError, match="not fully approved"):
        mask_service.load_v2_release_approval(rejected)
    approved = tmp_path / "release_approved.json"
    approved.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "approved": True,
                "gates": {
                    gate: True
                    for gate in mask_service.check_v2_release_gates.RELEASE_GATE_NAMES
                },
            }
        )
    )
    assert mask_service.load_v2_release_approval(approved)["approved"] is True

    incomplete = tmp_path / "release_incomplete.json"
    incomplete.write_text(
        json.dumps({
            "schema_version": 2,
            "approved": True,
            "gates": {"language": True},
        })
    )
    with pytest.raises(RuntimeError, match="not fully approved"):
        mask_service.load_v2_release_approval(incomplete)


def test_v2_routes_exist_alongside_unchanged_v1_routes():
    paths = {route.path for route in mask_service.app.routes}
    assert {"/v2/interpret", "/v2/segment", "/v1/segment", "/segment"} <= paths


def test_v2_evaluation_mode_breaks_no_motion_release_evidence_cycle():
    assert mask_service.v2_segmentation_available(
        argparse.Namespace(v2_enabled=False, v2_evaluation_mode=True)
    )
    assert mask_service.v2_segmentation_available(
        argparse.Namespace(v2_enabled=True, v2_evaluation_mode=False)
    )
    assert not mask_service.v2_segmentation_available(
        argparse.Namespace(v2_enabled=False, v2_evaluation_mode=False)
    )


def test_v2_pipeline_refusal_writes_no_motion_audit_result(tmp_path):
    sealed = grounding_v2.seal_command_envelope(interpretation_value())
    frame = tmp_path / "frame.png"
    full = tmp_path / "full.png"
    frame.write_bytes(b"frame")
    full.write_bytes(b"full")
    frame_info = {
        "saved_frame": str(frame),
        "saved_full_frame": str(full),
        "crop": {
            "enabled": False,
            "full_width": 8,
            "full_height": 6,
            "applied_xyxy": [0, 0, 8, 6],
        },
    }

    class FakeSam:
        def predict_boxes(self):
            pass

        def clear_embedding_cache(self):
            pass

        def cache_stats(self):
            return {"cached_views": 0}

    state = {
        "args": argparse.Namespace(
            v2_enabled=False,
            v2_evaluation_mode=True,
            sam_backend="image",
        ),
        "sam": FakeSam(),
        "run_dir": tmp_path,
        "v2_interpretations": {
            sealed["envelope_hash"]: {
                "command_envelope": sealed,
                "qwen_interpretation": {"status": "accepted", "elapsed_s": 1.25},
            }
        },
        "sam_image_parity": {"approved": True},
    }
    arrays = (
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.ones((6, 8), dtype=np.float32),
        np.ones((6, 8, 3), dtype=np.float32),
        frame_info,
    )
    refusal = grounding_v2.GroundingV2Error(
        "bad verifier JSON",
        code="invalid_verification_response",
    )
    with mock.patch.dict(mask_service.STATE, state, clear=True):
        with mock.patch.object(mask_service, "capture_request_frame", return_value=arrays):
            with mock.patch.object(
                mask_service.v2_pipeline,
                "segment_frame",
                side_effect=refusal,
            ):
                with pytest.raises(grounding_v2.GroundingV2Error) as caught:
                    mask_service.segment_v2_once(sealed)
    result_path = Path(caught.value.details["result_json"])
    result = json.loads(result_path.read_text())
    assert result["reason"] == "invalid_verification_response"
    assert result["robot_target"] is None
    assert result["motion_permitted"] is False
    assert result["elapsed_s"] >= 1.25
