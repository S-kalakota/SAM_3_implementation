import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_v2_release_gates as release


def passing_inputs():
    language = {
        "schema_version": 2,
        "cases": 120,
        "actual_whisper_count": 1,
        "entity_graph_accuracy": 0.98,
    }
    frozen = {
        "schema_version": 2,
        "reviewed_scene_count": 60,
        "target_candidate_recall": 0.95,
        "anchor_candidate_recall": 0.95,
        "valid_final_selection_accuracy": 0.95,
        "wrong_object_acceptances_safety": 0,
        "p95_latency_s": 30.0,
    }
    parity = {
        "schema_version": 1,
        "approved": True,
        "cases": 60,
        "min_iou": 0.95,
    }
    thresholds = {
        "schema_version": 1,
        "safety_false_accepts": 0,
        "cases": 60,
        "relationship_case_counts": {
            relationship: 6 for relationship in release.RELATIONSHIPS
        },
        "relationship_safety_case_counts": {
            relationship: 1 for relationship in release.RELATIONSHIPS
        },
    }
    live = [
        {
            "schema_version": 2,
            "relationship": relationship,
            "expected": "present",
            "reference_mask": "/reviewed/reference.png",
            "min_iou": 0.5,
            "summary": {
                "round_count": 20,
                "passed_rounds": 19,
                "acceptance_19_of_20": True,
            },
        }
        for relationship in sorted(release.RELATIONSHIPS)
    ]
    return language, frozen, parity, thresholds, live


def test_all_release_boundaries_pass_at_specified_limits():
    language, frozen, parity, thresholds, live = passing_inputs()
    report = release.check_release_gates(
        language=language,
        frozen=frozen,
        image_parity=parity,
        relation_thresholds=thresholds,
        live_reports=live,
    )
    assert report["approved"] is True
    assert all(report["gates"].values())


def test_any_wrong_safety_acceptance_blocks_release():
    language, frozen, parity, thresholds, live = passing_inputs()
    frozen["wrong_object_acceptances_safety"] = 1
    report = release.check_release_gates(
        language=language,
        frozen=frozen,
        image_parity=parity,
        relation_thresholds=thresholds,
        live_reports=live,
    )
    assert report["approved"] is False
    assert report["gates"]["zero_wrong_object_safety_acceptances"] is False


def test_missing_relationship_live_report_blocks_release():
    language, frozen, parity, thresholds, live = passing_inputs()
    report = release.check_release_gates(
        language=language,
        frozen=frozen,
        image_parity=parity,
        relation_thresholds=thresholds,
        live_reports=live[:-1],
    )
    assert report["approved"] is False
    assert report["metrics"]["missing_live_relationships"]


def test_live_report_without_reference_mask_cannot_approve():
    language, frozen, parity, thresholds, live = passing_inputs()
    live[0]["reference_mask"] = None
    report = release.check_release_gates(
        language=language,
        frozen=frozen,
        image_parity=parity,
        relation_thresholds=thresholds,
        live_reports=live,
    )
    assert report["approved"] is False
    assert report["gates"]["all_relationships_live_19_of_20"] is False


def test_transcript_style_seed_cannot_replace_actual_whisper_capture():
    language, frozen, parity, thresholds, live = passing_inputs()
    language["actual_whisper_count"] = 0
    report = release.check_release_gates(
        language=language,
        frozen=frozen,
        image_parity=parity,
        relation_thresholds=thresholds,
        live_reports=live,
    )
    assert report["approved"] is False
    assert report["gates"]["actual_whisper_variations_present"] is False
