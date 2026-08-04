#!/usr/bin/env python3
"""Combine reviewed evaluation reports into the v2 rollout approval artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RELATIONSHIPS = {
    "inside",
    "on",
    "left_of",
    "right_of",
    "above",
    "below",
    "near",
    "next_to",
    "in_front_of",
    "behind",
}
RELEASE_GATE_NAMES = {
    "at_least_120_reviewed_language_commands",
    "actual_whisper_variations_present",
    "at_least_60_reviewed_frozen_scenes",
    "entity_graph_accuracy_at_least_0_98",
    "target_candidate_recall_at_least_0_95",
    "anchor_candidate_recall_at_least_0_95",
    "valid_final_selection_at_least_0_95",
    "zero_wrong_object_safety_acceptances",
    "all_relationships_live_19_of_20",
    "p95_latency_no_more_than_30_s",
    "sam_image_output_parity",
    "relation_threshold_zero_safety_accepts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-report", required=True, type=Path)
    parser.add_argument("--frozen-report", required=True, type=Path)
    parser.add_argument("--image-parity-report", required=True, type=Path)
    parser.add_argument("--relation-threshold-report", required=True, type=Path)
    parser.add_argument("--live-report", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be an object: {resolved}")
    return value


def check_release_gates(
    *,
    language: dict[str, Any],
    frozen: dict[str, Any],
    image_parity: dict[str, Any],
    relation_thresholds: dict[str, Any],
    live_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    identities = {
        "language": language.get("schema_version") == 2,
        "frozen": frozen.get("schema_version") == 2,
        "image_parity": image_parity.get("schema_version") == 1,
        "relation_thresholds": relation_thresholds.get("schema_version") == 1,
    }
    if not all(identities.values()):
        raise ValueError(f"release input report schema mismatch: {identities}")
    live_by_relation: dict[str, bool] = {}
    for report in live_reports:
        relationship = report.get("relationship")
        if relationship not in RELATIONSHIPS or relationship in live_by_relation:
            raise ValueError("live reports need one unique supported relationship each")
        summary = report.get("summary") or {}
        live_by_relation[relationship] = bool(
            report.get("schema_version") == 2
            and report.get("expected") == "present"
            and isinstance(report.get("reference_mask"), str)
            and bool(report["reference_mask"].strip())
            and float(report.get("min_iou", 0.0)) >= 0.5
            and summary.get("round_count") == 20
            and int(summary.get("passed_rounds", 0)) >= 19
            and summary.get("acceptance_19_of_20") is True
        )
    missing_live = sorted(RELATIONSHIPS - set(live_by_relation))
    metrics = {
        "reviewed_language_command_count": int(language.get("cases", 0)),
        "actual_whisper_command_count": int(
            language.get("actual_whisper_count", 0)
        ),
        "reviewed_frozen_scene_count": int(frozen.get("reviewed_scene_count", 0)),
        "entity_graph_accuracy": float(language.get("entity_graph_accuracy", 0.0)),
        "target_candidate_recall": float(frozen.get("target_candidate_recall", 0.0)),
        "anchor_candidate_recall": float(frozen.get("anchor_candidate_recall", 0.0)),
        "valid_final_selection_accuracy": float(
            frozen.get("valid_final_selection_accuracy", 0.0)
        ),
        "wrong_object_acceptances_safety": int(
            frozen.get("wrong_object_acceptances_safety", -1)
        ),
        "p95_latency_s": float(frozen.get("p95_latency_s", 1.0e99)),
        "image_parity_approved": image_parity.get("approved") is True
        and int(image_parity.get("cases", 0)) >= 60
        and float(image_parity.get("min_iou", 0.0)) >= 0.95,
        "relation_safety_false_accepts": int(
            relation_thresholds.get("safety_false_accepts", -1)
        ),
        "relation_calibration_cases": int(relation_thresholds.get("cases", 0)),
        "relation_calibration_full_coverage": (
            set(relation_thresholds.get("relationship_case_counts") or {})
            == RELATIONSHIPS
            and all(
                isinstance(value, int) and value >= 1
                for value in (
                    relation_thresholds.get("relationship_case_counts") or {}
                ).values()
            )
            and set(relation_thresholds.get("relationship_safety_case_counts") or {})
            == RELATIONSHIPS
            and all(
                isinstance(value, int) and value >= 1
                for value in (
                    relation_thresholds.get("relationship_safety_case_counts") or {}
                ).values()
            )
        ),
        "live_relationships_19_of_20": live_by_relation,
        "missing_live_relationships": missing_live,
    }
    gates = {
        "at_least_120_reviewed_language_commands": metrics[
            "reviewed_language_command_count"
        ]
        >= 120,
        "actual_whisper_variations_present": metrics[
            "actual_whisper_command_count"
        ]
        > 0,
        "at_least_60_reviewed_frozen_scenes": metrics[
            "reviewed_frozen_scene_count"
        ]
        >= 60,
        "entity_graph_accuracy_at_least_0_98": metrics["entity_graph_accuracy"] >= 0.98,
        "target_candidate_recall_at_least_0_95": metrics["target_candidate_recall"] >= 0.95,
        "anchor_candidate_recall_at_least_0_95": metrics["anchor_candidate_recall"] >= 0.95,
        "valid_final_selection_at_least_0_95": metrics[
            "valid_final_selection_accuracy"
        ]
        >= 0.95,
        "zero_wrong_object_safety_acceptances": metrics[
            "wrong_object_acceptances_safety"
        ]
        == 0,
        "all_relationships_live_19_of_20": not missing_live
        and all(live_by_relation.values()),
        "p95_latency_no_more_than_30_s": metrics["p95_latency_s"] <= 30.0,
        "sam_image_output_parity": metrics["image_parity_approved"],
        "relation_threshold_zero_safety_accepts": metrics[
            "relation_safety_false_accepts"
        ]
        == 0
        and metrics["relation_calibration_cases"] >= 60
        and metrics["relation_calibration_full_coverage"],
    }
    if set(gates) != RELEASE_GATE_NAMES:  # pragma: no cover - developer invariant.
        raise AssertionError("release gate set drifted from the authenticated contract")
    return {
        "schema_version": 2,
        "approved": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    report = check_release_gates(
        language=_read(args.language_report),
        frozen=_read(args.frozen_report),
        image_parity=_read(args.image_parity_report),
        relation_thresholds=_read(args.relation_threshold_report),
        live_reports=[_read(path) for path in args.live_report],
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
