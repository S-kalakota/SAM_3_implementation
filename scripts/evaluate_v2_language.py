#!/usr/bin/env python3
"""Evaluate Qwen-only v2 interpretation against the reviewed 120-command corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import local_qwen
import mask_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "evaluation/v2_language_commands.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qwen-model", default=local_qwen.DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-device-map", default=local_qwen.DEFAULT_DEVICE_MAP)
    parser.add_argument("--max-new-tokens", default=1024, type=int)
    parser.add_argument("--allow-qwen-downloads", action="store_true")
    parser.add_argument("--fail-below", default=0.98, type=float)
    return parser.parse_args()


def _attribute_map(entity: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for attribute in entity["attributes"]:
        result.setdefault(attribute["type"], []).append(attribute["value"])
    return result


def compare_case(case: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    differences: dict[str, Any] = {}
    if envelope["action"]["type"] != expected["action"]:
        differences["action"] = {
            "expected": expected["action"],
            "actual": envelope["action"]["type"],
        }
    if envelope["destination"] != expected["destination"]:
        differences["destination"] = {
            "expected": expected["destination"],
            "actual": envelope["destination"],
        }
    actual_target = envelope["target"]
    expected_target = expected["target"]
    target_actual = {
        "mention": actual_target["mention"],
        "head_noun": actual_target["head_noun"],
        "noun_modifiers": actual_target["noun_modifiers"],
        "attributes": _attribute_map(actual_target),
        "selector": None
        if actual_target["selector"] is None
        else actual_target["selector"]["type"],
    }
    if target_actual != expected_target:
        differences["target"] = {"expected": expected_target, "actual": target_actual}
    actual_anchors = [
        {
            "mention": anchor["mention"],
            "head_noun": anchor["head_noun"],
            "noun_modifiers": anchor["noun_modifiers"],
            "attributes": _attribute_map(anchor),
            "selector": None
            if anchor["selector"] is None
            else anchor["selector"]["type"],
        }
        for anchor in envelope["anchors"]
    ]
    if actual_anchors != expected["anchors"]:
        differences["anchors"] = {
            "expected": expected["anchors"],
            "actual": actual_anchors,
        }
    actual_relationships = [
        {
            "type": relationship["type"],
            "anchor_index": int(relationship["anchor_id"].split("_")[-1]) - 1,
            "evidence": relationship["evidence"],
        }
        for relationship in envelope["relationships"]
    ]
    if actual_relationships != expected["relationships"]:
        differences["relationships"] = {
            "expected": expected["relationships"],
            "actual": actual_relationships,
        }
    return {"correct": not differences, "differences": differences}


def validate_corpus(corpus: Any) -> list[dict[str, Any]]:
    if not isinstance(corpus, dict) or set(corpus) != {
        "schema_version",
        "name",
        "reviewed_count",
        "actual_whisper_count",
        "notes",
        "cases",
    }:
        raise ValueError("v2 language corpus keys are invalid")
    if corpus["schema_version"] != 2:
        raise ValueError("v2 language corpus schema_version must be 2")
    cases = corpus["cases"]
    if not isinstance(cases, list) or len(cases) < 120:
        raise ValueError("v2 language corpus must contain at least 120 cases")
    required_case_keys = {
        "case_id",
        "command",
        "expected",
        "coverage",
        "reviewed",
    }
    if any(
        not isinstance(case, dict) or set(case) != required_case_keys
        for case in cases
    ):
        raise ValueError("language corpus case keys are invalid")
    if any(
        not isinstance(case["coverage"], list)
        or any(not isinstance(tag, str) or not tag for tag in case["coverage"])
        for case in cases
    ):
        raise ValueError("language corpus coverage tags are invalid")
    if corpus["reviewed_count"] != len(cases) or any(
        case.get("reviewed") is not True for case in cases
    ):
        raise ValueError("every corpus case must be marked reviewed")
    actual_whisper_count = sum(
        "actual_whisper" in case.get("coverage", []) for case in cases
    )
    if (
        isinstance(corpus["actual_whisper_count"], bool)
        or not isinstance(corpus["actual_whisper_count"], int)
        or corpus["actual_whisper_count"] != actual_whisper_count
    ):
        raise ValueError("actual_whisper_count must match actual capture coverage tags")
    seen_ids: set[str] = set()
    for case in cases:
        if (
            not isinstance(case["case_id"], str)
            or not case["case_id"]
            or case["case_id"] in seen_ids
        ):
            raise ValueError("language corpus case ids must be unique")
        seen_ids.add(case["case_id"])
        if not isinstance(case["command"], str) or not case["command"].strip():
            raise ValueError("language corpus commands must be non-empty")
        expected = case["expected"]
        if not isinstance(expected, dict) or set(expected) != {
            "outcome",
            "action",
            "destination",
            "target",
            "anchors",
            "relationships",
        }:
            raise ValueError("language corpus expected graph keys are invalid")
        if expected["outcome"] != "accept":
            raise ValueError("this reviewed graph corpus contains accepted cases only")
        entity_expectation_keys = {
            "mention",
            "head_noun",
            "noun_modifiers",
            "attributes",
            "selector",
        }
        if (
            not isinstance(expected["target"], dict)
            or set(expected["target"]) != entity_expectation_keys
        ):
            raise ValueError("language corpus target expectation is invalid")
        if not isinstance(expected["anchors"], list) or any(
            not isinstance(anchor, dict)
            or set(anchor) != entity_expectation_keys
            for anchor in expected["anchors"]
        ):
            raise ValueError("language corpus anchor expectation is invalid")
        for entity in [expected["target"], *expected["anchors"]]:
            if (
                not isinstance(entity["head_noun"], str)
                or not entity["head_noun"]
                or not isinstance(entity["noun_modifiers"], list)
                or any(
                    not isinstance(modifier, str) or not modifier
                    for modifier in entity["noun_modifiers"]
                )
            ):
                raise ValueError("language corpus entity grammar is invalid")
        if not isinstance(expected["relationships"], list) or any(
            not isinstance(relationship, dict)
            or set(relationship) != {"type", "anchor_index", "evidence"}
            or relationship["anchor_index"] < 0
            or relationship["anchor_index"] >= len(expected["anchors"])
            for relationship in expected["relationships"]
        ):
            raise ValueError("language corpus relationship expectation is invalid")
    required_relations = {
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
    covered = {
        relationship["type"]
        for case in cases
        for relationship in case["expected"]["relationships"]
    }
    if not required_relations <= covered:
        raise ValueError(f"corpus is missing relationships: {sorted(required_relations - covered)}")
    return cases


def main() -> None:
    args = parse_args()
    corpus_path = args.corpus.expanduser().resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    cases = validate_corpus(corpus)
    runtime_args = argparse.Namespace(
        qwen_model=args.qwen_model,
        qwen_device_map=args.qwen_device_map,
        allow_qwen_downloads=args.allow_qwen_downloads,
        v2_interpret_max_new_tokens=args.max_new_tokens,
    )
    records = []
    correct = 0
    malformed = 0
    for case in cases:
        try:
            envelope, parser_record = mask_service.qwen_command_envelope(
                case["command"], runtime_args
            )
            comparison = compare_case(case, envelope)
            correct += int(comparison["correct"])
            records.append(
                {
                    "case_id": case["case_id"],
                    **comparison,
                    "envelope": envelope,
                    "parser": parser_record,
                }
            )
        except Exception as exc:
            malformed += 1
            records.append(
                {
                    "case_id": case["case_id"],
                    "correct": False,
                    "error": repr(exc),
                }
            )
    accuracy = correct / len(cases)
    report = {
        "schema_version": 2,
        "corpus": str(corpus_path),
        "model": args.qwen_model,
        "cases": len(cases),
        "actual_whisper_count": int(corpus["actual_whisper_count"]),
        "correct_entity_graphs": correct,
        "entity_graph_accuracy": accuracy,
        "malformed_or_refused_responses": malformed,
        "release_gate": {
            "minimum_accuracy": args.fail_below,
            "passed": accuracy >= args.fail_below,
        },
        "records": records,
    }
    output = args.output or PROJECT_ROOT / "outputs/v2_language_evaluation.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if not report["release_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
