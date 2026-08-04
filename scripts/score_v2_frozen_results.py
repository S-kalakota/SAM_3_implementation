#!/usr/bin/env python3
"""Score human-reviewed frozen-scene v2 result artifacts.

The scorer deliberately consumes saved ``result.json`` files instead of
running models.  This keeps the reviewed labels immutable and makes a release
report reproducible after the camera/model process has exited.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
RELATION_STATUSES = {"pass", "fail", "unavailable"}
MANIFEST_KEYS = {"schema_version", "name", "cases"}
CASE_KEYS = {
    "id",
    "reviewed",
    "scenario",
    "result_json",
    "expected_accept",
    "safety_case",
    "target_mask",
    "anchor_masks",
    "relationships",
}
RELATION_KEYS = {"type", "anchor_id", "expected_status"}


class FrozenScoreError(RuntimeError):
    pass


def _resolve_existing(value: str, base: Path, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FrozenScoreError(f"{description} does not exist: {path}")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenScoreError(f"cannot read frozen manifest {resolved}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise FrozenScoreError("manifest keys must be schema_version, name, and cases")
    if value["schema_version"] != 2:
        raise FrozenScoreError("manifest schema_version must be 2")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise FrozenScoreError("manifest name must be non-empty")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise FrozenScoreError("manifest cases must be a non-empty list")

    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw in value["cases"]:
        if not isinstance(raw, dict) or set(raw) != CASE_KEYS:
            raise FrozenScoreError(f"frozen case keys are invalid: {raw!r}")
        case_id = raw["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise FrozenScoreError(f"case id is empty or duplicated: {case_id!r}")
        if raw["reviewed"] is not True:
            raise FrozenScoreError(f"case {case_id}: reviewed must be true")
        if not isinstance(raw["scenario"], str) or not raw["scenario"].strip():
            raise FrozenScoreError(f"case {case_id}: scenario must be non-empty")
        if not isinstance(raw["expected_accept"], bool) or not isinstance(
            raw["safety_case"], bool
        ):
            raise FrozenScoreError(
                f"case {case_id}: expected_accept and safety_case must be booleans"
            )
        if raw["safety_case"] and raw["expected_accept"]:
            raise FrozenScoreError(
                f"case {case_id}: a safety case cannot expect acceptance"
            )
        result_path = _resolve_existing(
            raw["result_json"], resolved.parent, f"case {case_id} result"
        )
        target_path = None
        if raw["target_mask"] is not None:
            if not isinstance(raw["target_mask"], str):
                raise FrozenScoreError(f"case {case_id}: target_mask must be a path or null")
            target_path = _resolve_existing(
                raw["target_mask"], resolved.parent, f"case {case_id} target mask"
            )
        anchors = raw["anchor_masks"]
        if not isinstance(anchors, dict) or len(anchors) > 3:
            raise FrozenScoreError(f"case {case_id}: anchor_masks must have at most 3 entries")
        anchor_paths: dict[str, str | None] = {}
        for entity_id, mask_path in anchors.items():
            if not isinstance(entity_id, str) or not entity_id.startswith("anchor_"):
                raise FrozenScoreError(f"case {case_id}: invalid anchor id {entity_id!r}")
            if mask_path is None:
                anchor_paths[entity_id] = None
            elif isinstance(mask_path, str):
                anchor_paths[entity_id] = str(
                    _resolve_existing(
                        mask_path,
                        resolved.parent,
                        f"case {case_id} {entity_id} mask",
                    )
                )
            else:
                raise FrozenScoreError(
                    f"case {case_id}: {entity_id} mask must be a path or null"
                )
        relationships = raw["relationships"]
        if not isinstance(relationships, list) or len(relationships) > 4:
            raise FrozenScoreError(f"case {case_id}: relationships must have at most 4 entries")
        relation_records: list[dict[str, str]] = []
        for relationship in relationships:
            if not isinstance(relationship, dict) or set(relationship) != RELATION_KEYS:
                raise FrozenScoreError(f"case {case_id}: relationship keys are invalid")
            if relationship["type"] not in RELATIONSHIPS:
                raise FrozenScoreError(f"case {case_id}: unsupported relationship")
            if relationship["anchor_id"] not in anchor_paths:
                raise FrozenScoreError(
                    f"case {case_id}: relationship references an unknown anchor"
                )
            if relationship["expected_status"] not in RELATION_STATUSES:
                raise FrozenScoreError(f"case {case_id}: invalid expected relation status")
            relation_records.append(dict(relationship))
        if raw["expected_accept"]:
            if target_path is None or any(path is None for path in anchor_paths.values()):
                raise FrozenScoreError(
                    f"case {case_id}: accepted cases require every entity mask"
                )
            if any(item["expected_status"] != "pass" for item in relation_records):
                raise FrozenScoreError(
                    f"case {case_id}: accepted cases require passing relationships"
                )
        seen.add(case_id)
        cases.append(
            {
                **raw,
                "result_json": str(result_path),
                "target_mask": None if target_path is None else str(target_path),
                "anchor_masks": anchor_paths,
                "relationships": relation_records,
            }
        )
    return {**value, "path": str(resolved), "cases": cases}


def read_mask(path: str | Path, *, relative_to: Path | None = None) -> np.ndarray:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and relative_to is not None:
        resolved = relative_to / resolved
    image = cv2.imread(str(resolved.resolve()), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FrozenScoreError(f"cannot read mask {resolved}")
    return image > 0


def iou_and_dice(predicted: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    if predicted.shape != expected.shape:
        raise FrozenScoreError(
            f"mask shapes differ: predicted {predicted.shape}, expected {expected.shape}"
        )
    intersection = int(np.count_nonzero(predicted & expected))
    union = int(np.count_nonzero(predicted | expected))
    denominator = int(np.count_nonzero(predicted)) + int(np.count_nonzero(expected))
    return (
        1.0 if union == 0 else intersection / union,
        1.0 if denominator == 0 else 2.0 * intersection / denominator,
    )


def _candidate_full_mask(
    record: dict[str, Any],
    response: dict[str, Any],
    result_dir: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(record.get("mask_path_crop"), str):
        raise FrozenScoreError("candidate is missing mask_path_crop")
    mask = read_mask(record["mask_path_crop"], relative_to=result_dir)
    if mask.shape == expected_shape:
        return mask
    crop = (response.get("zed_frame") or {}).get("crop")
    if not isinstance(crop, dict) or not crop.get("enabled"):
        raise FrozenScoreError("candidate mask shape differs and crop metadata is unavailable")
    full_shape = (int(crop["full_height"]), int(crop["full_width"]))
    if full_shape != expected_shape:
        raise FrozenScoreError(
            f"crop full shape {full_shape} does not match label {expected_shape}"
        )
    x0, y0, x1, y1 = [int(value) for value in crop["applied_xyxy"]]
    if mask.shape != (y1 - y0, x1 - x0):
        raise FrozenScoreError("candidate mask does not match the recorded crop")
    full = np.zeros(full_shape, dtype=bool)
    full[y0:y1, x0:x1] = mask
    return full


def _best_candidate(
    records: Any,
    expected: np.ndarray,
    response: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    best = {"label": None, "iou": 0.0, "dice": 0.0}
    if not isinstance(records, list):
        return best
    for record in records:
        if not isinstance(record, dict):
            continue
        predicted = _candidate_full_mask(record, response, result_dir, expected.shape)
        iou, dice = iou_and_dice(predicted, expected)
        if iou > best["iou"]:
            best = {"label": record.get("label"), "iou": iou, "dice": dice}
    return best


def _final_mask_score(
    record: Any,
    expected: np.ndarray,
    result_dir: Path,
) -> dict[str, float]:
    if not isinstance(record, dict) or not isinstance(record.get("mask_full"), str):
        return {"iou": 0.0, "dice": 0.0}
    predicted = read_mask(record["mask_full"], relative_to=result_dir)
    iou, dice = iou_and_dice(predicted, expected)
    return {"iou": iou, "dice": dice}


def _malformed_qwen_response(response: dict[str, Any]) -> bool:
    detail = response.get("detail")
    error = response.get("error")
    code = (
        detail.get("code")
        if isinstance(detail, dict)
        else error.get("code")
        if isinstance(error, dict)
        else response.get("code")
    )
    return code in {
        "invalid_interpretation_json",
        "invalid_visual_grounding_response",
        "invalid_verification_response",
    }


def score_case(case: dict[str, Any], *, match_iou: float) -> dict[str, Any]:
    result_path = Path(case["result_json"])
    try:
        response = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenScoreError(f"cannot read result {result_path}: {exc}") from exc
    if not isinstance(response, dict):
        raise FrozenScoreError(f"case {case['id']}: result must contain a JSON object")
    pools = response.get("raw_candidate_pools") or {}
    if not isinstance(pools, dict):
        pools = {}
    expected_target = (
        None if case["target_mask"] is None else read_mask(case["target_mask"])
    )
    expected_anchors = {
        entity_id: None if path is None else read_mask(path)
        for entity_id, path in case["anchor_masks"].items()
    }
    target_best = (
        None
        if expected_target is None
        else _best_candidate(
            pools.get("target"), expected_target, response, result_path.parent
        )
    )
    anchor_best = {
        entity_id: None
        if expected is None
        else _best_candidate(
            pools.get(entity_id), expected, response, result_path.parent
        )
        for entity_id, expected in expected_anchors.items()
    }
    final_target = (
        None
        if expected_target is None
        else _final_mask_score(response.get("target_mask"), expected_target, result_path.parent)
    )
    response_anchor_masks = response.get("anchor_masks") or {}
    final_anchors = {
        entity_id: None
        if expected is None
        else _final_mask_score(
            response_anchor_masks.get(entity_id), expected, result_path.parent
        )
        for entity_id, expected in expected_anchors.items()
    }
    accepted = response.get("accepted") is True
    final_entities_match = (
        final_target is not None
        and final_target["iou"] >= match_iou
        and set(response_anchor_masks) == set(expected_anchors)
        and all(
            score is not None and score["iou"] >= match_iou
            for score in final_anchors.values()
        )
    )

    relationship_scores: list[dict[str, Any]] = []
    matrix = response.get("relationship_measurements") or []
    for index, expected_relation in enumerate(case["relationships"]):
        anchor = anchor_best[expected_relation["anchor_id"]]
        candidate_match = (
            target_best is not None
            and target_best["iou"] >= match_iou
            and anchor is not None
            and anchor["iou"] >= match_iou
        )
        measured = None
        if candidate_match:
            measured = next(
                (
                    item
                    for item in matrix
                    if isinstance(item, dict)
                    and item.get("relationship_index") == index
                    and item.get("relationship") == expected_relation["type"]
                    and item.get("anchor_id") == expected_relation["anchor_id"]
                    and item.get("target_candidate") == target_best["label"]
                    and item.get("anchor_candidate") == anchor["label"]
                ),
                None,
            )
        relationship_scores.append(
            {
                **expected_relation,
                "actual_status": None if measured is None else measured.get("status"),
                "correct": measured is not None
                and measured.get("status") == expected_relation["expected_status"],
            }
        )

    valid_selection_correct = bool(
        case["expected_accept"] and accepted and final_entities_match
    )
    elapsed = response.get("elapsed_s")
    elapsed_s = float(elapsed) if isinstance(elapsed, (int, float)) else math.nan
    return {
        "case_id": case["id"],
        "scenario": case["scenario"],
        "expected_accept": case["expected_accept"],
        "actual_accept": accepted,
        "safety_case": case["safety_case"],
        "correct_outcome": accepted == case["expected_accept"]
        and (not accepted or final_entities_match),
        "target_candidate": target_best,
        "target_candidate_recalled": target_best is not None
        and target_best["iou"] >= match_iou,
        "anchor_candidates": anchor_best,
        "anchor_candidates_recalled": {
            key: value is not None and value["iou"] >= match_iou
            for key, value in anchor_best.items()
            if expected_anchors[key] is not None
        },
        "valid_final_selection_correct": valid_selection_correct,
        "final_target": final_target,
        "final_anchors": final_anchors,
        "relationship_scores": relationship_scores,
        "malformed_qwen_response": _malformed_qwen_response(response),
        "elapsed_s": elapsed_s,
    }


def aggregate_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_records = [item for item in records if item["target_candidate"] is not None]
    anchor_recalled = [
        recalled
        for item in records
        for recalled in item["anchor_candidates_recalled"].values()
    ]
    valid = [item for item in records if item["expected_accept"]]
    safety = [item for item in records if item["safety_case"]]
    relationship_records = [
        relation for item in records for relation in item["relationship_scores"]
    ]
    final_mask_scores = [
        score
        for item in valid
        for score in [item["final_target"], *item["final_anchors"].values()]
        if score is not None
    ]
    latencies = [item["elapsed_s"] for item in records if math.isfinite(item["elapsed_s"])]
    missing_latencies = len(records) - len(latencies)
    return {
        "reviewed_scene_count": len(records),
        "target_candidate_recall": (
            sum(item["target_candidate_recalled"] for item in target_records)
            / len(target_records)
            if target_records
            else 0.0
        ),
        "anchor_candidate_recall": (
            sum(anchor_recalled) / len(anchor_recalled) if anchor_recalled else 0.0
        ),
        "relationship_pair_accuracy": (
            sum(item["correct"] for item in relationship_records)
            / len(relationship_records)
            if relationship_records
            else 0.0
        ),
        "valid_final_selection_accuracy": (
            sum(item["valid_final_selection_correct"] for item in valid) / len(valid)
            if valid
            else 0.0
        ),
        "wrong_object_acceptances_safety": sum(item["actual_accept"] for item in safety),
        "overall_case_accuracy": sum(item["correct_outcome"] for item in records)
        / len(records),
        "mean_final_iou": (
            sum(item["iou"] for item in final_mask_scores) / len(final_mask_scores)
            if final_mask_scores
            else 0.0
        ),
        "mean_final_dice": (
            sum(item["dice"] for item in final_mask_scores) / len(final_mask_scores)
            if final_mask_scores
            else 0.0
        ),
        "malformed_qwen_responses": sum(
            item["malformed_qwen_response"] for item in records
        ),
        "missing_latency_count": missing_latencies,
        "p95_latency_s": (
            float(np.percentile(latencies, 95))
            if latencies and missing_latencies == 0
            else 1.0e99
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match-iou", default=0.5, type=float)
    args = parser.parse_args()
    if not 0.0 < args.match_iou <= 1.0:
        parser.error("--match-iou must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    records = [score_case(case, match_iou=args.match_iou) for case in manifest["cases"]]
    metrics = aggregate_scores(records)
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest["path"],
        "manifest_name": manifest["name"],
        "match_iou": args.match_iou,
        **metrics,
        "case_reports": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
