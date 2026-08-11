#!/usr/bin/env python3
"""Run and score repeatable frozen-frame SAM/Qwen A/B evaluations."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import candidate_generation
import grounding_intent
import task5_zed_live_prompt as task5


MANIFEST_KEYS = {"schema_version", "name", "cases"}
CASE_KEYS = {"id", "image", "source_phrase", "expected_present", "expected_mask"}


class EvaluationError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot load manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise EvaluationError("manifest must contain exactly schema_version, name, cases")
    if value["schema_version"] != 1:
        raise EvaluationError("manifest schema_version must be 1")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise EvaluationError("manifest name must be non-empty")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise EvaluationError("manifest cases must be a non-empty list")
    seen_ids: set[str] = set()
    cases = []
    for raw in value["cases"]:
        if not isinstance(raw, dict) or set(raw) != CASE_KEYS:
            raise EvaluationError(f"case keys are invalid: {raw!r}")
        case_id = raw["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise EvaluationError(f"case id is empty or duplicated: {case_id!r}")
        image = (path.parent / raw["image"]).resolve()
        if not image.is_file():
            raise EvaluationError(f"case {case_id}: image does not exist: {image}")
        expected_present = raw["expected_present"]
        if not isinstance(expected_present, bool):
            raise EvaluationError(f"case {case_id}: expected_present must be boolean")
        expected_mask = raw["expected_mask"]
        if expected_mask is not None:
            if not isinstance(expected_mask, str):
                raise EvaluationError(f"case {case_id}: expected_mask must be a path or null")
            expected_mask = (path.parent / expected_mask).resolve()
            if not expected_mask.is_file():
                raise EvaluationError(
                    f"case {case_id}: expected mask does not exist: {expected_mask}"
                )
        if expected_present and expected_mask is None:
            raise EvaluationError(
                f"case {case_id}: present targets require a full-frame expected_mask"
            )
        if not isinstance(raw["source_phrase"], str) or not raw["source_phrase"].strip():
            raise EvaluationError(f"case {case_id}: source_phrase must be non-empty")
        seen_ids.add(case_id)
        cases.append(
            {
                **raw,
                "id": case_id,
                "image": str(image),
                "expected_mask": None if expected_mask is None else str(expected_mask),
            }
        )
    return {**value, "path": str(path), "cases": cases}


def parse_crop(value: str) -> tuple[int, int, int, int] | None:
    if value.lower() in {"none", "full", "full-frame"}:
        return None
    return task5.parse_crop(value)


def service_url(host: str, path: str) -> str:
    base = host.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base + path


def get_health(host: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(service_url(host, "/health"), timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise EvaluationError("service health response is invalid")
    return value


def post_evaluation(host: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        service_url(host, "/v1/evaluate"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EvaluationError(f"evaluation service returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict):
        raise EvaluationError("evaluation response is invalid")
    return value


def read_binary_mask(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise EvaluationError(f"cannot read mask {path}")
    return image > 0


def iou_and_dice(predicted: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    if predicted.shape != expected.shape:
        raise EvaluationError(
            f"mask shapes differ: predicted {predicted.shape}, expected {expected.shape}"
        )
    intersection = int(np.count_nonzero(predicted & expected))
    predicted_area = int(np.count_nonzero(predicted))
    expected_area = int(np.count_nonzero(expected))
    union = predicted_area + expected_area - intersection
    iou = 1.0 if union == 0 else intersection / union
    denominator = predicted_area + expected_area
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    return iou, dice


def candidate_full_masks(response: dict[str, Any]) -> list[np.ndarray]:
    crop_info = response["zed_frame"]["crop"]
    records = response.get("candidate_generation", {}).get("candidate_manifest", [])
    full_masks = []
    for record in records:
        crop_mask = read_binary_mask(record["mask_path_crop"])
        full_masks.append(candidate_generation.expand_crop_mask_to_full(crop_mask, crop_info))
    return full_masks


def score_case(
    case: dict[str, Any],
    response: dict[str, Any],
    *,
    match_iou: float,
) -> dict[str, Any]:
    expected_present = case["expected_present"]
    expected = (
        read_binary_mask(case["expected_mask"])
        if case["expected_mask"] is not None
        else None
    )
    raw_ious: list[float] = []
    raw_dices: list[float] = []
    if expected is not None:
        for candidate in candidate_full_masks(response):
            iou, dice = iou_and_dice(candidate, expected)
            raw_ious.append(iou)
            raw_dices.append(dice)
    final_ious: list[float] = []
    final_dices: list[float] = []
    if expected is not None:
        for record in response.get("final_masks", []):
            predicted = read_binary_mask(record["mask_full"])
            iou, dice = iou_and_dice(predicted, expected)
            final_ious.append(iou)
            final_dices.append(dice)

    max_raw_iou = max(raw_ious, default=0.0)
    max_final_iou = max(final_ious, default=0.0)
    selected_any = int(response.get("num_kept", 0)) > 0
    raw_candidate_recalled = expected_present and max_raw_iou >= match_iou
    correct_selection = (
        max_final_iou >= match_iou if expected_present else not selected_any
    )
    verification = response.get("verification") or {}
    attempts = verification.get("attempts") or []
    malformed_attempts = sum(1 for attempt in attempts if attempt.get("error"))
    intent_attempts = (
        ((response.get("intent_parser") or {}).get("qwen") or {}).get("attempts")
        or []
    )
    malformed_intent_attempts = sum(
        1 for attempt in intent_attempts if attempt.get("error")
    )
    return {
        "case_id": case["id"],
        "expected_present": expected_present,
        "raw_candidate_count": int(
            response.get("candidate_generation", {}).get("merged_candidate_count", 0)
        ),
        "raw_candidate_recalled": raw_candidate_recalled,
        "selected_count": int(response.get("num_kept", 0)),
        "correct_selection": correct_selection,
        "absent_false_positive": not expected_present and selected_any,
        "max_raw_iou": max_raw_iou,
        "max_raw_dice": max(raw_dices, default=0.0),
        "max_final_iou": max_final_iou,
        "max_final_dice": max(final_dices, default=0.0),
        "verification_status": verification.get("status"),
        "malformed_verifier_attempts": malformed_attempts,
        "malformed_intent_attempts": malformed_intent_attempts,
        "malformed_final_response": verification.get("status") == "error",
        "elapsed_s": float(response.get("elapsed_s", math.nan)),
        "result_json": response.get("result_json"),
        "overlay": (response.get("overlay") or {}).get("output"),
    }


def aggregate_scores(cases: list[dict[str, Any]]) -> dict[str, Any]:
    present = [case for case in cases if case["expected_present"]]
    absent = [case for case in cases if not case["expected_present"]]
    finite_latencies = [case["elapsed_s"] for case in cases if math.isfinite(case["elapsed_s"])]
    return {
        "case_count": len(cases),
        "present_case_count": len(present),
        "absent_case_count": len(absent),
        "raw_candidate_recall": (
            sum(case["raw_candidate_recalled"] for case in present) / len(present)
            if present else None
        ),
        "correct_object_selection_accuracy": (
            sum(case["correct_selection"] for case in cases) / len(cases)
            if cases else None
        ),
        "no_object_false_positive_rate": (
            sum(case["absent_false_positive"] for case in absent) / len(absent)
            if absent else None
        ),
        "mean_final_iou_present": (
            sum(case["max_final_iou"] for case in present) / len(present)
            if present else None
        ),
        "mean_final_dice_present": (
            sum(case["max_final_dice"] for case in present) / len(present)
            if present else None
        ),
        "malformed_verifier_attempts": sum(
            case["malformed_verifier_attempts"] for case in cases
        ),
        "malformed_intent_attempts": sum(
            case.get("malformed_intent_attempts", 0) for case in cases
        ),
        "malformed_final_responses": sum(
            case["malformed_final_response"] for case in cases
        ),
        "mean_latency_s": (
            sum(finite_latencies) / len(finite_latencies)
            if finite_latencies else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--host", default="127.0.0.1:8765")
    parser.add_argument("--crop", default="448,360,384,360", type=parse_crop)
    parser.add_argument(
        "--multiscale", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--full-frame-context", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--expected-qwen-model")
    parser.add_argument("--match-iou", default=0.5, type=float)
    parser.add_argument("--timeout", default=600.0, type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation_reports"),
    )
    args = parser.parse_args()
    if not 0.0 < args.match_iou <= 1.0:
        parser.error("--match-iou must be in (0, 1]")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    try:
        health = get_health(args.host, min(args.timeout, 10.0))
    except Exception as exc:
        raise SystemExit(f"Cannot reach resident service: {exc}") from exc
    if args.expected_qwen_model and health.get("qwen_model") != args.expected_qwen_model:
        raise SystemExit(
            f"Service model is {health.get('qwen_model')!r}, expected "
            f"{args.expected_qwen_model!r}; refusing to label the wrong A/B variant"
        )

    output_dir = args.output_dir.expanduser().resolve() / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_cases = []
    responses = []
    for case in manifest["cases"]:
        try:
            intent = grounding_intent.parse_grounding_intent(case["source_phrase"])
        except grounding_intent.GroundingIntentError as exc:
            raise SystemExit(f"Case {case['id']} has invalid intent: {exc}") from exc
        payload = {
            "schema_version": 1,
            "case_id": case["id"],
            "source_phrase": intent["source_phrase"],
            "grounding_intent": intent,
            "intent_hash": grounding_intent.intent_hash(intent),
            "image_path": case["image"],
            "crop_xywh": None if args.crop is None else list(args.crop),
            "use_multiscale": bool(args.multiscale),
            "use_full_frame": bool(args.full_frame_context),
        }
        response = post_evaluation(args.host, payload, args.timeout)
        (output_dir / f"{case['id']}.json").write_text(
            json.dumps(response, indent=2) + "\n",
            encoding="utf-8",
        )
        responses.append(response)
        scored_cases.append(score_case(case, response, match_iou=args.match_iou))

    report = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": manifest["path"],
        "manifest_name": manifest["name"],
        "variant": args.variant,
        "service_health": health,
        "configuration": {
            "crop_xywh": None if args.crop is None else list(args.crop),
            "multiscale": bool(args.multiscale),
            "full_frame_context": bool(args.full_frame_context),
            "match_iou": args.match_iou,
        },
        "metrics": aggregate_scores(scored_cases),
        "cases": scored_cases,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
