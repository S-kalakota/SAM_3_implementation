#!/usr/bin/env python3
"""Fail-closed Qwen-VL verification for numbered SAM mask candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


REQUIRED_RESPONSE_KEYS = {
    "decision",
    "selected_candidate_ids",
    "confidence",
    "reason",
}
MASK_COLORS_RGB = (
    (255, 64, 64),
    (64, 192, 255),
    (255, 192, 64),
    (128, 255, 96),
    (224, 96, 255),
    (64, 255, 224),
)


class VerifierResponseError(ValueError):
    """Raised when Qwen does not return the required verifier schema."""


def parse_verifier_response(text: str, candidate_count: int) -> dict[str, Any]:
    """Parse exactly one strict verifier JSON object."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("candidate_count must be an integer")
    if candidate_count < 0:
        raise ValueError("candidate_count must not be negative")
    if not isinstance(text, str) or not text.strip():
        raise VerifierResponseError("response is empty")
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise VerifierResponseError(f"response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifierResponseError("response must be a JSON object")
    keys = set(value)
    if keys != REQUIRED_RESPONSE_KEYS:
        missing = sorted(REQUIRED_RESPONSE_KEYS - keys)
        extra = sorted(keys - REQUIRED_RESPONSE_KEYS)
        raise VerifierResponseError(
            f"response keys are invalid; missing={missing}, extra={extra}"
        )

    decision = value["decision"]
    if decision not in {"select", "no_match"}:
        raise VerifierResponseError("decision must be 'select' or 'no_match'")
    selected = value["selected_candidate_ids"]
    if not isinstance(selected, list):
        raise VerifierResponseError("selected_candidate_ids must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in selected):
        raise VerifierResponseError("candidate IDs must be integers")
    if len(selected) != len(set(selected)):
        raise VerifierResponseError("candidate IDs must be unique")
    if any(item < 1 or item > candidate_count for item in selected):
        raise VerifierResponseError(
            f"candidate IDs must be in the range 1..{candidate_count}"
        )
    if decision == "select" and not selected:
        raise VerifierResponseError("select requires at least one candidate ID")
    if decision == "no_match" and selected:
        raise VerifierResponseError("no_match requires an empty candidate list")

    confidence_value = value["confidence"]
    if isinstance(confidence_value, bool) or not isinstance(
        confidence_value, (int, float)
    ):
        raise VerifierResponseError("confidence must be a number")
    confidence = float(confidence_value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise VerifierResponseError("confidence must be in [0, 1]")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise VerifierResponseError("reason must be a non-empty string")

    return {
        "decision": decision,
        "selected_candidate_ids": sorted(selected),
        "confidence": confidence,
        "reason": reason.strip(),
    }


def render_numbered_candidates(
    rgb_np: np.ndarray,
    kept: list[tuple[np.ndarray, float]],
    output_path: Path,
) -> dict[str, Any]:
    """Render strongly colored, outlined, one-based candidate IDs."""

    if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {rgb_np.shape}")
    rendered = rgb_np.copy()
    for position, (mask, _score) in enumerate(kept):
        if mask.shape != rgb_np.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image {rgb_np.shape[:2]}"
            )
        if not mask.any():
            continue
        color = MASK_COLORS_RGB[position % len(MASK_COLORS_RGB)]
        color_array = np.asarray(color, dtype=np.float32)
        rendered[mask] = (
            0.65 * rendered[mask].astype(np.float32) + 0.35 * color_array
        ).astype(np.uint8)
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rendered, contours, -1, color, 2, cv2.LINE_AA)
        ys, xs = np.where(mask)
        min_x = int(xs.min())
        min_y = int(ys.min())
        max_y = int(ys.max())
        label_x = min(max(min_x + 14, 14), rgb_np.shape[1] - 15)
        if min_y >= 30:
            label_y = min_y - 16
        else:
            label_y = min(max_y + 16, rgb_np.shape[0] - 15)
        label_center = (label_x, label_y)
        cv2.line(rendered, label_center, (min_x, min_y), color, 2, cv2.LINE_AA)
        cv2.circle(rendered, label_center, 14, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(rendered, label_center, 11, color, -1, cv2.LINE_AA)
        label = str(position + 1)
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        origin = (
            label_center[0] - label_w // 2,
            label_center[1] + label_h // 2,
        )
        cv2.putText(
            rendered,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered[:, :, ::-1]):
        raise OSError(f"Failed to write verifier overlay: {output_path}")
    return {
        "output": str(output_path),
        "num_masks_drawn": len(kept),
        "label_indexing": "one_based",
    }


def render_candidate_zooms(
    rgb_np: np.ndarray,
    kept: list[tuple[np.ndarray, float]],
    output_path: Path,
) -> dict[str, Any]:
    """Render enlarged candidate crops without hiding the masked object."""

    if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {rgb_np.shape}")
    if not kept:
        raise ValueError("At least one candidate is required")

    tile_width = 320
    tile_height = 280
    header_height = 36
    columns = min(2, len(kept))
    rows = math.ceil(len(kept) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        28,
        dtype=np.uint8,
    )

    for position, (mask, score) in enumerate(kept):
        if mask.shape != rgb_np.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image {rgb_np.shape[:2]}"
            )
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        padding = max(12, int(0.15 * max(max_x - min_x + 1, max_y - min_y + 1)))
        crop_x0 = max(0, min_x - padding)
        crop_y0 = max(0, min_y - padding)
        crop_x1 = min(rgb_np.shape[1], max_x + padding + 1)
        crop_y1 = min(rgb_np.shape[0], max_y + padding + 1)
        crop_rgb = rgb_np[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

        available_width = tile_width - 16
        available_height = tile_height - header_height - 12
        scale = min(
            available_width / crop_rgb.shape[1],
            available_height / crop_rgb.shape[0],
        )
        resized_width = max(1, int(round(crop_rgb.shape[1] * scale)))
        resized_height = max(1, int(round(crop_rgb.shape[0] * scale)))
        resized_rgb = cv2.resize(
            crop_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = MASK_COLORS_RGB[position % len(MASK_COLORS_RGB)]
        color_array = np.asarray(color, dtype=np.float32)
        rendered_crop = resized_rgb.copy()
        rendered_crop[resized_mask] = (
            0.72 * rendered_crop[resized_mask].astype(np.float32)
            + 0.28 * color_array
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            resized_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(rendered_crop, contours, -1, color, 3, cv2.LINE_AA)

        row = position // columns
        column = position % columns
        tile_x = column * tile_width
        tile_y = row * tile_height
        image_x = tile_x + (tile_width - resized_width) // 2
        image_y = tile_y + header_height + (available_height - resized_height) // 2
        sheet[
            image_y : image_y + resized_height,
            image_x : image_x + resized_width,
        ] = rendered_crop
        cv2.putText(
            sheet,
            f"Candidate {position + 1}  SAM {score:.2f}",
            (tile_x + 10, tile_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet[:, :, ::-1]):
        raise OSError(f"Failed to write verifier zoom sheet: {output_path}")
    return {
        "output": str(output_path),
        "num_candidates_drawn": len(kept),
        "label_indexing": "one_based",
    }


def build_verifier_messages(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_overlay_path: Path,
    candidate_zoom_path: Path,
    candidate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the three-image, strict-JSON visual verification prompt."""

    system_prompt = (
        "You are the final safety-critical visual verifier for a robot pick target. "
        "Image 1 is the unmodified camera crop. Image 2 shows SAM candidate masks "
        "with one-based numeric IDs. Select only candidates whose mask closely "
        "covers one complete, movable instance of the requested target. Reject masks "
        "that mainly cover a storage bin, shelf, cart, robot, fixture, background, or "
        "a large surrounding structure. A requested object merely appearing somewhere "
        "inside the colored region is not a match: the colored mask boundary itself "
        "must closely follow that object's outer boundary. For example, if a small box "
        "is visible inside a mask covering an entire storage bin, reject the candidate. "
        "Reject partial, merged, or semantically wrong masks even if SAM confidence is "
        "high. Ignore the SAM score when deciding and do not copy it as verifier "
        "confidence. When uncertain, return no_match. "
        "Do not apply relative selectors such as topmost or rightmost; deterministic "
        "geometry will apply them after semantic verification. Return JSON only, with "
        "exactly these keys: decision, selected_candidate_ids, confidence, reason. "
        "decision must be select or no_match. selected_candidate_ids must contain "
        "every semantic target candidate, or [] for no_match. confidence must be a "
        "number from 0 to 1. The reason must name the physical object or structure "
        "whose outer boundary the colored mask actually follows; merely repeating the "
        "requested target name is not visual evidence. Do not use markdown or add any "
        "other text. The response will already be prefixed with the JSON text "
        '{"decision":. Continue that object with a quoted decision value, then the '
        "other three required keys, and close it with }. Do not repeat the prefix."
    )
    details = json.dumps(candidate_records, separators=(",", ":"))
    selector_text = selector if selector is not None else "none"
    user_text = (
        f"Original robot request: {request!r}. Semantic target phrase: "
        f"{target_phrase!r}. Deferred spatial selector: {selector_text!r}. "
        f"Candidate metadata: {details}. For each candidate, inspect the colored outer "
        "outline and identify the physical object or structure it encloses. Do not "
        "credit a candidate for a smaller requested object that is merely visible "
        "inside a much larger colored mask. Decide whether those outline-traced "
        "objects are complete, movable instances of the semantic target. A tight mask "
        "around a small or thin target is valid; do not reject it merely because the "
        "target occupies little of the full camera image. Think through that boundary "
        "test silently, then return only the required JSON object."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1: unmodified camera crop."},
                {"type": "image", "image": str(frame_path)},
                {"type": "text", "text": "Image 2: numbered SAM candidates."},
                {"type": "image", "image": str(candidate_overlay_path)},
                {
                    "type": "text",
                    "text": "Image 3: enlarged views of the same numbered candidates.",
                },
                {"type": "image", "image": str(candidate_zoom_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def run_visual_verifier(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_overlay_path: Path,
    candidate_zoom_path: Path,
    candidate_records: list[dict[str, Any]],
    send_generate_request: Callable[[list[dict[str, Any]]], str],
    min_select_confidence: float,
) -> dict[str, Any]:
    """Ask Qwen once, allow one format-only retry, and otherwise fail closed."""

    if not 0.0 <= min_select_confidence <= 1.0:
        raise ValueError("min_select_confidence must be in [0, 1]")
    if not candidate_records:
        raise ValueError("candidate_records must not be empty")

    messages = build_verifier_messages(
        request=request,
        target_phrase=target_phrase,
        selector=selector,
        frame_path=frame_path,
        candidate_overlay_path=candidate_overlay_path,
        candidate_zoom_path=candidate_zoom_path,
        candidate_records=candidate_records,
    )
    attempts = []
    for attempt_number in (1, 2):
        try:
            raw = send_generate_request(messages)
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": None,
                    "error": repr(exc),
                }
            )
            return {
                "status": "error",
                "decision": None,
                "selected_candidate_ids": [],
                "model_selected_candidate_ids": [],
                "confidence": None,
                "reason": f"Qwen verifier inference failed: {exc!r}",
                "attempts": attempts,
            }
        try:
            parsed = parse_verifier_response(raw, len(candidate_records))
        except VerifierResponseError as exc:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": raw,
                    "error": str(exc),
                }
            )
            if attempt_number == 1:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": str(raw)[:1000],
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Invalid verifier response: {exc}. Retry once. "
                                "Return exactly one valid JSON object with only "
                                "decision, selected_candidate_ids, confidence, reason. "
                                "The response is already prefixed with "
                                '{"decision":. Continue with a quoted select or '
                                "no_match value, then selected_candidate_ids as an "
                                "integer list, confidence as a number, and reason as a "
                                "quoted visual explanation. Close the JSON with }. Do "
                                "not repeat the prefix and do not output shorthand."
                            ),
                        },
                    ]
                )
                continue
            return {
                "status": "error",
                "decision": None,
                "selected_candidate_ids": [],
                "model_selected_candidate_ids": [],
                "confidence": None,
                "reason": "Qwen verifier returned invalid JSON twice",
                "attempts": attempts,
            }

        attempts.append(
            {
                "attempt": attempt_number,
                "raw_response": raw,
                "error": None,
            }
        )
        if parsed["decision"] == "no_match":
            status = "no_match"
            selected = []
        elif parsed["confidence"] < min_select_confidence:
            status = "low_confidence"
            selected = []
        else:
            status = "selected"
            selected = parsed["selected_candidate_ids"]
        return {
            "status": status,
            "decision": parsed["decision"],
            "selected_candidate_ids": selected,
            "model_selected_candidate_ids": parsed["selected_candidate_ids"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "attempts": attempts,
        }

    raise AssertionError("unreachable verifier retry state")


def apply_max_area_fraction_policy(
    verification: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    max_area_fraction: float,
) -> dict[str, Any]:
    """Reject Qwen selections too large to be a safe pick target."""

    if not 0.0 < max_area_fraction <= 1.0:
        raise ValueError("max_area_fraction must be in (0, 1]")
    result = dict(verification)
    qwen_decision = result.get("decision")
    qwen_selected = list(
        result.get(
            "model_selected_candidate_ids",
            result.get("selected_candidate_ids", []),
        )
    )
    result["qwen_decision"] = qwen_decision
    result["qwen_selected_candidate_ids"] = qwen_selected
    result["max_area_fraction"] = float(max_area_fraction)
    result["policy_rejections"] = []
    if result.get("status") != "selected":
        result["final_decision"] = qwen_decision
        return result

    records_by_id = {
        int(record["candidate_id"]): record for record in candidate_records
    }
    retained = []
    for candidate_id in qwen_selected:
        record = records_by_id[candidate_id]
        area_fraction = float(record["area_fraction"])
        if area_fraction > max_area_fraction:
            result["policy_rejections"].append(
                {
                    "candidate_id": candidate_id,
                    "reason": "area_fraction_exceeds_pick_target_limit",
                    "area_fraction": area_fraction,
                    "max_area_fraction": float(max_area_fraction),
                }
            )
        else:
            retained.append(candidate_id)

    result["selected_candidate_ids"] = retained
    if retained:
        result["final_decision"] = "select"
        if result["policy_rejections"]:
            result["status"] = "selected_with_policy_rejections"
    else:
        result["status"] = "policy_rejected"
        result["decision"] = "no_match"
        result["final_decision"] = "no_match"
        rejected_ids = [
            item["candidate_id"] for item in result["policy_rejections"]
        ]
        result["reason"] = (
            f"Qwen selected candidate IDs {rejected_ids}, but each exceeded the "
            f"configured pick-target area limit {max_area_fraction:.3f}."
        )
    return result


def apply_verifier_selection(
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    verification: dict[str, Any],
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]], list[int]]:
    """Keep only Qwen-selected candidates while retaining original SAM indices."""

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    if len(kept_candidates) != len(kept):
        raise ValueError("kept masks and candidate metadata are inconsistent")
    selected_positions = {
        candidate_id - 1 for candidate_id in verification["selected_candidate_ids"]
    }
    selected_sam_indices = [
        kept_candidates[position]["index"] for position in sorted(selected_positions)
    ]
    selected_index_set = set(selected_sam_indices)
    selected_kept = [
        item for position, item in enumerate(kept) if position in selected_positions
    ]

    status = verification["status"]
    status_reason = {
        "no_match": "qwen_verifier_no_match",
        "low_confidence": "qwen_verifier_low_confidence",
        "error": "qwen_verifier_error",
        "policy_rejected": "qwen_verifier_not_selected",
    }.get(status, "qwen_verifier_not_selected")
    policy_rejected_ids = {
        int(item["candidate_id"])
        for item in verification.get("policy_rejections", [])
    }
    candidate_id_by_sam_index = {
        int(candidate["index"]): position + 1
        for position, candidate in enumerate(kept_candidates)
    }
    updated_candidates = []
    for candidate in candidates:
        candidate = dict(candidate)
        if candidate["kept"] and candidate["index"] not in selected_index_set:
            candidate_id = candidate_id_by_sam_index[int(candidate["index"])]
            reject_reason = (
                "qwen_verifier_oversized_candidate"
                if candidate_id in policy_rejected_ids
                else status_reason
            )
            candidate["kept"] = False
            candidate["reject_reasons"] = list(candidate["reject_reasons"]) + [
                reject_reason
            ]
        updated_candidates.append(candidate)
    return selected_kept, updated_candidates, selected_sam_indices
