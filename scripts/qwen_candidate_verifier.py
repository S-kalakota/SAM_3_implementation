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
IDENTITY_RESPONSE_KEYS = {
    "decision",
    "selected_candidate_ids",
    "candidate_assessments",
    "confidence",
}
IDENTITY_ASSESSMENT_KEYS = {
    "candidate_id",
    "most_likely_object",
    "matches_target",
}
RANKED_RESPONSE_KEYS = {
    "decision",
    "selected_candidate_ids",
    "candidate_assessments",
    "confidence",
    "reason",
}
RANKED_ASSESSMENT_KEYS = {
    "candidate_id",
    "most_likely_object",
    "matches_target",
    "match_score",
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


def identity_verifier_json_schema(candidate_count: int) -> dict[str, Any]:
    """Return the strict generation schema for one identity-verifier call."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("candidate_count must be an integer")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    candidate_id = {
        "type": "integer",
        "minimum": 1,
        "maximum": candidate_count,
    }
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["select", "no_match"]},
            "selected_candidate_ids": {
                "type": "array",
                "items": candidate_id,
                "maxItems": candidate_count,
                "uniqueItems": True,
            },
            "candidate_assessments": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": candidate_id,
                        "most_likely_object": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "matches_target": {"type": "boolean"},
                    },
                    "required": sorted(IDENTITY_ASSESSMENT_KEYS),
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": sorted(IDENTITY_RESPONSE_KEYS),
        "additionalProperties": False,
    }


def ranked_mask_verifier_json_schema(candidate_count: int) -> dict[str, Any]:
    """Return a strict schema that permits at most one selected mask."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("candidate_count must be an integer")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    candidate_id = {
        "type": "integer",
        "minimum": 1,
        "maximum": candidate_count,
    }
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["select", "no_match"]},
            "selected_candidate_ids": {
                "type": "array",
                "items": candidate_id,
                "maxItems": 1,
                "uniqueItems": True,
            },
            "candidate_assessments": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": candidate_id,
                        "most_likely_object": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "matches_target": {"type": "boolean"},
                        "match_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": sorted(RANKED_ASSESSMENT_KEYS),
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "required": sorted(RANKED_RESPONSE_KEYS),
        "additionalProperties": False,
    }
def _identity_candidate_id(value: Any, *, field: str) -> int:
    """Normalize an unambiguous identity ID while rejecting ambiguous text."""

    if isinstance(value, bool):
        raise VerifierResponseError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise VerifierResponseError(f"{field} must be an integer")


def _identity_boolean(value: Any, *, field: str) -> bool:
    """Normalize exact JSON-boolean strings emitted by Qwen."""

    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise VerifierResponseError(f"{field} must be a boolean")


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


def parse_identity_verifier_response(
    text: str,
    candidate_count: int,
) -> dict[str, Any]:
    """Parse an identity-only response and enforce internal consistency."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("candidate_count must be an integer")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not isinstance(text, str) or not text.strip():
        raise VerifierResponseError("response is empty")
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise VerifierResponseError(f"response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifierResponseError("response must be a JSON object")
    keys = set(value)
    if keys != IDENTITY_RESPONSE_KEYS:
        missing = sorted(IDENTITY_RESPONSE_KEYS - keys)
        extra = sorted(keys - IDENTITY_RESPONSE_KEYS)
        raise VerifierResponseError(
            f"response keys are invalid; missing={missing}, extra={extra}"
        )

    decision = value["decision"]
    if decision not in {"select", "no_match"}:
        raise VerifierResponseError("decision must be 'select' or 'no_match'")
    selected_values = value["selected_candidate_ids"]
    if not isinstance(selected_values, list):
        raise VerifierResponseError("selected_candidate_ids must be a list")
    selected = [
        _identity_candidate_id(item, field="candidate ID")
        for item in selected_values
    ]
    if len(selected) != len(set(selected)):
        raise VerifierResponseError("candidate IDs must be unique")
    if any(item < 1 or item > candidate_count for item in selected):
        raise VerifierResponseError(
            f"candidate IDs must be in the range 1..{candidate_count}"
        )

    assessments = value["candidate_assessments"]
    if not isinstance(assessments, list):
        raise VerifierResponseError("candidate_assessments must be a list")
    parsed_assessments = []
    seen_ids = set()
    matching_ids = []
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise VerifierResponseError("each candidate assessment must be an object")
        assessment_keys = set(assessment)
        if assessment_keys != IDENTITY_ASSESSMENT_KEYS:
            missing = sorted(IDENTITY_ASSESSMENT_KEYS - assessment_keys)
            extra = sorted(assessment_keys - IDENTITY_ASSESSMENT_KEYS)
            raise VerifierResponseError(
                "candidate assessment keys are invalid; "
                f"missing={missing}, extra={extra}"
            )
        candidate_id = _identity_candidate_id(
            assessment["candidate_id"],
            field="assessment candidate_id",
        )
        if candidate_id < 1 or candidate_id > candidate_count:
            raise VerifierResponseError(
                f"assessment candidate_id must be in 1..{candidate_count}"
            )
        if candidate_id in seen_ids:
            raise VerifierResponseError("assessment candidate IDs must be unique")
        seen_ids.add(candidate_id)
        label = assessment["most_likely_object"]
        if not isinstance(label, str) or not label.strip():
            raise VerifierResponseError(
                "most_likely_object must be a non-empty string"
            )
        matches_target = _identity_boolean(
            assessment["matches_target"],
            field="matches_target",
        )
        if matches_target:
            matching_ids.append(candidate_id)
        parsed_assessments.append(
            {
                "candidate_id": candidate_id,
                "most_likely_object": label.strip(),
                "matches_target": matches_target,
            }
        )
    expected_ids = set(range(1, candidate_count + 1))
    if seen_ids != expected_ids:
        raise VerifierResponseError(
            "candidate_assessments must contain every candidate exactly once"
        )

    selected = sorted(selected)
    matching_ids = sorted(matching_ids)
    if selected != matching_ids:
        raise VerifierResponseError(
            "selected_candidate_ids must exactly match assessments with "
            "matches_target=true"
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
    assessment_reason = "; ".join(
        "candidate "
        f"{assessment['candidate_id']}: {assessment['most_likely_object']} "
        f"({'matches target' if assessment['matches_target'] else 'not target'})"
        for assessment in sorted(
            parsed_assessments,
            key=lambda item: item["candidate_id"],
        )
    )

    return {
        "decision": decision,
        "selected_candidate_ids": selected,
        "candidate_assessments": sorted(
            parsed_assessments,
            key=lambda item: item["candidate_id"],
        ),
        "confidence": confidence,
        "reason": f"Identity assessments: {assessment_reason}.",
    }


def _finite_candidate_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerifierResponseError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise VerifierResponseError(f"{field} must be a finite number")
    return number


def spatial_winner_ids(
    candidate_records: list[dict[str, Any]],
    eligible_ids: list[int],
    selector: str,
) -> set[int]:
    """Return IDs tied at the measured spatial/size selector extreme."""

    records_by_id = {
        int(record["candidate_id"]): record for record in candidate_records
    }
    if not eligible_ids:
        raise VerifierResponseError("a spatial selector has no matching candidates")
    missing = [candidate_id for candidate_id in eligible_ids if candidate_id not in records_by_id]
    if missing:
        raise VerifierResponseError(f"candidate metadata is missing IDs {missing}")

    values: dict[int, float] = {}
    for candidate_id in eligible_ids:
        record = records_by_id[candidate_id]
        if selector in {"leftmost", "rightmost", "topmost", "bottommost"}:
            center = record.get("center_xy_crop_pixels")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise VerifierResponseError(
                    f"candidate {candidate_id} has no measured center pixels"
                )
            axis = 0 if selector in {"leftmost", "rightmost"} else 1
            value = center[axis]
            field = f"candidate {candidate_id} center pixel"
        elif selector in {"nearest", "farthest"}:
            value = record.get("median_depth_m")
            field = f"candidate {candidate_id} median depth"
        elif selector in {"largest", "smallest"}:
            value = record.get("area_pixels")
            field = f"candidate {candidate_id} mask area"
        else:
            raise VerifierResponseError(f"unsupported selector {selector!r}")
        values[candidate_id] = _finite_candidate_number(value, field=field)

    choose_max = selector in {"rightmost", "bottommost", "farthest", "largest"}
    extreme = (max if choose_max else min)(values.values())
    return {
        candidate_id
        for candidate_id, value in values.items()
        if math.isclose(value, extreme, rel_tol=0.0, abs_tol=1e-9)
    }


def parse_ranked_mask_verifier_response(
    text: str,
    candidate_records: list[dict[str, Any]],
    selector: str | None,
) -> dict[str, Any]:
    """Parse one ranked mask choice and verify its measured location rule."""

    candidate_count = len(candidate_records)
    if candidate_count < 1:
        raise ValueError("candidate_records must not be empty")
    if not isinstance(text, str) or not text.strip():
        raise VerifierResponseError("response is empty")
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise VerifierResponseError(f"response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifierResponseError("response must be a JSON object")
    keys = set(value)
    if keys != RANKED_RESPONSE_KEYS:
        missing = sorted(RANKED_RESPONSE_KEYS - keys)
        extra = sorted(keys - RANKED_RESPONSE_KEYS)
        raise VerifierResponseError(
            f"response keys are invalid; missing={missing}, extra={extra}"
        )

    decision = value["decision"]
    if decision not in {"select", "no_match"}:
        raise VerifierResponseError("decision must be 'select' or 'no_match'")
    selected_values = value["selected_candidate_ids"]
    if not isinstance(selected_values, list):
        raise VerifierResponseError("selected_candidate_ids must be a list")
    selected = [
        _identity_candidate_id(item, field="candidate ID")
        for item in selected_values
    ]
    if len(selected) > 1:
        raise VerifierResponseError("at most one candidate may be selected")
    if any(item < 1 or item > candidate_count for item in selected):
        raise VerifierResponseError(
            f"candidate IDs must be in the range 1..{candidate_count}"
        )
    if decision == "select" and len(selected) != 1:
        raise VerifierResponseError("select requires exactly one candidate ID")
    if decision == "no_match" and selected:
        raise VerifierResponseError("no_match requires an empty candidate list")

    assessments = value["candidate_assessments"]
    if not isinstance(assessments, list):
        raise VerifierResponseError("candidate_assessments must be a list")
    parsed_assessments = []
    seen_ids = set()
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise VerifierResponseError("each candidate assessment must be an object")
        assessment_keys = set(assessment)
        if assessment_keys != RANKED_ASSESSMENT_KEYS:
            missing = sorted(RANKED_ASSESSMENT_KEYS - assessment_keys)
            extra = sorted(assessment_keys - RANKED_ASSESSMENT_KEYS)
            raise VerifierResponseError(
                "candidate assessment keys are invalid; "
                f"missing={missing}, extra={extra}"
            )
        candidate_id = _identity_candidate_id(
            assessment["candidate_id"],
            field="assessment candidate_id",
        )
        if candidate_id < 1 or candidate_id > candidate_count:
            raise VerifierResponseError(
                f"assessment candidate_id must be in 1..{candidate_count}"
            )
        if candidate_id in seen_ids:
            raise VerifierResponseError("assessment candidate IDs must be unique")
        seen_ids.add(candidate_id)
        label = assessment["most_likely_object"]
        if not isinstance(label, str) or not label.strip():
            raise VerifierResponseError(
                "most_likely_object must be a non-empty string"
            )
        matches_target = _identity_boolean(
            assessment["matches_target"],
            field="matches_target",
        )
        match_score = _finite_candidate_number(
            assessment["match_score"],
            field="match_score",
        )
        if not 0.0 <= match_score <= 1.0:
            raise VerifierResponseError("match_score must be in [0, 1]")
        parsed_assessments.append(
            {
                "candidate_id": candidate_id,
                "most_likely_object": label.strip(),
                "matches_target": matches_target,
                "match_score": match_score,
            }
        )
    if seen_ids != set(range(1, candidate_count + 1)):
        raise VerifierResponseError(
            "candidate_assessments must contain every candidate exactly once"
        )

    matching = [
        assessment for assessment in parsed_assessments
        if assessment["matches_target"]
    ]
    matching_ids = [assessment["candidate_id"] for assessment in matching]
    if decision == "no_match":
        if matching:
            raise VerifierResponseError(
                "no_match requires every matches_target value to be false"
            )
    else:
        selected_id = selected[0]
        if selected_id not in matching_ids:
            raise VerifierResponseError(
                "the selected candidate must have matches_target=true"
            )
        if selector is None:
            best_score = max(assessment["match_score"] for assessment in matching)
            best_ids = {
                assessment["candidate_id"]
                for assessment in matching
                if math.isclose(
                    assessment["match_score"],
                    best_score,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            }
            if selected_id not in best_ids:
                raise VerifierResponseError(
                    "selected candidate does not have the highest visual match score"
                )
        else:
            winners = spatial_winner_ids(
                candidate_records,
                matching_ids,
                selector,
            )
            if selected_id not in winners:
                raise VerifierResponseError(
                    f"selected candidate contradicts measured selector {selector!r}; "
                    f"expected one of {sorted(winners)}"
                )

    confidence = _finite_candidate_number(value["confidence"], field="confidence")
    if not 0.0 <= confidence <= 1.0:
        raise VerifierResponseError("confidence must be in [0, 1]")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise VerifierResponseError("reason must be a non-empty string")
    lowered_reason = reason.lower()
    forbidden_score_reasons = (
        "sam score",
        "sam confidence",
        "dino score",
        "dino confidence",
        "model confidence",
    )
    if any(phrase in lowered_reason for phrase in forbidden_score_reasons):
        raise VerifierResponseError(
            "reason must use visual mask evidence, not SAM/DINO confidence"
        )
    return {
        "decision": decision,
        "selected_candidate_ids": selected,
        "candidate_assessments": sorted(
            parsed_assessments,
            key=lambda item: item["candidate_id"],
        ),
        "confidence": confidence,
        "reason": reason.strip(),
        "selector": selector,
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
            f"Candidate {position + 1}",
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


def render_clean_dino_candidate_crops(
    rgb_np: np.ndarray,
    candidate_records: list[dict[str, Any]],
    output_path: Path,
    *,
    context_padding_fraction: float = 0.25,
) -> dict[str, Any]:
    """Render clean, labeled crops around the original Grounding DINO boxes.

    Candidate labels are placed in a header outside each crop. No tint, contour,
    rectangle, or other annotation is drawn over the camera pixels Qwen inspects.
    """

    if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {rgb_np.shape}")
    if not candidate_records:
        raise ValueError("At least one candidate record is required")
    if not 0.0 <= context_padding_fraction <= 1.0:
        raise ValueError("context_padding_fraction must be in [0, 1]")

    tile_width = 384
    tile_height = 300
    header_height = 40
    columns = min(2, len(candidate_records))
    rows = math.ceil(len(candidate_records) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        28,
        dtype=np.uint8,
    )
    rendered_records = []
    image_height, image_width = rgb_np.shape[:2]

    for position, record in enumerate(candidate_records):
        expected_id = position + 1
        candidate_id = record.get("candidate_id")
        if candidate_id != expected_id:
            raise ValueError(
                "candidate records must use contiguous one-based IDs in order"
            )
        box = record.get("dino_box_xyxy_crop_pixels")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"Candidate {candidate_id} is missing its original DINO box"
            )
        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Candidate {candidate_id} has a non-numeric DINO box"
            ) from exc
        if not all(np.isfinite(value) for value in (x0, y0, x1, y1)):
            raise ValueError(f"Candidate {candidate_id} has a non-finite DINO box")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Candidate {candidate_id} has an invalid DINO box")

        box_width = x1 - x0
        box_height = y1 - y0
        padding = max(
            12,
            int(round(context_padding_fraction * max(box_width, box_height))),
        )
        crop_x0 = max(0, int(math.floor(x0)) - padding)
        crop_y0 = max(0, int(math.floor(y0)) - padding)
        crop_x1 = min(image_width, int(math.ceil(x1)) + padding)
        crop_y1 = min(image_height, int(math.ceil(y1)) + padding)
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise ValueError(f"Candidate {candidate_id} produced an empty crop")
        crop_rgb = rgb_np[crop_y0:crop_y1, crop_x0:crop_x1]

        available_width = tile_width - 16
        available_height = tile_height - header_height - 12
        scale = min(
            available_width / crop_rgb.shape[1],
            available_height / crop_rgb.shape[0],
        )
        resized_width = max(1, int(round(crop_rgb.shape[1] * scale)))
        resized_height = max(1, int(round(crop_rgb.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        rendered_crop = cv2.resize(
            crop_rgb,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

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
            f"Candidate {candidate_id}",
            (tile_x + 12, tile_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        rendered_records.append(
            {
                "candidate_id": candidate_id,
                "dino_box_xyxy_crop_pixels": [x0, y0, x1, y1],
                "clean_crop_xyxy_crop_pixels": [
                    crop_x0,
                    crop_y0,
                    crop_x1,
                    crop_y1,
                ],
            }
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet[:, :, ::-1]):
        raise OSError(f"Failed to write clean DINO candidate crops: {output_path}")
    return {
        "output": str(output_path),
        "num_candidates_drawn": len(candidate_records),
        "label_indexing": "one_based",
        "input_mode": "clean_dino_box_crops",
        "candidates": rendered_records,
    }


def build_identity_verifier_messages(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_crop_path: Path,
    candidate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the clean-crop, identity-only Grounding DINO verifier prompt."""

    candidate_ids = [int(record["candidate_id"]) for record in candidate_records]
    system_prompt = (
        "You are the semantic identity verifier for a robot pick target. Image 1 "
        "is the unmodified camera crop. Image 2 is a sheet of clean camera crops "
        "made from Grounding DINO boxes. Candidate labels appear only in the dark "
        "header outside each crop; no segmentation mask, tint, contour, or rectangle "
        "is drawn over the objects. For every candidate, identify the single most "
        "likely primary physical object framed by that crop. Then decide whether that "
        "primary object is one movable instance of the requested semantic target. A "
        "requested object merely visible in the background or inside a crop whose "
        "primary object is a shelf, bin, cart, robot, fixture, or larger surrounding "
        "structure is not a match. Use visible colors, shape, material, printed text, "
        "and logos as evidence. Do not judge SAM mask boundaries, segmentation "
        "quality, depth quality, graspability, or robot motion; separate deterministic "
        "gates handle those concerns. Do not apply relative selectors such as topmost "
        "or rightmost; deterministic geometry applies them after semantic identity. "
        "Return JSON only with exactly these keys: decision, selected_candidate_ids, "
        "candidate_assessments, confidence. candidate_assessments must contain "
        "one object for every candidate with exactly candidate_id, most_likely_object, "
        "and matches_target. most_likely_object must name what you actually see, not "
        "just repeat the request. selected_candidate_ids must exactly equal the IDs "
        "whose matches_target value is true. decision must be select when that list is "
        "non-empty and no_match when it is empty. confidence must be a number from 0 "
        "to 1. Candidate IDs must be JSON integers without quotes. matches_target must "
        "be a JSON boolean true or false without quotes. Do not use markdown or add "
        "other text. Return one complete JSON object and nothing else."
    )
    selector_text = selector if selector is not None else "none"
    user_text = (
        f"Original robot request: {request!r}. Semantic target phrase: "
        f"{target_phrase!r}. Deferred spatial selector: {selector_text!r}. "
        f"Candidate IDs shown in Image 2: {candidate_ids}. Inspect each clean crop, "
        "name its most likely primary object, and mark whether it matches the semantic "
        "target. Think silently, then return only the required JSON object."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1: unmodified camera crop."},
                {"type": "image", "image": str(frame_path)},
                {
                    "type": "text",
                    "text": (
                        "Image 2: clean Grounding DINO candidate crops with labels "
                        "outside the camera pixels."
                    ),
                },
                {"type": "image", "image": str(candidate_crop_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_verifier_messages(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_overlay_path: Path,
    candidate_zoom_path: Path,
    candidate_records: list[dict[str, Any]],
    grounding_intent_value: dict[str, Any] | None = None,
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
        "Use visible printed text, logos, colors, shape, material, and the requested "
        "source region when those fields are present in the structured intent. Never "
        "invent an attribute that is not in that intent. "
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
    intent_details = json.dumps(
        grounding_intent_value,
        separators=(",", ":"),
    ) if grounding_intent_value is not None else "null"
    selector_text = selector if selector is not None else "none"
    user_text = (
        f"Original robot request: {request!r}. Semantic target phrase: "
        f"{target_phrase!r}. Deferred spatial selector: {selector_text!r}. "
        f"Canonical structured grounding intent: {intent_details}. "
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


def build_ranked_mask_verifier_messages(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_overlay_path: Path,
    candidate_zoom_path: Path,
    candidate_records: list[dict[str, Any]],
    grounding_intent_value: dict[str, Any] | None = None,
    candidate_crop_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build a full-intent, single-choice prompt over numbered SAM masks."""

    system_prompt = (
        "You are the final visual ranker for a robot pick target. Image 1 is the "
        "unmodified camera crop. Image 2 shows numbered SAM masks with colored "
        "boundaries. Image 3 shows enlarged views of those same masks. If supplied, "
        "Image 4 contains clean Grounding DINO crops for identity details. Inspect "
        "what physical object each mask boundary actually encloses; a requested "
        "object merely visible inside a mask around a shelf, bin, cart, fixture, or "
        "background is not a match. Use the canonical structured intent as the "
        "authoritative breakdown of category and visible attributes. Assess every "
        "candidate with a semantic-and-boundary match_score from 0 to 1. If no "
        "candidate matches, return no_match. Otherwise select exactly one candidate. "
        "Ignore SAM confidence, DINO confidence, proposal order, and mask area unless "
        "the explicit selector is largest or smallest. Never copy a model confidence "
        "into match_score; match_score must be your independent visual judgment of "
        "semantic identity and how tightly the colored boundary follows that object. "
        "Without a selector, select the matching candidate with the highest "
        "match_score. With a selector, first identify the semantic matches and then "
        "apply the supplied measured metadata: smaller center x is leftmost, larger "
        "center x is rightmost, smaller center y is topmost, larger center y is "
        "bottommost, smaller median_depth_m is nearest, larger median_depth_m is "
        "farthest, and area_pixels determines largest or smallest. Pixel origin is "
        "the top-left. Never estimate or invent pixel/depth values; use only the "
        "candidate metadata. Return JSON only with exactly decision, "
        "selected_candidate_ids, candidate_assessments, confidence, reason. "
        "selected_candidate_ids must contain exactly one integer for select or be [] "
        "for no_match. candidate_assessments must contain every candidate exactly "
        "once with only candidate_id, most_likely_object, matches_target, and "
        "match_score. Use unquoted JSON integers, booleans, and numbers. Do not use "
        "markdown or add other text."
    )
    intent_details = (
        json.dumps(grounding_intent_value, separators=(",", ":"))
        if grounding_intent_value is not None
        else "null"
    )
    qwen_metadata_keys = {
        "candidate_id",
        "area_pixels",
        "area_fraction",
        "center_xy_crop_pixels",
        "bbox_xywh_crop_pixels",
        "crop_size_wh_pixels",
        "median_depth_m",
        "valid_depth_fraction",
    }
    qwen_candidate_records = [
        {
            key: record[key]
            for key in qwen_metadata_keys
            if key in record
        }
        for record in candidate_records
    ]
    candidate_details = json.dumps(
        qwen_candidate_records,
        separators=(",", ":"),
    )
    selector_text = selector if selector is not None else "none"
    user_text = (
        f"Original robot request: {request!r}. Semantic target phrase: "
        f"{target_phrase!r}. Canonical structured intent: {intent_details}. "
        f"Requested selector: {selector_text!r}. Measured candidate metadata: "
        f"{candidate_details}. Compare every numbered mask to the prompt breakdown, "
        "then return the one best candidate or no_match using only the required JSON."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Image 1: unmodified camera crop."},
        {"type": "image", "image": str(frame_path)},
        {"type": "text", "text": "Image 2: numbered SAM mask boundaries."},
        {"type": "image", "image": str(candidate_overlay_path)},
        {"type": "text", "text": "Image 3: enlarged numbered SAM masks."},
        {"type": "image", "image": str(candidate_zoom_path)},
    ]
    if candidate_crop_path is not None:
        content.extend(
            [
                {
                    "type": "text",
                    "text": "Image 4: clean numbered DINO crops for identity detail.",
                },
                {"type": "image", "image": str(candidate_crop_path)},
            ]
        )
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def run_ranked_mask_verifier(
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
    grounding_intent_value: dict[str, Any] | None = None,
    candidate_crop_path: Path | None = None,
) -> dict[str, Any]:
    """Ask Qwen for one best mask and fail closed on schema/geometry drift."""

    if not 0.0 <= min_select_confidence <= 1.0:
        raise ValueError("min_select_confidence must be in [0, 1]")
    if not candidate_records:
        raise ValueError("candidate_records must not be empty")
    messages = build_ranked_mask_verifier_messages(
        request=request,
        target_phrase=target_phrase,
        selector=selector,
        frame_path=frame_path,
        candidate_overlay_path=candidate_overlay_path,
        candidate_zoom_path=candidate_zoom_path,
        candidate_records=candidate_records,
        grounding_intent_value=grounding_intent_value,
        candidate_crop_path=candidate_crop_path,
    )
    attempts = []
    for attempt_number in (1, 2):
        try:
            raw = send_generate_request(messages)
        except Exception as exc:
            attempts.append(
                {"attempt": attempt_number, "raw_response": None, "error": repr(exc)}
            )
            return {
                "status": "error",
                "decision": None,
                "selected_candidate_ids": [],
                "model_selected_candidate_ids": [],
                "candidate_assessments": [],
                "confidence": None,
                "reason": f"Qwen ranked-mask verifier inference failed: {exc!r}",
                "attempts": attempts,
            }
        try:
            parsed = parse_ranked_mask_verifier_response(
                raw,
                candidate_records,
                selector,
            )
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
                        {"role": "assistant", "content": str(raw)[:1500]},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid single-choice response: {exc}. Retry once. "
                                "Assess every candidate, use the exact measured "
                                "metadata for any selector, and return zero or one "
                                "selected candidate in the required JSON schema."
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
                "candidate_assessments": [],
                "confidence": None,
                "reason": "Qwen ranked-mask verifier returned invalid JSON twice",
                "attempts": attempts,
            }

        attempts.append(
            {"attempt": attempt_number, "raw_response": raw, "error": None}
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
            "candidate_assessments": parsed["candidate_assessments"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "selector": parsed["selector"],
            "attempts": attempts,
        }

    raise AssertionError("unreachable ranked-mask verifier retry state")


def run_identity_verifier(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    frame_path: Path,
    candidate_crop_path: Path,
    candidate_records: list[dict[str, Any]],
    send_generate_request: Callable[[list[dict[str, Any]]], str],
    min_select_confidence: float,
) -> dict[str, Any]:
    """Ask Qwen to classify clean DINO crops and fail closed on uncertainty."""

    if not 0.0 <= min_select_confidence <= 1.0:
        raise ValueError("min_select_confidence must be in [0, 1]")
    if not candidate_records:
        raise ValueError("candidate_records must not be empty")

    messages = build_identity_verifier_messages(
        request=request,
        target_phrase=target_phrase,
        selector=selector,
        frame_path=frame_path,
        candidate_crop_path=candidate_crop_path,
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
                "candidate_assessments": [],
                "confidence": None,
                "reason": f"Qwen identity verifier inference failed: {exc!r}",
                "attempts": attempts,
            }
        try:
            parsed = parse_identity_verifier_response(
                raw,
                len(candidate_records),
            )
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
                        {"role": "assistant", "content": str(raw)[:1500]},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid identity response: {exc}. Retry once. Return "
                                "exactly one JSON object with only decision, "
                                "selected_candidate_ids, candidate_assessments, "
                                "confidence. Include every candidate exactly "
                                "once in candidate_assessments. Each assessment must "
                                "contain only candidate_id, most_likely_object, and "
                                "matches_target. selected_candidate_ids must exactly "
                                "match the true matches_target assessments. Use JSON "
                                "integers for IDs and unquoted JSON true/false values. "
                                "Return one complete JSON object and nothing else."
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
                "candidate_assessments": [],
                "confidence": None,
                "reason": "Qwen identity verifier returned invalid JSON twice",
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
            "candidate_assessments": parsed["candidate_assessments"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"],
            "attempts": attempts,
        }

    raise AssertionError("unreachable identity verifier retry state")


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
    grounding_intent_value: dict[str, Any] | None = None,
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
        grounding_intent_value=grounding_intent_value,
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
