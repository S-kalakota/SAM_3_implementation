#!/usr/bin/env python3
"""Pure helpers for multiview SAM candidate projection and deduplication."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


REGION_DIRECTION_WORDS = {
    "top": "topmost",
    "upper": "topmost",
    "bottom": "bottommost",
    "lower": "bottommost",
    "left": "leftmost",
    "right": "rightmost",
    "center": "center",
    "middle": "center",
}


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def padded_mask_roi(
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.15,
) -> tuple[int, int, int, int]:
    """Return a padded, image-clamped xyxy ROI for one non-empty mask."""

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {source.shape}")
    if not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("padding_fraction must be in [0, 1]")
    x, y, width, height = mask_bbox_xywh(source)
    if width == 0 or height == 0:
        raise ValueError("cannot create a region ROI from an empty mask")
    pad_x = int(np.ceil(width * padding_fraction))
    pad_y = int(np.ceil(height * padding_fraction))
    image_height, image_width = source.shape
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(image_width, x + width + pad_x),
        min(image_height, y + height + pad_y),
    )


def source_region_direction(source_region: str) -> str | None:
    """Resolve the directional modifier embedded in a structured source region."""

    words = str(source_region).strip().lower().split()
    directions = {
        REGION_DIRECTION_WORDS[word]
        for word in words
        if word in REGION_DIRECTION_WORDS
    }
    if len(directions) > 1:
        raise ValueError(f"source region has conflicting directions: {source_region!r}")
    return next(iter(directions), None)


def source_region_prompt(source_region: str) -> str:
    """Return the region noun phrase without directional selection words."""

    words = str(source_region).strip().lower().split()
    prompt = " ".join(word for word in words if word not in REGION_DIRECTION_WORDS)
    if not prompt:
        raise ValueError(f"source region has no semantic noun: {source_region!r}")
    return prompt


def select_source_region_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    source_region: str,
    conf_threshold: float,
    min_area: int,
) -> int | None:
    """Select one valid region mask using its phrase direction, then SAM score."""

    mask_array = np.asarray(masks, dtype=bool)
    if mask_array.ndim != 3:
        raise ValueError(f"masks must have shape NxHxW, got {mask_array.shape}")
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")
    if min_area < 0:
        raise ValueError("min_area must not be negative")

    eligible: list[dict[str, float | int]] = []
    image_height, image_width = mask_array.shape[1:]
    image_center_x = (image_width - 1) / 2.0
    image_center_y = (image_height - 1) / 2.0
    for index, mask in enumerate(mask_array):
        score = float(score_array[index]) if index < score_array.size else 0.0
        area = int(np.count_nonzero(mask))
        if score <= conf_threshold or area <= min_area:
            continue
        ys, xs = np.where(mask)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        eligible.append(
            {
                "index": index,
                "score": score,
                "area": area,
                "center_x": center_x,
                "center_y": center_y,
                "center_distance_sq": (
                    (center_x - image_center_x) ** 2
                    + (center_y - image_center_y) ** 2
                ),
            }
        )
    if not eligible:
        return None

    direction = source_region_direction(source_region)
    if direction == "topmost":
        key = lambda item: (item["center_y"], -item["score"], -item["area"])
    elif direction == "bottommost":
        key = lambda item: (-item["center_y"], -item["score"], -item["area"])
    elif direction == "leftmost":
        key = lambda item: (item["center_x"], -item["score"], -item["area"])
    elif direction == "rightmost":
        key = lambda item: (-item["center_x"], -item["score"], -item["area"])
    elif direction == "center":
        key = lambda item: (
            item["center_distance_sq"],
            -item["score"],
            -item["area"],
        )
    else:
        key = lambda item: (-item["score"], -item["area"], item["index"])
    return int(min(eligible, key=key)["index"])


def project_mask_to_canonical(
    mask: np.ndarray,
    *,
    source_roi_xyxy: tuple[int, int, int, int],
    canonical_roi_xyxy: tuple[int, int, int, int],
    canonical_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Map a source-view mask into the canonical workspace-crop image."""

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {source.shape}")
    source_x0, source_y0, source_x1, source_y1 = source_roi_xyxy
    target_x0, target_y0, target_x1, target_y1 = canonical_roi_xyxy
    if not (
        0 <= source_x0 < source_x1 <= source.shape[1]
        and 0 <= source_y0 < source_y1 <= source.shape[0]
    ):
        raise ValueError("source ROI is outside the mask")
    canonical_height, canonical_width = canonical_shape_hw
    if not (
        0 <= target_x0 < target_x1 <= canonical_width
        and 0 <= target_y0 < target_y1 <= canonical_height
    ):
        raise ValueError("canonical ROI is outside the workspace crop")

    source_crop = source[source_y0:source_y1, source_x0:source_x1]
    target_width = target_x1 - target_x0
    target_height = target_y1 - target_y0
    if source_crop.shape != (target_height, target_width):
        source_crop = cv2.resize(
            source_crop.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    projected = np.zeros((canonical_height, canonical_width), dtype=bool)
    projected[target_y0:target_y1, target_x0:target_x1] = source_crop
    return projected


def overlapping_tile_rois(
    width: int,
    height: int,
    *,
    scale: float = 0.72,
) -> list[tuple[int, int, int, int]]:
    """Return four overlapping corner tiles in canonical xyxy coordinates."""

    if isinstance(width, bool) or isinstance(height, bool) or width < 2 or height < 2:
        raise ValueError("tile image dimensions must be at least 2x2")
    if not 0.5 <= scale < 1.0:
        raise ValueError("tile scale must be in [0.5, 1.0)")
    tile_width = min(width, max(2, int(round(width * scale))))
    tile_height = min(height, max(2, int(round(height * scale))))
    x_positions = sorted({0, width - tile_width})
    y_positions = sorted({0, height - tile_height})
    return [
        (x, y, x + tile_width, y + tile_height)
        for y in y_positions
        for x in x_positions
    ]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("mask shapes differ")
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(first & second))
    return intersection / union


def deduplicate_candidates(
    raw_candidates: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Keep the highest-scoring representative of near-identical masks."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered = sorted(
        raw_candidates,
        key=lambda item: (-float(item["score"]), int(np.count_nonzero(item["mask"]))),
    )
    merged: list[dict[str, Any]] = []
    for raw in ordered:
        mask = np.asarray(raw["mask"], dtype=bool)
        if mask.ndim != 2 or not mask.any():
            continue
        provenance = {
            "prompt": str(raw["prompt"]),
            "view_id": str(raw["view_id"]),
            "view_kind": str(raw["view_kind"]),
            "source_candidate_index": int(raw["source_candidate_index"]),
            "source_score": float(raw["score"]),
            "source_json": str(raw["source_json"]),
            "workspace_retained_fraction": float(
                raw.get("workspace_retained_fraction", 1.0)
            ),
        }
        duplicate = next(
            (
                candidate
                for candidate in merged
                if mask_iou(mask, candidate["mask"]) >= iou_threshold
            ),
            None,
        )
        if duplicate is not None:
            duplicate["provenance"].append(provenance)
            duplicate["duplicate_count"] = len(duplicate["provenance"])
            duplicate["max_workspace_retained_fraction"] = max(
                float(duplicate["max_workspace_retained_fraction"]),
                provenance["workspace_retained_fraction"],
            )
            continue
        merged.append(
            {
                "mask": mask,
                "score": float(raw["score"]),
                "area_pixels": int(np.count_nonzero(mask)),
                "bbox_xywh_crop_pixels": mask_bbox_xywh(mask),
                "provenance": [provenance],
                "duplicate_count": 1,
                "max_workspace_retained_fraction": provenance[
                    "workspace_retained_fraction"
                ],
            }
        )
    return merged


def gate_merged_candidates(
    merged: list[dict[str, Any]],
    *,
    conf_threshold: float,
    min_area: int,
    max_candidates: int,
    min_workspace_retained_fraction: float = 0.90,
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]]]:
    """Apply score/area gates while preserving prompt and view provenance."""

    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")
    if min_area < 0:
        raise ValueError("min_area must not be negative")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not 0.0 <= min_workspace_retained_fraction <= 1.0:
        raise ValueError("min_workspace_retained_fraction must be in [0, 1]")

    kept: list[tuple[np.ndarray, float]] = []
    candidates: list[dict[str, Any]] = []
    eligible_seen = 0
    for index, candidate in enumerate(merged):
        score = float(candidate["score"])
        area = int(candidate["area_pixels"])
        reject_reasons: list[str] = []
        if score <= conf_threshold:
            reject_reasons.append("low_score")
        if area <= min_area:
            reject_reasons.append("small_area")
        retained_fraction = float(candidate["max_workspace_retained_fraction"])
        if retained_fraction < min_workspace_retained_fraction:
            reject_reasons.append("mask_extends_outside_workspace")
        if not reject_reasons:
            eligible_seen += 1
            if eligible_seen > max_candidates:
                reject_reasons.append("candidate_limit")
        is_kept = not reject_reasons
        record = {
            "index": index,
            "score": score,
            "area_pixels": area,
            "bbox_xywh_crop_pixels": list(candidate["bbox_xywh_crop_pixels"]),
            "duplicate_count": int(candidate["duplicate_count"]),
            "max_workspace_retained_fraction": retained_fraction,
            "provenance": list(candidate["provenance"]),
            "kept": is_kept,
            "reject_reasons": reject_reasons,
        }
        candidates.append(record)
        if is_kept:
            kept.append((np.asarray(candidate["mask"], dtype=bool), score))
    return kept, candidates


def expand_crop_mask_to_full(
    mask: np.ndarray,
    crop_info: dict[str, Any],
) -> np.ndarray:
    """Expand one canonical crop mask into its full camera resolution."""

    crop_mask = np.asarray(mask, dtype=bool)
    if not crop_info.get("enabled", False):
        expected = (
            int(crop_info["full_height"]),
            int(crop_info["full_width"]),
        )
        if crop_mask.shape != expected:
            raise ValueError(f"mask shape {crop_mask.shape} does not match {expected}")
        return crop_mask.copy()
    x0, y0, x1, y1 = [int(value) for value in crop_info["applied_xyxy"]]
    expected_crop_shape = (y1 - y0, x1 - x0)
    if crop_mask.shape != expected_crop_shape:
        raise ValueError(
            f"mask shape {crop_mask.shape} does not match crop {expected_crop_shape}"
        )
    full = np.zeros(
        (int(crop_info["full_height"]), int(crop_info["full_width"])),
        dtype=bool,
    )
    full[y0:y1, x0:x1] = crop_mask
    return full


def crop_bbox_to_full(
    bbox_xywh: list[int],
    crop_info: dict[str, Any],
) -> list[int]:
    x, y, width, height = [int(value) for value in bbox_xywh]
    if crop_info.get("enabled", False):
        crop_x, crop_y = [int(value) for value in crop_info["applied_xyxy"][:2]]
        x += crop_x
        y += crop_y
    return [x, y, width, height]
