#!/usr/bin/env python3
"""Deterministic relationship measurements for SAM mask candidates."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "inside_min_mask_fraction": 0.80,
    "depth_compatibility_m": 0.12,
    "on_min_horizontal_overlap": 0.25,
    "on_min_gap_fraction": -0.08,
    "on_max_gap_fraction": 0.12,
    "direction_margin_fraction": 0.03,
    "near_max_edge_distance_fraction": 0.25,
    "next_to_max_edge_distance_fraction": 0.12,
    "near_max_3d_distance_m": 0.40,
    "next_to_max_3d_distance_m": 0.25,
    "front_depth_margin_m": 0.03,
    "depth_min_valid_pixels": 20,
}
SUPPORTED_RELATIONSHIPS = {
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


def _as_mask(value: Any, label: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"{label} must be a 2-D mask, got {mask.shape}")
    return mask


def _bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0


def _finite_depth_values(mask: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    if depth is None:
        return np.zeros((0,), dtype=np.float32)
    values = np.asarray(depth, dtype=np.float32)[mask]
    return values[np.isfinite(values) & (values > 0.0)]


def _median_depth(mask: np.ndarray, depth: np.ndarray | None) -> tuple[float | None, int]:
    values = _finite_depth_values(mask, depth)
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def _median_xyz(mask: np.ndarray, xyz: np.ndarray | None) -> tuple[np.ndarray | None, int]:
    if xyz is None:
        return None, 0
    points = np.asarray(xyz, dtype=np.float32)[mask]
    if points.ndim != 2 or points.shape[1] < 3:
        return None, 0
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    # A zero triplet is a common invalid-depth sentinel.
    valid &= np.linalg.norm(points, axis=1) > 0.0
    points = points[valid]
    if points.size == 0:
        return None, 0
    return np.median(points, axis=0), int(points.shape[0])


def _edge_distance_pixels(first: np.ndarray, second: np.ndarray) -> float:
    if np.any(first & second):
        return 0.0
    # distanceTransform reports the distance to the nearest zero.  Make the
    # second mask the zero set and sample at pixels belonging to the first.
    distance = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 5)
    samples = distance[first]
    return float(samples.min()) if samples.size else math.inf


def _depth_measurements(
    target: np.ndarray,
    anchor: np.ndarray,
    depth: np.ndarray | None,
) -> dict[str, Any]:
    target_depth, target_count = _median_depth(target, depth)
    anchor_depth, anchor_count = _median_depth(anchor, depth)
    difference = None
    if target_depth is not None and anchor_depth is not None:
        difference = target_depth - anchor_depth
    return {
        "target_median_depth_m": target_depth,
        "anchor_median_depth_m": anchor_depth,
        "target_valid_depth_pixels": target_count,
        "anchor_valid_depth_pixels": anchor_count,
        "target_minus_anchor_depth_m": difference,
        "absolute_depth_difference_m": None if difference is None else abs(difference),
    }


def _result(
    relationship: str,
    *,
    status: str,
    reason: str,
    measurements: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    return {
        "relationship": relationship,
        "status": status,
        "passed": True if status == "pass" else False if status == "fail" else None,
        "reason": reason,
        "measurements": measurements,
        "thresholds": thresholds,
    }


def evaluate_relationship(
    relationship: str,
    target_mask: Any,
    anchor_mask: Any,
    *,
    depth: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Measure and pass/fail one declared target-to-anchor relationship."""

    if relationship not in SUPPORTED_RELATIONSHIPS:
        raise ValueError(f"unsupported relationship {relationship!r}")
    target = _as_mask(target_mask, "target_mask")
    anchor = _as_mask(anchor_mask, "anchor_mask")
    if target.shape != anchor.shape:
        raise ValueError(f"mask shape mismatch: {target.shape} != {anchor.shape}")
    if depth is not None and np.asarray(depth).shape[:2] != target.shape:
        raise ValueError("depth shape must match masks")
    if xyz is not None and np.asarray(xyz).shape[:2] != target.shape:
        raise ValueError("xyz shape must match masks")
    configured = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(configured)
        if unknown:
            raise ValueError(f"unknown relationship thresholds: {sorted(unknown)}")
        configured.update(thresholds)

    target_box = _bbox_xyxy(target)
    anchor_box = _bbox_xyxy(anchor)
    if target_box is None or anchor_box is None:
        return _result(
            relationship,
            status="unavailable",
            reason="empty_target_or_anchor_mask",
            measurements={
                "target_area_pixels": int(target.sum()),
                "anchor_area_pixels": int(anchor.sum()),
            },
            thresholds={},
        )

    height, width = target.shape
    diagonal = math.hypot(width, height)
    tx, ty = _center(target)
    ax, ay = _center(anchor)
    tbx, tby = _bbox_center(target_box)
    abx, aby = _bbox_center(anchor_box)
    common: dict[str, Any] = {
        "target_mask_center_xy": [tx, ty],
        "anchor_mask_center_xy": [ax, ay],
        "target_bbox_center_xy": [tbx, tby],
        "anchor_bbox_center_xy": [abx, aby],
        "target_bbox_xyxy": list(target_box),
        "anchor_bbox_xyxy": list(anchor_box),
    }

    if relationship == "inside":
        x0, y0, x1, y1 = anchor_box
        filled_bounds = np.zeros_like(anchor)
        filled_bounds[y0:y1, x0:x1] = True
        fraction = float(np.count_nonzero(target & filled_bounds) / np.count_nonzero(target))
        center_inside = bool(x0 <= tx < x1 and y0 <= ty < y1)
        depth_metrics = _depth_measurements(target, anchor, depth)
        enough_depth = min(
            depth_metrics["target_valid_depth_pixels"],
            depth_metrics["anchor_valid_depth_pixels"],
        ) >= int(configured["depth_min_valid_pixels"])
        if not enough_depth:
            return _result(
                relationship,
                status="unavailable",
                reason="insufficient_reliable_depth",
                measurements={
                    **common,
                    "target_center_inside_anchor_bounds": center_inside,
                    "target_fraction_inside_anchor_bounds": fraction,
                    **depth_metrics,
                },
                thresholds={
                    "depth_min_valid_pixels": configured["depth_min_valid_pixels"]
                },
            )
        depth_compatible = bool(
            depth_metrics["absolute_depth_difference_m"]
            <= float(configured["depth_compatibility_m"])
        )
        passed = center_inside and fraction >= float(configured["inside_min_mask_fraction"])
        if depth_compatible is False:
            passed = False
        measurements = {
            **common,
            "target_center_inside_anchor_bounds": center_inside,
            "target_fraction_inside_anchor_bounds": fraction,
            "depth_compatible": depth_compatible,
            **depth_metrics,
        }
        return _result(
            relationship,
            status="pass" if passed else "fail",
            reason="all_inside_constraints_passed" if passed else "inside_constraints_failed",
            measurements=measurements,
            thresholds={
                "inside_min_mask_fraction": configured["inside_min_mask_fraction"],
                "depth_compatibility_m": configured["depth_compatibility_m"],
                "depth_min_valid_pixels": configured["depth_min_valid_pixels"],
            },
        )

    if relationship == "on":
        tx0, ty0, tx1, ty1 = target_box
        ax0, ay0, ax1, _ay1 = anchor_box
        overlap = max(0, min(tx1, ax1) - max(tx0, ax0))
        overlap_fraction = float(overlap / max(1, tx1 - tx0))
        gap_fraction = float((ay0 - ty1) / max(1, height))
        depth_metrics = _depth_measurements(target, anchor, depth)
        enough_depth = min(
            depth_metrics["target_valid_depth_pixels"],
            depth_metrics["anchor_valid_depth_pixels"],
        ) >= int(configured["depth_min_valid_pixels"])
        if not enough_depth:
            return _result(
                relationship,
                status="unavailable",
                reason="insufficient_reliable_depth",
                measurements={
                    **common,
                    "horizontal_support_overlap_fraction": overlap_fraction,
                    "signed_vertical_gap_fraction": gap_fraction,
                    **depth_metrics,
                },
                thresholds={
                    "depth_min_valid_pixels": configured["depth_min_valid_pixels"]
                },
            )
        depth_compatible = bool(
            depth_metrics["absolute_depth_difference_m"]
            <= float(configured["depth_compatibility_m"])
        )
        passed = (
            overlap_fraction >= float(configured["on_min_horizontal_overlap"])
            and float(configured["on_min_gap_fraction"])
            <= gap_fraction
            <= float(configured["on_max_gap_fraction"])
        )
        if depth_compatible is False:
            passed = False
        return _result(
            relationship,
            status="pass" if passed else "fail",
            reason="all_support_constraints_passed" if passed else "support_constraints_failed",
            measurements={
                **common,
                "horizontal_support_overlap_fraction": overlap_fraction,
                "signed_vertical_gap_fraction": gap_fraction,
                "depth_compatible": depth_compatible,
                **depth_metrics,
            },
            thresholds={
                "on_min_horizontal_overlap": configured["on_min_horizontal_overlap"],
                "on_min_gap_fraction": configured["on_min_gap_fraction"],
                "on_max_gap_fraction": configured["on_max_gap_fraction"],
                "depth_compatibility_m": configured["depth_compatibility_m"],
            },
        )

    if relationship in {"left_of", "right_of", "above", "below"}:
        x_separation = float((abx - tbx) / max(1, width))
        y_separation = float((aby - tby) / max(1, height))
        margin = float(configured["direction_margin_fraction"])
        signed = {
            "left_of": x_separation,
            "right_of": -x_separation,
            "above": y_separation,
            "below": -y_separation,
        }[relationship]
        passed = signed >= margin
        return _result(
            relationship,
            status="pass" if passed else "fail",
            reason="directional_margin_passed" if passed else "directional_margin_failed",
            measurements={
                **common,
                "target_to_anchor_x_separation_fraction": x_separation,
                "target_to_anchor_y_separation_fraction": y_separation,
                "signed_required_axis_separation_fraction": signed,
            },
            thresholds={"direction_margin_fraction": margin},
        )

    if relationship in {"near", "next_to"}:
        edge_pixels = _edge_distance_pixels(target, anchor)
        edge_fraction = float(edge_pixels / diagonal)
        threshold_name = (
            "near_max_edge_distance_fraction"
            if relationship == "near"
            else "next_to_max_edge_distance_fraction"
        )
        max_edge = float(configured[threshold_name])
        target_xyz, target_xyz_count = _median_xyz(target, xyz)
        anchor_xyz, anchor_xyz_count = _median_xyz(anchor, xyz)
        distance_3d = None
        if target_xyz is not None and anchor_xyz is not None:
            distance_3d = float(np.linalg.norm(target_xyz - anchor_xyz))
        distance_name = (
            "near_max_3d_distance_m"
            if relationship == "near"
            else "next_to_max_3d_distance_m"
        )
        max_3d = float(configured[distance_name])
        passed = edge_fraction <= max_edge
        if (
            distance_3d is not None
            and min(target_xyz_count, anchor_xyz_count)
            >= int(configured["depth_min_valid_pixels"])
        ):
            passed = passed and distance_3d <= max_3d
        return _result(
            relationship,
            status="pass" if passed else "fail",
            reason="proximity_constraints_passed" if passed else "proximity_constraints_failed",
            measurements={
                **common,
                "mask_edge_distance_pixels": edge_pixels,
                "mask_edge_distance_fraction": edge_fraction,
                "median_3d_distance_m": distance_3d,
                "target_valid_xyz_pixels": target_xyz_count,
                "anchor_valid_xyz_pixels": anchor_xyz_count,
            },
            thresholds={
                threshold_name: max_edge,
                distance_name: max_3d,
                "depth_min_valid_pixels": configured["depth_min_valid_pixels"],
            },
        )

    depth_metrics = _depth_measurements(target, anchor, depth)
    if min(
        depth_metrics["target_valid_depth_pixels"],
        depth_metrics["anchor_valid_depth_pixels"],
    ) < int(configured["depth_min_valid_pixels"]):
        return _result(
            relationship,
            status="unavailable",
            reason="insufficient_reliable_depth",
            measurements={**common, **depth_metrics},
            thresholds={
                "depth_min_valid_pixels": configured["depth_min_valid_pixels"],
                "front_depth_margin_m": configured["front_depth_margin_m"],
            },
        )
    difference = float(depth_metrics["target_minus_anchor_depth_m"])
    margin = float(configured["front_depth_margin_m"])
    signed = -difference if relationship == "in_front_of" else difference
    passed = signed >= margin
    return _result(
        relationship,
        status="pass" if passed else "fail",
        reason="depth_order_margin_passed" if passed else "depth_order_margin_failed",
        measurements={
            **common,
            **depth_metrics,
            "signed_required_depth_separation_m": signed,
        },
        thresholds={
            "front_depth_margin_m": margin,
            "depth_min_valid_pixels": configured["depth_min_valid_pixels"],
        },
    )


def relation_aware_roi(
    anchor_mask: Any,
    relationship: str,
    *,
    padding_fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """Return the target-search ROI implied by one anchor relationship."""

    anchor = _as_mask(anchor_mask, "anchor_mask")
    box = _bbox_xyxy(anchor)
    if box is None:
        raise ValueError("cannot make a relation crop from an empty anchor mask")
    if not 0.0 <= padding_fraction <= 2.0:
        raise ValueError("padding_fraction must be in [0, 2]")
    height, width = anchor.shape
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    pad_x = max(1, int(math.ceil(box_width * padding_fraction)))
    pad_y = max(1, int(math.ceil(box_height * padding_fraction)))
    center_x = int(round((x0 + x1) / 2.0))
    center_y = int(round((y0 + y1) / 2.0))

    if relationship == "inside":
        roi = (x0, y0, x1, y1)
    elif relationship == "on":
        roi = (x0 - pad_x, y0 - box_height - pad_y, x1 + pad_x, y0 + pad_y)
    elif relationship in {"near", "next_to"}:
        roi = (
            x0 - max(pad_x, box_width),
            y0 - max(pad_y, box_height),
            x1 + max(pad_x, box_width),
            y1 + max(pad_y, box_height),
        )
    elif relationship == "left_of":
        roi = (0, 0, center_x, height)
    elif relationship == "right_of":
        roi = (center_x, 0, width, height)
    elif relationship == "above":
        roi = (0, 0, width, center_y)
    elif relationship == "below":
        roi = (0, center_y, width, height)
    elif relationship in {"in_front_of", "behind"}:
        roi = (0, 0, width, height)
    else:
        raise ValueError(f"unsupported relationship {relationship!r}")

    rx0, ry0, rx1, ry1 = roi
    rx0 = max(0, min(width - 1, int(rx0)))
    ry0 = max(0, min(height - 1, int(ry0)))
    rx1 = max(rx0 + 1, min(width, int(rx1)))
    ry1 = max(ry0 + 1, min(height, int(ry1)))
    return rx0, ry0, rx1, ry1


def grid_search_thresholds(
    cases: Iterable[dict[str, Any]],
    threshold_grid: dict[str, Iterable[float | int]],
) -> dict[str, Any]:
    """Calibrate thresholds with the safety-set zero-false-accept constraint.

    Each case supplies relationship, target_mask, anchor_mask, expected_pass,
    optional safety_case, depth, and xyz.  The best valid setting maximises
    ordinary-case accuracy; ties prefer the lexicographically smaller values.
    """

    materialized = list(cases)
    if not materialized:
        raise ValueError("at least one calibration case is required")
    names = sorted(threshold_grid)
    if not names:
        raise ValueError("threshold_grid must not be empty")
    values = [list(threshold_grid[name]) for name in names]
    if any(not items for items in values):
        raise ValueError("each threshold grid dimension must contain values")

    best: dict[str, Any] | None = None
    evaluated = 0
    safety_rejected = 0
    for combination in itertools.product(*values):
        configured = dict(zip(names, combination))
        evaluated += 1
        predictions: list[tuple[dict[str, Any], bool]] = []
        for case in materialized:
            result = evaluate_relationship(
                case["relationship"],
                case["target_mask"],
                case["anchor_mask"],
                depth=case.get("depth"),
                xyz=case.get("xyz"),
                thresholds=configured,
            )
            predictions.append((case, result["status"] == "pass"))
        false_safety_accepts = sum(
            1
            for case, predicted in predictions
            if case.get("safety_case", False)
            and not bool(case["expected_pass"])
            and predicted
        )
        if false_safety_accepts:
            safety_rejected += 1
            continue
        ordinary = [
            (case, predicted)
            for case, predicted in predictions
            if not case.get("safety_case", False)
        ]
        correct = sum(
            predicted == bool(case["expected_pass"])
            for case, predicted in ordinary
        )
        accuracy = correct / len(ordinary) if ordinary else 1.0
        candidate = {
            "thresholds": configured,
            "valid_case_accuracy": accuracy,
            "valid_case_correct": correct,
            "valid_case_count": len(ordinary),
            "safety_false_accepts": 0,
        }
        ranking = (accuracy, correct, tuple(-float(value) for value in combination))
        if best is None or ranking > best["_ranking"]:
            best = {**candidate, "_ranking": ranking}
    if best is None:
        raise ValueError("no threshold combination achieved zero safety false acceptance")
    best.pop("_ranking")
    best.update(
        {
            "grid_combinations_evaluated": evaluated,
            "grid_combinations_rejected_by_safety": safety_rejected,
        }
    )
    return best
