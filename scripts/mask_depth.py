#!/usr/bin/env python3
"""Object depth extraction from a segmentation mask and ZED depth measures.

The ZED depth map (MEASURE.DEPTH) and point cloud (MEASURE.XYZ) are registered
pixel-for-pixel to the LEFT image (use the *_RIGHT measures for the right
view), so a mask from that view indexes them directly — no reprojection.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from stereo_mask_warp import mask_centroid_xy


DEFAULT_REFINEMENT_MIN_VALID_PIXELS = 64
DEFAULT_REFINEMENT_MIN_VALID_FRACTION = 0.50
DEFAULT_REFINEMENT_CENTER_PATCH_FRACTION = 0.25
DEFAULT_REFINEMENT_MIN_ANCHOR_PIXELS = 16
DEFAULT_REFINEMENT_MIN_MASK_SEED_PIXELS = 16
DEFAULT_REFINEMENT_MIN_JUMP_M = 0.04
DEFAULT_REFINEMENT_RELATIVE_JUMP = 0.03
DEFAULT_REFINEMENT_MAD_SCALE = 4.0
DEFAULT_REFINEMENT_MAX_LOCAL_JUMP_M = 0.15
DEFAULT_REFINEMENT_GLOBAL_JUMP_MULTIPLIER = 3.0
DEFAULT_REFINEMENT_MAX_GLOBAL_DRIFT_M = 0.30
DEFAULT_REFINEMENT_MIN_RETAINED_FRACTION = 0.20
DEFAULT_REFINEMENT_MIN_REMOVED_PIXELS = 4

_EIGHT_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def valid_depth_values(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape != depth.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match depth shape {depth.shape[:2]}"
        )
    values = depth[mask.astype(bool, copy=False)].astype(np.float32, copy=False)
    valid = np.isfinite(values)
    valid &= values > 0.0
    return values[valid]


def mask_depth_stats(mask: np.ndarray, depth: np.ndarray) -> dict:
    """Robust per-object depth from the depth map under the mask.

    Use the median, not the mean: mask edges bleed onto the background and
    occlusion holes return NaN, both of which wreck a mean.
    """
    mask_pixels = int(np.count_nonzero(mask))
    values = valid_depth_values(mask, depth)
    if values.size == 0:
        return {
            "mask_pixels": mask_pixels,
            "valid_depth_pixels": 0,
            "valid_fraction": 0.0,
            "median": None,
            "mean": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": int(values.size),
        "valid_fraction": float(values.size / max(mask_pixels, 1)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _center_patch_bounds(
    box_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    fraction: float,
) -> tuple[int, int, int, int, float, float]:
    if len(box_xyxy) != 4:
        raise ValueError("anchor_box_xyxy must contain four values")
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("anchor box coordinates must be finite")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("anchor box must have positive width and height")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("center_patch_fraction must be in (0, 1]")

    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    patch_width = max(5.0, (x1 - x0) * fraction)
    patch_height = max(5.0, (y1 - y0) * fraction)
    patch_x0 = max(0, min(image_width, int(math.floor(center_x - patch_width / 2.0))))
    patch_y0 = max(0, min(image_height, int(math.floor(center_y - patch_height / 2.0))))
    patch_x1 = max(0, min(image_width, int(math.ceil(center_x + patch_width / 2.0))))
    patch_y1 = max(0, min(image_height, int(math.ceil(center_y + patch_height / 2.0))))
    if patch_x1 <= patch_x0 or patch_y1 <= patch_y0:
        raise ValueError("anchor box center does not overlap the depth image")
    return patch_x0, patch_y0, patch_x1, patch_y1, center_x, center_y


def _anchor_depth_cluster(
    depth: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    center_xy: tuple[float, float],
    min_anchor_pixels: int,
    min_jump_m: float,
    relative_jump: float,
) -> tuple[np.ndarray, float, int]:
    """Select the depth cluster nearest the exact DINO-box center."""

    x0, y0, x1, y1 = bounds
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0)
    ys, xs = np.where(valid)
    if xs.size < min_anchor_pixels:
        return np.zeros((0,), dtype=np.float32), 0.0, int(xs.size)

    values = patch[ys, xs].astype(np.float32, copy=False)
    center_x, center_y = center_xy
    distances = np.square(xs.astype(np.float64) + x0 - center_x)
    distances += np.square(ys.astype(np.float64) + y0 - center_y)
    nearest_count = min(values.size, max(min_anchor_pixels, 9))
    nearest_indices = np.argpartition(distances, nearest_count - 1)[:nearest_count]
    center_hint = float(np.median(values[nearest_indices]))
    initial_band = max(float(min_jump_m), float(relative_jump) * center_hint)
    cluster = values[np.abs(values - center_hint) <= initial_band]
    if cluster.size >= min_anchor_pixels:
        return cluster, center_hint, int(values.size)

    # If the nearest samples are sparse/noisy, use the densest narrow interval
    # in the center patch, preferring the interval closest to the center hint.
    ordered = np.sort(values)
    maximum_span = 2.0 * initial_band
    best_start = 0
    best_end = 0
    best_count = 0
    best_distance = math.inf
    start = 0
    for end in range(ordered.size):
        while ordered[end] - ordered[start] > maximum_span:
            start += 1
        count = end - start + 1
        interval_median = float(np.median(ordered[start : end + 1]))
        distance = abs(interval_median - center_hint)
        if count > best_count or (count == best_count and distance < best_distance):
            best_start = start
            best_end = end + 1
            best_count = count
            best_distance = distance
    return ordered[best_start:best_end], center_hint, int(values.size)


def _grow_depth_connected(
    eligible: np.ndarray,
    seed: np.ndarray,
    depth: np.ndarray,
    maximum_neighbor_jump: float,
) -> np.ndarray:
    reached = np.asarray(seed, dtype=bool).copy()
    height, width = reached.shape
    queue = deque((int(y), int(x)) for y, x in np.argwhere(reached))
    while queue:
        y, x = queue.popleft()
        current_depth = float(depth[y, x])
        for offset_y, offset_x in _EIGHT_NEIGHBORS:
            neighbor_y = y + offset_y
            neighbor_x = x + offset_x
            if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                continue
            if reached[neighbor_y, neighbor_x] or not eligible[neighbor_y, neighbor_x]:
                continue
            if abs(float(depth[neighbor_y, neighbor_x]) - current_depth) > maximum_neighbor_jump:
                continue
            reached[neighbor_y, neighbor_x] = True
            queue.append((neighbor_y, neighbor_x))
    return reached


def _grow_binary_region(eligible: np.ndarray, seed: np.ndarray) -> np.ndarray:
    reached = np.asarray(seed, dtype=bool).copy()
    height, width = reached.shape
    queue = deque((int(y), int(x)) for y, x in np.argwhere(reached))
    while queue:
        y, x = queue.popleft()
        for offset_y, offset_x in _EIGHT_NEIGHBORS:
            neighbor_y = y + offset_y
            neighbor_x = x + offset_x
            if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                continue
            if reached[neighbor_y, neighbor_x] or not eligible[neighbor_y, neighbor_x]:
                continue
            reached[neighbor_y, neighbor_x] = True
            queue.append((neighbor_y, neighbor_x))
    return reached


def refine_mask_at_depth_discontinuities(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    anchor_box_xyxy: Sequence[float],
    min_valid_pixels: int = DEFAULT_REFINEMENT_MIN_VALID_PIXELS,
    min_valid_fraction: float = DEFAULT_REFINEMENT_MIN_VALID_FRACTION,
    center_patch_fraction: float = DEFAULT_REFINEMENT_CENTER_PATCH_FRACTION,
    min_anchor_pixels: int = DEFAULT_REFINEMENT_MIN_ANCHOR_PIXELS,
    min_mask_seed_pixels: int = DEFAULT_REFINEMENT_MIN_MASK_SEED_PIXELS,
    min_jump_m: float = DEFAULT_REFINEMENT_MIN_JUMP_M,
    relative_jump: float = DEFAULT_REFINEMENT_RELATIVE_JUMP,
    mad_scale: float = DEFAULT_REFINEMENT_MAD_SCALE,
    max_local_jump_m: float = DEFAULT_REFINEMENT_MAX_LOCAL_JUMP_M,
    global_jump_multiplier: float = DEFAULT_REFINEMENT_GLOBAL_JUMP_MULTIPLIER,
    max_global_drift_m: float = DEFAULT_REFINEMENT_MAX_GLOBAL_DRIFT_M,
    min_retained_fraction: float = DEFAULT_REFINEMENT_MIN_RETAINED_FRACTION,
    min_removed_pixels: int = DEFAULT_REFINEMENT_MIN_REMOVED_PIXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cut mask pixels across a ZED depth discontinuity from the DINO center.

    The center of the original DINO box supplies an object-depth reference even
    when SAM returns a hollow mask. Valid mask pixels grow from that reference
    only across locally continuous depth. Unknown-depth pixels are preserved
    when they remain spatially attached to accepted pixels. The returned mask
    is always a subset of the input mask; guardrails return the original mask
    when the depth evidence is weak or the proposed cut is too destructive.
    """

    mask_np = np.asarray(mask, dtype=bool)
    depth_np = np.asarray(depth)
    if mask_np.ndim != 2:
        raise ValueError("mask must be an HxW array")
    if depth_np.ndim != 2 or depth_np.shape != mask_np.shape:
        raise ValueError(
            f"Depth shape {depth_np.shape} does not match mask shape {mask_np.shape}"
        )
    if min_valid_pixels < 1 or min_anchor_pixels < 1 or min_mask_seed_pixels < 1:
        raise ValueError("depth pixel thresholds must be positive")
    if min_removed_pixels < 1:
        raise ValueError("min_removed_pixels must be positive")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    if not 0.0 < min_retained_fraction <= 1.0:
        raise ValueError("min_retained_fraction must be in (0, 1]")
    if min_jump_m <= 0.0 or relative_jump < 0.0 or mad_scale < 0.0:
        raise ValueError("depth jump thresholds must be non-negative and nonzero")
    if max_local_jump_m < min_jump_m:
        raise ValueError("max_local_jump_m must be at least min_jump_m")
    if global_jump_multiplier < 1.0 or max_global_drift_m < min_jump_m:
        raise ValueError("global depth thresholds are inconsistent")

    original_area = int(mask_np.sum())
    before_stats = mask_depth_stats(mask_np, depth_np)
    valid_depth = np.isfinite(depth_np) & (depth_np > 0.0)
    valid_mask = mask_np & valid_depth
    valid_pixels = int(valid_mask.sum())
    valid_fraction = float(valid_pixels / max(original_area, 1))
    report: dict[str, Any] = {
        "algorithm": "dino_center_depth_connectivity_v1",
        "status": "not_run",
        "applied": False,
        "adds_pixels": False,
        "original_area_pixels": original_area,
        "refined_area_pixels": original_area,
        "removed_pixels": 0,
        "valid_mask_depth_pixels": valid_pixels,
        "valid_mask_depth_fraction": valid_fraction,
        "before_depth_stats_m": before_stats,
        "after_depth_stats_m": before_stats,
        "thresholds": {
            "min_valid_pixels": int(min_valid_pixels),
            "min_valid_fraction": float(min_valid_fraction),
            "center_patch_fraction": float(center_patch_fraction),
            "min_anchor_pixels": int(min_anchor_pixels),
            "min_mask_seed_pixels": int(min_mask_seed_pixels),
            "min_jump_m": float(min_jump_m),
            "relative_jump": float(relative_jump),
            "mad_scale": float(mad_scale),
            "max_local_jump_m": float(max_local_jump_m),
            "global_jump_multiplier": float(global_jump_multiplier),
            "max_global_drift_m": float(max_global_drift_m),
            "min_retained_fraction": float(min_retained_fraction),
            "min_removed_pixels": int(min_removed_pixels),
        },
    }
    if original_area == 0:
        report.update(
            status="skipped_empty_mask",
            reason="the input mask has no pixels",
        )
        return mask_np.copy(), report
    if valid_pixels < min_valid_pixels or valid_fraction < min_valid_fraction:
        report.update(
            status="skipped_insufficient_valid_depth",
            reason="too few mask pixels have valid registered ZED depth",
        )
        return mask_np.copy(), report

    height, width = mask_np.shape
    patch_x0, patch_y0, patch_x1, patch_y1, center_x, center_y = _center_patch_bounds(
        anchor_box_xyxy,
        image_width=width,
        image_height=height,
        fraction=center_patch_fraction,
    )
    anchor_values, center_hint, anchor_patch_valid = _anchor_depth_cluster(
        depth_np,
        (patch_x0, patch_y0, patch_x1, patch_y1),
        center_xy=(center_x, center_y),
        min_anchor_pixels=min_anchor_pixels,
        min_jump_m=min_jump_m,
        relative_jump=relative_jump,
    )
    report.update(
        anchor_box_xyxy_pixels=[float(value) for value in anchor_box_xyxy],
        anchor_center_xy_pixels=[float(center_x), float(center_y)],
        anchor_patch_xyxy_pixels=[patch_x0, patch_y0, patch_x1, patch_y1],
        anchor_patch_valid_pixels=int(anchor_patch_valid),
        anchor_cluster_pixels=int(anchor_values.size),
        anchor_center_hint_depth_m=None if center_hint <= 0.0 else float(center_hint),
    )
    if anchor_values.size < min_anchor_pixels:
        report.update(
            status="skipped_insufficient_anchor_depth",
            reason="the center of the DINO box lacks a stable depth cluster",
        )
        return mask_np.copy(), report

    reference_depth = float(np.median(anchor_values))
    anchor_mad = float(np.median(np.abs(anchor_values - reference_depth)))
    robust_sigma = 1.4826 * anchor_mad
    local_jump = min(
        max(
            float(min_jump_m),
            float(relative_jump) * reference_depth,
            float(mad_scale) * robust_sigma,
        ),
        float(max_local_jump_m),
    )
    global_drift = min(
        local_jump * float(global_jump_multiplier),
        float(max_global_drift_m),
    )
    report.update(
        anchor_reference_depth_m=reference_depth,
        anchor_mad_m=anchor_mad,
        anchor_robust_sigma_m=float(robust_sigma),
        local_neighbor_jump_m=float(local_jump),
        global_reference_drift_m=float(global_drift),
    )

    distance_from_reference = np.abs(depth_np - reference_depth)
    seed = valid_mask & (distance_from_reference <= local_jump)
    seed_pixels = int(seed.sum())
    report["mask_seed_pixels"] = seed_pixels
    if seed_pixels < min_mask_seed_pixels:
        report.update(
            status="skipped_insufficient_mask_seed",
            reason="too little of the SAM mask agrees with the DINO-center depth",
        )
        return mask_np.copy(), report

    eligible = valid_mask & (distance_from_reference <= global_drift)
    connected_valid = _grow_depth_connected(
        eligible,
        seed,
        depth_np,
        local_jump,
    )
    unknown_mask = mask_np & ~valid_depth
    retainable = connected_valid | unknown_mask
    proposed = _grow_binary_region(retainable, connected_valid)
    proposed &= mask_np

    connected_valid_pixels = int(connected_valid.sum())
    retained_valid_fraction = float(connected_valid_pixels / max(valid_pixels, 1))
    proposed_area = int(proposed.sum())
    retained_mask_fraction = float(proposed_area / original_area)
    removed_valid = valid_mask & ~connected_valid
    removed_nearer = removed_valid & (depth_np < reference_depth - local_jump)
    removed_farther = removed_valid & (depth_np > reference_depth + local_jump)
    removed_edge_separated = removed_valid & ~removed_nearer & ~removed_farther
    proposed_removed = mask_np & ~proposed
    proposed_removed_pixels = int(proposed_removed.sum())
    proposed_after_stats = mask_depth_stats(proposed, depth_np)
    report.update(
        connected_valid_pixels=connected_valid_pixels,
        retained_valid_fraction=retained_valid_fraction,
        proposed_area_pixels=proposed_area,
        proposed_retained_mask_fraction=retained_mask_fraction,
        proposed_removed_pixels=proposed_removed_pixels,
        proposed_removed_valid_pixels=int(removed_valid.sum()),
        proposed_removed_nearer_pixels=int(removed_nearer.sum()),
        proposed_removed_farther_pixels=int(removed_farther.sum()),
        proposed_removed_edge_separated_pixels=int(removed_edge_separated.sum()),
        proposed_removed_unknown_depth_pixels=int((unknown_mask & ~proposed).sum()),
        proposed_after_depth_stats_m=proposed_after_stats,
    )

    if int(removed_valid.sum()) < min_removed_pixels:
        report.update(
            status="no_depth_discontinuity",
            reason="no material depth-separated mask region was found",
        )
        return mask_np.copy(), report
    if (
        retained_valid_fraction < min_retained_fraction
        or retained_mask_fraction < min_retained_fraction
    ):
        report.update(
            status="skipped_retention_guardrail",
            reason="the proposed depth cut would remove too much of the mask",
        )
        return mask_np.copy(), report

    report.update(
        status="applied",
        reason="removed pixels separated from the DINO-center object depth",
        applied=True,
        refined_area_pixels=proposed_area,
        removed_pixels=proposed_removed_pixels,
        removed_valid_pixels=int(removed_valid.sum()),
        removed_nearer_pixels=int(removed_nearer.sum()),
        removed_farther_pixels=int(removed_farther.sum()),
        removed_edge_separated_pixels=int(removed_edge_separated.sum()),
        removed_unknown_depth_pixels=int((unknown_mask & ~proposed).sum()),
        after_depth_stats_m=proposed_after_stats,
    )
    return proposed, report


def mask_xyz_centroid(mask: np.ndarray, xyz: np.ndarray) -> list[float] | None:
    """Per-axis median 3D point (camera frame) of the point cloud under the mask.

    This is the value to hand to a grasp/reach planner: MEASURE.DEPTH alone is
    only the perpendicular Z distance, not where the object is in 3D.
    """
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise ValueError(f"Unexpected XYZ shape: {xyz.shape}")
    if mask.shape != xyz.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match XYZ shape {xyz.shape[:2]}"
        )
    points = xyz[mask.astype(bool, copy=False)][:, :3].astype(np.float32, copy=False)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        return None
    return [
        float(np.median(points[:, 0])),
        float(np.median(points[:, 1])),
        float(np.median(points[:, 2])),
    ]


def triangulate_mask_depth(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    *,
    fx_pixels: float,
    baseline: float,
) -> dict | None:
    """Depth from the two masks alone: Z = fx * B / (x_left - x_right).

    Independent of the SDK depth map, so it doubles as a consistency check on
    the mask pair. `baseline` sets the output unit (meters in → meters out).
    A large vertical_offset_pixels means the two masks disagree (rectified
    views should differ only horizontally) — distrust the result then.
    """
    left_centroid = mask_centroid_xy(left_mask)
    right_centroid = mask_centroid_xy(right_mask)
    if left_centroid is None or right_centroid is None:
        return None
    disparity = left_centroid[0] - right_centroid[0]
    result = {
        "disparity_pixels": float(disparity),
        "vertical_offset_pixels": float(left_centroid[1] - right_centroid[1]),
        "depth": None,
    }
    if disparity > 0:
        result["depth"] = float(fx_pixels * baseline / disparity)
    return result
