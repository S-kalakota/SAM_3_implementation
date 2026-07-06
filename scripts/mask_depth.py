#!/usr/bin/env python3
"""Object depth extraction from a segmentation mask and ZED depth measures.

The ZED depth map (MEASURE.DEPTH) and point cloud (MEASURE.XYZ) are registered
pixel-for-pixel to the LEFT image (use the *_RIGHT measures for the right
view), so a mask from that view indexes them directly — no reprojection.
"""

from __future__ import annotations

import numpy as np

from stereo_mask_warp import mask_centroid_xy


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
