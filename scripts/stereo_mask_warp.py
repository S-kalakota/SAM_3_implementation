#!/usr/bin/env python3
"""Disparity-based mask transfer between rectified ZED stereo views."""

from __future__ import annotations

import cv2
import numpy as np


def mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask.astype(bool, copy=False))
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid_xy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool, copy=False))
    if ys.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def valid_disparity_values(disparity: np.ndarray) -> np.ndarray:
    values = disparity.astype(np.float32, copy=False).reshape(-1)
    valid = np.isfinite(values)
    valid &= np.abs(values) < 10000.0
    return values[valid]


def disparity_image_stats(disparity: np.ndarray) -> dict:
    values = valid_disparity_values(disparity)
    total_pixels = int(disparity.size)
    if values.size == 0:
        return {
            "total_pixels": total_pixels,
            "valid_pixels": 0,
            "nonzero_valid_pixels": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "total_pixels": total_pixels,
        "valid_pixels": int(values.size),
        "nonzero_valid_pixels": int(np.count_nonzero(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def disparity_stats(values: np.ndarray, *, shift_sign: int) -> dict | None:
    if values.size == 0:
        return None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_horizontal_shift_pixels": float(shift_sign * values.mean()),
        "median_horizontal_shift_pixels": float(shift_sign * np.median(values)),
    }


def opposite_view(view_name: str) -> str:
    if view_name == "LEFT":
        return "RIGHT"
    if view_name == "RIGHT":
        return "LEFT"
    raise ValueError(f"Unsupported stereo view: {view_name}")


def disparity_measure_name(view_name: str) -> str:
    if view_name == "LEFT":
        return "DISPARITY"
    if view_name == "RIGHT":
        return "DISPARITY_RIGHT"
    raise ValueError(f"Unsupported stereo view: {view_name}")


def default_shift_sign(view_name: str) -> int:
    if view_name == "LEFT":
        return -1
    if view_name == "RIGHT":
        return 1
    raise ValueError(f"Unsupported stereo view: {view_name}")


def resolve_shift_sign(view_name: str, configured_sign: int | None) -> int:
    if configured_sign is None:
        return default_shift_sign(view_name)
    if configured_sign not in {-1, 1}:
        raise ValueError("configured_sign must be -1, 1, or None")
    return configured_sign


def cleanup_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or not mask.any():
        return mask.astype(bool, copy=False)

    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def warp_mask_by_disparity(
    mask: np.ndarray,
    disparity: np.ndarray,
    *,
    shift_sign: int,
    cleanup_kernel: int = 3,
) -> tuple[np.ndarray, dict]:
    """Transfer one source-view mask into the opposite rectified stereo view."""

    if mask.shape != disparity.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match disparity shape {disparity.shape[:2]}"
        )
    if shift_sign not in {-1, 1}:
        raise ValueError("shift_sign must be -1 or 1")

    mask_bool = mask.astype(bool, copy=False)
    ys, xs = np.where(mask_bool)
    source_pixels = int(ys.size)
    source_bbox = mask_bbox_xyxy(mask_bool)
    source_centroid = mask_centroid_xy(mask_bool)
    target = np.zeros(mask_bool.shape, dtype=bool)
    if source_pixels == 0:
        return target, {
            "source_pixels": 0,
            "source_bbox_xyxy": None,
            "source_centroid_xy": None,
            "valid_disparity_pixels": 0,
            "disparity_pixels": None,
            "projected_pixels": 0,
            "target_pixels": 0,
            "target_bbox_xyxy": None,
            "target_centroid_xy": None,
            "centroid_shift_xy": None,
        }

    d = disparity[ys, xs].astype(np.float32, copy=False)
    valid = np.isfinite(d)
    valid &= np.abs(d) < 10000.0
    if not valid.any():
        return target, {
            "source_pixels": source_pixels,
            "source_bbox_xyxy": source_bbox,
            "source_centroid_xy": source_centroid,
            "valid_disparity_pixels": 0,
            "disparity_pixels": None,
            "projected_pixels": 0,
            "target_pixels": 0,
            "target_bbox_xyxy": None,
            "target_centroid_xy": None,
            "centroid_shift_xy": None,
        }

    valid_disparities = d[valid]
    xr = np.rint(xs[valid].astype(np.float32) + shift_sign * valid_disparities).astype(
        np.int32
    )
    yr = ys[valid]
    inside = (xr >= 0) & (xr < mask_bool.shape[1])
    target[yr[inside], xr[inside]] = True
    target = cleanup_mask(target, cleanup_kernel)
    target_bbox = mask_bbox_xyxy(target)
    target_centroid = mask_centroid_xy(target)
    if source_centroid is None or target_centroid is None:
        centroid_shift = None
    else:
        centroid_shift = [
            float(target_centroid[0] - source_centroid[0]),
            float(target_centroid[1] - source_centroid[1]),
        ]

    return target, {
        "source_pixels": source_pixels,
        "source_bbox_xyxy": source_bbox,
        "source_centroid_xy": source_centroid,
        "valid_disparity_pixels": int(valid.sum()),
        "disparity_pixels": disparity_stats(valid_disparities, shift_sign=shift_sign),
        "projected_pixels": int(inside.sum()),
        "target_pixels": int(target.sum()),
        "target_bbox_xyxy": target_bbox,
        "target_centroid_xy": target_centroid,
        "centroid_shift_xy": centroid_shift,
    }


def warp_kept_masks(
    kept: list[tuple[np.ndarray, float]],
    disparity: np.ndarray,
    *,
    shift_sign: int,
    cleanup_kernel: int = 3,
) -> tuple[list[tuple[np.ndarray, float]], list[dict]]:
    warped = []
    stats = []
    for index, (mask, score) in enumerate(kept):
        warped_mask, mask_stats = warp_mask_by_disparity(
            mask,
            disparity,
            shift_sign=shift_sign,
            cleanup_kernel=cleanup_kernel,
        )
        warped.append((warped_mask, score))
        stats.append({"index": index, **mask_stats})
    return warped, stats
