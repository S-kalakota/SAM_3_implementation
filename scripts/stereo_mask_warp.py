#!/usr/bin/env python3
"""Disparity-based mask transfer between rectified ZED stereo views."""

from __future__ import annotations

import cv2
import numpy as np


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
    target = np.zeros(mask_bool.shape, dtype=bool)
    if source_pixels == 0:
        return target, {
            "source_pixels": 0,
            "valid_disparity_pixels": 0,
            "projected_pixels": 0,
            "target_pixels": 0,
        }

    d = disparity[ys, xs].astype(np.float32, copy=False)
    valid = np.isfinite(d)
    valid &= np.abs(d) < 10000.0
    if not valid.any():
        return target, {
            "source_pixels": source_pixels,
            "valid_disparity_pixels": 0,
            "projected_pixels": 0,
            "target_pixels": 0,
        }

    xr = np.rint(xs[valid].astype(np.float32) + shift_sign * d[valid]).astype(np.int32)
    yr = ys[valid]
    inside = (xr >= 0) & (xr < mask_bool.shape[1])
    target[yr[inside], xr[inside]] = True
    target = cleanup_mask(target, cleanup_kernel)

    return target, {
        "source_pixels": source_pixels,
        "valid_disparity_pixels": int(valid.sum()),
        "projected_pixels": int(inside.sum()),
        "target_pixels": int(target.sum()),
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
