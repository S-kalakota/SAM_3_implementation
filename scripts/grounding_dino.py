#!/usr/bin/env python3
"""Cached Grounding DINO proposal generation and geometry helpers.

The runtime adapter is deliberately local-cache-only.  Model downloads belong in
``cache_grounding_dino.py`` and never in a camera request.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = os.environ.get(
    "GROUNDING_DINO_MODEL_ID",
    "IDEA-Research/grounding-dino-base",
)
DEFAULT_DEVICE = os.environ.get("GROUNDING_DINO_DEVICE", "cuda:0")
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.20
DEFAULT_NMS_IOU = 0.50
DEFAULT_MAX_PROPOSALS = 3
DEFAULT_BOX_PADDING = 0.05
MAX_PHRASES = 5


class GroundingDinoError(RuntimeError):
    """Base error for the bounded proposal stage."""


class GroundingDinoCacheError(GroundingDinoError):
    """Raised when the configured offline snapshot cannot be loaded."""


class GroundingDinoOutputError(GroundingDinoError):
    """Raised when Transformers returns an inconsistent detection payload."""


_HEAD_CATEGORIES = (
    "flat rectangular item",
    "storage bin",
    "water bottle",
    "tape roll",
    "robot arm",
    "cardboard box",
    "shipping box",
    "package",
    "carton",
    "box",
    "bottle",
    "container",
    "bin",
    "cup",
    "mug",
    "can",
    "filter",
    "tool",
    "object",
    "item",
)

_CONTROLLED_SYNONYMS = {
    "box": ("box", "package", "carton", "flat rectangular item"),
    "cardboard box": ("box", "package", "carton", "flat rectangular item"),
    "shipping box": ("box", "package", "carton", "flat rectangular item"),
    "package": ("box", "package", "carton", "flat rectangular item"),
    "carton": ("box", "package", "carton", "flat rectangular item"),
    "flat rectangular item": (
        "box",
        "package",
        "carton",
        "flat rectangular item",
    ),
    "cup": ("cup", "mug"),
    "mug": ("mug", "cup"),
}


def _normalized_phrase(value: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else " " for character in value
    )
    return " ".join(cleaned.split())


def head_category(target_phrase: str) -> str:
    """Return a deterministic category head for a parsed noun phrase."""

    normalized = _normalized_phrase(target_phrase)
    if not normalized:
        raise ValueError("target_phrase must contain at least one letter or number")
    for category in _HEAD_CATEGORIES:
        if normalized == category or normalized.endswith(f" {category}"):
            return category
    return normalized.rsplit(" ", 1)[-1]


def build_phrase_family(target_phrase: str, max_phrases: int = MAX_PHRASES) -> list[str]:
    """Build the exact phrase, category head, and bounded controlled synonyms."""

    if not 1 <= max_phrases <= MAX_PHRASES:
        raise ValueError(f"max_phrases must be in [1, {MAX_PHRASES}]")
    exact = _normalized_phrase(target_phrase)
    if not exact:
        raise ValueError("target_phrase must not be empty")
    category = head_category(exact)
    candidates = [exact, category, *_CONTROLLED_SYNONYMS.get(category, ())]
    phrases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalized_phrase(candidate)
        if normalized and normalized not in seen:
            phrases.append(normalized)
            seen.add(normalized)
        if len(phrases) == max_phrases:
            break
    return phrases


def match_submitted_phrase(text_label: str, phrases: Sequence[str]) -> str:
    """Map DINO's decoded token phrase back to one submitted phrase."""

    if not phrases:
        raise ValueError("phrases must not be empty")
    normalized_label = _normalized_phrase(text_label)
    normalized_phrases = [(_normalized_phrase(phrase), phrase) for phrase in phrases]
    for normalized, original in normalized_phrases:
        if normalized_label == normalized:
            return original
    related = [
        (len(normalized), original)
        for normalized, original in normalized_phrases
        if normalized
        and (
            normalized in normalized_label
            or (normalized_label and normalized_label in normalized)
        )
    ]
    if related:
        return max(related, key=lambda item: item[0])[1]
    return text_label.strip() or phrases[0]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def decode_dino_result(
    result: Mapping[str, Any],
    phrases: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate and serialize one Transformers DINO post-process result."""

    if not isinstance(result, Mapping):
        raise GroundingDinoOutputError("DINO result must be a mapping")
    missing = {"scores", "boxes"} - set(result)
    if missing:
        raise GroundingDinoOutputError(
            f"DINO result is missing required keys: {sorted(missing)}"
        )
    labels_value = result.get("text_labels", result.get("labels"))
    if labels_value is None:
        raise GroundingDinoOutputError("DINO result is missing text labels")

    scores = _as_numpy(result["scores"]).astype(np.float64, copy=False).reshape(-1)
    boxes = _as_numpy(result["boxes"]).astype(np.float64, copy=False)
    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise GroundingDinoOutputError(
            f"DINO boxes must have shape Nx4, got {boxes.shape}"
        )
    labels = list(labels_value)
    if len(scores) != len(boxes) or len(labels) != len(boxes):
        raise GroundingDinoOutputError(
            "DINO scores, boxes, and text labels have different lengths: "
            f"{len(scores)}, {len(boxes)}, {len(labels)}"
        )

    detections = []
    for index, (score, box, text_label) in enumerate(zip(scores, boxes, labels)):
        label = str(text_label).strip()
        detections.append(
            {
                "dino_index": int(index),
                "phrase": match_submitted_phrase(label, phrases),
                "text_label": label,
                "dino_score": float(score),
                "box_xyxy_crop_pixels": [float(value) for value in box.tolist()],
            }
        )
    return detections


def box_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def pixel_xyxy_to_normalized_sam_xywh(
    box_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert a bounded crop-local pixel box to normalized SAM xywh."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain four values")
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be finite")
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError("box coordinates must already be clamped to the image")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return [
        float(x1 / image_width),
        float(y1 / image_height),
        float((x2 - x1) / image_width),
        float((y2 - y1) / image_height),
    ]


def mask_bbox_xywh_normalized(mask: np.ndarray) -> list[float]:
    """Return the mask-derived normalized xywh box used by robot consumers."""

    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.ndim != 2 or not mask_np.any():
        raise ValueError("mask must be a non-empty HxW array")
    height, width = mask_np.shape
    ys, xs = np.where(mask_np)
    return pixel_xyxy_to_normalized_sam_xywh(
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        width,
        height,
    )


def mask_center_xy(mask: np.ndarray) -> tuple[float, float]:
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.ndim != 2 or not mask_np.any():
        raise ValueError("mask must be a non-empty HxW array")
    ys, xs = np.where(mask_np)
    return float(xs.mean()), float(ys.mean())


def box_contains_point(box_xyxy: Sequence[float], point_xy: Sequence[float]) -> bool:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x, y = [float(value) for value in point_xy]
    return x1 <= x < x2 and y1 <= y < y2


def crop_point_to_full(
    point_xy: Sequence[float],
    crop_xywh: Sequence[int] | None,
) -> list[float]:
    """Translate one crop-local point into the full ZED image."""

    x, y = [float(value) for value in point_xy]
    if crop_xywh is None:
        return [x, y]
    if len(crop_xywh) != 4:
        raise ValueError("crop_xywh must contain x, y, width, height")
    crop_x, crop_y, width, height = [int(value) for value in crop_xywh]
    if crop_x < 0 or crop_y < 0 or width <= 0 or height <= 0:
        raise ValueError("crop_xywh is invalid")
    return [x + crop_x, y + crop_y]


def prepare_proposals(
    detections: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    padding_fraction: float = DEFAULT_BOX_PADDING,
    dino_inference_s: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Threshold, clamp, cross-phrase NMS, pad, and cap DINO boxes."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= box_threshold <= 1.0:
        raise ValueError("box_threshold must be in [0, 1]")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("nms_iou must be in [0, 1]")
    if not 1 <= max_proposals <= DEFAULT_MAX_PROPOSALS:
        raise ValueError(
            f"max_proposals must be in [1, {DEFAULT_MAX_PROPOSALS}]"
        )
    if not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("padding_fraction must be in [0, 1]")

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for position, detection_value in enumerate(detections):
        detection = dict(detection_value)
        detection.setdefault("dino_index", position)
        try:
            score = float(detection["dino_score"])
            raw_box = [float(value) for value in detection["box_xyxy_crop_pixels"]]
            if len(raw_box) != 4:
                raise ValueError("box must contain four values")
            if not math.isfinite(score) or not all(
                math.isfinite(value) for value in raw_box
            ):
                raise ValueError("score and box coordinates must be finite")
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {**detection, "reject_reason": f"malformed_detection:{exc}"}
            )
            continue
        if score < box_threshold:
            rejected.append({**detection, "reject_reason": "below_box_threshold"})
            continue

        x1, y1, x2, y2 = raw_box
        clamped = [
            min(max(x1, 0.0), float(image_width)),
            min(max(y1, 0.0), float(image_height)),
            min(max(x2, 0.0), float(image_width)),
            min(max(y2, 0.0), float(image_height)),
        ]
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            rejected.append({**detection, "reject_reason": "empty_after_clamp"})
            continue
        valid.append(
            {
                **detection,
                "dino_score": score,
                "raw_box_xyxy_crop_pixels": raw_box,
                "original_box_xyxy_crop_pixels": clamped,
            }
        )

    valid.sort(key=lambda item: (-item["dino_score"], int(item["dino_index"])))
    deduplicated: list[dict[str, Any]] = []
    for detection in valid:
        suppressor = next(
            (
                kept
                for kept in deduplicated
                if box_iou_xyxy(
                    detection["original_box_xyxy_crop_pixels"],
                    kept["original_box_xyxy_crop_pixels"],
                )
                > nms_iou
            ),
            None,
        )
        if suppressor is not None:
            rejected.append(
                {
                    **detection,
                    "reject_reason": "cross_phrase_nms",
                    "suppressed_by_dino_index": int(suppressor["dino_index"]),
                }
            )
            continue
        deduplicated.append(detection)

    selected: list[dict[str, Any]] = []
    for detection in deduplicated:
        if len(selected) >= max_proposals:
            rejected.append({**detection, "reject_reason": "proposal_limit"})
            continue
        x1, y1, x2, y2 = detection["original_box_xyxy_crop_pixels"]
        pad_x = (x2 - x1) * padding_fraction
        pad_y = (y2 - y1) * padding_fraction
        padded = [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(image_width), x2 + pad_x),
            min(float(image_height), y2 + pad_y),
        ]
        proposal = {
            **detection,
            "proposal_id": len(selected) + 1,
            "padded_box_xyxy_crop_pixels": padded,
            "sam_box_xywh_normalized": pixel_xyxy_to_normalized_sam_xywh(
                padded,
                image_width,
                image_height,
            ),
        }
        if dino_inference_s is not None:
            proposal["dino_inference_s"] = float(dino_inference_s)
        selected.append(proposal)
    return selected, rejected


class GroundingDinoAdapter:
    """Resident CUDA/BF16 adapter for Transformers Grounding DINO."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = DEFAULT_DEVICE,
        processor_class: Any | None = None,
        model_class: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if processor_class is None or model_class is None:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            processor_class = processor_class or AutoProcessor
            model_class = model_class or AutoModelForZeroShotObjectDetection
        if torch_module is None:
            import torch as torch_module

        if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
            raise GroundingDinoError(
                "Grounding DINO requires a CUDA device for the configured BF16 runtime"
            )

        self.model_id = model_id
        self.device = str(device)
        self.local_files_only = True
        self.torch = torch_module
        self.dtype = torch_module.bfloat16
        try:
            self.processor = processor_class.from_pretrained(
                model_id,
                local_files_only=True,
            )
            self.model = model_class.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                local_files_only=True,
                use_safetensors=True,
            )
        except Exception as exc:
            raise GroundingDinoCacheError(
                f"Cached Grounding DINO snapshot {model_id!r} is missing or invalid; "
                "run scripts/cache_grounding_dino.py before starting offline: "
                f"{exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    def _synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def runtime_metadata(self) -> dict[str, Any]:
        hf_device_map = getattr(self.model, "hf_device_map", {})
        devices = sorted({str(value) for value in hf_device_map.values()})
        if not devices and hasattr(self.model, "device"):
            devices = [str(self.model.device)]
        first_parameter = next(self.model.parameters(), None)
        dtype = str(self.dtype if first_parameter is None else first_parameter.dtype)
        return {
            "loaded": True,
            "model_id": self.model_id,
            "model_class": type(self.model).__name__,
            "dtype": dtype,
            "devices": devices or [self.device],
            "configured_device": self.device,
            "local_files_only": True,
            "evaluation_mode": not bool(getattr(self.model, "training", False)),
        }

    def detect(
        self,
        rgb_np: np.ndarray,
        phrases: Sequence[str],
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> dict[str, Any]:
        """Run one multi-phrase pass and return JSON-safe detections."""

        image = np.asarray(rgb_np)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image shaped HxWx3, got {image.shape}")
        if not phrases or len(phrases) > MAX_PHRASES:
            raise ValueError(f"phrases must contain between 1 and {MAX_PHRASES} items")
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB")

        started = time.monotonic()
        inputs = self.processor(
            images=pil_image,
            text=list(phrases),
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        self._synchronize()
        inference_started = time.monotonic()
        with self.torch.inference_mode():
            with self.torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = self.model(**inputs)
        self._synchronize()
        inference_s = time.monotonic() - inference_started
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(height, width)],
        )
        if not isinstance(processed, Sequence) or len(processed) != 1:
            raise GroundingDinoOutputError(
                "DINO post-processing must return exactly one image result"
            )
        detections = decode_dino_result(processed[0], phrases)
        return {
            "phrases": list(phrases),
            "detections": detections,
            "inference_s": float(inference_s),
            "total_s": float(time.monotonic() - started),
        }

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        self.torch.cuda.empty_cache()
