"""Visual grounding boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_VISUAL_CONFIDENCE = 0.80


@dataclass(frozen=True)
class VisualGrounding:
    visible: bool
    confidence: float
    bbox_xyxy: list[int] | None
    image_size: list[int] | None
    object: str


@dataclass(frozen=True)
class VisualVerification:
    approved: bool
    reason: str
    grounding: VisualGrounding | None


def verify_visual_grounding(payload: dict[str, Any]) -> VisualVerification:
    """Build grounding from VLM payload and validate visual evidence."""

    if payload.get("action") == "return_home":
        return VisualVerification(True, "no object visibility required", None)

    grounding = VisualGrounding(
        visible=bool(payload.get("visible", False)),
        confidence=float(payload.get("confidence", 0.0)),
        bbox_xyxy=payload.get("bbox_xyxy"),
        image_size=payload.get("image_size"),
        object=str(payload.get("object", "")),
    )
    return check_visual_grounding(grounding)


def check_visual_grounding(
    grounding: VisualGrounding,
    *,
    min_confidence: float = MIN_VISUAL_CONFIDENCE,
) -> VisualVerification:
    """Validate object existence evidence without robot pose assumptions."""

    if not grounding.visible:
        return VisualVerification(False, "object not visually verified", grounding)
    if grounding.confidence < min_confidence:
        return VisualVerification(False, "visual confidence below threshold", grounding)
    if not _bbox_sane(grounding.bbox_xyxy, grounding.image_size):
        return VisualVerification(False, "visual bounding box is invalid", grounding)
    return VisualVerification(True, "object visually verified", grounding)


def _bbox_sane(bbox: list[int] | None, image_size: list[int] | None) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(value, int) for value in bbox):
        return False
    if not isinstance(image_size, list) or len(image_size) != 2:
        return False
    if not all(isinstance(value, int) for value in image_size):
        return False

    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return False

    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    return x2 <= width and y2 <= height
