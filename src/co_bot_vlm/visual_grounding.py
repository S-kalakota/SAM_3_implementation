"""Visual grounding boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    """Minimal dependency-free visual check for the Agent 1 skeleton."""

    grounding = VisualGrounding(
        visible=bool(payload.get("visible", False)),
        confidence=float(payload.get("confidence", 0.0)),
        bbox_xyxy=payload.get("bbox_xyxy"),
        image_size=payload.get("image_size"),
        object=str(payload.get("object", "")),
    )

    if not grounding.visible:
        return VisualVerification(False, "object not visually verified", grounding)
    if grounding.confidence < 0.8:
        return VisualVerification(False, "visual confidence below threshold", grounding)
    if not _bbox_sane(grounding.bbox_xyxy, grounding.image_size):
        return VisualVerification(False, "visual bounding box is invalid", grounding)
    return VisualVerification(True, "object visually verified", grounding)


def _bbox_sane(bbox: list[int] | None, image_size: list[int] | None) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if not all(isinstance(value, int) for value in bbox):
        return False

    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return False

    if isinstance(image_size, list) and len(image_size) == 2:
        width, height = image_size
        if isinstance(width, int) and isinstance(height, int):
            return x2 <= width and y2 <= height

    return True
