"""VLM adapter boundary and skeleton backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import BackendUnavailableError, ValidationError
from .image_source import ImageFrame
from .transcript import Transcript


MOTION_CONTROL_FIELDS = {
    "joint_angles",
    "pose",
    "object_pose",
    "velocity",
    "gripper",
    "robot_command",
    "trajectory",
}


@dataclass(frozen=True)
class VLMResponse:
    backend: str
    model: str
    output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class VLMBackend(Protocol):
    name: str
    model: str

    def analyze(self, transcript: Transcript, image: ImageFrame) -> VLMResponse:
        """Analyze one transcript and one RGB image."""


def create_vlm_backend(name: str) -> VLMBackend:
    if name == "mock":
        return MockVLMBackend()
    if name == "qwen":
        return QwenVLMBackend()
    raise ValidationError(
        code="unknown_vlm_backend",
        message=f"Unknown VLM backend: {name}",
        details={"vlm_backend": name},
    )


class MockVLMBackend:
    name = "mock"
    model = "deterministic-skeleton"

    def analyze(self, transcript: Transcript, image: ImageFrame) -> VLMResponse:
        width = image.width or 640
        height = image.height or 480
        object_name = _infer_object(transcript.text)
        destination = _infer_destination(transcript.text)
        output = {
            "action": "pick_and_place",
            "object": object_name,
            "destination": destination,
            "visible": True,
            "confidence": 0.9,
            "bbox_xyxy": [
                max(0, width // 4),
                max(0, height // 4),
                max(1, width // 2),
                max(1, height // 2),
            ],
            "image_size": [width, height],
        }
        return VLMResponse(
            backend=self.name,
            model=self.model,
            output=validate_vlm_payload(output),
            metadata={"deterministic": True},
        )


class QwenVLMBackend:
    name = "qwen"
    model = "Qwen2.5-VL"

    def analyze(self, transcript: Transcript, image: ImageFrame) -> VLMResponse:
        raise BackendUnavailableError(
            code="qwen_backend_unavailable",
            message=(
                "Qwen backend unavailable: install Transformers, PyTorch, and "
                "Qwen2.5-VL model weights before selecting --vlm-backend qwen."
            ),
            details={"backend": self.name, "model": self.model},
        )


def validate_vlm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the current VLM contract free of out-of-phase motion fields."""

    blocked = sorted(MOTION_CONTROL_FIELDS.intersection(payload))
    if blocked:
        raise ValidationError(
            code="motion_fields_rejected",
            message=f"VLM output contains out-of-phase motion field(s): {', '.join(blocked)}",
            details={"blocked_fields": blocked},
        )
    return payload


def _infer_object(text: str) -> str:
    lowered = text.lower()
    for candidate in ("red cup", "blue cube", "green bottle", "cup", "cube", "bottle"):
        if candidate in lowered:
            return candidate
    return "requested object"


def _infer_destination(text: str) -> str:
    lowered = text.lower()
    match = re.search(r"\b(?:to|into|onto)\s+the\s+(.+)$", lowered)
    if match:
        return match.group(1).strip().rstrip(".")
    return "drop zone"
