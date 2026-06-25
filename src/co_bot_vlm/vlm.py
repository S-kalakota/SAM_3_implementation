"""VLM adapter boundary and skeleton backends."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .command import SUPPORTED_OBJECT_ALIASES, TaskCommand, parse_vla_output, supported_objects
from .errors import BackendUnavailableError, ValidationError
from .image_source import ImageFrame
from .transcript import Transcript
from .visual_grounding import VisualGrounding


VLM_SCHEMA_VERSION = "object-existence-v1"
QWEN_MODEL_ID = os.environ.get(
    "CO_BOT_VLM_QWEN_MODEL_ID",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
QWEN_MAX_NEW_TOKENS = 256

PICK_AND_PLACE_REQUIRED_FIELDS = (
    "action",
    "object",
    "destination",
    "visible",
    "confidence",
    "bbox_xyxy",
    "image_size",
)
RETURN_HOME_REQUIRED_FIELDS = ("action",)
ALLOWED_FIELDS = set(PICK_AND_PLACE_REQUIRED_FIELDS)
MOTION_CONTROL_FIELDS = {
    "joint_angles",
    "pose",
    "object_pose",
    "velocity",
    "gripper",
    "robot_command",
    "trajectory",
}
SUPPORTED_OBJECTS = tuple(supported_objects())


@dataclass(frozen=True)
class VLMValidatedOutput:
    payload: dict[str, Any]
    command: TaskCommand
    grounding: VisualGrounding | None


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


def create_vlm_backend(name: str, *, qwen_model: str | None = None) -> VLMBackend:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockVLMBackend()
    if normalized == "qwen":
        return QwenVLMBackend(model_id=qwen_model or QWEN_MODEL_ID)
    raise ValidationError(
        code="unknown_vlm_backend",
        message=f"Unknown VLM backend: {name}",
        details={"vlm_backend": name},
    )


class MockVLMBackend:
    name = "mock"
    model = "deterministic-contract-v1"

    def analyze(self, transcript: Transcript, image: ImageFrame) -> VLMResponse:
        action = _infer_action(transcript.text)
        if action == "return_home":
            return build_vlm_response(
                backend=self.name,
                model=self.model,
                payload={"action": "return_home"},
                metadata={"deterministic": True},
            )

        width = image.width or 640
        height = image.height or 480
        object_name = _infer_object(transcript.text)
        destination = _infer_destination(transcript.text)
        visible, confidence = _infer_visual_presence(transcript.text)
        output = {
            "action": action,
            "object": object_name,
            "destination": destination,
            "visible": visible,
            "confidence": confidence,
            "bbox_xyxy": [
                max(0, width // 4),
                max(0, height // 4),
                max(1, width // 2),
                max(1, height // 2),
            ],
            "image_size": [width, height],
        }
        return build_vlm_response(
            backend=self.name,
            model=self.model,
            payload=output,
            metadata={"deterministic": True},
        )


class QwenVLMBackend:
    name = "qwen"
    model = QWEN_MODEL_ID

    def __init__(
        self,
        *,
        model_id: str = QWEN_MODEL_ID,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
        local_files_only: bool | None = None,
        device_map: str | None = None,
    ) -> None:
        self.model = model_id
        self.max_new_tokens = max_new_tokens
        self.local_files_only = (
            _env_flag("CO_BOT_VLM_QWEN_LOCAL_ONLY", default=True)
            if local_files_only is None
            else local_files_only
        )
        self.device_map = (
            os.environ.get("CO_BOT_VLM_QWEN_DEVICE_MAP")
            if device_map is None
            else device_map
        ) or _default_qwen_device_map()

    def analyze(self, transcript: Transcript, image: ImageFrame) -> VLMResponse:
        raw_output = self._generate_text(transcript, image)
        return build_vlm_response_from_text(
            backend=self.name,
            model=self.model,
            raw_output=raw_output,
            fallback_image_size=image.image_size,
            metadata={
                "max_new_tokens": self.max_new_tokens,
                "local_files_only": self.local_files_only,
                "device_map": self.device_map,
                "image_size_source": "vlm_or_input_image",
            },
        )

    def _generate_text(self, transcript: Transcript, image: ImageFrame) -> str:
        log_buffer = _StdoutToStderr()
        with contextlib.redirect_stdout(log_buffer):
            return self._generate_text_with_redirected_logs(transcript, image)

    def _generate_text_with_redirected_logs(
        self,
        transcript: Transcript,
        image: ImageFrame,
    ) -> str:
        if image.path is None:
            raise ValidationError(
                code="qwen_requires_image_file",
                message="Qwen backend currently requires an image file path.",
                details={"source_type": image.source_type},
            )

        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise BackendUnavailableError(
                code="qwen_backend_unavailable",
                message=(
                    "Qwen backend unavailable: install Transformers, PyTorch, "
                    "qwen-vl-utils, and Qwen2.5-VL model weights before selecting "
                    "--vlm-backend qwen."
                ),
                details={"backend": self.name, "model": self.model, "missing": str(exc)},
            ) from exc

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model,
                torch_dtype="auto",
                device_map=self.device_map,
                local_files_only=self.local_files_only,
            )
            processor = AutoProcessor.from_pretrained(
                self.model,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            mode = "local cache" if self.local_files_only else "configured model source"
            raise BackendUnavailableError(
                code="qwen_backend_unavailable",
                message=(
                    "Qwen backend unavailable: could not load Qwen2.5-VL from the "
                    f"{mode}. Install the model weights or set "
                    "CO_BOT_VLM_QWEN_LOCAL_ONLY=0 to allow Transformers downloads."
                ),
                details={"backend": self.name, "model": self.model, "error": str(exc)},
            ) from exc

        image_uri = Path(image.path).resolve().as_uri()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API. Return raw JSON only. "
                    "Do not use Markdown, code fences, explanations, or prose."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {
                        "type": "text",
                        "text": build_vlm_prompt(transcript.text, image.image_size),
                    },
                ],
            }
        ]

        try:
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if hasattr(model, "device"):
                inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise BackendUnavailableError(
                code="qwen_inference_failed",
                message=f"Qwen backend failed while generating VLM output: {exc}",
                details={"backend": self.name, "model": self.model},
            ) from exc

        if not output_text:
            raise ValidationError(
                code="vlm_output_empty",
                message="Qwen backend returned no text.",
                details={"backend": self.name, "model": self.model},
            )
        return output_text[0]


def build_vlm_response(
    *,
    backend: str,
    model: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> VLMResponse:
    validated = validate_vlm_payload(payload)
    return VLMResponse(
        backend=backend,
        model=model,
        output=validated,
        metadata=_metadata(metadata),
    )


def build_vlm_response_from_text(
    *,
    backend: str,
    model: str,
    raw_output: str,
    fallback_image_size: list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> VLMResponse:
    payload = parse_vlm_json_output(raw_output)
    payload = _fill_missing_image_size(payload, fallback_image_size)
    return build_vlm_response(
        backend=backend,
        model=model,
        payload=payload,
        metadata=metadata,
    )


def parse_vlm_json_output(raw_output: str) -> dict[str, Any]:
    """Parse a JSON-only VLM response into one top-level object."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValidationError(
            code="vlm_output_not_json",
            message="VLM output must be a non-empty JSON object string.",
            details={"type": type(raw_output).__name__},
        )

    stripped_output = raw_output.strip()
    try:
        payload, _end_index = json.JSONDecoder(
            parse_constant=_reject_json_constant,
        ).raw_decode(stripped_output)
    except ValueError as exc:
        raise ValidationError(
            code="vlm_output_not_json",
            message="VLM output must contain JSON only, with no prose or code fences.",
            details={"error": str(exc)},
        ) from exc

    if not isinstance(payload, dict):
        raise ValidationError(
            code="vlm_output_not_object",
            message="VLM output JSON must be one object.",
            details={"type": type(payload).__name__},
        )
    return payload


def _fill_missing_image_size(
    payload: dict[str, Any],
    fallback_image_size: list[int] | None,
) -> dict[str, Any]:
    if "image_size" in payload or fallback_image_size is None:
        return payload
    if payload.get("action") != "pick_and_place":
        return payload
    return {**payload, "image_size": fallback_image_size}


def validate_vlm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the restricted object-existence schema."""

    if not isinstance(payload, dict):
        raise ValidationError(
            code="invalid_vlm_payload",
            message="VLM payload must be a JSON object.",
            details={"type": type(payload).__name__},
        )
    blocked = sorted(MOTION_CONTROL_FIELDS.intersection(payload))
    if blocked:
        raise ValidationError(
            code="motion_fields_rejected",
            message=f"VLM output contains out-of-phase motion field(s): {', '.join(blocked)}",
            details={"blocked_fields": blocked},
        )

    action = _require_string(payload, "action")
    if action == "return_home":
        return _validate_return_home_payload(payload)
    if action != "pick_and_place":
        parse_vla_output({"action": action})

    missing = [key for key in PICK_AND_PLACE_REQUIRED_FIELDS if key not in payload]
    if missing:
        raise ValidationError(
            code="vlm_required_fields_missing",
            message=f"VLM output is missing required field(s): {', '.join(missing)}",
            details={"missing": missing, "schema_version": VLM_SCHEMA_VERSION},
        )

    unexpected = sorted(set(payload) - ALLOWED_FIELDS)
    if unexpected:
        raise ValidationError(
            code="vlm_unexpected_fields",
            message=f"VLM output contains unsupported field(s): {', '.join(unexpected)}",
            details={"unexpected": unexpected, "schema_version": VLM_SCHEMA_VERSION},
        )

    canonical_object = _canonicalize_object(payload["object"])
    destination = _require_string(payload, "destination")
    visible = _require_bool(payload, "visible")
    confidence = _require_confidence(payload["confidence"])
    image_size = _validate_image_size(payload["image_size"])
    bbox = _validate_bbox(payload["bbox_xyxy"], image_size)

    validated = {
        "action": action,
        "object": canonical_object,
        "destination": destination,
        "visible": visible,
        "confidence": confidence,
        "bbox_xyxy": bbox,
        "image_size": image_size,
    }
    parse_vla_output(validated)
    return validated


def validate_vlm_output(payload: dict[str, Any]) -> VLMValidatedOutput:
    """Return the command and grounding objects produced from one VLM payload."""

    validated = validate_vlm_payload(payload)
    command = parse_vla_output(validated)
    grounding = None
    if command.action == "pick_and_place":
        grounding = VisualGrounding(
            visible=validated["visible"],
            confidence=validated["confidence"],
            bbox_xyxy=validated["bbox_xyxy"],
            image_size=validated["image_size"],
            object=validated["object"],
        )
    return VLMValidatedOutput(payload=validated, command=command, grounding=grounding)


def build_vlm_prompt(transcript_text: str, image_size: list[int] | None = None) -> str:
    image_size_instruction = (
        f"The input image_size is exactly {image_size}; use this value unchanged. "
        if image_size is not None
        else ""
    )
    return (
        "You are verifying whether a requested object exists in one RGB image. "
        "Return exactly one raw JSON object and nothing else. "
        "The first character must be { and the last character must be }. "
        "Do not wrap the answer in ```json or any Markdown fence. "
        "Allowed schema for a pick-and-place command: "
        '{"action":"pick_and_place","object":"red cup|blue box|green bottle",'
        '"destination":"drop zone","visible":true|false,'
        '"confidence":0.0-1.0,"bbox_xyxy":[x1,y1,x2,y2],'
        '"image_size":[width,height]}. '
        f"{image_size_instruction}"
        "Use the destination phrase from the transcript without angle brackets. "
        "Use integer pixel coordinates within image_size. "
        "Allowed object synonyms are cup/red mug for red cup and bottle for "
        "green bottle. There are no synonyms for blue box. "
        "Do not include joint_angles, pose, object_pose, velocity, gripper, "
        "robot_command, trajectory, or any other fields. "
        f"Transcript: {transcript_text}"
    )


def _infer_action(text: str) -> str:
    lowered = _normalize_text(text)
    if re.search(r"\b(?:return home|go home|home position)\b", lowered):
        return "return_home"
    return "pick_and_place"


def _infer_visual_presence(text: str) -> tuple[bool, float]:
    lowered = _normalize_text(text)
    absence_patterns = (
        "not visible",
        "not present",
        "not in the image",
        "not in image",
        "cannot see",
        "can't see",
        "missing",
        "absent",
    )
    if any(pattern in lowered for pattern in absence_patterns):
        return False, 0.2
    return True, 0.9


def _infer_object(text: str) -> str:
    lowered = _normalize_text(text)
    for candidate in sorted(SUPPORTED_OBJECT_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            return candidate

    match = re.search(
        r"\b(?:pick up|pick|grab|move)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:to|into|onto)\b|$)",
        lowered,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return "requested object"


def _infer_destination(text: str) -> str:
    lowered = text.lower()
    match = re.search(r"\b(?:to|into|onto)\s+the\s+(.+)$", lowered)
    if match:
        return match.group(1).strip().rstrip(".")
    return "drop zone"


def _validate_return_home_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unexpected = sorted(set(payload) - set(RETURN_HOME_REQUIRED_FIELDS))
    if unexpected:
        raise ValidationError(
            code="vlm_unexpected_fields",
            message=(
                "return_home VLM output must not include object, visual, or "
                f"extra field(s): {', '.join(unexpected)}"
            ),
            details={"unexpected": unexpected, "schema_version": VLM_SCHEMA_VERSION},
        )
    return {"action": "return_home"}


def _canonicalize_object(value: Any) -> str:
    object_name = _normalize_text(_require_string({"object": value}, "object"))
    canonical = SUPPORTED_OBJECT_ALIASES.get(object_name)
    if canonical is None:
        raise ValidationError(
            code="unsupported_object",
            message=(
                f"Unsupported object in VLM output: {value}. "
                f"Supported objects are: {', '.join(SUPPORTED_OBJECTS)}."
            ),
            details={"object": value, "supported_objects": list(SUPPORTED_OBJECTS)},
        )
    return canonical


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            code="invalid_vlm_field",
            message=f"VLM field '{key}' must be a non-empty string.",
            details={"field": key, "type": type(value).__name__},
        )
    return value.strip()


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValidationError(
            code="invalid_vlm_field",
            message=f"VLM field '{key}' must be a JSON boolean.",
            details={"field": key, "type": type(value).__name__},
        )
    return value


def _require_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            code="invalid_vlm_field",
            message="VLM field 'confidence' must be a number from 0.0 to 1.0.",
            details={"field": "confidence", "type": type(value).__name__},
        )
    confidence = float(value)
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValidationError(
            code="invalid_vlm_field",
            message="VLM field 'confidence' must be between 0.0 and 1.0.",
            details={"field": "confidence", "value": value},
        )
    return confidence


def _validate_image_size(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValidationError(
            code="invalid_image_size",
            message="VLM field 'image_size' must be [width, height] integers.",
            details={"image_size": value},
        )
    width, height = value
    if width <= 0 or height <= 0:
        raise ValidationError(
            code="invalid_image_size",
            message="VLM image_size values must be positive.",
            details={"image_size": value},
        )
    return [width, height]


def _validate_bbox(value: Any, image_size: list[int]) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValidationError(
            code="invalid_bbox",
            message="VLM field 'bbox_xyxy' must be [x1, y1, x2, y2] integers.",
            details={"bbox_xyxy": value},
        )
    x1, y1, x2, y2 = value
    width, height = image_size
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValidationError(
            code="invalid_bbox",
            message="VLM bbox_xyxy must be within image bounds and have positive area.",
            details={"bbox_xyxy": value, "image_size": image_size},
        )
    return [x1, y1, x2, y2]


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower()).strip(" .")
    for prefix in ("the ", "a ", "an "):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(metadata or {})
    values["schema_version"] = VLM_SCHEMA_VERSION
    values["required_fields"] = list(PICK_AND_PLACE_REQUIRED_FIELDS)
    values["motion_control_fields_rejected"] = True
    values["supported_objects"] = list(SUPPORTED_OBJECTS)
    return values


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _default_qwen_device_map() -> str:
    if sys.platform == "darwin":
        return "cpu"
    return "auto"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


class _StdoutToStderr(io.TextIOBase):
    def write(self, value: str) -> int:
        sys.stderr.write(value)
        return len(value)

    def flush(self) -> None:
        sys.stderr.flush()
