"""Command parsing boundary for VLM task output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .errors import ValidationError


@dataclass(frozen=True)
class TaskCommand:
    action: str
    object: str | None = None
    destination: str | None = None


SUPPORTED_OBJECT_ALIASES = {
    "red cup": "red cup",
    "cup": "red cup",
    "red mug": "red cup",
    "mug": "red cup",
    "blue box": "blue box",
    "green bottle": "green bottle",
    "bottle": "green bottle",
}


def parse_vla_output(payload: dict[str, Any]) -> TaskCommand:
    """Validate only the command fields promised for this phase."""

    if not payload.get("action"):
        raise ValidationError(
            code="invalid_command",
            message="VLM output is missing command field(s): action",
            details={"missing": ["action"]},
        )

    action = str(payload["action"]).strip()

    if action == "return_home":
        return TaskCommand(action=action)

    missing = [key for key in ("object", "destination") if not payload.get(key)]
    if missing:
        raise ValidationError(
            code="invalid_command",
            message=f"VLM output is missing command field(s): {', '.join(missing)}",
            details={"missing": missing},
        )

    object_name = str(payload["object"]).strip()
    destination = str(payload["destination"]).strip()

    if action != "pick_and_place":
        raise ValidationError(
            code="unsupported_action",
            message=f"Unsupported action for this phase: {action}",
            details={"action": action},
        )

    return TaskCommand(action=action, object=object_name, destination=destination)


def parse_transcript_command(text: str) -> TaskCommand:
    """Parse the user's transcript before any VLM-generated command rewriting."""

    normalized = _normalize_text(text)
    if re.search(r"\b(?:return home|go home|home position)\b", normalized):
        return TaskCommand(action="return_home")

    object_name = _extract_supported_object(normalized)
    if object_name is None:
        requested_object = _extract_requested_object(normalized)
        raise ValidationError(
            code="unsupported_object",
            message=(
                f"Unsupported object in transcript: {requested_object}. "
                f"Supported objects are: {', '.join(supported_objects())}."
            ),
            details={
                "object": requested_object,
                "supported_objects": supported_objects(),
            },
        )

    destination = _extract_destination(normalized)
    return TaskCommand(
        action="pick_and_place",
        object=object_name,
        destination=destination,
    )


def supported_objects() -> list[str]:
    return sorted(set(SUPPORTED_OBJECT_ALIASES.values()))


def _extract_supported_object(text: str) -> str | None:
    for candidate in sorted(SUPPORTED_OBJECT_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", text):
            return SUPPORTED_OBJECT_ALIASES[candidate]
    return None


def _extract_requested_object(text: str) -> str:
    match = re.search(
        r"\b(?:pick up|pick|grab|move)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:to|into|onto)\b|$)",
        text,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return "requested object"


def _extract_destination(text: str) -> str:
    match = re.search(r"\b(?:to|into|onto)\s+(?:the\s+)?(.+)$", text)
    if match:
        return match.group(1).strip().rstrip(".")
    return "drop zone"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).strip(" .")
