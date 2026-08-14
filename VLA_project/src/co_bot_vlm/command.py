"""Command parsing boundary for transcript-derived task intent."""

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
    source: str | None = None



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
    source = str(payload["source"]).strip() if payload.get("source") else None
    destination = str(payload["destination"]).strip()

    if action != "pick_and_place":
        raise ValidationError(
            code="unsupported_action",
            message=f"Unsupported action for this phase: {action}",
            details={"action": action},
        )

    return TaskCommand(
        action=action,
        object=object_name,
        destination=destination,
        source=source,
    )


def parse_transcript_command(text: str) -> TaskCommand:
    """Parse the user's transcript before any VLM-generated command rewriting."""

    normalized = _normalize_text(text)
    if re.search(r"\b(?:return home|go home|home position)\b", normalized):
        return TaskCommand(action="return_home")

    object_name = _extract_requested_object(normalized)
    if object_name == "requested object":
        raise ValidationError(
            code="missing_object",
            message="Could not identify the requested object in the transcript.",
            details={"transcript": text},
        )

    destination = _extract_destination(normalized)
    source = _extract_source(normalized)
    return TaskCommand(
        action="pick_and_place",
        object=object_name,
        destination=destination,
        source=source,
    )



def _extract_requested_object(text: str) -> str:
    match = re.search(
        r"\b(?:pick up|pick|grab|move|put|place)\s+"
        r"(?:the\s+|a\s+|an\s+)?(.+?)"
        r"(?=\s+(?:from|to|into|onto)\b|"
        r"\s+and\s+(?:place|put|move|drop|set)\b|$)",
        text,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return "requested object"


def _extract_source(text: str) -> str | None:
    match = re.search(
        r"\bfrom\s+(?:the\s+)?(.+?)"
        r"(?=\s+(?:to|into|onto)\b|"
        r"\s+and\s+(?:place|put|move|drop|set)\b|$)",
        text,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def _extract_destination(text: str) -> str:
    action_match = re.search(
        r"\band\s+(?:place|put|move|drop|set)(?:\s+it)?\s+"
        r"(?:to|in|into|onto)\s+(?:the\s+)?(.+)$",
        text,
    )
    if action_match:
        return action_match.group(1).strip().rstrip(".")
    match = re.search(r"\b(?:to|into|onto)\s+(?:the\s+)?(.+)$", text)
    if match:
        return match.group(1).strip().rstrip(".")
    return "drop zone"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).strip(" .")
