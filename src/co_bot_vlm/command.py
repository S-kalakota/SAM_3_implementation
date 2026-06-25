"""Command parsing boundary for VLM task output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ValidationError


@dataclass(frozen=True)
class TaskCommand:
    action: str
    object: str
    destination: str


def parse_vla_output(payload: dict[str, Any]) -> TaskCommand:
    """Validate only the command fields promised for this phase."""

    missing = [key for key in ("action", "object", "destination") if not payload.get(key)]
    if missing:
        raise ValidationError(
            code="invalid_command",
            message=f"VLM output is missing command field(s): {', '.join(missing)}",
            details={"missing": missing},
        )

    action = str(payload["action"]).strip()
    object_name = str(payload["object"]).strip()
    destination = str(payload["destination"]).strip()

    if action != "pick_and_place":
        raise ValidationError(
            code="unsupported_action",
            message=f"Unsupported action for this phase: {action}",
            details={"action": action},
        )

    return TaskCommand(action=action, object=object_name, destination=destination)
