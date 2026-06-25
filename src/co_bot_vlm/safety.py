"""Safety decision boundary for the current no-motion phase."""

from __future__ import annotations

from dataclasses import dataclass

from .command import TaskCommand
from .visual_grounding import VisualGrounding, VisualVerification, check_visual_grounding

NON_OBJECT_ACTIONS = {"return_home"}


@dataclass(frozen=True)
class SafetyDecision:
    approved: bool
    reason: str


def check_safety(command: TaskCommand, verification: VisualVerification) -> SafetyDecision:
    """Approve only parsed commands whose requested object is visually verified."""

    return check_verified_command(command, verification.grounding)


def check_verified_command(
    command: TaskCommand,
    grounding: VisualGrounding | None,
) -> SafetyDecision:
    """Approve valid commands only when object commands have matching visual proof."""

    if command.action in NON_OBJECT_ACTIONS:
        return SafetyDecision(True, "command is valid; no object visibility required")

    if grounding is None:
        return SafetyDecision(False, "blocked by safety: object grounding is missing")

    verification = check_visual_grounding(grounding)
    if not verification.approved:
        return SafetyDecision(False, f"blocked by safety: {verification.reason}")

    if command.object is None:
        return SafetyDecision(False, "blocked by safety: object command is missing an object")

    grounded_object = grounding.object
    if grounded_object.strip().lower() != command.object.strip().lower():
        return SafetyDecision(False, "blocked by safety: grounded object and command object differ")

    return SafetyDecision(True, "command is valid and object visually verified")
