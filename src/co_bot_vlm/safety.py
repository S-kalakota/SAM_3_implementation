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
    """Approve object commands only from visual existence evidence."""

    if command.action in NON_OBJECT_ACTIONS:
        return SafetyDecision(True, "command is valid; no object visibility required")
    if verification.approved:
        return SafetyDecision(True, "object is present in the frame")
    return SafetyDecision(False, f"object not approved: {verification.reason}")


def check_verified_command(
    command: TaskCommand,
    grounding: VisualGrounding | None,
) -> SafetyDecision:
    """Compatibility wrapper around the visual existence gate."""

    if command.action in NON_OBJECT_ACTIONS:
        return SafetyDecision(True, "command is valid; no object visibility required")

    if grounding is None:
        return SafetyDecision(False, "object not approved: object grounding is missing")

    verification = check_visual_grounding(grounding)
    if not verification.approved:
        return SafetyDecision(False, f"object not approved: {verification.reason}")

    return SafetyDecision(True, "object is present in the frame")
