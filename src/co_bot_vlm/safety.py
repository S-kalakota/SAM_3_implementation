"""Safety decision boundary for the current no-motion phase."""

from __future__ import annotations

from dataclasses import dataclass

from .command import TaskCommand
from .visual_grounding import VisualVerification


@dataclass(frozen=True)
class SafetyDecision:
    approved: bool
    reason: str


def check_safety(command: TaskCommand, verification: VisualVerification) -> SafetyDecision:
    """Approve only parsed commands whose requested object is visually verified."""

    if not verification.approved:
        return SafetyDecision(False, f"blocked by safety: {verification.reason}")

    grounded_object = verification.grounding.object if verification.grounding else ""
    if grounded_object.strip().lower() != command.object.strip().lower():
        return SafetyDecision(False, "blocked by safety: VLM object and command object differ")

    return SafetyDecision(True, "command is valid and object visually verified")
