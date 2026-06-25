"""Public JSON envelope formatting."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .command import TaskCommand
from .errors import PipelineError
from .image_source import ImageFrame
from .safety import SafetyDecision
from .transcript import Transcript
from .visual_grounding import VisualVerification
from .vlm import VLMResponse


TOP_LEVEL_KEYS = ("input", "vlm", "visual_verification", "safety", "next")


def success_envelope(
    *,
    transcript: Transcript,
    image: ImageFrame,
    vlm: VLMResponse,
    command: TaskCommand,
    visual_verification: VisualVerification,
    safety: SafetyDecision,
) -> dict[str, Any]:
    next_step = "ready_for_later_phase" if safety.approved else "blocked"
    return {
        "input": {
            "transcript": asdict(transcript),
            "image": image.to_public_dict(),
        },
        "vlm": {
            "backend": vlm.backend,
            "model": vlm.model,
            "output": vlm.output,
            "metadata": vlm.metadata,
            "command": asdict(command),
        },
        "visual_verification": asdict(visual_verification),
        "safety": asdict(safety),
        "next": {
            "status": next_step,
            "description": (
                "No robot motion is produced in this phase."
                if safety.approved
                else "Fix the input image, transcript, or backend before continuing."
            ),
        },
    }


def error_envelope(error: PipelineError) -> dict[str, Any]:
    return {
        "input": {},
        "vlm": {},
        "visual_verification": {},
        "safety": {
            "approved": False,
            "reason": "blocked by safety: pipeline could not complete",
        },
        "next": {
            "status": "error",
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            },
        },
    }
