"""Transcript input boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BackendUnavailableError, ValidationError


@dataclass(frozen=True)
class Transcript:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def get_transcript(
    *,
    text: str | None,
    audio_file: Path | None = None,
    voice: bool = False,
) -> Transcript:
    """Return a transcript from the selected source.

    Audio and live voice are public interface placeholders for later agents.
    """

    if text:
        return Transcript(text=text, source="text", metadata={})

    if audio_file is not None:
        raise BackendUnavailableError(
            code="audio_backend_unavailable",
            message=(
                "Audio transcript backend unavailable: speech-to-text is not "
                "implemented in this skeleton."
            ),
            details={"audio_file": str(audio_file)},
        )

    if voice:
        raise BackendUnavailableError(
            code="voice_backend_unavailable",
            message=(
                "Voice transcript backend unavailable: live microphone capture "
                "is not implemented in this skeleton."
            ),
            details={},
        )

    raise ValidationError(
        code="missing_transcript",
        message="No transcript source provided. Pass --text for the skeleton path.",
        details={},
    )
