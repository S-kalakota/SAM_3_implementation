"""Shared structured errors for the verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineError(Exception):
    """Base error that can be rendered into the public JSON envelope."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class BackendUnavailableError(PipelineError):
    """Raised when an optional backend is selected but not installed yet."""


class ValidationError(PipelineError):
    """Raised when pipeline data does not meet the skeleton contract."""
