#!/usr/bin/env python3
"""Shared path discovery for the integrated Grounding DINO/FR5 repository."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the monorepo root, with an environment override for installations."""

    configured = os.environ.get("GROUNDED_COBOT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    location = (start or Path(__file__)).expanduser().resolve()
    for candidate in (location.parent, *location.parents):
        if (candidate / "sam3-dino").is_file() and (
            candidate / "robot_ws"
        ).is_dir():
            return candidate

    # Compatibility with the original multi-repository workstation layout.
    return (Path.home() / "VLA_Model_Work").resolve()


def calibration_dir(start: Path | None = None) -> Path:
    """Return the configured or repository-local FR5 calibration directory."""

    configured = os.environ.get("FR5_CALIB_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return find_repo_root(start) / "robot_ws" / "calib"
