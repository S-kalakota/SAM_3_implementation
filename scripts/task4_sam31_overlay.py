#!/usr/bin/env python3
"""Task 4 smoke test: prompt SAM 3.1, gate masks, and draw the overlay."""

from __future__ import annotations

import task2_sam31_image_prompt as task2


DEFAULT_OUTPUT = task2.PROJECT_ROOT / "outputs/task4_sam31_overlay.json"
DEFAULT_OVERLAY_OUTPUT = task2.PROJECT_ROOT / "outputs/result.png"


def main() -> None:
    task2.main(
        description=(
            "Run SAM 3.1 on a saved image, filter masks with the presence "
            "gate, then draw every kept mask onto an overlay image."
        ),
        default_output=DEFAULT_OUTPUT,
        default_overlay_output=DEFAULT_OVERLAY_OUTPUT,
    )


if __name__ == "__main__":
    main()
