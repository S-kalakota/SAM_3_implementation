#!/usr/bin/env python3
"""Task 3 smoke test: prompt SAM 3.1 and apply the presence gate."""

from __future__ import annotations

import task2_sam31_image_prompt as task2


DEFAULT_OUTPUT = task2.PROJECT_ROOT / "outputs/task3_sam31_presence_gate.json"


def main() -> None:
    task2.main(
        description=(
            "Run SAM 3.1 on a saved image, then filter masks with the "
            "Task 3 presence gate."
        ),
        default_output=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    main()
