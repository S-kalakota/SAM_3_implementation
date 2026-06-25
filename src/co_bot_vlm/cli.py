"""Command line entrypoint for the Co-Bot VLM skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import PipelineError
from .pipeline import run_pipeline
from .result import error_envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="co-bot-vlm",
        description="Verify that a commanded object visually exists in one RGB image.",
    )
    parser.add_argument("--text", help="Text transcript command.")
    parser.add_argument("--audio-file", type=Path, help="Audio transcript source placeholder.")
    parser.add_argument("--voice", action="store_true", help="Live voice source placeholder.")
    parser.add_argument("--image-file", type=Path, help="RGB image file to verify.")
    parser.add_argument(
        "--camera-index",
        type=int,
        help="Generic RGB camera index for one live OpenCV snapshot.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=("mock", "qwen"),
        default="mock",
        help="VLM backend to use.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON envelope.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        envelope = run_pipeline(
            text=args.text,
            image_file=args.image_file,
            camera_index=args.camera_index,
            vlm_backend=args.vlm_backend,
            audio_file=args.audio_file,
            voice=args.voice,
        )
        status = 0
    except PipelineError as error:
        envelope = error_envelope(error)
        status = 2

    indent = 2 if args.pretty else None
    print(json.dumps(envelope, indent=indent, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
