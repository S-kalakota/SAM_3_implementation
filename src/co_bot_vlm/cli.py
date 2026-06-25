"""Command line entrypoint for the Co-Bot VLM skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import PipelineError, ValidationError
from .pipeline import run_live_pipeline, run_pipeline
from .result import error_envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="co-bot-vlm",
        description="Verify that a commanded object visually exists in one RGB image.",
    )
    parser.add_argument("--text", help="Text transcript command.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Record a microphone command and transcribe it with Whisper.",
    )
    parser.add_argument(
        "--voice-duration",
        type=float,
        default=5.0,
        help="Seconds to record when --voice is selected.",
    )
    parser.add_argument(
        "--whisper-model",
        help=(
            "Hugging Face Whisper model id for --voice. "
            "Defaults to CO_BOT_VLM_WHISPER_MODEL_ID or openai/whisper-tiny.en."
        ),
    )
    parser.add_argument("--image-file", type=Path, help="RGB image file to verify.")
    parser.add_argument(
        "--camera-index",
        type=int,
        help="Generic RGB camera index for one live OpenCV snapshot.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Continuously verify frames from a generic RGB camera.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=1.0,
        help="Seconds to wait between live camera frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop live mode after this many frames.",
    )
    parser.add_argument(
        "--stop-on-approved",
        action="store_true",
        help="Stop live mode after the first approved frame.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=("mock", "qwen"),
        default="mock",
        help="VLM backend to use.",
    )
    parser.add_argument(
        "--qwen-model",
        help=(
            "Hugging Face model id for --vlm-backend qwen. "
            "Defaults to CO_BOT_VLM_QWEN_MODEL_ID or Qwen/Qwen2.5-VL-3B-Instruct."
        ),
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
    indent = 2 if args.pretty else None

    try:
        if args.live:
            if args.image_file is not None:
                raise ValidationError(
                    code="live_image_file_conflict",
                    message="Live mode reads from --camera-index and cannot use --image-file.",
                    details={"image_file": str(args.image_file)},
                )
            camera_index = 0 if args.camera_index is None else args.camera_index
            for envelope in run_live_pipeline(
                text=args.text,
                camera_index=camera_index,
                vlm_backend=args.vlm_backend,
                qwen_model=args.qwen_model,
                interval_seconds=args.live_interval,
                max_frames=args.max_frames,
                voice=args.voice,
                voice_duration_seconds=args.voice_duration,
                whisper_model=args.whisper_model,
            ):
                print(json.dumps(envelope, indent=indent, sort_keys=True), flush=True)
                if args.stop_on_approved and envelope["safety"]["approved"]:
                    return 0
            return 0

        envelope = run_pipeline(
            text=args.text,
            image_file=args.image_file,
            camera_index=args.camera_index,
            vlm_backend=args.vlm_backend,
            qwen_model=args.qwen_model,
            voice=args.voice,
            voice_duration_seconds=args.voice_duration,
            whisper_model=args.whisper_model,
        )
        status = 0
    except PipelineError as error:
        envelope = error_envelope(error)
        status = 2

    print(json.dumps(envelope, indent=indent, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
