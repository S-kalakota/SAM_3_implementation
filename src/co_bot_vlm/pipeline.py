"""End-to-end object existence verification pipeline."""

from __future__ import annotations

from pathlib import Path

from .command import parse_vla_output
from .image_source import get_image_frame
from .result import success_envelope
from .safety import check_safety
from .transcript import get_transcript
from .visual_grounding import verify_visual_grounding
from .vlm import create_vlm_backend


def run_pipeline(
    *,
    text: str | None,
    image_file: Path | None,
    camera_index: int | None,
    vlm_backend: str,
    audio_file: Path | None = None,
    voice: bool = False,
) -> dict:
    transcript = get_transcript(text=text, audio_file=audio_file, voice=voice)
    image = get_image_frame(image_file=image_file, camera_index=camera_index)
    backend = create_vlm_backend(vlm_backend)
    vlm = backend.analyze(transcript, image)
    command = parse_vla_output(vlm.output)
    visual_verification = verify_visual_grounding(vlm.output)
    safety = check_safety(command, visual_verification)
    return success_envelope(
        transcript=transcript,
        image=image,
        vlm=vlm,
        command=command,
        visual_verification=visual_verification,
        safety=safety,
    )
