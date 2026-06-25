"""End-to-end object existence verification pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .command import parse_transcript_command
from .image_source import ImageFrame, get_image_frame, iter_camera_frames
from .result import success_envelope
from .safety import check_safety
from .transcript import Transcript, get_transcript
from .visual_grounding import VisualVerification, verify_visual_grounding
from .vlm import VLMBackend, VLMResponse, create_vlm_backend


def run_pipeline(
    *,
    text: str | None,
    image_file: Path | None,
    camera_index: int | None,
    vlm_backend: str,
    qwen_model: str | None = None,
    voice: bool = False,
    voice_duration_seconds: float = 5.0,
    whisper_model: str | None = None,
) -> dict:
    transcript = get_transcript(
        text=text,
        voice=voice,
        voice_duration_seconds=voice_duration_seconds,
        whisper_model=whisper_model,
    )
    image = get_image_frame(image_file=image_file, camera_index=camera_index)
    backend = create_vlm_backend(vlm_backend, qwen_model=qwen_model)
    return run_verification(transcript=transcript, image=image, backend=backend)


def run_live_pipeline(
    *,
    text: str | None,
    camera_index: int,
    vlm_backend: str,
    qwen_model: str | None = None,
    interval_seconds: float,
    max_frames: int | None = None,
    voice: bool = False,
    voice_duration_seconds: float = 5.0,
    whisper_model: str | None = None,
) -> Iterator[dict]:
    transcript = get_transcript(
        text=text,
        voice=voice,
        voice_duration_seconds=voice_duration_seconds,
        whisper_model=whisper_model,
    )
    backend = create_vlm_backend(vlm_backend, qwen_model=qwen_model)
    for image in iter_camera_frames(
        camera_index=camera_index,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    ):
        yield run_verification(transcript=transcript, image=image, backend=backend)


def run_verification(
    *,
    transcript: Transcript,
    image: ImageFrame,
    backend: VLMBackend,
) -> dict:
    command = parse_transcript_command(transcript.text)
    if command.action == "return_home":
        vlm = VLMResponse(
            backend=backend.name,
            model=backend.model,
            output={},
            metadata={
                "skipped": True,
                "reason": "return_home command has no target object to ground",
            },
        )
        visual_verification = VisualVerification(
            approved=True,
            reason="no object visibility required",
            grounding=None,
        )
    else:
        vlm = backend.ground(command, image)
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
