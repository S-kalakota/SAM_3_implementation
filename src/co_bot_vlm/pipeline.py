"""End-to-end object existence verification pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .command import TaskCommand, parse_transcript_command, parse_vla_output
from .errors import ValidationError
from .image_source import ImageFrame, get_image_frame, iter_camera_frames
from .result import success_envelope
from .safety import check_safety
from .transcript import Transcript, get_transcript
from .visual_grounding import verify_visual_grounding
from .vlm import VLMBackend, create_vlm_backend


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
    requested_command = parse_transcript_command(transcript.text)
    vlm = backend.analyze(transcript, image)
    vlm_command = parse_vla_output(vlm.output)
    command = _require_matching_commands(requested_command, vlm_command)
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


def _require_matching_commands(
    requested_command: TaskCommand,
    vlm_command: TaskCommand,
) -> TaskCommand:
    if requested_command.action != vlm_command.action:
        raise ValidationError(
            code="command_mismatch",
            message="VLM command action does not match the user transcript.",
            details={
                "requested_action": requested_command.action,
                "vlm_action": vlm_command.action,
            },
        )
    if requested_command.object != vlm_command.object:
        raise ValidationError(
            code="command_mismatch",
            message="VLM command object does not match the user transcript.",
            details={
                "requested_object": requested_command.object,
                "vlm_object": vlm_command.object,
            },
        )
    if requested_command.destination != vlm_command.destination:
        raise ValidationError(
            code="command_mismatch",
            message="VLM command destination does not match the user transcript.",
            details={
                "requested_destination": requested_command.destination,
                "vlm_destination": vlm_command.destination,
            },
        )
    return requested_command
