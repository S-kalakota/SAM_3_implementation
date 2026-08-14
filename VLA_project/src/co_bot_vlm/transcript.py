"""Transcript input boundary."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Any

from .errors import BackendUnavailableError, ValidationError

DEFAULT_WHISPER_MODEL_ID = os.environ.get(
    "CO_BOT_VLM_WHISPER_MODEL_ID",
    "openai/whisper-tiny.en",
)
DEFAULT_VOICE_DURATION_SECONDS = 5.0
VOICE_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Transcript:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def get_transcript(
    *,
    text: str | None,
    voice: bool = False,
    voice_duration_seconds: float = DEFAULT_VOICE_DURATION_SECONDS,
    whisper_model: str | None = None,
) -> Transcript:
    """Return a transcript from typed text or live Whisper voice capture."""

    if text:
        return Transcript(text=text, source="text", metadata={})

    if voice:
        return transcribe_voice(
            duration_seconds=voice_duration_seconds,
            model_id=whisper_model or DEFAULT_WHISPER_MODEL_ID,
        )

    raise ValidationError(
        code="missing_transcript",
        message="No transcript source provided. Pass --text or --voice.",
        details={},
    )


def transcribe_voice(
    *,
    duration_seconds: float,
    model_id: str,
) -> Transcript:
    """Record one microphone clip with SoX and transcribe it with Whisper."""

    print(
        f"Recording voice command for {duration_seconds:.1f} seconds...",
        file=sys.stderr,
        flush=True,
    )
    audio_path = _record_microphone_clip(duration_seconds)
    try:
        print("Transcribing voice command with Whisper...", file=sys.stderr, flush=True)
        text, metadata = _transcribe_wav_with_whisper(audio_path, model_id=model_id)
    finally:
        audio_path.unlink(missing_ok=True)

    if not text:
        raise ValidationError(
            code="empty_voice_transcript",
            message="Whisper returned an empty transcript. Try speaking closer to the microphone.",
            details={"model": model_id, "duration_seconds": duration_seconds},
        )

    metadata["duration_seconds"] = duration_seconds
    return Transcript(text=text, source="voice", metadata=metadata)


def _record_microphone_clip(duration_seconds: float) -> "Path":
    from pathlib import Path

    if duration_seconds <= 0:
        raise ValidationError(
            code="invalid_voice_duration",
            message="Voice recording duration must be greater than zero.",
            details={"duration_seconds": duration_seconds},
        )

    rec_path = shutil.which("rec")
    if rec_path is None:
        raise BackendUnavailableError(
            code="voice_recorder_unavailable",
            message="Voice recorder unavailable: install SoX so the 'rec' command is available.",
            details={"dependency": "sox"},
        )

    with tempfile.NamedTemporaryFile(
        prefix="co_bot_vlm_voice_",
        suffix=".wav",
        delete=False,
    ) as handle:
        audio_path = Path(handle.name)

    command = [
        rec_path,
        "-q",
        "-r",
        str(VOICE_SAMPLE_RATE),
        "-c",
        "1",
        "-b",
        "16",
        str(audio_path),
        "trim",
        "0",
        f"{duration_seconds:.3f}",
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        audio_path.unlink(missing_ok=True)
        raise BackendUnavailableError(
            code="voice_capture_failed",
            message=(
                "Voice capture failed. On macOS, verify microphone permission for "
                "the terminal or app running this command."
            ),
            details={
                "command": command,
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
            },
        )

    return audio_path


def _transcribe_wav_with_whisper(audio_path: "Path", *, model_id: str) -> tuple[str, dict[str, Any]]:
    log_buffer = _StdoutToStderr()
    with contextlib.redirect_stdout(log_buffer):
        return _transcribe_wav_with_whisper_redirected(audio_path, model_id=model_id)


def _transcribe_wav_with_whisper_redirected(
    audio_path: "Path",
    *,
    model_id: str,
) -> tuple[str, dict[str, Any]]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise BackendUnavailableError(
            code="whisper_backend_unavailable",
            message=(
                "Whisper backend unavailable: install torch, transformers, and numpy "
                "before using --voice."
            ),
            details={"model": model_id, "missing": str(exc)},
        ) from exc

    audio, sample_rate = _read_wav_float32(audio_path, np)
    local_files_only = _env_flag("CO_BOT_VLM_WHISPER_LOCAL_ONLY", default=True)
    device = os.environ.get("CO_BOT_VLM_WHISPER_DEVICE", "cpu")
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype="auto",
            local_files_only=local_files_only,
        )
        if device != "cpu":
            model = model.to(device)
    except Exception as exc:
        mode = "local cache" if local_files_only else "configured model source"
        raise BackendUnavailableError(
            code="whisper_backend_unavailable",
            message=(
                f"Whisper backend unavailable: could not load {model_id} from the "
                f"{mode}. Download the model or set CO_BOT_VLM_WHISPER_LOCAL_ONLY=0."
            ),
            details={"model": model_id, "error": str(exc)},
        ) from exc

    try:
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                max_new_tokens=96,
            )
        transcript = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
    except Exception as exc:
        raise BackendUnavailableError(
            code="whisper_transcription_failed",
            message=f"Whisper failed while transcribing microphone audio: {exc}",
            details={"model": model_id},
        ) from exc

    return transcript, {
        "backend": "whisper",
        "model": model_id,
        "sample_rate": sample_rate,
        "local_files_only": local_files_only,
        "device": device,
    }


def _read_wav_float32(audio_path: "Path", np: Any) -> tuple[Any, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw_audio = wav_file.readframes(frames)

    if sample_width != 2:
        raise ValidationError(
            code="unsupported_voice_audio_format",
            message="Voice capture produced unsupported audio; expected 16-bit PCM WAV.",
            details={"sample_width": sample_width},
        )

    audio = np.frombuffer(raw_audio, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class _StdoutToStderr(io.TextIOBase):
    def write(self, value: str) -> int:
        sys.stderr.write(value)
        return len(value)

    def flush(self) -> None:
        sys.stderr.flush()
