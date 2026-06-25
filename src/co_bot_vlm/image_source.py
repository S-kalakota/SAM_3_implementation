"""Generic RGB image source boundary."""

from __future__ import annotations

import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from .errors import BackendUnavailableError, ValidationError

CAMERA_WARMUP_FRAMES = 3
CAMERA_READ_ATTEMPTS = 10


@dataclass(frozen=True)
class ImageFrame:
    source_type: str
    path: str | None
    width: int | None
    height: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    image: Any = field(default=None, repr=False, compare=False)

    @property
    def image_size(self) -> list[int] | None:
        if self.width is None or self.height is None:
            return None
        return [self.width, self.height]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }


def get_image_frame(
    *,
    image_file: Path | None,
    camera_index: int | None,
) -> ImageFrame:
    """Return one RGB frame from a file or generic camera device."""

    if image_file is not None and camera_index is not None:
        raise ValidationError(
            code="multiple_image_sources",
            message="Pass either --image-file or --camera-index, not both.",
            details={"image_file": str(image_file), "camera_index": camera_index},
        )

    if image_file is not None:
        return _load_image_file(image_file)

    if camera_index is not None:
        return _capture_camera_frame(camera_index)

    raise ValidationError(
        code="missing_image_source",
        message="No image source provided. Pass --image-file for the skeleton path.",
        details={},
    )


def _load_image_file(path: Path) -> ImageFrame:
    expanded = path.expanduser()
    if not expanded.exists():
        raise ValidationError(
            code="image_file_missing",
            message=f"Image file does not exist: {path}",
            details={"image_file": str(path)},
        )
    if not expanded.is_file():
        raise ValidationError(
            code="image_file_not_file",
            message=f"Image path is not a file: {path}",
            details={"image_file": str(path)},
        )

    try:
        width, height, image_format = _read_dimensions(expanded)
    except OSError as exc:
        raise ValidationError(
            code="image_file_unreadable",
            message=f"Image file could not be read: {path}",
            details={"image_file": str(path), "error": str(exc)},
        ) from exc

    if (
        width is None
        or height is None
        or width <= 0
        or height <= 0
        or image_format is None
    ):
        raise ValidationError(
            code="invalid_image_file",
            message=(
                "Image file is unreadable or not a supported RGB image. "
                "Use a PNG, JPEG, or GIF frame."
            ),
            details={"image_file": str(path)},
        )

    return ImageFrame(
        source_type="image_file",
        path=str(expanded.resolve()),
        width=width,
        height=height,
        metadata={
            "format": image_format,
            "dimensions_known": True,
        },
    )


def _capture_camera_frame(camera_index: int) -> ImageFrame:
    if camera_index < 0:
        raise ValidationError(
            code="invalid_camera_index",
            message="Camera index must be zero or greater.",
            details={"camera_index": camera_index},
        )

    with _open_configured_camera(camera_index) as camera:
        return _capture_frame_from_open_camera(
            camera.cv2,
            camera.capture,
            camera_index,
            camera.opencv_capture_backend,
            frame_number=1,
        )


def iter_camera_frames(
    *,
    camera_index: int,
    interval_seconds: float,
    max_frames: int | None = None,
) -> Iterator[ImageFrame]:
    """Yield RGB frames from one generic camera until stopped."""

    if camera_index < 0:
        raise ValidationError(
            code="invalid_camera_index",
            message="Camera index must be zero or greater.",
            details={"camera_index": camera_index},
        )
    if interval_seconds < 0:
        raise ValidationError(
            code="invalid_live_interval",
            message="Live camera interval must be zero or greater.",
            details={"interval_seconds": interval_seconds},
        )
    if max_frames is not None and max_frames <= 0:
        raise ValidationError(
            code="invalid_max_frames",
            message="Live camera max frame count must be greater than zero.",
            details={"max_frames": max_frames},
        )

    frame_number = 0
    with _open_configured_camera(camera_index) as camera:
        while max_frames is None or frame_number < max_frames:
            frame_number += 1
            yield _capture_frame_from_open_camera(
                camera.cv2,
                camera.capture,
                camera_index,
                camera.opencv_capture_backend,
                frame_number=frame_number,
            )
            if interval_seconds:
                time.sleep(interval_seconds)


@dataclass(frozen=True)
class _OpenCamera:
    cv2: Any
    capture: Any
    opencv_capture_backend: str

    def __enter__(self) -> "_OpenCamera":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.capture.release()


def _open_configured_camera(camera_index: int) -> _OpenCamera:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: install OpenCV (cv2) for generic "
                "RGB camera capture, or use --image-file."
            ),
            details={"camera_index": camera_index, "dependency": "opencv-python"},
        ) from exc

    capture, opencv_capture_backend = _open_video_capture(cv2, camera_index)
    _configure_capture(cv2, capture)
    if not capture.isOpened():
        capture.release()
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: could not open generic RGB "
                f"camera index {camera_index}. Use --image-file for "
                "repeatable checks. On macOS, verify camera permission "
                "for the terminal or app running this command."
            ),
            details={"camera_index": camera_index},
        )

    return _OpenCamera(
        cv2=cv2,
        capture=capture,
        opencv_capture_backend=opencv_capture_backend,
    )


def _capture_frame_from_open_camera(
    cv2: Any,
    capture: Any,
    camera_index: int,
    opencv_capture_backend: str,
    *,
    frame_number: int,
) -> ImageFrame:
    frame, read_metadata = _read_live_frame(capture, camera_index)
    height, width = _frame_dimensions(frame, camera_index)
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: captured frame could not be "
                "converted to RGB."
            ),
            details={"camera_index": camera_index, "error": str(exc)},
        ) from exc
    snapshot_path = _write_camera_snapshot(cv2, frame, camera_index)
    return ImageFrame(
        source_type="camera",
        path=snapshot_path,
        width=width,
        height=height,
        metadata={
            "camera_index": camera_index,
            "backend": "opencv",
            "opencv_capture_backend": opencv_capture_backend,
            "color_space": "RGB",
            "snapshot_path": snapshot_path,
            "frame_number": frame_number,
            **read_metadata,
        },
        image=rgb_frame,
    )


def _open_video_capture(cv2: Any, camera_index: int) -> tuple[Any, str]:
    backend = None
    backend_name = "default"
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backend = cv2.CAP_AVFOUNDATION
        backend_name = "avfoundation"

    try:
        if backend is not None:
            return cv2.VideoCapture(camera_index, backend), backend_name
        return cv2.VideoCapture(camera_index), backend_name
    except TypeError:
        return cv2.VideoCapture(camera_index), "default"
    except Exception as exc:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: could not initialize generic RGB "
                "camera capture."
            ),
            details={"camera_index": camera_index, "error": str(exc)},
        ) from exc


def _configure_capture(cv2: Any, capture: Any) -> None:
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            return


def _read_live_frame(capture: Any, camera_index: int) -> tuple[Any, dict[str, Any]]:
    last_frame = None
    successful_reads = 0
    failed_reads = 0
    total_reads = CAMERA_WARMUP_FRAMES + CAMERA_READ_ATTEMPTS

    for read_number in range(1, total_reads + 1):
        ok, frame = capture.read()
        if ok and frame is not None:
            successful_reads += 1
            last_frame = frame
            if successful_reads > CAMERA_WARMUP_FRAMES:
                return frame, {
                    "warmup_frames": CAMERA_WARMUP_FRAMES,
                    "read_attempts": read_number,
                    "successful_reads": successful_reads,
                    "failed_reads": failed_reads,
                }
        else:
            failed_reads += 1

    if last_frame is not None:
        return last_frame, {
            "warmup_frames": CAMERA_WARMUP_FRAMES,
            "read_attempts": total_reads,
            "successful_reads": successful_reads,
            "failed_reads": failed_reads,
            "used_last_available_frame": True,
        }

    raise BackendUnavailableError(
        code="camera_backend_unavailable",
        message=(
            "Camera backend unavailable: generic RGB camera capture returned "
            f"no frames for index {camera_index}. Use --image-file for "
            "repeatable checks."
        ),
        details={
            "camera_index": camera_index,
            "warmup_frames": CAMERA_WARMUP_FRAMES,
            "read_attempts": total_reads,
            "failed_reads": failed_reads,
        },
    )


def _frame_dimensions(frame: Any, camera_index: int) -> tuple[int, int]:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: captured frame had no readable "
                "width or height."
            ),
            details={"camera_index": camera_index},
        )
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: captured frame had invalid "
                "width or height."
            ),
            details={"camera_index": camera_index, "width": width, "height": height},
        )
    return height, width


def _write_camera_snapshot(cv2: Any, frame: Any, camera_index: int) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="co_bot_vlm_camera_",
        suffix=".jpg",
        delete=False,
    ) as handle:
        snapshot_path = Path(handle.name)

    try:
        wrote = cv2.imwrite(str(snapshot_path), frame)
    except Exception as exc:
        snapshot_path.unlink(missing_ok=True)
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message="Camera backend unavailable: captured frame could not be saved.",
            details={"camera_index": camera_index, "error": str(exc)},
        ) from exc

    if not wrote:
        snapshot_path.unlink(missing_ok=True)
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message="Camera backend unavailable: captured frame could not be saved.",
            details={"camera_index": camera_index},
        )

    return str(snapshot_path)


def _read_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    """Best-effort PNG/JPEG/GIF dimension reader using only the stdlib."""

    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return width, height, "png"
        if header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10:
            width, height = struct.unpack("<HH", header[6:10])
            return width, height, "gif"
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            width, height = _read_jpeg_dimensions(handle)
            return width, height, "jpeg" if width is not None and height is not None else None
    return None, None, None


def _read_jpeg_dimensions(handle: BinaryIO) -> tuple[int | None, int | None]:
    while True:
        marker_prefix = handle.read(1)
        if not marker_prefix:
            return None, None
        if marker_prefix != b"\xff":
            continue

        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker or marker in {b"\xd8", b"\xd9"}:
            continue

        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            return None, None
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            return None, None

        if marker in {
            b"\xc0",
            b"\xc1",
            b"\xc2",
            b"\xc3",
            b"\xc5",
            b"\xc6",
            b"\xc7",
            b"\xc9",
            b"\xca",
            b"\xcb",
            b"\xcd",
            b"\xce",
            b"\xcf",
        }:
            data = handle.read(5)
            if len(data) != 5:
                return None, None
            height, width = struct.unpack(">HH", data[1:5])
            return width, height

        handle.seek(segment_length - 2, 1)
