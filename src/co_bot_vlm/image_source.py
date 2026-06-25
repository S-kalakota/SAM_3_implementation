"""Generic RGB image source boundary."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BackendUnavailableError, ValidationError


@dataclass(frozen=True)
class ImageFrame:
    source_type: str
    path: str | None
    width: int | None
    height: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_size(self) -> list[int] | None:
        if self.width is None or self.height is None:
            return None
        return [self.width, self.height]


def get_image_frame(
    *,
    image_file: Path | None,
    camera_index: int | None,
) -> ImageFrame:
    """Return one RGB frame from a file or generic camera placeholder."""

    if image_file is not None and camera_index is not None:
        raise ValidationError(
            code="multiple_image_sources",
            message="Pass either --image-file or --camera-index, not both.",
            details={"image_file": str(image_file), "camera_index": camera_index},
        )

    if image_file is not None:
        return _load_image_file(image_file)

    if camera_index is not None:
        raise BackendUnavailableError(
            code="camera_backend_unavailable",
            message=(
                "Camera backend unavailable: generic RGB capture is not "
                "implemented in this skeleton. Use --image-file for now."
            ),
            details={"camera_index": camera_index},
        )

    raise ValidationError(
        code="missing_image_source",
        message="No image source provided. Pass --image-file for the skeleton path.",
        details={},
    )


def _load_image_file(path: Path) -> ImageFrame:
    if not path.exists():
        raise ValidationError(
            code="image_file_missing",
            message=f"Image file does not exist: {path}",
            details={"image_file": str(path)},
        )
    if not path.is_file():
        raise ValidationError(
            code="image_file_not_file",
            message=f"Image path is not a file: {path}",
            details={"image_file": str(path)},
        )

    width, height = _read_dimensions(path)
    return ImageFrame(
        source_type="image_file",
        path=str(path),
        width=width,
        height=height,
        metadata={"dimensions_known": width is not None and height is not None},
    )


def _read_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Best-effort PNG/JPEG/GIF dimension reader using only the stdlib."""

    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            return struct.unpack(">II", header[16:24])
        if header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10:
            return struct.unpack("<HH", header[6:10])
        if header.startswith(b"\xff\xd8"):
            return _read_jpeg_dimensions(handle)
    return None, None


def _read_jpeg_dimensions(handle) -> tuple[int | None, int | None]:
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
