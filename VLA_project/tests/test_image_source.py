from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from co_bot_vlm.errors import BackendUnavailableError, ValidationError
from co_bot_vlm.image_source import get_image_frame


def minimal_png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def minimal_jpeg_with_app_segment(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        b"\xff\xe0"
        b"\x00\x10"
        b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xc0"
        b"\x00\x11"
        b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )


class ImageSourceTests(unittest.TestCase):
    def test_load_image_file_returns_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png(1280, 720))

            frame = get_image_frame(image_file=image, camera_index=None)

        self.assertEqual(frame.source_type, "image_file")
        self.assertEqual(frame.width, 1280)
        self.assertEqual(frame.height, 720)
        self.assertEqual(frame.image_size, [1280, 720])
        self.assertEqual(frame.metadata["format"], "png")

    def test_load_jpeg_with_app_segment_returns_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.jpeg"
            image.write_bytes(minimal_jpeg_with_app_segment(5712, 4284))

            frame = get_image_frame(image_file=image, camera_index=None)

        self.assertEqual(frame.source_type, "image_file")
        self.assertEqual(frame.width, 5712)
        self.assertEqual(frame.height, 4284)
        self.assertEqual(frame.image_size, [5712, 4284])
        self.assertEqual(frame.metadata["format"], "jpeg")

    def test_missing_image_file_errors_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.jpg"

            with self.assertRaises(ValidationError) as raised:
                get_image_frame(image_file=missing, camera_index=None)

        self.assertEqual(raised.exception.code, "image_file_missing")
        self.assertIn("does not exist", raised.exception.message)

    def test_invalid_image_file_errors_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.jpg"
            image.write_bytes(b"not really an image")

            with self.assertRaises(ValidationError) as raised:
                get_image_frame(image_file=image, camera_index=None)

        self.assertEqual(raised.exception.code, "invalid_image_file")
        self.assertIn("supported RGB image", raised.exception.message)

    def test_camera_capture_returns_dimensions_with_mocked_opencv(self) -> None:
        frame = self._capture_fake_camera_frame(width=32, height=24)

        self.assertEqual(frame.source_type, "camera")
        self.assertEqual(frame.width, 32)
        self.assertEqual(frame.height, 24)
        self.assertEqual(frame.metadata["camera_index"], 2)
        self.assertEqual(frame.metadata["backend"], "opencv")
        self.assertEqual(frame.metadata["successful_reads"], 4)
        self.assertFalse(frame.metadata["stereo_crop_applied"])
        self.assertEqual(frame.image["code"], "BGR2RGB")
        self.assertIsNotNone(frame.path)
        self.assertTrue(Path(frame.path).exists())
        self.assertNotIn("image", frame.to_public_dict())
        Path(frame.path).unlink()

    def test_camera_capture_crops_wide_stereo_frame_to_left_view(self) -> None:
        frame = self._capture_fake_camera_frame(width=1344, height=376)

        self.assertEqual(frame.width, 672)
        self.assertEqual(frame.height, 376)
        self.assertTrue(frame.metadata["stereo_crop_applied"])
        self.assertEqual(frame.metadata["stereo_crop_view"], "left")
        self.assertEqual(frame.metadata["original_width"], 1344)
        self.assertEqual(frame.metadata["original_height"], 376)
        self.assertEqual(frame.image["frame"].shape, (376, 672, 3))
        Path(frame.path).unlink()

    def _capture_fake_camera_frame(self, *, width: int, height: int):
        class FakeFrame:
            def __init__(self, width: int, height: int) -> None:
                self.width = width
                self.height = height
                self.shape = (height, width, 3)

            def __getitem__(self, key):
                rows, columns = key
                if not isinstance(columns, slice):
                    raise AssertionError("expected column slice")
                stop = self.width if columns.stop is None else columns.stop
                start = 0 if columns.start is None else columns.start
                return FakeFrame(stop - start, self.height)

        class FakeCapture:
            def __init__(self, camera_index: int) -> None:
                self.camera_index = camera_index
                self.released = False
                self.read_count = 0

            def isOpened(self) -> bool:
                return True

            def read(self):
                self.read_count += 1
                return True, FakeFrame(width, height)

            def set(self, property_id, value) -> bool:
                return True

            def release(self) -> None:
                self.released = True

        def fake_imwrite(path: str, frame: FakeFrame) -> bool:
            Path(path).write_bytes(f"fake jpeg {frame.width}x{frame.height}".encode())
            return True

        fake_cv2 = types.SimpleNamespace(
            COLOR_BGR2RGB="BGR2RGB",
            CAP_PROP_BUFFERSIZE="CAP_PROP_BUFFERSIZE",
            VideoCapture=FakeCapture,
            cvtColor=lambda frame, code: {"frame": frame, "code": code},
            imwrite=fake_imwrite,
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            return get_image_frame(image_file=None, camera_index=2)

    def test_camera_without_opencv_errors_cleanly(self) -> None:
        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(BackendUnavailableError) as raised:
                get_image_frame(image_file=None, camera_index=0)

        self.assertEqual(raised.exception.code, "camera_backend_unavailable")
        self.assertIn("install OpenCV", raised.exception.message)


if __name__ == "__main__":
    unittest.main()
