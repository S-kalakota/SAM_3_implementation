from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from co_bot_vlm.errors import ValidationError
from co_bot_vlm.image_source import ImageFrame
from co_bot_vlm.pipeline import run_live_pipeline, run_pipeline
from co_bot_vlm.transcript import Transcript
from co_bot_vlm.vlm import build_vlm_response


def minimal_png(width: int = 640, height: int = 480) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_uses_image_file_and_mock_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png(800, 600))

            envelope = run_pipeline(
                text="pick up the red cup to the drop zone",
                image_file=image,
                camera_index=None,
                vlm_backend="mock",
            )

        self.assertEqual(envelope["input"]["image"]["source_type"], "image_file")
        self.assertEqual(envelope["input"]["image"]["width"], 800)
        self.assertEqual(envelope["input"]["image"]["height"], 600)
        self.assertEqual(envelope["vlm"]["backend"], "mock")
        self.assertEqual(envelope["vlm"]["output"]["image_size"], [800, 600])

    def test_pipeline_output_has_no_fake_perception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            envelope = run_pipeline(
                text="pick up the blue box to the inspection bin",
                image_file=image,
                camera_index=None,
                vlm_backend="mock",
            )

        serialized = json.dumps(envelope)
        forbidden_fields = (
            "fake_perception",
            "fake_perception_bundle",
            "object_pose",
            "robot_coordinates",
            "joint_angles",
            "movement_command",
            "robot_command",
            "trajectory",
            "velocity",
            "gripper",
        )
        for field in forbidden_fields:
            self.assertNotIn(field, serialized)

    def test_pipeline_approved_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            envelope = run_pipeline(
                text="pick up the red cup to the drop zone",
                image_file=image,
                camera_index=None,
                vlm_backend="mock",
            )

        self.assertTrue(envelope["visual_verification"]["approved"])
        self.assertTrue(envelope["safety"]["approved"])
        self.assertEqual(envelope["visual_verification"]["reason"], "object visually verified")
        self.assertIn("object visually verified", envelope["safety"]["reason"])
        self.assertIn("object visually verified", envelope["next"]["description"])
        self.assertEqual(envelope["next"]["status"], "ready_for_later_phase")

    def test_pipeline_blocked_by_visual_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            envelope = run_pipeline(
                text="pick up the red cup that is not visible to the drop zone",
                image_file=image,
                camera_index=None,
                vlm_backend="mock",
            )

        self.assertFalse(envelope["visual_verification"]["approved"])
        self.assertFalse(envelope["safety"]["approved"])
        self.assertEqual(envelope["visual_verification"]["reason"], "object not visually verified")
        self.assertIn("blocked by safety", envelope["safety"]["reason"])
        self.assertIn("blocked by safety", envelope["next"]["description"])
        self.assertEqual(envelope["next"]["status"], "blocked_by_safety")

    def test_pipeline_accepts_blue_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            envelope = run_pipeline(
                text="pick up the blue box to the drop zone",
                image_file=image,
                camera_index=None,
                vlm_backend="mock",
            )

        self.assertTrue(envelope["safety"]["approved"])
        self.assertEqual(envelope["vlm"]["command"]["object"], "blue box")
        self.assertEqual(envelope["vlm"]["output"]["object"], "blue box")

    def test_pipeline_rejects_blue_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            with self.assertRaises(ValidationError) as context:
                run_pipeline(
                    text="pick up the blue cube to the drop zone",
                    image_file=image,
                    camera_index=None,
                    vlm_backend="mock",
                )

        self.assertEqual(context.exception.code, "unsupported_object")

    def test_pipeline_rejects_vlm_object_rewrite_from_unsupported_transcript(self) -> None:
        class RewritingBackend:
            name = "rewriting"
            model = "test"

            def analyze(self, transcript: Transcript, image: ImageFrame):
                return build_vlm_response(
                    backend=self.name,
                    model=self.model,
                    payload={
                        "action": "pick_and_place",
                        "object": "blue box",
                        "destination": "drop zone",
                        "visible": True,
                        "confidence": 0.95,
                        "bbox_xyxy": [0, 10, 100, 200],
                        "image_size": [640, 480],
                    },
                )

        transcript = Transcript(
            text="pick up the orange object to the drop zone",
            source="text",
        )
        image = ImageFrame(
            source_type="image_file",
            path="/tmp/frame.png",
            width=640,
            height=480,
        )

        with self.assertRaises(ValidationError) as context:
            from co_bot_vlm.pipeline import run_verification

            run_verification(
                transcript=transcript,
                image=image,
                backend=RewritingBackend(),
            )

        self.assertEqual(context.exception.code, "unsupported_object")
        self.assertEqual(context.exception.details["object"], "orange object")

    def test_voice_transcript_enters_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(minimal_png())

            with patch(
                "co_bot_vlm.transcript.transcribe_voice",
                return_value=Transcript(
                    text="pick up the blue box to the bin",
                    source="voice",
                    metadata={"backend": "whisper", "model": "test-whisper"},
                ),
            ):
                envelope = run_pipeline(
                    text=None,
                    voice=True,
                    voice_duration_seconds=1.0,
                    whisper_model="test-whisper",
                    image_file=image,
                    camera_index=None,
                    vlm_backend="mock",
                )

        self.assertEqual(envelope["input"]["transcript"]["source"], "voice")
        self.assertEqual(envelope["input"]["transcript"]["metadata"]["backend"], "whisper")
        self.assertEqual(envelope["vlm"]["command"]["object"], "blue box")

    def test_live_camera_pipeline_yields_bounded_frames(self) -> None:
        frames = [
            ImageFrame(
                source_type="camera",
                path="/tmp/frame1.jpg",
                width=640,
                height=480,
                metadata={"camera_index": 0, "frame_number": 1},
            ),
            ImageFrame(
                source_type="camera",
                path="/tmp/frame2.jpg",
                width=640,
                height=480,
                metadata={"camera_index": 0, "frame_number": 2},
            ),
        ]

        with patch("co_bot_vlm.pipeline.iter_camera_frames", return_value=iter(frames)):
            envelopes = list(
                run_live_pipeline(
                    text="pick up the green bottle to the shelf",
                    camera_index=0,
                    vlm_backend="mock",
                    interval_seconds=0,
                    max_frames=2,
                )
            )

        self.assertEqual(len(envelopes), 2)
        self.assertEqual(envelopes[0]["input"]["image"]["source_type"], "camera")
        self.assertEqual(envelopes[0]["input"]["image"]["metadata"]["frame_number"], 1)
        self.assertEqual(envelopes[1]["input"]["image"]["metadata"]["frame_number"], 2)
        self.assertEqual(envelopes[0]["vlm"]["command"]["object"], "green bottle")


if __name__ == "__main__":
    unittest.main()
