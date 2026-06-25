from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "co_bot_vlm.cli", *args],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )


class CliSkeletonTests(unittest.TestCase):
    def test_cli_help_loads_without_optional_dependencies(self) -> None:
        result = run_cli("--help")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--vlm-backend", result.stdout)
        self.assertIn("--image-file", result.stdout)

    def test_mock_pipeline_runs_without_camera_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x02\x80"
                b"\x00\x00\x01\xe0"
                b"\x08\x02\x00\x00\x00"
            )

            result = run_cli(
                "--text",
                "pick up the red cup to the drop zone",
                "--image-file",
                str(image),
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        envelope = json.loads(result.stdout)
        self.assertEqual(list(envelope.keys()), sorted(envelope.keys()))
        self.assertTrue(envelope["safety"]["approved"])
        self.assertEqual(envelope["vlm"]["backend"], "mock")
        self.assertEqual(envelope["vlm"]["output"]["image_size"], [640, 480])
        self.assertNotIn("object_pose", json.dumps(envelope))
        self.assertNotIn("joint_angles", json.dumps(envelope))

    def test_qwen_backend_unavailable_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.jpg"
            image.write_bytes(b"not really an image")

            result = run_cli(
                "--text",
                "pick up the red cup",
                "--image-file",
                str(image),
                "--vlm-backend",
                "qwen",
            )

        self.assertEqual(result.returncode, 2)
        envelope = json.loads(result.stdout)
        self.assertEqual(envelope["next"]["error"]["code"], "qwen_backend_unavailable")
        self.assertIn("Qwen backend unavailable", envelope["next"]["error"]["message"])

    def test_camera_backend_unavailable_error_is_clear(self) -> None:
        result = run_cli(
            "--text",
            "pick up the red cup",
            "--camera-index",
            "0",
        )

        self.assertEqual(result.returncode, 2)
        envelope = json.loads(result.stdout)
        self.assertEqual(envelope["next"]["error"]["code"], "camera_backend_unavailable")
        self.assertIn("Camera backend unavailable", envelope["next"]["error"]["message"])

    def test_skeleton_output_schema_has_expected_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.gif"
            image.write_bytes(b"GIF89a\x20\x03\x58\x02")

            result = run_cli(
                "--text",
                "pick up the blue cube to the inspection bin",
                "--image-file",
                str(image),
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        envelope = json.loads(result.stdout)
        self.assertEqual(
            set(envelope),
            {"input", "vlm", "visual_verification", "safety", "next"},
        )
        self.assertEqual(envelope["vlm"]["command"]["object"], "blue cube")
        self.assertEqual(envelope["next"]["status"], "ready_for_later_phase")


if __name__ == "__main__":
    unittest.main()
