from __future__ import annotations

import json
import unittest

from co_bot_vlm.errors import ValidationError
from co_bot_vlm.image_source import ImageFrame
from co_bot_vlm.transcript import Transcript
from co_bot_vlm.vlm import (
    MockVLMBackend,
    VLM_SCHEMA_VERSION,
    build_vlm_response_from_text,
    parse_vlm_json_output,
    validate_vlm_output,
)


def vlm_payload(**overrides):
    values = {
        "action": "pick_and_place",
        "object": "red cup",
        "destination": "drop zone",
        "visible": True,
        "confidence": 0.86,
        "bbox_xyxy": [280, 190, 360, 310],
        "image_size": [1280, 720],
    }
    values.update(overrides)
    return values


class VLMAdapterTests(unittest.TestCase):
    def test_vlm_accepts_valid_visible_object(self) -> None:
        raw_output = json.dumps(vlm_payload(object="red cup"))

        result = validate_vlm_output(parse_vlm_json_output(raw_output))

        self.assertEqual(result.command.object, "red cup")
        self.assertIsNotNone(result.grounding)
        self.assertEqual(result.grounding.object, "red cup")
        self.assertEqual(result.payload["bbox_xyxy"], [280, 190, 360, 310])

    def test_vlm_accepts_blue_box(self) -> None:
        result = validate_vlm_output(vlm_payload(object="blue box"))

        self.assertEqual(result.command.object, "blue box")
        self.assertEqual(result.grounding.object, "blue box")
        self.assertEqual(result.payload["object"], "blue box")

    def test_vlm_rejects_blue_cube(self) -> None:
        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(vlm_payload(object="blue cube"))

        self.assertEqual(context.exception.code, "unsupported_object")

    def test_vlm_rejects_blue_block(self) -> None:
        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(vlm_payload(object="blue block"))

        self.assertEqual(context.exception.code, "unsupported_object")

    def test_vlm_rejects_unsupported_object(self) -> None:
        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(vlm_payload(object="banana"))

        self.assertEqual(context.exception.code, "unsupported_object")

    def test_vlm_rejects_missing_bbox(self) -> None:
        payload = vlm_payload()
        del payload["bbox_xyxy"]

        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(payload)

        self.assertEqual(context.exception.code, "vlm_required_fields_missing")
        self.assertEqual(context.exception.details["missing"], ["bbox_xyxy"])

    def test_vlm_response_fills_missing_image_size_from_image_source(self) -> None:
        payload = vlm_payload(bbox_xyxy=[280, 190, 360, 310])
        del payload["image_size"]

        response = build_vlm_response_from_text(
            backend="qwen",
            model="test-model",
            raw_output=json.dumps(payload),
            fallback_image_size=[1280, 720],
        )

        self.assertEqual(response.output["image_size"], [1280, 720])

    def test_vlm_rejects_invalid_bbox(self) -> None:
        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(vlm_payload(bbox_xyxy=[280, 190, 2000, 310]))

        self.assertEqual(context.exception.code, "invalid_bbox")

    def test_vlm_rejects_motion_control_fields(self) -> None:
        with self.assertRaises(ValidationError) as context:
            validate_vlm_output(vlm_payload(robot_command={"move": "arm"}))

        self.assertEqual(context.exception.code, "motion_fields_rejected")
        self.assertEqual(context.exception.details["blocked_fields"], ["robot_command"])

    def test_vlm_rejects_non_json(self) -> None:
        with self.assertRaises(ValidationError) as context:
            parse_vlm_json_output('Here is the result: {"action":"pick_and_place"}')

        self.assertEqual(context.exception.code, "vlm_output_not_json")

    def test_vlm_accepts_first_json_object_with_trailing_model_output(self) -> None:
        payload = parse_vlm_json_output(
            json.dumps(vlm_payload(object="blue box"))
            + "\nThe requested object is visible."
        )

        self.assertEqual(payload["object"], "blue box")
        self.assertEqual(payload["action"], "pick_and_place")

    def test_vlm_rejects_non_standard_json_constant(self) -> None:
        with self.assertRaises(ValidationError) as context:
            parse_vlm_json_output('{"action":"pick_and_place","confidence":NaN}')

        self.assertEqual(context.exception.code, "vlm_output_not_json")

    def test_mock_backend_records_backend_model_and_schema_metadata(self) -> None:
        response = MockVLMBackend().analyze(
            Transcript(text="pick up the cup to the drop zone", source="text"),
            ImageFrame(
                source_type="image_file",
                path="/tmp/frame.png",
                width=640,
                height=480,
            ),
        )

        self.assertEqual(response.backend, "mock")
        self.assertEqual(response.model, "deterministic-contract-v1")
        self.assertEqual(response.output["object"], "red cup")
        self.assertEqual(response.metadata["schema_version"], VLM_SCHEMA_VERSION)
        self.assertTrue(response.metadata["motion_control_fields_rejected"])


if __name__ == "__main__":
    unittest.main()
