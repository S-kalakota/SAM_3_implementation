from __future__ import annotations

import unittest

from co_bot_vlm.command import TaskCommand, parse_vla_output
from co_bot_vlm.safety import check_verified_command
from co_bot_vlm.visual_grounding import VisualGrounding, check_visual_grounding


def grounding(**overrides) -> VisualGrounding:
    values = {
        "visible": True,
        "confidence": 0.86,
        "bbox_xyxy": [280, 190, 360, 310],
        "image_size": [1280, 720],
        "object": "red cup",
    }
    values.update(overrides)
    return VisualGrounding(**values)


class VisualSafetyTests(unittest.TestCase):
    def test_valid_grounding_approved(self) -> None:
        command = TaskCommand(
            action="pick_and_place",
            object="red cup",
            destination="drop zone",
        )

        decision = check_verified_command(command, grounding())

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "object is present in the frame")

    def test_not_visible_rejected(self) -> None:
        decision = check_verified_command(
            TaskCommand("pick_and_place", "red cup", "drop zone"),
            grounding(visible=False),
        )

        self.assertFalse(decision.approved)
        self.assertIn("object not visually verified", decision.reason)

    def test_object_name_mismatch_does_not_override_visual_existence(self) -> None:
        decision = check_verified_command(
            TaskCommand("pick_and_place", "green bottle", "drop zone"),
            grounding(object="red cup"),
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "object is present in the frame")

    def test_low_confidence_does_not_override_visible_answer(self) -> None:
        verification = check_visual_grounding(grounding(confidence=0.0))

        self.assertTrue(verification.approved)
        self.assertEqual(verification.reason, "object visually verified")

    def test_bbox_is_not_required_for_presence_only_check(self) -> None:
        verification = check_visual_grounding(grounding(bbox_xyxy=None))

        self.assertTrue(verification.approved)
        self.assertEqual(verification.reason, "object visually verified")

    def test_return_home_does_not_require_grounding(self) -> None:
        command = parse_vla_output({"action": "return_home"})

        decision = check_verified_command(command, None)

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "command is valid; no object visibility required")


if __name__ == "__main__":
    unittest.main()
