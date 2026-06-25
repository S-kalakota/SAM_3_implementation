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
        self.assertEqual(decision.reason, "command is valid and object visually verified")

    def test_not_visible_rejected(self) -> None:
        decision = check_verified_command(
            TaskCommand("pick_and_place", "red cup", "drop zone"),
            grounding(visible=False),
        )

        self.assertFalse(decision.approved)
        self.assertIn("object not visually verified", decision.reason)

    def test_wrong_object_rejected(self) -> None:
        decision = check_verified_command(
            TaskCommand("pick_and_place", "blue cube", "drop zone"),
            grounding(object="red cup"),
        )

        self.assertFalse(decision.approved)
        self.assertIn("object and command object differ", decision.reason)

    def test_low_confidence_rejected(self) -> None:
        verification = check_visual_grounding(grounding(confidence=0.79))

        self.assertFalse(verification.approved)
        self.assertEqual(verification.reason, "visual confidence below threshold")

    def test_bbox_out_of_bounds_rejected(self) -> None:
        verification = check_visual_grounding(
            grounding(bbox_xyxy=[10, 10, 1300, 100], image_size=[1280, 720])
        )

        self.assertFalse(verification.approved)
        self.assertEqual(verification.reason, "visual bounding box is invalid")

    def test_return_home_does_not_require_grounding(self) -> None:
        command = parse_vla_output({"action": "return_home"})

        decision = check_verified_command(command, None)

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "command is valid; no object visibility required")


if __name__ == "__main__":
    unittest.main()
