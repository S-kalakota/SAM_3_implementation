from __future__ import annotations

import unittest

from co_bot_vlm.command import parse_transcript_command
from co_bot_vlm.errors import ValidationError


class CommandParserTests(unittest.TestCase):
    def test_parse_transcript_accepts_blue_box(self) -> None:
        command = parse_transcript_command("pick up the blue box to the drop zone")

        self.assertEqual(command.action, "pick_and_place")
        self.assertEqual(command.object, "blue box")
        self.assertEqual(command.destination, "drop zone")

    def test_parse_transcript_rejects_orange_object(self) -> None:
        with self.assertRaises(ValidationError) as context:
            parse_transcript_command("pick up the orange object to the drop zone")

        self.assertEqual(context.exception.code, "unsupported_object")
        self.assertEqual(context.exception.details["object"], "orange object")

    def test_parse_transcript_rejects_blue_cube(self) -> None:
        with self.assertRaises(ValidationError) as context:
            parse_transcript_command("pick up the blue cube to the drop zone")

        self.assertEqual(context.exception.code, "unsupported_object")
        self.assertEqual(context.exception.details["object"], "blue cube")


if __name__ == "__main__":
    unittest.main()
