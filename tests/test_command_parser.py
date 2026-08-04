from __future__ import annotations

import unittest

from co_bot_vlm.command import parse_transcript_command
class CommandParserTests(unittest.TestCase):
    def test_parse_transcript_accepts_blue_box(self) -> None:
        command = parse_transcript_command("pick up the blue box to the drop zone")

        self.assertEqual(command.action, "pick_and_place")
        self.assertEqual(command.object, "blue box")
        self.assertEqual(command.destination, "drop zone")
        self.assertIsNone(command.source)

    def test_parse_transcript_extracts_source_and_destination(self) -> None:
        command = parse_transcript_command(
            "put the blue box from the left bin into the right bin"
        )

        self.assertEqual(command.action, "pick_and_place")
        self.assertEqual(command.object, "blue box")
        self.assertEqual(command.source, "left bin")
        self.assertEqual(command.destination, "right bin")

    def test_parse_transcript_accepts_open_vocabulary_object(self) -> None:
        command = parse_transcript_command("pick up the orange object to the drop zone")

        self.assertEqual(command.action, "pick_and_place")
        self.assertEqual(command.object, "orange object")
        self.assertEqual(command.destination, "drop zone")

    def test_parse_transcript_accepts_blue_cube(self) -> None:
        command = parse_transcript_command("pick up the blue cube to the drop zone")

        self.assertEqual(command.action, "pick_and_place")
        self.assertEqual(command.object, "blue cube")
        self.assertEqual(command.destination, "drop zone")


if __name__ == "__main__":
    unittest.main()
