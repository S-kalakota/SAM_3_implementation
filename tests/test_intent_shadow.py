#!/usr/bin/env python3
"""Ensure the committed 40-command shadow corpus is deterministic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_intent_shadow as shadow  # noqa: E402


class IntentShadowCorpusTests(unittest.TestCase):
    def test_all_40_expectations_match(self) -> None:
        corpus = shadow.load_commands(
            PROJECT_ROOT / "evaluation" / "intent_shadow_commands.json"
        )
        self.assertEqual(len(corpus["commands"]), 40)
        mismatches = []
        for command in corpus["commands"]:
            actual = shadow.parse_one_deterministically(command["text"])["accepted"]
            if actual != command["accepted"]:
                mismatches.append(command["text"])
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
