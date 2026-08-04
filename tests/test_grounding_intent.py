#!/usr/bin/env python3
"""Contract and parser tests for structured visual grounding intents."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_intent as grounding  # noqa: E402


class DeterministicIntentTests(unittest.TestCase):
    def test_required_valid_examples(self) -> None:
        examples = {
            "pick the box": ("box", None, None),
            "pick the top box": ("box", "topmost", None),
            "pick the small orange and white box": ("box", None, None),
            "grab the largest blue carton": ("carton", "largest", None),
            "pick the flat orange package on the upper shelf": (
                "package",
                None,
                "upper shelf",
            ),
            "pick the red box from the top shelf and place it in the right bin": (
                "box",
                None,
                "top shelf",
            ),
        }
        for phrase, expected in examples.items():
            with self.subTest(phrase=phrase):
                intent = grounding.parse_grounding_intent(phrase)
                self.assertEqual(
                    (intent["category"], intent["selector"], intent["source_region"]),
                    expected,
                )
                self.assertEqual(intent["schema_version"], 1)
                self.assertEqual(intent["ambiguities"], [])

    def test_attributes_are_typed_and_primary_prompt_is_stable(self) -> None:
        intent = grounding.parse_grounding_intent(
            "pick the small orange and white rectangular cardboard box"
        )
        self.assertEqual(
            intent["attributes"],
            [
                {"type": "size", "value": "small"},
                {"type": "color", "value": "orange"},
                {"type": "color", "value": "white"},
                {"type": "shape", "value": "rectangular"},
                {"type": "material", "value": "cardboard"},
            ],
        )
        self.assertEqual(
            grounding.construct_primary_prompt(intent),
            "small orange and white cardboard rectangular box",
        )

    def test_prompt_family_is_bounded_and_contains_safe_synonyms(self) -> None:
        intent = grounding.parse_grounding_intent(
            "pick the small orange and white box"
        )
        family = grounding.build_prompt_family(intent, max_prompts=5)
        self.assertLessEqual(len(family), 5)
        self.assertEqual(family[0], "small orange and white box")
        self.assertIn("box", family)
        self.assertTrue({"package", "carton"}.intersection(family))

    def test_required_refusals(self) -> None:
        phrases = (
            "pick it up",
            "pick the red box or the blue box",
            "pick the leftmost nearest box",
            "pick something over there",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                with self.assertRaises(grounding.GroundingIntentError):
                    grounding.parse_grounding_intent(phrase)

    def test_locative_bin_is_not_part_of_category(self) -> None:
        intent = grounding.parse_grounding_intent("air filter box in the bin")
        self.assertEqual(intent["category"], "air filter box")
        self.assertEqual(intent["source_region"], "bin")


class IntentValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = grounding.parse_grounding_intent("topmost orange box")

    def test_hash_is_stable_and_identity_sensitive(self) -> None:
        first = grounding.intent_hash(self.intent)
        second = grounding.intent_hash(dict(self.intent))
        self.assertEqual(first, second)
        changed = grounding.parse_grounding_intent("topmost blue box")
        self.assertNotEqual(first, grounding.intent_hash(changed))

    def test_rejects_extra_keys(self) -> None:
        invalid = {**self.intent, "destination": "drop zone"}
        with self.assertRaises(grounding.GroundingIntentError):
            grounding.validate_grounding_intent(invalid)

    def test_rejects_invented_qwen_attribute(self) -> None:
        invalid = {
            **self.intent,
            "attributes": [
                {"type": "color", "value": "orange"},
                {"type": "material", "value": "metal"},
            ],
        }
        with self.assertRaisesRegex(
            grounding.GroundingIntentError,
            "not anchored",
        ):
            grounding.parse_qwen_grounding_intent(
                json.dumps(invalid),
                self.intent["source_phrase"],
            )

    def test_rejects_changed_source_phrase(self) -> None:
        with self.assertRaises(grounding.GroundingIntentError):
            grounding.validate_grounding_intent(
                self.intent,
                expected_source_phrase="bottommost orange box",
            )

    def test_shadow_comparison_names_changed_fields(self) -> None:
        shadow = grounding.parse_grounding_intent("topmost blue box")
        comparison = grounding.compare_intents(self.intent, shadow)
        self.assertFalse(comparison["matches"])
        self.assertIn("attributes", comparison["differing_fields"])


if __name__ == "__main__":
    unittest.main()
