import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_v2_language
import grounding_v2


def test_reviewed_corpus_has_120_cases_all_relationships_and_required_coverage():
    corpus = json.loads(
        (PROJECT_ROOT / "evaluation/v2_language_commands.json").read_text()
    )
    cases = evaluate_v2_language.validate_corpus(corpus)
    assert len(cases) == 120
    assert len({case["case_id"] for case in cases}) == 120
    coverage = {tag for case in cases for tag in case["coverage"]}
    assert {
        "entity_scoped_selectors",
        "attribute_ownership",
        "destination",
        "filler",
        "repeated_head_nouns",
        "whisper_style",
        "three_anchors",
    } <= coverage


def test_language_comparison_checks_attribute_ownership_and_relationship_anchor():
    corpus = json.loads(
        (PROJECT_ROOT / "evaluation/v2_language_commands.json").read_text()
    )
    case = corpus["cases"][0]
    envelope = {
        "schema_version": 2,
        "raw_command": case["command"],
        "visual_source_phrase": "red box inside the blue bin",
        "action": {"type": "identify", "evidence": "find"},
        "destination": None,
        "target": {
            "id": "target",
            "mention": "red box",
            "head_noun": "box",
            "noun_modifiers": [],
            "attributes": [
                {"type": "color", "value": "red", "evidence": "red"}
            ],
            "selector": None,
        },
        "anchors": [
            {
                "id": "anchor_1",
                "mention": "blue bin",
                "head_noun": "bin",
                "noun_modifiers": [],
                "attributes": [
                    {"type": "color", "value": "blue", "evidence": "blue"}
                ],
                "selector": None,
            }
        ],
        "relationships": [
            {
                "type": "inside",
                "target_id": "target",
                "anchor_id": "anchor_1",
                "evidence": "inside",
            }
        ],
    }
    sealed = grounding_v2.seal_command_envelope(envelope)
    assert evaluate_v2_language.compare_case(case, sealed)["correct"]
    sealed["target"]["attributes"], sealed["anchors"][0]["attributes"] = (
        sealed["anchors"][0]["attributes"],
        sealed["target"]["attributes"],
    )
    assert not evaluate_v2_language.compare_case(case, sealed)["correct"]


def test_language_comparison_checks_head_noun_and_noun_modifiers():
    corpus = json.loads(
        (PROJECT_ROOT / "evaluation/v2_language_commands.json").read_text()
    )
    case = next(item for item in corpus["cases"] if item["case_id"] == "inside_06")
    target = case["expected"]["target"]
    assert target["head_noun"] == "box"
    assert target["noun_modifiers"] == ["filter"]
    actual = {
        "action": {"type": case["expected"]["action"]},
        "destination": case["expected"]["destination"],
        "target": {
            "mention": target["mention"],
            "head_noun": "filter",
            "noun_modifiers": ["box"],
            "attributes": [
                {"type": kind, "value": value}
                for kind, values in target["attributes"].items()
                for value in values
            ],
            "selector": None,
        },
        "anchors": [
            {
                "mention": anchor["mention"],
                "head_noun": anchor["head_noun"],
                "noun_modifiers": anchor["noun_modifiers"],
                "attributes": [
                    {"type": kind, "value": value}
                    for kind, values in anchor["attributes"].items()
                    for value in values
                ],
                "selector": None,
            }
            for anchor in case["expected"]["anchors"]
        ],
        "relationships": [
            {
                "type": relationship["type"],
                "anchor_id": f"anchor_{relationship['anchor_index'] + 1}",
                "evidence": relationship["evidence"],
            }
            for relationship in case["expected"]["relationships"]
        ],
    }
    comparison = evaluate_v2_language.compare_case(case, actual)
    assert not comparison["correct"]
    assert "target" in comparison["differences"]
