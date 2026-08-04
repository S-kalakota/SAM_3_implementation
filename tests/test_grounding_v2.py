import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grounding_v2


def base_envelope():
    return {
        "schema_version": 2,
        "raw_command": (
            "Please grab the small red box inside the blue bin and put it in the drop zone"
        ),
        "visual_source_phrase": "small red box inside the blue bin",
        "action": {"type": "pick", "evidence": "grab"},
        "destination": "drop zone",
        "target": {
            "id": "target",
            "mention": "small red box",
            "head_noun": "box",
            "noun_modifiers": [],
            "attributes": [
                {"type": "size", "value": "small", "evidence": "small"},
                {"type": "color", "value": "red", "evidence": "red"},
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


def semantic_value():
    return {
        "action": {"type": "pick", "evidence": "grab"},
        "destination": "drop zone",
        "target": {
            "mention": "small red box",
            "head_noun": "box",
            "noun_modifiers": [],
            "attributes": [
                {"type": "size", "evidence": "small"},
                {"type": "color", "evidence": "red"},
            ],
            "selector": None,
        },
        "anchors": [
            {
                "mention": "blue bin",
                "head_noun": "bin",
                "noun_modifiers": [],
                "attributes": [{"type": "color", "evidence": "blue"}],
                "selector": None,
            }
        ],
        "relationships": [
            {"type": "inside", "anchor_index": 0, "evidence": "inside"}
        ],
    }


def test_seals_and_authenticates_canonical_envelope():
    sealed = grounding_v2.seal_command_envelope(base_envelope())
    assert len(sealed["canonical_intent_hash"]) == 64
    assert len(sealed["envelope_hash"]) == 64
    assert grounding_v2.validate_command_envelope(sealed) == sealed


def test_hashes_cover_visual_intent_and_full_envelope_separately():
    first = grounding_v2.seal_command_envelope(base_envelope())
    changed = base_envelope()
    changed["destination"] = "the drop zone"
    changed["raw_command"] = changed["raw_command"].replace("drop zone", "the drop zone")
    second = grounding_v2.seal_command_envelope(changed)
    assert first["canonical_intent_hash"] == second["canonical_intent_hash"]
    assert first["envelope_hash"] != second["envelope_hash"]


def test_tampered_envelope_is_rejected():
    sealed = grounding_v2.seal_command_envelope(base_envelope())
    sealed["target"]["head_noun"] = "bin"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.validate_command_envelope(sealed)
    assert caught.value.code in {"missing_source_evidence", "grounding_identity_mismatch"}


def test_relationship_aliases_are_normalized_from_declared_evidence():
    value = base_envelope()
    value["raw_command"] = "Find the red box within the blue bin"
    value["visual_source_phrase"] = "red box within the blue bin"
    value["action"] = {"type": "identify", "evidence": "Find"}
    value["destination"] = None
    value["target"]["mention"] = "red box"
    value["target"]["attributes"] = [
        {"type": "color", "value": "red", "evidence": "red"}
    ]
    value["relationships"][0].update({"type": "within", "evidence": "within"})
    sealed = grounding_v2.seal_command_envelope(value)
    assert sealed["relationships"][0]["type"] == "inside"


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("in", "inside"),
        ("inside", "inside"),
        ("within", "inside"),
        ("beside", "next_to"),
        ("on top of", "on"),
        ("to the left of", "left_of"),
        ("in front of", "in_front_of"),
    ],
)
def test_relationship_operator_normalization(source, canonical):
    assert grounding_v2.normalize_relationship(source) == canonical


def test_from_never_proves_containment():
    value = base_envelope()
    value["raw_command"] = "grab the red box from the blue bin"
    value["visual_source_phrase"] = "red box from the blue bin"
    value["destination"] = None
    value["target"]["mention"] = "red box"
    value["target"]["attributes"] = [
        {"type": "color", "value": "red", "evidence": "red"}
    ]
    value["relationships"][0]["evidence"] = "from"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "unsupported_relationship"


def test_visual_phrase_is_resolved_from_earliest_to_latest_visual_evidence():
    value = base_envelope()
    value["visual_source_phrase"] = "the small red box inside the blue bin"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "grounding_identity_mismatch"
    assert caught.value.details["expected"] == "small red box inside the blue bin"


def test_missing_and_repeated_evidence_fail_closed():
    missing = base_envelope()
    missing["target"]["mention"] = "green box"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(missing)
    assert caught.value.code == "missing_source_evidence"

    repeated = base_envelope()
    repeated["raw_command"] = "grab the small red box inside the blue bin, not the small red box"
    repeated["destination"] = None
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(repeated)
    assert caught.value.code == "ambiguous_source_evidence"


def test_target_and_anchor_mentions_cannot_differ_only_by_case():
    value = base_envelope()
    value["raw_command"] = "find red box left of Red Box"
    value["visual_source_phrase"] = "red box left of Red Box"
    value["action"] = {"type": "identify", "evidence": "find"}
    value["destination"] = None
    value["target"]["mention"] = "red box"
    value["target"]["attributes"] = [
        {"type": "color", "value": "red", "evidence": "red"}
    ]
    value["anchors"] = [{
        "id": "anchor_1",
        "mention": "Red Box",
        "head_noun": "Box",
        "noun_modifiers": [],
        "attributes": [
            {"type": "color", "value": "Red", "evidence": "Red"}
        ],
        "selector": None,
    }]
    value["relationships"] = [{
        "type": "left_of",
        "target_id": "target",
        "anchor_id": "anchor_1",
        "evidence": "left of",
    }]
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "ambiguous_source_evidence"


def test_short_operator_evidence_does_not_match_inside_anchor_words():
    value = base_envelope()
    value["raw_command"] = "grab the red box in the blue bin"
    value["visual_source_phrase"] = "red box in the blue bin"
    value["destination"] = None
    value["target"]["mention"] = "red box"
    value["target"]["attributes"] = [
        {"type": "color", "value": "red", "evidence": "red"}
    ]
    value["relationships"][0].update({"type": "in", "evidence": "in"})
    sealed = grounding_v2.seal_command_envelope(value)
    assert sealed["relationships"][0]["type"] == "inside"


def test_unresolved_reference_fails_closed():
    value = {
        "schema_version": 2,
        "raw_command": "pick it",
        "visual_source_phrase": "it",
        "action": {"type": "pick", "evidence": "pick"},
        "destination": None,
        "target": {
            "id": "target",
            "mention": "it",
            "head_noun": "it",
            "noun_modifiers": [],
            "attributes": [],
            "selector": None,
        },
        "anchors": [],
        "relationships": [],
    }
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "unsupported_reference"


def test_entity_scoped_selector_must_be_inside_its_mention():
    value = base_envelope()
    value["target"]["selector"] = {"type": "leftmost", "evidence": "leftmost"}
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "missing_source_evidence"


def test_selector_evidence_must_support_declared_selector():
    value = base_envelope()
    value["raw_command"] = value["raw_command"].replace(
        "small red box", "leftmost small red box"
    )
    value["visual_source_phrase"] = value["visual_source_phrase"].replace(
        "small red box", "leftmost small red box"
    )
    value["target"]["mention"] = "leftmost small red box"
    value["target"]["selector"] = {
        "type": "rightmost",
        "evidence": "leftmost",
    }
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(value)
    assert caught.value.code == "invented_source_evidence"


def test_from_can_introduce_context_anchor_without_claiming_containment():
    value = base_envelope()
    value["raw_command"] = "grab the red box from the top shelf"
    value["visual_source_phrase"] = "red box from the top shelf"
    value["destination"] = None
    value["target"]["mention"] = "red box"
    value["target"]["attributes"] = [
        {"type": "color", "value": "red", "evidence": "red"}
    ]
    value["anchors"] = [
        {
            "id": "anchor_1",
            "mention": "top shelf",
            "head_noun": "shelf",
            "noun_modifiers": ["top"],
            "attributes": [],
            "selector": None,
        }
    ]
    value["relationships"] = []
    sealed = grounding_v2.seal_command_envelope(value)
    assert sealed["anchors"][0]["mention"] == "top shelf"
    assert sealed["relationships"] == []


def test_anchor_complexity_limit_fails_closed():
    too_many = base_envelope()
    too_many["anchors"].extend([copy.deepcopy(too_many["anchors"][0])] * 3)
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(too_many)
    assert caught.value.code == "request_complexity_limit"


def test_prompt_family_is_open_vocabulary_bounded_and_deterministic():
    entity = {
        "id": "target",
        "mention": "striped orange air filter box",
        "head_noun": "box",
        "noun_modifiers": ["air filter"],
        "attributes": [
            {"type": "pattern", "value": "striped", "evidence": "striped"},
            {"type": "color", "value": "orange", "evidence": "orange"},
        ],
        "selector": None,
    }
    assert grounding_v2.build_entity_prompts(entity) == [
        "striped orange air filter box",
        "striped orange box",
        "air filter box",
        "box",
    ]


def test_qwen_semantics_build_the_existing_public_envelope_deterministically():
    actual = grounding_v2.envelope_from_qwen_semantics(
        semantic_value(),
        base_envelope()["raw_command"],
    )
    expected = grounding_v2.seal_command_envelope(base_envelope())
    assert actual == expected


def test_qwen_semantics_preserve_attribute_evidence_case_as_value():
    semantics = semantic_value()
    semantics["target"] = {
        "mention": "box marked A17",
        "head_noun": "box",
        "noun_modifiers": [],
        "attributes": [{"type": "marking", "evidence": "A17"}],
        "selector": None,
    }
    semantics["anchors"] = []
    semantics["relationships"] = []
    semantics["destination"] = None
    semantics["action"] = {"type": "identify", "evidence": "find"}
    envelope = grounding_v2.envelope_from_qwen_semantics(
        semantics,
        "find the box marked A17",
    )
    assert envelope["target"]["attributes"][0] == {
        "type": "marking",
        "value": "A17",
        "evidence": "A17",
    }


@pytest.mark.parametrize("missing", ["selector", "attributes", "noun_modifiers"])
def test_qwen_semantic_entity_requires_every_field(missing):
    semantics = semantic_value()
    del semantics["target"][missing]
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.envelope_from_qwen_semantics(
            semantics,
            base_envelope()["raw_command"],
        )
    assert caught.value.retryable_format


def test_qwen_semantics_reject_deterministic_fields_and_bad_anchor_indexes():
    extra = semantic_value()
    extra["raw_command"] = base_envelope()["raw_command"]
    with pytest.raises(grounding_v2.GroundingV2Error):
        grounding_v2.envelope_from_qwen_semantics(
            extra,
            base_envelope()["raw_command"],
        )

    invalid_reference = semantic_value()
    invalid_reference["relationships"][0]["anchor_index"] = 2
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.envelope_from_qwen_semantics(
            invalid_reference,
            base_envelope()["raw_command"],
        )
    assert caught.value.code == "invalid_entity_reference"


@pytest.mark.parametrize("modifier", ["red", "small red"])
def test_qwen_semantics_reject_cross_role_evidence_duplication(modifier):
    semantics = semantic_value()
    semantics["target"]["noun_modifiers"] = [modifier]
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.envelope_from_qwen_semantics(
            semantics,
            base_envelope()["raw_command"],
        )
    assert caught.value.code == "invalid_attribute_partition"
    assert grounding_v2.qwen_interpretation_error_is_retryable(caught.value)


def test_public_envelope_rejects_cross_role_phrase_overlap():
    envelope = base_envelope()
    envelope["target"]["noun_modifiers"] = ["small red"]
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.seal_command_envelope(envelope)

    assert caught.value.code == "invalid_attribute_partition"
    assert caught.value.details["conflicting_evidence"] == "small red"


def test_direct_qwen_semantic_conversion_enforces_noun_modifier_limit():
    modifiers = [
        f"modifier{index}" for index in range(grounding_v2.MAX_NOUN_MODIFIERS)
    ]
    semantics = {
        "action": {"type": "identify", "evidence": "find"},
        "destination": None,
        "target": {
            "mention": " ".join([*modifiers, "widget"]),
            "head_noun": "widget",
            "noun_modifiers": modifiers,
            "attributes": [],
            "selector": None,
        },
        "anchors": [],
        "relationships": [],
    }
    raw_command = f"find the {semantics['target']['mention']}"

    envelope = grounding_v2.envelope_from_qwen_semantics(semantics, raw_command)
    assert len(envelope["target"]["noun_modifiers"]) == grounding_v2.MAX_NOUN_MODIFIERS

    extra_modifier = "overflowmodifier"
    semantics["target"]["noun_modifiers"] = [*modifiers, extra_modifier]
    semantics["target"]["mention"] = " ".join(
        [*semantics["target"]["noun_modifiers"], "widget"]
    )
    raw_command = f"find the {semantics['target']['mention']}"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.envelope_from_qwen_semantics(semantics, raw_command)

    assert caught.value.code == "request_complexity_limit"
    assert caught.value.details == {
        "entity_id": "target",
        "noun_modifiers": grounding_v2.MAX_NOUN_MODIFIERS + 1,
        "limit": grounding_v2.MAX_NOUN_MODIFIERS,
    }


def test_direct_qwen_semantic_conversion_enforces_attribute_limit():
    typed_evidence = [
        ("color", "red"),
        ("size", "small"),
        ("shape", "round"),
        ("material", "metal"),
        ("marking", "A17"),
        ("pattern", "striped"),
        ("text", "OPEN"),
        ("state", "sealed"),
    ]
    assert len(typed_evidence) == grounding_v2.MAX_ATTRIBUTES_PER_ENTITY
    attributes = [
        {"type": attribute_type, "evidence": evidence}
        for attribute_type, evidence in typed_evidence
    ]
    semantics = {
        "action": {"type": "identify", "evidence": "find"},
        "destination": None,
        "target": {
            "mention": " ".join([*(item["evidence"] for item in attributes), "widget"]),
            "head_noun": "widget",
            "noun_modifiers": [],
            "attributes": attributes,
            "selector": None,
        },
        "anchors": [],
        "relationships": [],
    }
    raw_command = f"find the {semantics['target']['mention']}"

    envelope = grounding_v2.envelope_from_qwen_semantics(semantics, raw_command)
    assert len(envelope["target"]["attributes"]) == grounding_v2.MAX_ATTRIBUTES_PER_ENTITY

    semantics["target"]["attributes"] = [
        *attributes,
        {"type": "color", "evidence": "green"},
    ]
    semantics["target"]["mention"] = " ".join(
        [
            *(item["evidence"] for item in semantics["target"]["attributes"]),
            "widget",
        ]
    )
    raw_command = f"find the {semantics['target']['mention']}"
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.envelope_from_qwen_semantics(semantics, raw_command)

    assert caught.value.code == "request_complexity_limit"
    assert caught.value.details == {
        "entity_id": "target",
        "attributes": grounding_v2.MAX_ATTRIBUTES_PER_ENTITY + 1,
        "limit": grounding_v2.MAX_ATTRIBUTES_PER_ENTITY,
    }


def test_qwen_semantic_schema_requires_shape_and_bounds_arrays():
    schema = grounding_v2.qwen_semantic_json_schema()
    validate(semantic_value(), schema)
    missing = semantic_value()
    del missing["target"]["selector"]
    with pytest.raises(ValidationError):
        validate(missing, schema)
    extra = semantic_value()
    extra["schema_version"] = 2
    with pytest.raises(ValidationError):
        validate(extra, schema)
    too_many = semantic_value()
    too_many["anchors"] *= grounding_v2.MAX_ANCHORS + 1
    with pytest.raises(ValidationError):
        validate(too_many, schema)


def test_qwen_prompt_uses_reduced_contract_examples_and_fresh_correction():
    messages = grounding_v2.qwen_interpretation_messages(
        "find the red box",
        correction={
            "code": "missing_source_evidence",
            "message": "bad evidence",
            "details": {"field": "target.mention"},
        },
    )
    assert len(messages) == 2
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "small orange box" in system
    assert "striped air filter box" in system
    assert "rightmost metal bolt" in system
    assert "Do not output schema_version, raw_command" in system
    assert "missing_source_evidence" in user
    assert "find the red box" in user


def test_qwen_semantic_parser_requires_strict_json():
    text = json.dumps(semantic_value())
    envelope = grounding_v2.parse_qwen_semantic_interpretation(
        text,
        base_envelope()["raw_command"],
    )
    assert envelope["schema_version"] == 2
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.parse_qwen_semantic_interpretation(
            f"```json\n{text}\n```",
            base_envelope()["raw_command"],
        )
    assert caught.value.code == "invalid_interpretation_json"
    assert caught.value.retryable_format


def test_qwen_parser_requires_strict_json_and_exact_raw_command():
    value = base_envelope()
    text = json.dumps(value)
    sealed = grounding_v2.parse_qwen_interpretation(text, value["raw_command"])
    assert sealed["schema_version"] == 2
    with pytest.raises(grounding_v2.GroundingV2Error) as caught:
        grounding_v2.parse_qwen_interpretation(f"```json\n{text}\n```", value["raw_command"])
    assert caught.value.code == "invalid_interpretation_json"
    assert caught.value.retryable_format
