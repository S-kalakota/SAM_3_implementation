#!/usr/bin/env python3
"""Fail-closed version-2 language contract for relational visual grounding.

Qwen is the only component that interprets the command.  The helpers in this
module validate the model's structured claim against literal source evidence,
normalise allowlisted operator names, enforce complexity limits, and seal the
result with stable hashes.  They deliberately do not contain a sentence parser.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


SCHEMA_VERSION = 2
MAX_ANCHORS = 3
MAX_RELATIONSHIPS = 4
MAX_PROMPTS_PER_ENTITY = 4
MAX_BOXES_PER_ENTITY = 4
MAX_NOUN_MODIFIERS = 4
MAX_ATTRIBUTES_PER_ENTITY = 8
MAX_QWEN_EVIDENCE_CHARS = 256
QWEN_SEMANTIC_CONTRACT_VERSION = 1

BASE_ENVELOPE_KEYS = {
    "schema_version",
    "raw_command",
    "visual_source_phrase",
    "action",
    "destination",
    "target",
    "anchors",
    "relationships",
}
HASH_KEYS = {"canonical_intent_hash", "envelope_hash"}
SEALED_ENVELOPE_KEYS = BASE_ENVELOPE_KEYS | HASH_KEYS
ACTION_KEYS = {"type", "evidence"}
ENTITY_KEYS = {
    "id",
    "mention",
    "head_noun",
    "noun_modifiers",
    "attributes",
    "selector",
}
ATTRIBUTE_KEYS = {"type", "value", "evidence"}
SELECTOR_KEYS = {"type", "evidence"}
RELATIONSHIP_KEYS = {"type", "target_id", "anchor_id", "evidence"}
QWEN_SEMANTIC_KEYS = {
    "action",
    "destination",
    "target",
    "anchors",
    "relationships",
}
QWEN_ENTITY_KEYS = {
    "mention",
    "head_noun",
    "noun_modifiers",
    "attributes",
    "selector",
}
QWEN_ATTRIBUTE_KEYS = {"type", "evidence"}
QWEN_RELATIONSHIP_KEYS = {"type", "anchor_index", "evidence"}

VALID_ACTIONS = {"identify", "pick", "move", "place"}
ACTION_EVIDENCE_ALIASES = {
    "identify": {
        "find",
        "identify",
        "choose",
        "detect",
        "locate",
        "look for",
        "point out",
        "select",
        "segment",
        "show",
        "show me",
    },
    "pick": {"get", "grab", "pick", "pick up", "take"},
    "move": {"bring", "move", "transfer"},
    "place": {"drop", "place", "put", "set"},
}

ATTRIBUTE_TYPES = {
    "color",
    "size",
    "shape",
    "material",
    "marking",
    "pattern",
    "text",
    "state",
}
VALID_SELECTORS = {
    "rightmost",
    "leftmost",
    "topmost",
    "bottommost",
    "nearest",
    "farthest",
    "largest",
    "smallest",
}
SELECTOR_EVIDENCE_ALIASES = {
    "rightmost": {"rightmost", "right most", "far right"},
    "leftmost": {"leftmost", "left most", "far left"},
    "topmost": {"topmost", "top most", "uppermost"},
    "bottommost": {"bottommost", "bottom most", "lowermost"},
    "nearest": {"nearest", "closest"},
    "farthest": {"farthest", "furthest"},
    "largest": {"largest", "biggest"},
    "smallest": {"smallest"},
}
VALID_RELATIONSHIPS = {
    "inside",
    "on",
    "left_of",
    "right_of",
    "above",
    "below",
    "near",
    "next_to",
    "in_front_of",
    "behind",
}
RELATIONSHIP_ALIASES = {
    "in": "inside",
    "inside": "inside",
    "within": "inside",
    "on": "on",
    "on top of": "on",
    "atop": "on",
    "left of": "left_of",
    "to the left of": "left_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "near": "near",
    "close to": "near",
    "next to": "next_to",
    "beside": "next_to",
    "in front of": "in_front_of",
    "behind": "behind",
}

UNRESOLVED_REFERENCE = re.compile(
    r"^(?:it|this|that|one|something|anything|thing|stuff|object|item)$",
    re.IGNORECASE,
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GroundingV2Error(ValueError):
    """A stable, API-safe version-2 refusal."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_command_envelope",
        details: dict[str, Any] | None = None,
        retryable_format: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.retryable_format = retryable_format


def qwen_semantic_json_schema() -> dict[str, Any]:
    """Return the bounded schema enforced during Qwen interpretation decoding."""

    nullable_string = {
        "anyOf": [
            {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_QWEN_EVIDENCE_CHARS,
            },
            {"type": "null"},
        ]
    }
    selector = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "evidence"],
                "properties": {
                    "type": {"type": "string", "enum": sorted(VALID_SELECTORS)},
                    "evidence": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QWEN_EVIDENCE_CHARS,
                    },
                },
            },
        ]
    }
    attribute = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "evidence"],
        "properties": {
            "type": {"type": "string", "enum": sorted(ATTRIBUTE_TYPES)},
            "evidence": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_QWEN_EVIDENCE_CHARS,
            },
        },
    }

    def entity_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "mention",
                "head_noun",
                "noun_modifiers",
                "attributes",
                "selector",
            ],
            "properties": {
                "mention": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QWEN_EVIDENCE_CHARS,
                },
                "head_noun": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QWEN_EVIDENCE_CHARS,
                },
                "noun_modifiers": {
                    "type": "array",
                    "maxItems": MAX_NOUN_MODIFIERS,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QWEN_EVIDENCE_CHARS,
                    },
                },
                "attributes": {
                    "type": "array",
                    "maxItems": MAX_ATTRIBUTES_PER_ENTITY,
                    "items": attribute,
                },
                "selector": selector,
            },
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "destination", "target", "anchors", "relationships"],
        "properties": {
            "action": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "evidence"],
                "properties": {
                    "type": {"type": "string", "enum": sorted(VALID_ACTIONS)},
                    "evidence": nullable_string,
                },
            },
            "destination": nullable_string,
            "target": entity_schema(),
            "anchors": {
                "type": "array",
                "maxItems": MAX_ANCHORS,
                "items": entity_schema(),
            },
            "relationships": {
                "type": "array",
                "maxItems": MAX_RELATIONSHIPS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "anchor_index", "evidence"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": sorted(VALID_RELATIONSHIPS),
                        },
                        "anchor_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_ANCHORS - 1,
                        },
                        "evidence": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QWEN_EVIDENCE_CHARS,
                        },
                    },
                },
            },
        },
    }


def collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_text(value: str) -> str:
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return collapse_space(value)


def _normalized_terms(value: str) -> set[str]:
    """Return normalized whole-word terms for cross-role evidence checks."""

    return set(normalize_text(value).split())


def _validate_entity_role_partition(
    *,
    entity_id: str,
    head_noun: str,
    noun_modifiers: list[str],
    attribute_evidence: list[str],
    selector_evidence: str | None,
) -> None:
    """Refuse semantic evidence duplicated across mutually exclusive roles."""

    lexical_roles = [("head_noun", head_noun)]
    lexical_roles.extend(
        ("noun_modifier", modifier) for modifier in noun_modifiers
    )
    for evidence in attribute_evidence:
        evidence_terms = _normalized_terms(evidence)
        for role, lexical_value in lexical_roles:
            if evidence_terms & _normalized_terms(lexical_value):
                raise GroundingV2Error(
                    "attribute evidence must not overlap a noun modifier or the head noun",
                    code="invalid_attribute_partition",
                    details={
                        "entity_id": entity_id,
                        "attribute_evidence": evidence,
                        "conflicting_role": role,
                        "conflicting_evidence": lexical_value,
                    },
                )

    if selector_evidence is None:
        return
    selector_terms = _normalized_terms(selector_evidence)
    selector_conflicts = list(lexical_roles)
    selector_conflicts.extend(
        ("attribute", evidence) for evidence in attribute_evidence
    )
    for role, conflicting_evidence in selector_conflicts:
        if selector_terms & _normalized_terms(conflicting_evidence):
            raise GroundingV2Error(
                "selector evidence must not overlap an attribute, noun modifier, or the head noun",
                code="invalid_attribute_partition",
                details={
                    "entity_id": entity_id,
                    "selector_evidence": selector_evidence,
                    "conflicting_role": role,
                    "conflicting_evidence": conflicting_evidence,
                },
            )


def normalize_relationship(value: str) -> str:
    """Normalize an explicitly supplied relationship name or evidence phrase."""

    if not isinstance(value, str) or not normalize_text(value):
        raise GroundingV2Error(
            "relationship operator must be a non-empty string",
            retryable_format=True,
        )
    normalized = normalize_text(value)
    canonical = RELATIONSHIP_ALIASES.get(normalized)
    if canonical is None and value in VALID_RELATIONSHIPS:
        canonical = value
    if canonical is None:
        raise GroundingV2Error(
            f"unsupported relationship operator {value!r}",
            code="unsupported_relationship",
        )
    return canonical


def _schema_error(message: str, *, details: dict[str, Any] | None = None) -> GroundingV2Error:
    return GroundingV2Error(
        message,
        details=details,
        retryable_format=True,
    )


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _schema_error(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise _schema_error(
            f"{label} keys are invalid",
            details={"missing": sorted(keys - actual), "extra": sorted(actual - keys)},
        )
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _schema_error(f"{label} must be a non-empty string")
    return value


def _literal_occurrences(source: str, evidence: str) -> list[tuple[int, int]]:
    escaped = re.escape(evidence)
    if evidence[0].isalnum():
        escaped = rf"(?<![A-Za-z0-9]){escaped}"
    if evidence[-1].isalnum():
        escaped = rf"{escaped}(?![A-Za-z0-9])"
    return [(match.start(), match.end()) for match in re.finditer(escaped, source)]


def resolve_unique_evidence(
    source: str,
    evidence: Any,
    *,
    field: str,
) -> tuple[int, int]:
    """Resolve literal copied evidence, refusing absent or repeated text."""

    evidence = _nonempty_string(evidence, field)
    spans = _literal_occurrences(source, evidence)
    if not spans:
        raise GroundingV2Error(
            f"{field} is not copied exactly from its source text",
            code="missing_source_evidence",
            details={"field": field, "evidence": evidence},
        )
    if len(spans) != 1:
        raise GroundingV2Error(
            f"{field} occurs more than once and is ambiguous",
            code="ambiguous_source_evidence",
            details={"field": field, "evidence": evidence, "occurrences": len(spans)},
        )
    return spans[0]


def _validate_action(value: Any, raw_command: str) -> dict[str, Any]:
    value = _require_exact_keys(value, ACTION_KEYS, "action")
    action_type = value["type"]
    if not isinstance(action_type, str) or action_type not in VALID_ACTIONS:
        raise GroundingV2Error(
            f"unsupported action {action_type!r}",
            code="unsupported_action",
        )
    evidence = value["evidence"]
    if evidence is None:
        if action_type != "identify":
            raise GroundingV2Error(
                f"action {action_type!r} requires source evidence",
                code="missing_source_evidence",
            )
    else:
        resolve_unique_evidence(raw_command, evidence, field="action.evidence")
        if normalize_text(evidence) not in ACTION_EVIDENCE_ALIASES[action_type]:
            raise GroundingV2Error(
                "action evidence does not support the declared action",
                code="unsupported_action",
                details={"type": action_type, "evidence": evidence},
            )
    return {"type": action_type, "evidence": evidence}


def _validate_selector(value: Any, mention: str, entity_id: str) -> dict[str, str] | None:
    if value is None:
        return None
    value = _require_exact_keys(value, SELECTOR_KEYS, f"{entity_id}.selector")
    selector_type = value["type"]
    if not isinstance(selector_type, str) or selector_type not in VALID_SELECTORS:
        raise GroundingV2Error(
            f"unsupported selector {selector_type!r}",
            code="unsupported_selector",
        )
    evidence = _nonempty_string(value["evidence"], f"{entity_id}.selector.evidence")
    resolve_unique_evidence(
        mention,
        evidence,
        field=f"{entity_id}.selector.evidence",
    )
    if normalize_text(evidence) not in SELECTOR_EVIDENCE_ALIASES[selector_type]:
        raise GroundingV2Error(
            "selector evidence does not support the declared selector",
            code="invented_source_evidence",
            details={"type": selector_type, "evidence": evidence},
        )
    return {"type": selector_type, "evidence": evidence}


def _validate_entity(
    value: Any,
    *,
    expected_id: str,
    raw_command: str,
) -> tuple[dict[str, Any], tuple[int, int]]:
    value = _require_exact_keys(value, ENTITY_KEYS, expected_id)
    entity_id = value["id"]
    if entity_id != expected_id:
        raise _schema_error(
            f"entity id must be {expected_id!r}",
            details={"actual": entity_id},
        )

    mention = _nonempty_string(value["mention"], f"{entity_id}.mention")
    mention_span = resolve_unique_evidence(
        raw_command,
        mention,
        field=f"{entity_id}.mention",
    )
    head_noun = _nonempty_string(value["head_noun"], f"{entity_id}.head_noun")
    resolve_unique_evidence(mention, head_noun, field=f"{entity_id}.head_noun")
    if UNRESOLVED_REFERENCE.fullmatch(normalize_text(head_noun)):
        raise GroundingV2Error(
            f"{entity_id} contains an unresolved reference",
            code="unsupported_reference",
        )

    raw_modifiers = value["noun_modifiers"]
    if not isinstance(raw_modifiers, list):
        raise _schema_error(f"{entity_id}.noun_modifiers must be a list")
    if len(raw_modifiers) > MAX_NOUN_MODIFIERS:
        raise GroundingV2Error(
            f"at most {MAX_NOUN_MODIFIERS} noun modifiers are allowed per entity",
            code="request_complexity_limit",
            details={
                "entity_id": entity_id,
                "noun_modifiers": len(raw_modifiers),
                "limit": MAX_NOUN_MODIFIERS,
            },
        )
    modifiers: list[str] = []
    seen_modifiers: set[str] = set()
    for index, raw_modifier in enumerate(raw_modifiers):
        modifier = _nonempty_string(
            raw_modifier,
            f"{entity_id}.noun_modifiers[{index}]",
        )
        key = normalize_text(modifier)
        if key in seen_modifiers:
            raise GroundingV2Error(f"duplicate noun modifier {modifier!r}")
        resolve_unique_evidence(
            mention,
            modifier,
            field=f"{entity_id}.noun_modifiers[{index}]",
        )
        seen_modifiers.add(key)
        modifiers.append(modifier)

    raw_attributes = value["attributes"]
    if not isinstance(raw_attributes, list):
        raise _schema_error(f"{entity_id}.attributes must be a list")
    if len(raw_attributes) > MAX_ATTRIBUTES_PER_ENTITY:
        raise GroundingV2Error(
            f"at most {MAX_ATTRIBUTES_PER_ENTITY} attributes are allowed per entity",
            code="request_complexity_limit",
            details={
                "entity_id": entity_id,
                "attributes": len(raw_attributes),
                "limit": MAX_ATTRIBUTES_PER_ENTITY,
            },
        )
    attributes: list[dict[str, str]] = []
    seen_attributes: set[tuple[str, str]] = set()
    for index, raw_attribute in enumerate(raw_attributes):
        attribute = _require_exact_keys(
            raw_attribute,
            ATTRIBUTE_KEYS,
            f"{entity_id}.attributes[{index}]",
        )
        attribute_type = attribute["type"]
        if not isinstance(attribute_type, str) or attribute_type not in ATTRIBUTE_TYPES:
            raise GroundingV2Error(
                f"unsupported attribute type {attribute_type!r}",
                code="unsupported_attribute_type",
            )
        attribute_value = _nonempty_string(
            attribute["value"],
            f"{entity_id}.attributes[{index}].value",
        )
        evidence = _nonempty_string(
            attribute["evidence"],
            f"{entity_id}.attributes[{index}].evidence",
        )
        resolve_unique_evidence(
            mention,
            evidence,
            field=f"{entity_id}.attributes[{index}].evidence",
        )
        if normalize_text(attribute_value) != normalize_text(evidence):
            raise GroundingV2Error(
                "attribute value must be the normalized copied evidence",
                code="invented_source_evidence",
                details={
                    "entity_id": entity_id,
                    "type": attribute_type,
                    "value": attribute_value,
                    "evidence": evidence,
                },
            )
        pair = (attribute_type, normalize_text(attribute_value))
        if pair in seen_attributes:
            raise GroundingV2Error(f"duplicate attribute {pair!r}")
        seen_attributes.add(pair)
        attributes.append(
            {"type": attribute_type, "value": attribute_value, "evidence": evidence}
        )

    selector = _validate_selector(value["selector"], mention, entity_id)
    _validate_entity_role_partition(
        entity_id=entity_id,
        head_noun=head_noun,
        noun_modifiers=modifiers,
        attribute_evidence=[attribute["evidence"] for attribute in attributes],
        selector_evidence=None if selector is None else selector["evidence"],
    )
    return (
        {
            "id": entity_id,
            "mention": mention,
            "head_noun": head_noun,
            "noun_modifiers": modifiers,
            "attributes": attributes,
            "selector": selector,
        },
        mention_span,
    )


def _validate_relationship(
    value: Any,
    *,
    index: int,
    raw_command: str,
    anchor_ids: set[str],
) -> tuple[dict[str, str], tuple[int, int]]:
    label = f"relationships[{index}]"
    value = _require_exact_keys(value, RELATIONSHIP_KEYS, label)
    if value["target_id"] != "target":
        raise _schema_error(f"{label}.target_id must be 'target'")
    anchor_id = value["anchor_id"]
    if anchor_id not in anchor_ids:
        raise GroundingV2Error(
            f"{label} references unknown anchor {anchor_id!r}",
            code="invalid_entity_reference",
        )
    declared = normalize_relationship(value["type"])
    evidence = _nonempty_string(value["evidence"], f"{label}.evidence")
    span = resolve_unique_evidence(raw_command, evidence, field=f"{label}.evidence")
    # Evidence is deliberately restricted to an operator phrase.  In particular,
    # "from" is never accepted as proof of containment.
    evidenced = normalize_relationship(evidence)
    if evidenced != declared:
        raise GroundingV2Error(
            "relationship evidence does not support the declared operator",
            code="invented_source_evidence",
            details={"declared": declared, "evidence": evidence},
        )
    return {
        "type": declared,
        "target_id": "target",
        "anchor_id": anchor_id,
        "evidence": evidence,
    }, span


def _base_envelope(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in BASE_ENVELOPE_KEYS}


def validate_command_envelope(
    value: Any,
    *,
    expected_raw_command: str | None = None,
    require_hashes: bool = True,
) -> dict[str, Any]:
    """Validate, canonicalise, and optionally authenticate one v2 envelope."""

    expected_keys = SEALED_ENVELOPE_KEYS if require_hashes else BASE_ENVELOPE_KEYS
    value = _require_exact_keys(value, expected_keys, "command envelope")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != SCHEMA_VERSION:
        raise _schema_error("command envelope schema_version must be 2")
    raw_command = _nonempty_string(value["raw_command"], "raw_command")
    if expected_raw_command is not None and raw_command != expected_raw_command:
        raise GroundingV2Error(
            "raw_command does not match the interpretation request",
            code="grounding_identity_mismatch",
            details={"expected": expected_raw_command, "actual": raw_command},
        )

    action = _validate_action(value["action"], raw_command)
    destination = value["destination"]
    if destination is not None:
        destination = _nonempty_string(destination, "destination")
        resolve_unique_evidence(raw_command, destination, field="destination")

    visual_spans: list[tuple[int, int]] = []
    target, target_span = _validate_entity(
        value["target"],
        expected_id="target",
        raw_command=raw_command,
    )
    visual_spans.append(target_span)

    raw_anchors = value["anchors"]
    if not isinstance(raw_anchors, list):
        raise _schema_error("anchors must be a list")
    if len(raw_anchors) > MAX_ANCHORS:
        raise GroundingV2Error(
            f"at most {MAX_ANCHORS} anchors are allowed",
            code="request_complexity_limit",
            details={"anchors": len(raw_anchors), "limit": MAX_ANCHORS},
        )
    anchors: list[dict[str, Any]] = []
    for index, raw_anchor in enumerate(raw_anchors, start=1):
        expected_id = f"anchor_{index}"
        anchor, span = _validate_entity(
            raw_anchor,
            expected_id=expected_id,
            raw_command=raw_command,
        )
        anchors.append(anchor)
        visual_spans.append(span)
    mentions = [target["mention"], *(anchor["mention"] for anchor in anchors)]
    normalized_mentions = [normalize_text(mention) for mention in mentions]
    if len(normalized_mentions) != len(set(normalized_mentions)):
        raise GroundingV2Error(
            "target and anchor mentions must be independently identifiable",
            code="ambiguous_source_evidence",
        )

    raw_relationships = value["relationships"]
    if not isinstance(raw_relationships, list):
        raise _schema_error("relationships must be a list")
    if len(raw_relationships) > MAX_RELATIONSHIPS:
        raise GroundingV2Error(
            f"at most {MAX_RELATIONSHIPS} relationships are allowed",
            code="request_complexity_limit",
            details={
                "relationships": len(raw_relationships),
                "limit": MAX_RELATIONSHIPS,
            },
        )
    anchor_ids = {anchor["id"] for anchor in anchors}
    relationships: list[dict[str, str]] = []
    seen_relationships: set[tuple[str, str]] = set()
    for index, raw_relationship in enumerate(raw_relationships):
        relationship, span = _validate_relationship(
            raw_relationship,
            index=index,
            raw_command=raw_command,
            anchor_ids=anchor_ids,
        )
        pair = (relationship["type"], relationship["anchor_id"])
        if pair in seen_relationships:
            raise GroundingV2Error(f"duplicate relationship {pair!r}")
        seen_relationships.add(pair)
        relationships.append(relationship)
        visual_spans.append(span)

    visual_start = min(start for start, _end in visual_spans)
    visual_end = max(end for _start, end in visual_spans)
    resolved_visual_phrase = raw_command[visual_start:visual_end]
    supplied_visual_phrase = _nonempty_string(
        value["visual_source_phrase"],
        "visual_source_phrase",
    )
    if supplied_visual_phrase != resolved_visual_phrase:
        raise GroundingV2Error(
            "visual_source_phrase does not equal the deterministic evidence span",
            code="grounding_identity_mismatch",
            details={
                "expected": resolved_visual_phrase,
                "actual": supplied_visual_phrase,
                "span": [visual_start, visual_end],
            },
        )

    canonical: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "raw_command": raw_command,
        "visual_source_phrase": resolved_visual_phrase,
        "action": action,
        "destination": destination,
        "target": target,
        "anchors": anchors,
        "relationships": relationships,
    }
    if require_hashes:
        supplied_intent_hash = value["canonical_intent_hash"]
        supplied_envelope_hash = value["envelope_hash"]
        if not isinstance(supplied_intent_hash, str) or not HEX_SHA256.fullmatch(
            supplied_intent_hash
        ):
            raise _schema_error("canonical_intent_hash must be a lowercase SHA-256")
        if not isinstance(supplied_envelope_hash, str) or not HEX_SHA256.fullmatch(
            supplied_envelope_hash
        ):
            raise _schema_error("envelope_hash must be a lowercase SHA-256")
        expected_intent_hash = canonical_intent_hash(canonical)
        expected_envelope_hash = command_envelope_hash(canonical)
        if supplied_intent_hash != expected_intent_hash or supplied_envelope_hash != expected_envelope_hash:
            raise GroundingV2Error(
                "command envelope hashes do not match its contents",
                code="grounding_identity_mismatch",
                details={
                    "canonical_intent_hash_matches": supplied_intent_hash
                    == expected_intent_hash,
                    "envelope_hash_matches": supplied_envelope_hash
                    == expected_envelope_hash,
                },
            )
        canonical.update(
            {
                "canonical_intent_hash": expected_intent_hash,
                "envelope_hash": expected_envelope_hash,
            }
        )
    return canonical


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_intent(envelope: dict[str, Any]) -> dict[str, Any]:
    canonical = validate_command_envelope(
        _base_envelope(envelope),
        require_hashes=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "visual_source_phrase": canonical["visual_source_phrase"],
        "target": canonical["target"],
        "anchors": canonical["anchors"],
        "relationships": canonical["relationships"],
    }


def canonical_intent_hash(envelope: dict[str, Any]) -> str:
    return _sha256_json(canonical_intent(envelope))


def command_envelope_hash(envelope: dict[str, Any]) -> str:
    base = validate_command_envelope(
        _base_envelope(envelope),
        require_hashes=False,
    )
    return _sha256_json(
        {
            "command_envelope": base,
            "canonical_intent_hash": canonical_intent_hash(base),
        }
    )


def seal_command_envelope(
    value: Any,
    *,
    expected_raw_command: str | None = None,
) -> dict[str, Any]:
    canonical = validate_command_envelope(
        value,
        expected_raw_command=expected_raw_command,
        require_hashes=False,
    )
    canonical["canonical_intent_hash"] = canonical_intent_hash(canonical)
    canonical["envelope_hash"] = command_envelope_hash(canonical)
    return canonical


def _qwen_entity_to_envelope_entity(
    value: Any,
    *,
    entity_id: str,
) -> dict[str, Any]:
    value = _require_exact_keys(value, QWEN_ENTITY_KEYS, entity_id)
    raw_attributes = value["attributes"]
    if not isinstance(raw_attributes, list):
        raise _schema_error(f"{entity_id}.attributes must be a list")
    if len(raw_attributes) > MAX_ATTRIBUTES_PER_ENTITY:
        raise GroundingV2Error(
            f"at most {MAX_ATTRIBUTES_PER_ENTITY} attributes are allowed per entity",
            code="request_complexity_limit",
            details={
                "entity_id": entity_id,
                "attributes": len(raw_attributes),
                "limit": MAX_ATTRIBUTES_PER_ENTITY,
            },
        )
    attributes: list[dict[str, str]] = []
    for index, raw_attribute in enumerate(raw_attributes):
        attribute = _require_exact_keys(
            raw_attribute,
            QWEN_ATTRIBUTE_KEYS,
            f"{entity_id}.attributes[{index}]",
        )
        evidence = _nonempty_string(
            attribute["evidence"],
            f"{entity_id}.attributes[{index}].evidence",
        )
        attributes.append(
            {
                "type": attribute["type"],
                "value": evidence,
                "evidence": evidence,
            }
        )

    raw_modifiers = value["noun_modifiers"]
    if not isinstance(raw_modifiers, list):
        raise _schema_error(f"{entity_id}.noun_modifiers must be a list")
    if len(raw_modifiers) > MAX_NOUN_MODIFIERS:
        raise GroundingV2Error(
            f"at most {MAX_NOUN_MODIFIERS} noun modifiers are allowed per entity",
            code="request_complexity_limit",
            details={
                "entity_id": entity_id,
                "noun_modifiers": len(raw_modifiers),
                "limit": MAX_NOUN_MODIFIERS,
            },
        )
    selector = value["selector"]
    _validate_entity_role_partition(
        entity_id=entity_id,
        head_noun=value["head_noun"] if isinstance(value["head_noun"], str) else "",
        noun_modifiers=[item for item in raw_modifiers if isinstance(item, str)],
        attribute_evidence=[attribute["evidence"] for attribute in attributes],
        selector_evidence=(
            selector.get("evidence")
            if isinstance(selector, dict) and isinstance(selector.get("evidence"), str)
            else None
        ),
    )

    return {
        "id": entity_id,
        "mention": value["mention"],
        "head_noun": value["head_noun"],
        "noun_modifiers": raw_modifiers,
        "attributes": attributes,
        "selector": selector,
    }


def envelope_from_qwen_semantics(value: Any, raw_command: str) -> dict[str, Any]:
    """Add only deterministic structure to Qwen's semantic interpretation."""

    raw_command = _nonempty_string(raw_command, "raw_command")
    value = _require_exact_keys(value, QWEN_SEMANTIC_KEYS, "Qwen semantics")
    action = _require_exact_keys(value["action"], ACTION_KEYS, "action")

    raw_anchors = value["anchors"]
    if not isinstance(raw_anchors, list):
        raise _schema_error("anchors must be a list")
    if len(raw_anchors) > MAX_ANCHORS:
        raise GroundingV2Error(
            f"at most {MAX_ANCHORS} anchors are allowed",
            code="request_complexity_limit",
            details={"anchors": len(raw_anchors), "limit": MAX_ANCHORS},
        )
    target = _qwen_entity_to_envelope_entity(value["target"], entity_id="target")
    anchors = [
        _qwen_entity_to_envelope_entity(anchor, entity_id=f"anchor_{index}")
        for index, anchor in enumerate(raw_anchors, start=1)
    ]

    raw_relationships = value["relationships"]
    if not isinstance(raw_relationships, list):
        raise _schema_error("relationships must be a list")
    if len(raw_relationships) > MAX_RELATIONSHIPS:
        raise GroundingV2Error(
            f"at most {MAX_RELATIONSHIPS} relationships are allowed",
            code="request_complexity_limit",
            details={
                "relationships": len(raw_relationships),
                "limit": MAX_RELATIONSHIPS,
            },
        )
    relationships: list[dict[str, Any]] = []
    for index, raw_relationship in enumerate(raw_relationships):
        relationship = _require_exact_keys(
            raw_relationship,
            QWEN_RELATIONSHIP_KEYS,
            f"relationships[{index}]",
        )
        anchor_index = relationship["anchor_index"]
        if (
            isinstance(anchor_index, bool)
            or not isinstance(anchor_index, int)
            or not 0 <= anchor_index < len(anchors)
        ):
            raise GroundingV2Error(
                "relationship anchor_index does not reference an emitted anchor",
                code="invalid_entity_reference",
                details={
                    "relationship_index": index,
                    "anchor_index": anchor_index,
                    "anchor_count": len(anchors),
                },
            )
        relationships.append(
            {
                "type": relationship["type"],
                "target_id": "target",
                "anchor_id": f"anchor_{anchor_index + 1}",
                "evidence": relationship["evidence"],
            }
        )

    visual_spans = [
        resolve_unique_evidence(
            raw_command,
            entity["mention"],
            field=f"{entity['id']}.mention",
        )
        for entity in [target, *anchors]
    ]
    visual_spans.extend(
        resolve_unique_evidence(
            raw_command,
            relationship["evidence"],
            field=f"relationships[{index}].evidence",
        )
        for index, relationship in enumerate(relationships)
    )
    visual_start = min(start for start, _end in visual_spans)
    visual_end = max(end for _start, end in visual_spans)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "raw_command": raw_command,
        "visual_source_phrase": raw_command[visual_start:visual_end],
        "action": action,
        "destination": value["destination"],
        "target": target,
        "anchors": anchors,
        "relationships": relationships,
    }
    return seal_command_envelope(envelope, expected_raw_command=raw_command)


def build_entity_prompts(entity: dict[str, Any], *, max_prompts: int = 4) -> list[str]:
    """Build the bounded open-vocabulary prompt family specified by v2."""

    if isinstance(max_prompts, bool) or not isinstance(max_prompts, int):
        raise ValueError("max_prompts must be an integer")
    if not 1 <= max_prompts <= MAX_PROMPTS_PER_ENTITY:
        raise ValueError(f"max_prompts must be in [1, {MAX_PROMPTS_PER_ENTITY}]")
    _require_exact_keys(entity, ENTITY_KEYS, "entity")
    mention = _nonempty_string(entity["mention"], "entity.mention")
    head = _nonempty_string(entity["head_noun"], "entity.head_noun")
    modifiers = entity["noun_modifiers"]
    attributes = entity["attributes"]
    if not isinstance(modifiers, list) or not isinstance(attributes, list):
        raise ValueError("entity modifiers and attributes must be lists")
    attribute_values = [str(item["value"]) for item in attributes]
    candidates = [
        mention,
        collapse_space(" ".join([*attribute_values, head])),
        collapse_space(" ".join([*(str(item) for item in modifiers), head])),
        head,
    ]
    prompts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalize_text(candidate)
        if key and key not in seen:
            prompts.append(candidate)
            seen.add(key)
        if len(prompts) == max_prompts:
            break
    return prompts


def entity_prompt_map(envelope: dict[str, Any]) -> dict[str, list[str]]:
    canonical = validate_command_envelope(envelope, require_hashes=True)
    entities = [canonical["target"], *canonical["anchors"]]
    return {entity["id"]: build_entity_prompts(entity) for entity in entities}


def qwen_interpretation_messages(
    raw_command: str,
    *,
    correction: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return the semantic-only, schema-constrained interpretation prompt."""

    raw_command = _nonempty_string(raw_command, "raw_command")
    examples = [
        (
            "identify the small orange box inside the blue bin",
            {
                "action": {"type": "identify", "evidence": "identify"},
                "destination": None,
                "target": {
                    "mention": "small orange box",
                    "head_noun": "box",
                    "noun_modifiers": [],
                    "attributes": [
                        {"type": "size", "evidence": "small"},
                        {"type": "color", "evidence": "orange"},
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
            },
        ),
        (
            "find the striped air filter box",
            {
                "action": {"type": "identify", "evidence": "find"},
                "destination": None,
                "target": {
                    "mention": "striped air filter box",
                    "head_noun": "box",
                    "noun_modifiers": ["air filter"],
                    "attributes": [{"type": "pattern", "evidence": "striped"}],
                    "selector": None,
                },
                "anchors": [],
                "relationships": [],
            },
        ),
        (
            "grab the rightmost metal bolt and put it in the drop zone",
            {
                "action": {"type": "pick", "evidence": "grab"},
                "destination": "drop zone",
                "target": {
                    "mention": "rightmost metal bolt",
                    "head_noun": "bolt",
                    "noun_modifiers": [],
                    "attributes": [{"type": "material", "evidence": "metal"}],
                    "selector": {"type": "rightmost", "evidence": "rightmost"},
                },
                "anchors": [],
                "relationships": [],
            },
        ),
    ]
    example_text = "\n\n".join(
        "EXAMPLE INPUT:\n"
        + command
        + "\nEXAMPLE OUTPUT:\n"
        + json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        for command, output in examples
    )
    system = f"""You are the sole language interpreter for a perception-only robot vision system.
Return exactly one JSON object matching the enforced schema and no commentary. Interpret the raw command only; never infer robot motion execution.

Output exactly these five top-level fields: action, destination, target, anchors, relationships. Do not output schema_version, raw_command, visual_source_phrase, ids, hashes, or attribute values; deterministic code supplies them.
Every action has exactly type and evidence. Type is identify, pick, move, or place. Evidence is the unique exact action phrase from the raw command, or null only for implicit identify.
Destination is a unique exact raw-command substring or null. A destination is not an anchor merely because it follows an action.
Target and every anchor have exactly mention, head_noun, noun_modifiers, attributes, selector. Always emit every field, using [] and null when absent.
Mention is the shortest complete identifying noun phrase copied exactly from the raw command, without a leading a/an/the. Mention must contain its head_noun, every noun modifier, every attribute evidence string, and selector evidence.
Use noun_modifiers only for lexical compound-noun taxonomy, such as "air filter" in "air filter box". Colors, sizes, shapes, materials, markings, patterns, text, and states belong only in attributes and must never be duplicated in noun_modifiers. Attributes have exactly type and evidence; evidence is copied exactly from mention.
Selector is null or has exactly type and evidence. It belongs only to the entity whose mention contains the evidence.
Each relationship has exactly type, anchor_index, evidence. anchor_index is the zero-based position in anchors. Type is one of inside, on, left_of, right_of, above, below, near, next_to, in_front_of, behind. Normalize aliases to that type but copy the unique exact operator phrase as evidence. "from" never proves inside and must not create a relationship.
Emit one target, at most {MAX_ANCHORS} anchors, and at most {MAX_RELATIONSHIPS} relationships. Never invent evidence. Never copy instructions, examples, diagnostics, or delimiters into a field.

{example_text}"""
    user_parts = [
        "RAW COMMAND AS A JSON STRING (the only source text):",
        json.dumps(raw_command, ensure_ascii=False),
    ]
    if correction is not None:
        user_parts.extend(
            [
                "PREVIOUS VALIDATION FAILURE (diagnostic only; do not copy it):",
                json.dumps(correction, ensure_ascii=False, separators=(",", ":")),
                "Regenerate the complete five-field semantic object and correct that failure.",
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def qwen_interpretation_error_is_retryable(error: GroundingV2Error) -> bool:
    """Retry Qwen placement/schema mistakes, but not unsafe input conditions."""

    return error.code not in {
        "ambiguous_source_evidence",
        "request_complexity_limit",
        "unsupported_reference",
    }


def parse_qwen_semantic_interpretation(text: str, raw_command: str) -> dict[str, Any]:
    """Parse Qwen's reduced semantic object and deterministically seal it."""

    if not isinstance(text, str) or not text.strip():
        raise GroundingV2Error(
            "Qwen returned an empty interpretation",
            code="invalid_interpretation_json",
            retryable_format=True,
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroundingV2Error(
            f"Qwen returned invalid JSON: {exc.msg}",
            code="invalid_interpretation_json",
            details={"line": exc.lineno, "column": exc.colno},
            retryable_format=True,
        ) from exc
    return envelope_from_qwen_semantics(value, raw_command)


def parse_qwen_interpretation(text: str, raw_command: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise GroundingV2Error(
            "Qwen returned an empty interpretation",
            code="invalid_interpretation_json",
            retryable_format=True,
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroundingV2Error(
            f"Qwen returned invalid JSON: {exc.msg}",
            code="invalid_interpretation_json",
            details={"line": exc.lineno, "column": exc.colno},
            retryable_format=True,
        ) from exc
    return seal_command_envelope(value, expected_raw_command=raw_command)


def relationship_types(envelope: dict[str, Any]) -> set[str]:
    canonical = validate_command_envelope(envelope, require_hashes=True)
    return {item["type"] for item in canonical["relationships"]}


def all_entities(envelope: dict[str, Any]) -> Iterable[dict[str, Any]]:
    canonical = validate_command_envelope(envelope, require_hashes=True)
    yield canonical["target"]
    yield from canonical["anchors"]
