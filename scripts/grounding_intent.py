#!/usr/bin/env python3
"""Versioned, fail-closed visual-grounding intent contract.

The contract deliberately contains only visual grounding information. Robot
actions and destinations belong to the task layer and must never be included
in an intent sent to SAM.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
INTENT_KEYS = {
    "schema_version",
    "source_phrase",
    "category",
    "attributes",
    "selector",
    "source_region",
    "relations",
    "ambiguities",
}
ATTRIBUTE_KEYS = {"type", "value"}
ATTRIBUTE_TYPES = {"color", "size", "shape", "material", "marking"}
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
RELATION_KEYS = {"type", "anchor"}
VALID_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
    "near",
    "in_front_of",
    "behind",
}

COLOR_ALIASES = {
    "red": ("red",),
    "orange": ("orange",),
    "yellow": ("yellow",),
    "green": ("green",),
    "blue": ("blue",),
    "purple": ("purple", "violet"),
    "white": ("white",),
    "black": ("black",),
    "gray": ("gray", "grey"),
    "brown": ("brown",),
    "pink": ("pink",),
    "silver": ("silver",),
    "gold": ("gold", "golden"),
    "beige": ("beige", "tan"),
}
SIZE_ALIASES = {
    "small": ("small", "little", "tiny"),
    "medium": ("medium", "medium sized"),
    "large": ("large", "big"),
}
SHAPE_ALIASES = {
    "rectangular": ("rectangular", "rectangle shaped"),
    "square": ("square",),
    "round": ("round", "circular"),
    "flat": ("flat",),
    "cylindrical": ("cylindrical", "cylinder shaped"),
    "oval": ("oval",),
}
MATERIAL_ALIASES = {
    "cardboard": ("cardboard", "corrugated"),
    "plastic": ("plastic",),
    "metal": ("metal", "metallic"),
    "wood": ("wood", "wooden"),
    "paper": ("paper",),
    "glass": ("glass",),
    "foam": ("foam",),
}
ATTRIBUTE_ALIASES = {
    "color": COLOR_ALIASES,
    "size": SIZE_ALIASES,
    "shape": SHAPE_ALIASES,
    "material": MATERIAL_ALIASES,
}

SELECTOR_ALIASES = {
    "rightmost": (
        r"\bright[ -]?most\b",
        r"\bon the (?:far )?right\b",
        r"\bright side\b",
        r"\bright(?:ern)? one\b",
    ),
    "leftmost": (
        r"\bleft[ -]?most\b",
        r"\bon the (?:far )?left\b",
        r"\bleft side\b",
        r"\bleft(?:ern)? one\b",
    ),
    "topmost": (
        r"\btop[ -]?most\b",
        r"\bat the top\b",
        r"\buppermost\b",
        r"\btop\b",
    ),
    "bottommost": (
        r"\bbottom[ -]?most\b",
        r"\bat the bottom\b",
        r"\blowermost\b",
        r"\bbottom\b",
    ),
    "nearest": (r"\bnearest\b", r"\bclosest\b"),
    "farthest": (r"\bfarthest\b", r"\bfurthest\b"),
    "largest": (r"\blargest\b", r"\bbiggest\b"),
    "smallest": (r"\bsmallest\b",),
}

CATEGORY_SYNONYMS = {
    "box": ("package", "carton"),
    "package": ("box", "carton"),
    "carton": ("box", "package"),
    "bottle": ("container",),
    "container": ("bottle",),
}

REGION_PATTERN = re.compile(
    r"\b(?:from|on|in|inside|within|at)\s+(?:the\s+)?"
    r"((?:(?:upper|top|lower|bottom|left|right|middle|center)\s+)?"
    r"(?:shelf|bin|tray|cart|rack|table|workspace|slot|compartment))\b"
)
MARKING_PATTERN = re.compile(
    r"\bwith\s+(?:a\s+|an\s+|the\s+)?"
    r"([a-z0-9][a-z0-9 -]{0,60}?)\s+(?:label|logo|marking|text)\b"
)
ALTERNATIVE_PATTERN = re.compile(r"\b(?:or|either)\b")
UNRESOLVED_PATTERN = re.compile(
    r"^(?:it|this|that|one|something|anything|thing|stuff|requested object)\b"
)


class GroundingIntentError(ValueError):
    """Raised when visual intent is missing, ambiguous, or schema-invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_grounding_intent",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_phrase(value: str) -> str:
    normalized = value.lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return collapse_space(normalized)


def _phrase_present(needle: str, haystack: str) -> bool:
    needle_tokens = normalize_phrase(needle).split()
    haystack_tokens = normalize_phrase(haystack).split()
    if not needle_tokens:
        return False
    width = len(needle_tokens)
    return any(
        haystack_tokens[index : index + width] == needle_tokens
        for index in range(len(haystack_tokens) - width + 1)
    )


def _attribute_is_anchored(attribute_type: str, value: str, source: str) -> bool:
    if attribute_type == "marking":
        return _phrase_present(value, source)
    aliases = ATTRIBUTE_ALIASES[attribute_type].get(value)
    return bool(aliases and any(_phrase_present(alias, source) for alias in aliases))


def _selector_is_anchored(selector: str, source: str) -> bool:
    normalized = normalize_phrase(source)
    return any(re.search(pattern, normalized) for pattern in SELECTOR_ALIASES[selector])


def validate_grounding_intent(
    value: Any,
    *,
    expected_source_phrase: str | None = None,
    require_unambiguous: bool = True,
) -> dict[str, Any]:
    """Validate and normalize one exact schema-v1 grounding intent."""

    if not isinstance(value, dict):
        raise GroundingIntentError("grounding intent must be a JSON object")
    keys = set(value)
    if keys != INTENT_KEYS:
        raise GroundingIntentError(
            "grounding intent keys are invalid",
            details={
                "missing": sorted(INTENT_KEYS - keys),
                "extra": sorted(keys - INTENT_KEYS),
            },
        )
    if isinstance(value["schema_version"], bool) or value["schema_version"] != SCHEMA_VERSION:
        raise GroundingIntentError(
            f"unsupported grounding intent schema {value['schema_version']!r}"
        )

    source_phrase = value["source_phrase"]
    if not isinstance(source_phrase, str) or not collapse_space(source_phrase):
        raise GroundingIntentError("source_phrase must be a non-empty string")
    source_phrase = collapse_space(source_phrase)
    if expected_source_phrase is not None and source_phrase != collapse_space(
        expected_source_phrase
    ):
        raise GroundingIntentError(
            "source_phrase does not match the request",
            code="grounding_identity_mismatch",
            details={
                "expected": collapse_space(expected_source_phrase),
                "actual": source_phrase,
            },
        )
    normalized_source = normalize_phrase(source_phrase)
    if ALTERNATIVE_PATTERN.search(normalized_source):
        raise GroundingIntentError(
            "alternative targets are ambiguous",
            code="ambiguous_grounding_intent",
        )

    category = value["category"]
    if not isinstance(category, str) or not normalize_phrase(category):
        raise GroundingIntentError("category must be a non-empty string")
    category = normalize_phrase(category)
    if UNRESOLVED_PATTERN.search(category):
        raise GroundingIntentError(
            "category contains an unresolved reference",
            code="unresolved_grounding_reference",
        )
    if not _phrase_present(category, normalized_source):
        raise GroundingIntentError(
            f"category {category!r} is not anchored in source_phrase",
            code="invented_grounding_evidence",
        )

    raw_attributes = value["attributes"]
    if not isinstance(raw_attributes, list):
        raise GroundingIntentError("attributes must be a list")
    attributes: list[dict[str, str]] = []
    seen_attributes: set[tuple[str, str]] = set()
    for raw_attribute in raw_attributes:
        if not isinstance(raw_attribute, dict) or set(raw_attribute) != ATTRIBUTE_KEYS:
            raise GroundingIntentError(
                "each attribute must contain exactly type and value"
            )
        attribute_type = raw_attribute["type"]
        attribute_value = raw_attribute["value"]
        if attribute_type not in ATTRIBUTE_TYPES:
            raise GroundingIntentError(
                f"unsupported attribute type {attribute_type!r}"
            )
        if not isinstance(attribute_value, str) or not normalize_phrase(attribute_value):
            raise GroundingIntentError("attribute value must be a non-empty string")
        attribute_value = normalize_phrase(attribute_value)
        pair = (attribute_type, attribute_value)
        if pair in seen_attributes:
            raise GroundingIntentError(f"duplicate attribute {pair!r}")
        if not _attribute_is_anchored(attribute_type, attribute_value, normalized_source):
            raise GroundingIntentError(
                f"attribute {attribute_type}:{attribute_value} is not anchored in source_phrase",
                code="invented_grounding_evidence",
            )
        seen_attributes.add(pair)
        attributes.append({"type": attribute_type, "value": attribute_value})

    selector = value["selector"]
    if selector is not None:
        if not isinstance(selector, str) or selector not in VALID_SELECTORS:
            raise GroundingIntentError(f"unsupported selector {selector!r}")
        if not _selector_is_anchored(selector, normalized_source):
            raise GroundingIntentError(
                f"selector {selector!r} is not anchored in source_phrase",
                code="invented_grounding_evidence",
            )

    source_region = value["source_region"]
    if source_region is not None:
        if not isinstance(source_region, str) or not normalize_phrase(source_region):
            raise GroundingIntentError("source_region must be null or a string")
        source_region = normalize_phrase(source_region)
        if not _phrase_present(source_region, normalized_source):
            raise GroundingIntentError(
                "source_region is not anchored in source_phrase",
                code="invented_grounding_evidence",
            )

    raw_relations = value["relations"]
    if not isinstance(raw_relations, list):
        raise GroundingIntentError("relations must be a list")
    relations: list[dict[str, str]] = []
    for raw_relation in raw_relations:
        if not isinstance(raw_relation, dict) or set(raw_relation) != RELATION_KEYS:
            raise GroundingIntentError(
                "each relation must contain exactly type and anchor"
            )
        relation_type = raw_relation["type"]
        anchor = raw_relation["anchor"]
        if relation_type not in VALID_RELATIONS:
            raise GroundingIntentError(f"unsupported relation {relation_type!r}")
        if not isinstance(anchor, str) or not normalize_phrase(anchor):
            raise GroundingIntentError("relation anchor must be a non-empty string")
        anchor = normalize_phrase(anchor)
        if not _phrase_present(anchor, normalized_source):
            raise GroundingIntentError(
                "relation anchor is not anchored in source_phrase",
                code="invented_grounding_evidence",
            )
        relations.append({"type": relation_type, "anchor": anchor})

    ambiguities = value["ambiguities"]
    if not isinstance(ambiguities, list) or any(
        not isinstance(item, str) or not item.strip() for item in ambiguities
    ):
        raise GroundingIntentError("ambiguities must be a list of non-empty strings")
    ambiguities = [collapse_space(item) for item in ambiguities]
    if require_unambiguous and ambiguities:
        raise GroundingIntentError(
            "grounding intent is ambiguous: " + "; ".join(ambiguities),
            code="ambiguous_grounding_intent",
            details={"ambiguities": ambiguities},
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_phrase": source_phrase,
        "category": category,
        "attributes": attributes,
        "selector": selector,
        "source_region": source_region,
        "relations": relations,
        "ambiguities": ambiguities,
    }


def intent_hash(intent: dict[str, Any]) -> str:
    canonical = validate_grounding_intent(intent)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_source_region(working: str) -> tuple[str, str | None]:
    matches = list(REGION_PATTERN.finditer(working))
    if len(matches) > 1:
        regions = sorted({normalize_phrase(match.group(1)) for match in matches})
        if len(regions) > 1:
            raise GroundingIntentError(
                "multiple source regions are ambiguous",
                code="ambiguous_grounding_intent",
                details={"source_regions": regions},
            )
    source_region = normalize_phrase(matches[0].group(1)) if matches else None
    return REGION_PATTERN.sub(" ", working), source_region


def _extract_selectors(working: str) -> tuple[str, str | None]:
    found: list[str] = []
    for selector, patterns in SELECTOR_ALIASES.items():
        if any(re.search(pattern, working) for pattern in patterns):
            found.append(selector)
    if len(found) > 1:
        raise GroundingIntentError(
            "multiple spatial selectors are ambiguous",
            code="ambiguous_grounding_intent",
            details={"selectors": sorted(found)},
        )
    selector = found[0] if found else None
    if selector:
        for pattern in SELECTOR_ALIASES[selector]:
            working = re.sub(pattern, " ", working)
    return working, selector


def _extract_attributes(working: str) -> tuple[str, list[dict[str, str]]]:
    found: list[tuple[int, str, str, str]] = []
    for attribute_type, values in ATTRIBUTE_ALIASES.items():
        for canonical_value, aliases in values.items():
            for alias in sorted(aliases, key=len, reverse=True):
                match = re.search(rf"\b{re.escape(alias)}\b", working)
                if match:
                    found.append(
                        (match.start(), attribute_type, canonical_value, match.group(0))
                    )
                    break

    marking_match = MARKING_PATTERN.search(working)
    if marking_match:
        marking_value = normalize_phrase(marking_match.group(1))
        found.append((marking_match.start(), "marking", marking_value, marking_match.group(0)))

    attributes: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []
    for _position, attribute_type, value, matched_text in sorted(found):
        match = re.search(rf"\b{re.escape(matched_text)}\b", working)
        if not match or any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        attributes.append({"type": attribute_type, "value": value})
        occupied.append((match.start(), match.end()))
        working = working[: match.start()] + " " * (match.end() - match.start()) + working[match.end() :]
    return working, attributes


def parse_grounding_intent(source_phrase: str) -> dict[str, Any]:
    """Conservatively parse common visual phrases without model inference."""

    if not isinstance(source_phrase, str) or not collapse_space(source_phrase):
        raise GroundingIntentError("source phrase must be a non-empty string")
    source_phrase = collapse_space(source_phrase)
    normalized_source = normalize_phrase(source_phrase)
    if ALTERNATIVE_PATTERN.search(normalized_source):
        raise GroundingIntentError(
            "alternative targets are ambiguous",
            code="ambiguous_grounding_intent",
        )

    working = normalized_source
    working = re.sub(
        r"^(?:please\s+)?(?:pick\s+up|pick|grab|get|select|find|choose|show(?:\s+me)?)\s+",
        "",
        working,
    )
    working = re.split(
        r"\s+and\s+(?:place|put|move|drop|set)\b",
        working,
        maxsplit=1,
    )[0]
    working = re.sub(
        r"\s+(?:to|into|onto)\s+(?:the\s+)?(?:drop\s+zone|left\s+bin|right\s+bin|destination|target\s+area).*$",
        "",
        working,
    )
    working, source_region = _extract_source_region(working)
    working, selector = _extract_selectors(working)
    working, attributes = _extract_attributes(working)
    working = re.sub(r"\b(?:the|a|an|please|me)\b", " ", working)
    working = re.sub(r"\b(?:and|with)\b", " ", working)
    category = collapse_space(working)
    category = re.sub(r"^(?:object|item)\s+", "", category)
    category = category.strip(" .,-")
    if not category or UNRESOLVED_PATTERN.search(category):
        raise GroundingIntentError(
            "could not resolve a concrete visual category",
            code="unresolved_grounding_reference",
        )
    if category in {"object", "item", "thing", "stuff"} and not attributes:
        raise GroundingIntentError(
            "generic target has no distinguishing visual evidence",
            code="unresolved_grounding_reference",
        )

    intent = {
        "schema_version": SCHEMA_VERSION,
        "source_phrase": source_phrase,
        "category": category,
        "attributes": attributes,
        "selector": selector,
        "source_region": source_region,
        "relations": [],
        "ambiguities": [],
    }
    return validate_grounding_intent(intent, expected_source_phrase=source_phrase)


def construct_primary_prompt(intent: dict[str, Any]) -> str:
    canonical = validate_grounding_intent(intent)
    grouped: dict[str, list[str]] = {key: [] for key in ATTRIBUTE_TYPES}
    for attribute in canonical["attributes"]:
        grouped[attribute["type"]].append(attribute["value"])
    pieces: list[str] = []
    pieces.extend(grouped["size"])
    if grouped["color"]:
        pieces.append(" and ".join(grouped["color"]))
    pieces.extend(grouped["material"])
    pieces.extend(grouped["shape"])
    pieces.extend(f"{value} label" for value in grouped["marking"])
    pieces.append(canonical["category"])
    return collapse_space(" ".join(pieces))


def _terminal_category_synonyms(category: str) -> list[str]:
    words = category.split()
    terminal = words[-1]
    prefix = " ".join(words[:-1])
    synonyms = []
    for replacement in CATEGORY_SYNONYMS.get(terminal, ()):
        synonyms.append(collapse_space(f"{prefix} {replacement}"))
    return synonyms


def build_prompt_family(
    intent: dict[str, Any],
    *,
    max_prompts: int = 5,
) -> list[str]:
    """Create a bounded, deterministic family of SAM prompts."""

    if isinstance(max_prompts, bool) or not isinstance(max_prompts, int) or max_prompts < 1:
        raise ValueError("max_prompts must be a positive integer")
    canonical = validate_grounding_intent(intent)
    primary = construct_primary_prompt(canonical)
    category = canonical["category"]
    colors = [
        item["value"] for item in canonical["attributes"] if item["type"] == "color"
    ]
    sizes = [
        item["value"] for item in canonical["attributes"] if item["type"] == "size"
    ]
    shapes = [
        item["value"] for item in canonical["attributes"] if item["type"] == "shape"
    ]
    candidates = [primary]
    if colors:
        candidates.append(collapse_space(f"{' and '.join(colors)} {category}"))
    if sizes or shapes:
        candidates.append(collapse_space(f"{' '.join(sizes + shapes)} {category}"))
    candidates.append(category)
    candidates.extend(_terminal_category_synonyms(category))

    family: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_phrase(candidate)
        if normalized and normalized not in seen:
            family.append(candidate)
            seen.add(normalized)
        if len(family) == max_prompts:
            break
    return family


def qwen_parser_messages(source_phrase: str) -> list[dict[str, str]]:
    source_phrase = collapse_space(source_phrase)
    schema_example = {
        "schema_version": SCHEMA_VERSION,
        "source_phrase": source_phrase,
        "category": "box",
        "attributes": [{"type": "color", "value": "orange"}],
        "selector": None,
        "source_region": None,
        "relations": [],
        "ambiguities": [],
    }
    return [
        {
            "role": "system",
            "content": (
                "Extract a visual grounding intent. Return exactly one JSON object "
                "with exactly these keys: schema_version, source_phrase, category, "
                "attributes, selector, source_region, relations, ambiguities. "
                "schema_version is 1. attributes may use only color, size, shape, "
                "material, marking. selector is null or rightmost, leftmost, "
                "topmost, bottommost, nearest, farthest, largest, smallest. Copy "
                "source_phrase exactly. Every category, attribute, region, and "
                "relation anchor must be explicitly supported by words in the "
                "source phrase. Never invent evidence. Put uncertainty, alternatives, "
                "pronouns, or conflicting selectors in ambiguities. Do not include "
                "robot actions or destinations. Return JSON only, without markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema shape example: {json.dumps(schema_example)}\n"
                f"Source phrase: {source_phrase}"
            ),
        },
    ]


def parse_qwen_grounding_intent(text: str, source_phrase: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise GroundingIntentError("Qwen intent response is empty")
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise GroundingIntentError(f"Qwen intent response is not JSON: {exc}") from exc
    return validate_grounding_intent(
        value,
        expected_source_phrase=source_phrase,
        require_unambiguous=True,
    )


def compare_intents(
    active: dict[str, Any],
    shadow: dict[str, Any] | None,
    *,
    shadow_error: str | None = None,
) -> dict[str, Any]:
    active = validate_grounding_intent(active)
    differing_fields: list[str] = []
    if shadow is not None:
        shadow = validate_grounding_intent(shadow)
        differing_fields = [
            key for key in sorted(INTENT_KEYS) if active[key] != shadow[key]
        ]
    return {
        "active_intent": active,
        "active_intent_hash": intent_hash(active),
        "shadow_intent": shadow,
        "shadow_intent_hash": None if shadow is None else intent_hash(shadow),
        "shadow_error": shadow_error,
        "matches": shadow is not None and not differing_fields,
        "differing_fields": differing_fields,
    }
