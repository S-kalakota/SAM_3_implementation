# SAM 3 version-2 perception API

## `POST /v2/interpret`

Request:

```json
{"raw_command":"grab the red box inside the blue bin and put it in the drop zone"}
```

Qwen returns the semantic fields. Deterministic code validates literal evidence,
normalizes declared operators, enforces limits, and adds the hashes:

```json
{
  "schema_version": 2,
  "raw_command": "grab the red box inside the blue bin and put it in the drop zone",
  "visual_source_phrase": "red box inside the blue bin",
  "action": {"type": "pick", "evidence": "grab"},
  "destination": "drop zone",
  "target": {
    "id": "target",
    "mention": "red box",
    "head_noun": "box",
    "noun_modifiers": [],
    "attributes": [{"type": "color", "value": "red", "evidence": "red"}],
    "selector": null
  },
  "anchors": [{
    "id": "anchor_1",
    "mention": "blue bin",
    "head_noun": "bin",
    "noun_modifiers": [],
    "attributes": [{"type": "color", "value": "blue", "evidence": "blue"}],
    "selector": null
  }],
  "relationships": [{
    "type": "inside",
    "target_id": "target",
    "anchor_id": "anchor_1",
    "evidence": "inside"
  }],
  "canonical_intent_hash": "<64 lowercase hex characters>",
  "envelope_hash": "<64 lowercase hex characters>"
}
```

The visual phrase is not free text: it must equal the exact raw-command slice
from the earliest to latest target, anchor, or relationship evidence. Every
entity mention and relationship operator must occur exactly once. Repeated
ambiguous or missing evidence is refused.

## `POST /v2/segment`

Send the exact `/v2/interpret` response as the JSON request body. Extra keys,
changed text, changed entities, or changed hashes are refused. This endpoint
does not invoke the language interpreter. The resident service also requires a
matching bounded interpretation attestation from its own `/v2/interpret` call;
a hand-built envelope or an envelope lost across a service restart is refused
with `interpretation_attestation_missing`.

An accepted response contains one target mask, selected anchor masks,
depth/XYZ metadata, selected identities, all relationship measurements, and
audit paths. A safe negative result has `status: "no_match"` and no mask.
Fail-closed pipeline errors also save an auditable no-motion `result.json` and
return its path in the error details. `elapsed_s` includes interpretation plus
the complete segmentation request.
Every response has:

```json
{"robot_target": null, "motion_permitted": false}
```

During stages 2–4 of rollout, `SAM3_V2_EVALUATION_MODE=1` makes this no-motion
endpoint available after image parity and relation calibration pass. Production
`SAM3_V2_ENABLED=1` additionally requires the fully approved aggregate release
report. This separation avoids using an unapproved report to collect the very
evidence that report must contain.

## Limits and operators

- Exactly one target, zero to three anchors, zero to four relationships, and
  no more than four distinct prompts per entity.
- Actions: `identify`, `pick`, `move`, `place`. They are descriptive only.
- Selectors: `leftmost`, `rightmost`, `topmost`, `bottommost`, `nearest`,
  `farthest`, `largest`, `smallest`, scoped to one entity mention.
- Relationships: `inside`, `on`, `left_of`, `right_of`, `above`, `below`,
  `near`, `next_to`, `in_front_of`, `behind`.
- `in`, `inside`, and `within` normalize to `inside`; `beside` normalizes to
  `next_to`; `on top of` normalizes to `on`. `from` may introduce a contextual
  anchor, but is never containment proof and creates no relationship by itself.

Stable refusal codes include `request_complexity_limit`,
`ambiguous_source_evidence`, `missing_source_evidence`,
`unsupported_reference`, `grounding_identity_mismatch`,
`grounding_parser_unavailable`, `visual_grounding_unavailable`,
`candidate_verification_unavailable`, `selector_geometry_unavailable`, and
`relation_geometry_unavailable`, plus `interpretation_attestation_missing`.
