# Segmentation Identification Improvement Plan

Date: 2026-08-04

Implementation status: phases 1 through 9 are implemented. The accepted
`448,360,384,360` HD720 crop is the centralized live default; structured intent,
prompt-family/multiscale candidate generation, Qwen verification, geometric
safety gates, ROS identity checks, frozen-frame A/B evaluation, and a 20-round
soak harness are present. Section 10 was added after a complex-language failure
was observed and is **planned but not implemented**. The empirical live
acceptance targets still require a fixed labeled scene and operator-reviewed
runs; they are not claimed from unit tests alone.

## Goal

Improve instruction-based object identification first, then improve final mask quality. The immediate target is reliable identification of small packages and boxes in the ZED workspace without accepting a confidently segmented but incorrect object.

## What the baseline system did before these changes

The resident service currently has two paths:

1. The fast path optionally converts some natural-language commands into a short noun phrase and sends that phrase directly to SAM 3.1.
2. The visual Qwen agent is used only when the fast path returns zero accepted masks.

This creates several limitations:

- Normal short prompts such as `orange package` bypass Qwen entirely.
- Qwen does not visually verify a non-empty direct result, even when SAM selected the wrong object.
- The fallback condition checks only whether the mask count is zero.
- The presence gate checks SAM score and pixel area, but not whether a candidate semantically matches the request.
- Spatial selectors cover only a limited set of relationships.
- The current service processes the complete 1280 x 720 frame, where the requested package may occupy only a thin, small region.

Recent saved runs support this diagnosis:

- `orange and white box`, `cardboard box`, `air filter box`, `orange package`, `rectangular package`, and `orange label` produced zero raw candidates.
- Large, common objects still worked: `robot arm` produced one mask and `shelf` produced three masks.
- A visual Qwen 3B agent run incorrectly concluded that the scene contained a robot but no car-filter box, even though the small package was visible on the cart.

## Crop and ROI values already saved

Crop coordinates use `x,y,width,height`, with `(0,0)` at the top-left of the 1280 x 720 frame.

### Standard true inference crop

```text
448,360,384,360
```

This means:

- Left edge: `x = 448`
- Top edge: `y = 360`
- Right edge: `x = 832`
- Bottom edge: `y = 720`
- Cropped image: `384 x 360`

This crop was used by earlier Task 5/Task 7 runs and the July 2 cropped test. In that cropped test, one `right most pasta box` round produced one mask with score `0.9375`; two later rounds produced no candidates. It is now the centralized default in `task5_zed_live_prompt.py` and is inherited by Task 7 and the resident mask service.

Evidence:

- `outputs/task5_live_green_object_crop_448_360_384_360.json`
- `outputs/06/02/2026-1_test/cropped/run_001/summary.json`

### Current service selection ROI

```text
430,380,430,170
```

This means:

- Left edge: `x = 430`
- Top edge: `y = 380`
- Right edge: `x = 860`
- Bottom edge: `y = 550`

This was previously configured in `docker-compose.mask-service.yml`, but it was only a post-segmentation selection ROI. It has been removed from the default service command because those full-frame coordinates are invalid after applying the standard true crop. A future selection ROI must use crop-local coordinates.

### Current service capture

The saved August 4 service results from before standardization record cropping as disabled:

```text
enabled: false
input/output: 1280 x 720
```

New service processes default to the standard crop after they are restarted. Historical result files remain unchanged.

## Recommended target pipeline

```text
Natural-language instruction
        |
        v
Structured intent
  - target category
  - visual attributes
  - relation/anchor
  - spatial selector
        |
        v
Candidate generation
  - full frame
  - workspace crop
  - optional overlapping/multiscale crops
  - several concise category synonyms
        |
        v
Merge and deduplicate candidate masks
        |
        v
Qwen-VL visual verification
  - inspect numbered masks
  - choose matching candidate(s)
  - or return no match
        |
        v
Deterministic geometry/depth selection
        |
        v
Full-frame mask + ZED depth/XYZ
```

## Implementation order

### 1. Confirm the workspace crop (complete)

Open the latest full frame and choose a crop that contains:

- The complete pickable workspace
- Enough surrounding context to recognize objects
- Minimal robot, wall, monitor, shelving, and other irrelevant scene content

The saved `448,360,384,360` crop has been accepted as the standard HD720 workspace crop. Reconfirm it if the camera or workspace moves.

### 2. Make the ROI a real inference crop (complete)

- Apply the crop to RGB before both SAM and Qwen.
- Apply the identical crop to depth and XYZ, or expand the final crop mask back into a full-frame zero mask before depth lookup.
- Record both crop-local and full-frame coordinates in every result.
- Keep the existing selection ROI as a separate concept if it remains useful.

RGB, depth, and XYZ now share the canonical crop. Context/tile masks are
projected back into crop coordinates, and each accepted mask is also expanded
to a zero-filled full-frame mask. Results record crop-local/full-frame boxes,
centers, mask paths, and coordinate-space semantics.

### 3. Add multiscale candidate generation (complete)

- Run the direct prompt on the confirmed workspace crop.
- Optionally retain a full-frame pass for large/contextual objects.
- For weak prompts, try a small controlled prompt family such as `box`, `package`, `carton`, and the requested visual noun phrase.
- Merge overlapping masks instead of accepting each prompt result independently.

The service now runs a bounded three-to-five phrase family on the workspace
crop, the primary/category prompts on four overlapping zoom tiles, and the
primary prompt on an optional full-frame context view. All masks are projected
into the canonical crop and IoU-deduplicated while retaining prompt, view,
score, source JSON, and duplicate provenance. Full-frame masks that primarily
lie outside the workspace fail closed.

### 4. Use Qwen as a candidate verifier (complete)

Run visual verification when any of these conditions holds:

- No candidate was found.
- More than one candidate was found.
- Candidate scores are close.
- Confidence is below the validated threshold.
- The instruction contains relationships, product attributes, or printed-label references.
- A candidate is outside the expected workspace.
- The result will be sent to the robot.

The resident service now sends every score/area-gated direct or agent candidate
through Qwen before spatial selection, depth extraction, or robot-coordinate
generation. Qwen receives the raw crop, a full-scene numbered overlay, and
enlarged candidate views. It must return strict JSON containing selected IDs or
`no_match`; malformed output gets one format-only retry and then fails closed.

The verifier records its raw response, decision, candidate-to-SAM index mapping,
confidence, reason, and image artifacts in each result. A verifier rejection or
error cannot trigger the agent fallback and therefore cannot bypass the gate.
Relative selectors are applied deterministically only after semantic approval.

Live paired control on 2026-08-04:

- A tight package mask covering `0.021021` of the crop was approved by Qwen at
  `0.84` confidence and produced one depth object.
- The erroneous full-bin `box` mask covering `0.331554` of the crop was refused
  by the final `0.25` pick-target area guard and produced zero depth objects.

The maximum area fraction and minimum Qwen confidence are configurable with
`SAM3_QWEN_VERIFIER_MAX_AREA_FRACTION` and
`SAM3_QWEN_VERIFIER_MIN_CONFIDENCE`.

### 5. Upgrade Qwen 3B to Qwen2.5-VL-7B (complete)

The resident Docker service now defaults to the locally cached
`Qwen/Qwen2.5-VL-7B-Instruct` snapshot. Docker runs Transformers and the
Hugging Face Hub in offline mode, mounts the persistent model cache, warms the
7B BF16 model onto `cuda:0` during service startup, and reports the loaded
runtime through `/health`. Agent generation is capped at 512 new tokens with a
1.15 repetition penalty.

An in-container inference check completed successfully. Step 4 now ensures that
non-empty direct and fallback SAM masks are verified before they can reach the
robot-target path.

Use the locally cached `Qwen/Qwen2.5-VL-7B-Instruct` before considering a much larger model.

Recommended generation controls:

- Deterministic generation for tool calls
- Maximum output around 256-512 tokens
- `SAM3_AGENT_QWEN_REPETITION_PENALTY=1.15`
- Strict JSON validation with one controlled retry

The current adapter is hard-coded for Qwen2.5-VL. Qwen2.5-VL-7B is therefore the low-risk upgrade. Qwen3-VL requires an adapter/model-class update rather than only changing the model ID.

The 7B model should improve visual verification and tool use, but it cannot improve direct requests in which Qwen is never called. It also cannot recover detail that is lost because the target is too small in the full frame.

### 6. Make relationships deterministic where possible (complete for current selector vocabulary)

- Use coordinates for leftmost/rightmost/topmost/bottommost.
- Use ZED depth for nearest/farthest.
- Use mask area for largest/smallest.
- Eventually perform robot-relative spatial selection in the robot base frame rather than image coordinates.
- Use OCR or VLM text recognition when the request depends on a brand, model, or printed label.

Image coordinates select left/right/top/bottom, ZED median depth selects
near/far, and mask area selects largest/smallest after semantic verification.
Printed markings are typed intent attributes and are explicitly given to the
Qwen-VL verifier. The version 1 `relations` field remains schema-reserved;
entity-scoped anchor relationships are now specified as planned work in
Section 10. Robot-base-frame relative selection remains a future extension.

### 7. Add semantic and safety checks (complete)

Do not accept a mask based only on SAM score and area. Require:

- Semantic verifier approval when the result is ambiguous
- Workspace/ROI membership
- Sufficient valid depth coverage
- Reasonable physical size and table height
- A rejection/no-match option

The active gates now include strict intent/hash identity, Qwen `no_match`,
workspace retention/ROI, score/area, valid-depth coverage and count, p90-p10
depth spread, verifier mask-area policy, projected physical extent, calibrated
surface height, calibration envelopes, and camera-XYZ disagreement.

### 8. Build a repeatable A/B evaluation (implemented; labeled run pending)

Freeze representative frames and expected targets, then compare:

1. Qwen 3B + full frame
2. Qwen 7B + full frame
3. Qwen 3B + workspace crop
4. Qwen 7B + workspace crop

Measure:

- Raw candidate recall
- Correct-object selection accuracy
- No-object false-positive rate
- Final mask IoU/Dice where labels exist
- Malformed agent responses
- End-to-end latency

This will show whether crop scale, Qwen size, or SAM candidate generation is the actual limiting factor.

`scripts/evaluate_segmentation.py` and `/v1/evaluate` run frozen RGB frames
through the same resident SAM/Qwen candidate path without producing a robot
target. The report computes every metric above. The four commands and manifest
schema are under `evaluation/`. A labeled, fixed manifest must be supplied
before numerical A/B claims can be made.

### 9. Convert speech into structured visual attributes (implemented)

Status: accepted and implemented. The deterministic schema is active, while
the Qwen text-parser comparison remains in fail-closed shadow mode by default.

The system should create one canonical visual intent and parse it only once.
The current pipeline interprets parts of the request independently in the VLA
command parser, ROS bridge, and mask service. That creates a risk that the three
components use different meanings for the same speech.

#### Proposed intent schema

Use typed attributes instead of an untyped list so prompt construction and
validation can distinguish color, size, shape, material, and markings:

```json
{
  "schema_version": 1,
  "source_phrase": "the small orange and white box at the top",
  "category": "box",
  "attributes": [
    {"type": "color", "value": "orange"},
    {"type": "color", "value": "white"},
    {"type": "size", "value": "small"}
  ],
  "selector": "topmost",
  "source_region": null,
  "relations": [],
  "ambiguities": []
}
```

Task execution information remains separate from visual grounding:

```json
{
  "action": "pick_and_place",
  "destination": "drop zone"
}
```

SAM should not interpret the destination.

#### Phase 1: Define and validate the contract (complete)

Create `scripts/grounding_intent.py` with:

- A versioned schema and strict parser
- Typed attributes: `color`, `size`, `shape`, `material`, and `marking`
- The existing selector vocabulary: `topmost`, `bottommost`, `leftmost`,
  `rightmost`, `nearest`, `farthest`, `largest`, and `smallest`
- Open-vocabulary object categories
- Normalization for safe synonyms such as `closest` to `nearest`
- Rejection of missing categories, conflicting selectors, unsupported keys,
  unresolved pronouns, and ambiguous alternatives
- Deterministic construction of the primary SAM phrase

#### Phase 2: Upgrade the Qwen text parser (complete)

Replace the current `target_phrase` and `selector` output with the structured
intent. Use:

- `Qwen/Qwen2.5-VL-7B-Instruct`, already warm in the resident service
- Deterministic generation with `do_sample=False`
- A strict JSON response prefix and exact key validation
- One format-only retry
- Evidence anchored in the original phrase so Qwen cannot invent a category or
  visual attribute
- Fail-closed behavior for ambiguity or parser failure

Simple requests may use a conservative deterministic fallback. Complex
requests must not silently lose attributes when Qwen is unavailable.

#### Phase 3: Construct controlled SAM prompt families (complete)

Generate a bounded prompt family from validated fields rather than letting the
language model invent unlimited prompts. For the example above:

```text
small orange and white box
orange and white box
small rectangular box
box
```

Limit a request to approximately three to five phrases. Run them through the
candidate generator, translate every mask into the standard crop coordinates,
and merge duplicate masks before Qwen visual verification.

#### Phase 4: Add a versioned service request (complete)

Keep the existing `/segment?request=...` interface during migration. Add a
versioned JSON request that carries the source phrase and structured intent.
The mask service should:

1. Parse the visual phrase once.
2. Validate and save the canonical intent.
3. Generate SAM prompt candidates deterministically.
4. Run SAM and merge duplicate masks.
5. Give the candidates and the same intent to the visual verifier.
6. Apply deterministic selectors only after semantic approval.
7. Echo the intent, schema version, and an intent hash in the result.

#### Phase 5: Update ROS identity and audit checks (complete)

Update `vla_pick_target.py` to:

- Verify the returned `source_phrase` matches the phrase sent to perception
- Validate the schema version and intent hash
- Store the complete structured intent in the robot target audit
- Refuse motion for malformed, changed, or ambiguous intents
- Continue keeping action and destination outside SAM grounding

#### Phase 6: Run in shadow mode (instrumented; model-backed review pending)

Before changing runtime mask prompts, parse each request using both the old and
new parsers while continuing to use the old result. Record their differences
over at least 30 to 50 representative text commands. Enable the new parser only
after reviewing the comparison.

Every live result can save `intent_shadow.json`. A committed 40-command corpus
and `scripts/evaluate_intent_shadow.py` provide the required comparison. The
deterministic expectations are covered by tests; run `./sam3 intent-shadow
--qwen` to collect the model-backed review before switching
`SAM3_INTENT_PARSER_MODE` from `shadow` to `qwen`.

#### Required command tests

Valid examples:

```text
pick the box
pick the top box
pick the small orange and white box
grab the largest blue carton
pick the flat orange package on the upper shelf
pick the red box from the top shelf and place it in the right bin
```

Required refusals:

```text
pick it up
pick the red box or the blue box
pick the leftmost nearest box
pick something over there
```

#### Structured-intent acceptance criteria

- Every accepted command produces schema-valid JSON.
- No category or attribute is invented.
- Destination words never become visual selectors or attributes.
- Ambiguous targets cause no SAM request and no robot motion.
- Existing simple commands remain backward compatible.
- The returned intent and hash match what the robot bridge sent.
- Every result and robot audit records the structured intent.
- Structured prompt families improve correct-candidate recall on frozen test
  frames without increasing wrong-object acceptance.
- Version 1 keeps `relations` reserved. The version 2 entity graph and migration
  required to implement requests such as "the orange box left of the white
  cup" are specified in Section 10.

### 10. Support flexible relational speech and attribute ownership (planned; not implemented)

Status: required next work. None of the behavior in this section should be
treated as active until its tests and rollout gates pass.

#### Failure that exposed the gap

The following no-motion bridge request was refused on 2026-08-04:

```text
Pick up the air filter box in the grey bin
```

The current deterministic parser produced this invalid category:

```text
air filter box in bin
```

It removed `grey` as though it were a target attribute, left the containment
words inside the category, and then correctly failed the evidence validator
because `air filter box in bin` was not a literal phrase in the source text.
The request stopped before SAM, Qwen visual verification, depth lookup, or
robot-target creation.

The safe refusal was preferable to guessing, but it confirms that the current
version 1 intent is too flat for complex speech. In the intended meaning:

- `air filter box` is the target category.
- `bin` is a separate anchor entity.
- `grey` describes the bin, not the target.
- `in` is a containment relationship between those two entities.

The larger Qwen model cannot repair this at present because the ROS bridge
constructs and validates the deterministic version 1 intent before the visual
request reaches Qwen. Qwen text parsing is only running in shadow mode.

#### Design goal

Accept natural variations of commands containing multiple described entities,
while preserving the existing fail-closed robot behavior. The system must
understand which entity each attribute describes, distinguish source-location
anchors from task destinations, and validate spatial relationships before a
mask can become a robot target.

The goal is flexible grounded language, not unrestricted guessing. Commands
that remain genuinely ambiguous must still be refused with a specific reason.

#### Version 2 entity-grounding schema

Replace the single flat `category` and `attributes` fields with an entity graph.
Every category, attribute, and relationship must retain evidence from the
original visual phrase.

For the phrase `air filter box in the grey bin`, the canonical intent should be
equivalent to:

```json
{
  "schema_version": 2,
  "source_phrase": "air filter box in the grey bin",
  "target_entity_id": "target",
  "entities": [
    {
      "entity_id": "target",
      "role": "target",
      "category": {
        "value": "air filter box",
        "evidence_text": "air filter box",
        "source_span": [0, 14]
      },
      "attributes": []
    },
    {
      "entity_id": "anchor_1",
      "role": "anchor",
      "category": {
        "value": "bin",
        "evidence_text": "bin",
        "source_span": [27, 30]
      },
      "attributes": [
        {
          "type": "color",
          "value": "grey",
          "evidence_text": "grey",
          "source_span": [22, 26]
        }
      ]
    }
  ],
  "relations": [
    {
      "subject_entity_id": "target",
      "predicate": "inside",
      "object_entity_id": "anchor_1",
      "evidence_text": "in",
      "source_span": [15, 17]
    }
  ],
  "selector": null,
  "ambiguities": []
}
```

Spans are half-open character offsets into `source_phrase`. Qwen should return
exact evidence text rather than calculate offsets itself. Deterministic code
must locate the evidence text, reject missing or non-unique matches, and attach
the offsets before hashing the intent.

Task execution remains outside the visual intent:

```json
{
  "action": "pick",
  "destination": null
}
```

For `pick the air filter box in the grey bin and place it in the right bin`,
`grey bin` is a visual source anchor and `right bin` is the task destination.
The destination must not appear as a target entity, visual attribute, selector,
or source relation.

#### Version 2 validation rules

The validator must enforce all of the following before SAM is called:

- Exactly one entity is designated as the target.
- Every other referenced entity has an explicit role such as `anchor`.
- Every attribute belongs to one entity; attributes cannot float globally.
- Target and anchor categories are open-vocabulary noun phrases, not complete
  prepositional phrases.
- Every category and attribute has exact evidence in `source_phrase`.
- Every relation references entity IDs that exist in the same intent.
- Every normalized relation has source evidence such as `in`, `inside`,
  `under`, or `next to`.
- Safe normalization, such as `gray` to `grey` or `within` to `inside`, records
  both the original evidence and normalized value.
- An anchor attribute can never silently migrate onto the target.
- Destination evidence can never be reused as visual-grounding evidence.
- Conflicting parses, multiple possible attribute owners, unresolved pronouns,
  unsupported relationships, and ambiguous alternatives fail closed.
- Validation errors use specific codes such as `attribute_scope_ambiguous`,
  `unresolved_anchor`, `unsupported_relation`, and `ambiguous_target_entity`.

Substring validation of a reconstructed category must be removed. Evidence is
validated field by field against its owning entity. This retains the protection
against invention without rejecting valid phrases merely because adjectives
or relationship words were separated into their proper fields.

#### Supported language in the first version 2 release

The initial controlled relationship vocabulary should include:

- Containment: `in`, `inside`, `within`, `from`
- Support: `on`, `on top of`
- Image-relative: `left of`, `right of`, `above`, `below`
- Proximity: `near`, `next to`, `beside`
- Depth-relative: `in front of`, `behind`
- Existing target selectors: topmost, bottommost, leftmost, rightmost, nearest,
  farthest, largest, and smallest

Synonyms should normalize to this controlled vocabulary. An unsupported
relationship should return a clear refusal instead of being discarded.

Cross-turn pronouns such as `pick that one` remain out of scope until the
system has an explicit, audited conversational-reference mechanism. Ordinary
politeness, filler words, word-order variation, and speech-to-text punctuation
should not cause a refusal.

#### Phase 10.1: Make one component the parsing authority

Stop independently reconstructing visual meaning in the VLA parser, ROS
bridge, and mask service.

Add a text-only `/v2/interpret` endpoint backed by the already warm Qwen 7B
model. It should return a validated command envelope containing:

- The allowlisted task action
- The task destination, if present
- The exact visual source phrase
- The version 2 entity-grounding intent
- The canonical intent hash
- Parser attempts, evidence mappings, ambiguities, and timing

The ROS bridge should call the interpreter once and pass the returned visual
intent unchanged to segmentation. The mask service validates and uses the
supplied intent but does not reinterpret it. The VLA parser must not build a
second competing target category.

Simple commands may continue through a deterministic fast parser only if it
produces the same version 2 schema. Relationship cues, multiple noun phrases,
or uncertain attribute ownership must route to Qwen. If Qwen is unavailable,
a complex request must fail with `complex_parser_unavailable`; it must not fall
back to a lossy flat phrase.

Qwen parsing requirements remain:

- `do_sample=False`
- Strict schema and exact-key validation
- One format-only retry
- No semantic rewriting during the retry
- Deterministic post-validation of evidence and ownership
- Explicit ambiguity output instead of a forced best guess

#### Phase 10.2: Separate target and anchor prompt generation

Build prompt families independently for each entity.

For the motivating command, target prompts may include:

```text
air filter box
filter box
air filter package
box
```

Anchor prompts may include:

```text
grey bin
bin
grey tote
```

Each candidate must record its entity role. Anchor masks may be large and must
use a separate area policy from pick-target masks. A valid full-bin anchor mask
must never be eligible to become the robot pick target.

Generate target candidates both inside the anchor-derived search region and in
the standard workspace crop. This avoids losing the target if the anchor mask
is incomplete while still gaining scale from the relational crop.

#### Phase 10.3: Implement deterministic relationship checks

After target and anchor candidates are generated, score the requested relation
for every valid target-anchor pair:

- `inside`: target center and most target pixels lie within the anchor interior
  region or validated container bounds.
- `on`: target bottom is adjacent to the anchor support surface, with compatible
  depth.
- `left_of`, `right_of`, `above`, `below`: compare mask centers in the agreed
  image coordinate space.
- `near` and `next_to`: compare normalized image distance and, when available,
  3D distance.
- `in_front_of` and `behind`: compare robust ZED depth statistics.

Container masks often cover walls and rims rather than the empty interior.
Therefore `inside` must use a documented interior-region calculation plus Qwen
visual confirmation; raw mask intersection alone is insufficient.

Every relationship check must save its measurements, thresholds, coordinate
space, pass/fail decision, and involved entity IDs. If the required geometry
cannot be measured reliably, fail with `relation_geometry_unavailable`.

#### Phase 10.4: Verify multiple entities visually

Create a verifier overlay that distinguishes target candidates from anchors,
for example `T1`, `T2`, `A1`, and `A2`. Give Qwen:

- The exact version 2 intent
- The unmodified crop
- The multi-entity overlay
- Enlarged target and anchor views
- Deterministic relationship measurements

Qwen must return strict JSON identifying the selected target candidate, the
matching anchor candidate, and whether the requested relationship is visually
satisfied. `no_match` remains valid. The service applies deterministic relation
and geometry gates after Qwen; Qwen cannot override a failed gate.

Saved artifacts should include:

```text
command_envelope.json
grounding_intent_v2.json
entity_candidates.json
relation_checks.json
entity_overlay.png
verifier_response.json
result.json
```

#### Phase 10.5: Add versioned service and ROS integration

Add `/v2/segment` while retaining `/v1/segment` during migration. Version 2
must echo the complete entity graph and canonical hash. Update
`vla_pick_target.py` to verify:

- Schema version 2
- Exact command and source-phrase identity
- Exact intent hash
- Selected candidate role is `target`, never `anchor`
- Every required relationship passed
- Destination is absent from the visual intent
- All existing depth, size, calibration, and surface-height gates still pass

The robot audit should preserve the command envelope, entity graph, parser
evidence, target/anchor candidate records, relationship decisions, verifier
response, and final target. Any mismatch produces no target and no motion.

Add a no-camera interpretation command for rapid debugging:

```text
./sam3 interpret "Pick up the air filter box in the grey bin"
```

This command is planned and does not exist yet. It should show the parsed task,
entities, attribute owners, relations, evidence, hash, and any refusal without
running SAM or using the robot.

#### Phase 10.6: Build the complex-language corpus

Add at least 50 to 100 reviewed commands with expected entity graphs. Include
wording, attribute-scope, relation, destination, ambiguity, and speech-like
variations.

Required valid cases include:

```text
Pick up the air filter box in the grey bin
Grab the air filter package inside the gray tote
Pick the small orange box inside the large grey bin
Pick the grey air filter box inside the orange bin
Pick the box with the blue label in the grey bin
Pick the topmost package in the left bin
Pick the orange box next to the white cup
Pick the carton behind the blue bottle
Pick the air filter box in the grey bin and place it in the right bin
Could you please pick up, uh, the filter box that's inside the gray bin
```

The test suite must prove that `grey` belongs to the anchor in the first
example, belongs to the target in `grey air filter box`, and never leaks from a
task destination.

Required refusals include:

```text
Pick it up
Pick the box in one of the bins
Pick the red box or the blue box
Pick the box next to it
Pick whichever box looks best
Pick the box in the grey bin
```

The final example is only a refusal when the observed scene contains multiple
matching boxes in the grey bin; language parsing can succeed, but visual
grounding must report an unresolved target.

#### Phase 10.7: Shadow rollout and activation

Roll out version 2 in stages:

1. Run `/v2/interpret` in shadow mode without changing active SAM prompts.
2. Review all parser differences and attribute-owner assignments on the fixed
   corpus and representative Whisper transcripts.
3. Run frozen-frame evaluation with labeled target and anchor masks.
4. Run live no-motion tests for each supported relationship.
5. Run present and relationship-violated absence soaks.
6. Enable version 2 for complex commands behind a configuration flag.
7. Keep version 1 available for rollback until the release gates remain stable.

Do not silently fall back from version 2 to version 1 for a complex command.
Fallback would discard the relationship that disambiguates the target.

#### Version 2 acceptance criteria

- At least 95% of reviewed valid complex commands produce the expected entity
  graph without manual prompt rewriting.
- Target-category exact match is at least 98% on the reviewed corpus.
- Attribute-owner accuracy is at least 98%, with zero known destination-to-
  target or anchor-to-target leakage in the safety set.
- Supported-relation extraction accuracy is at least 95%.
- All evidence fields resolve to the original source phrase; no invented entity,
  category, attribute, or relationship is accepted.
- Every intentionally ambiguous or unsupported safety case is refused before a
  robot target is written.
- The motivating command parses as target `air filter box`, anchor `grey bin`,
  and relation `inside` in every deterministic repeat.
- The correct target is selected in at least 19 of 20 static-scene trials.
- Zero targets are accepted when the requested anchor is absent or the required
  relationship is false.
- Anchor masks are never emitted as pick targets.
- No destination words enter SAM target or anchor prompt families.
- Intent, entity, relation, verifier, and geometry identities match across the
  interpreter, mask service, and ROS audit.
- Any parser, anchor, relationship, verifier, depth, or identity failure writes
  no new target and requests no robot motion.
- End-to-end p95 latency is measured and reported; model or crop changes are
  accepted only with evidence that accuracy gains justify the added latency.

### Post-implementation runtime smoke check

After activating the new service on 2026-08-04, a no-motion request for
`orange and blue box` completed through `/v1/segment` in 23.86 seconds. The
pipeline merged three raw masks into two candidates, Qwen rejected the
oversized bin mask, and selected the tight package mask with `0.80` semantic
confidence. The selected mask had `93.98%` valid depth and a `24.43 mm`
p10-p90 depth spread, so it passed the configured geometry gate. The result,
candidate provenance, masks, verifier record, and overlays are saved under:

```text
outputs/service/20260804_194451_257515/
```

This is a deployment smoke check, not a substitute for the labeled A/B run or
the operator-reviewed 20-round acceptance gate below.

## Initial acceptance criteria

Automation status: `scripts/segmentation_soak.py` implements the 20-round
present/absent and malformed-response checks, and the frozen evaluator covers
candidate recall, selection, false positives, IoU/Dice, and latency. These
criteria require live/labeled evidence and therefore remain release gates, not
unit-test assertions.

- Correct target selected in at least 19 of 20 repeated static-scene requests.
- No wrong object accepted when the target is absent.
- Correct behavior for multiple similar boxes and spatial requests.
- No malformed Qwen tool calls in a 20-request soak test.
- Crop masks remain correctly aligned with ZED depth and XYZ.
- Every result records crop, prompt candidates, selected mask, verifier decision, scores, and latency.

## Image inspection commands

Open the latest full 1280 x 720 service frame:

```bash
xdg-open /home/team/VLA_Model_Work/SAM_3_implementation/outputs/service/20260804_103012_681818/frame.png
```

Open an earlier frame that already used the saved `448,360,384,360` crop:

```bash
xdg-open /home/team/VLA_Model_Work/SAM_3_implementation/outputs/06/02/2026-1_test/cropped/run_001/round_001/frame.png
```

When reporting a new crop, provide either `x,y,width,height` or the number of pixels to remove from the left, top, right, and bottom. Convert the latter using:

```text
x = left removal
y = top removal
width = 1280 - left removal - right removal
height = 720 - top removal - bottom removal
```
