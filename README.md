# SAM 3.1 ZED Segmentation

## Quick start

Two resident perception modes are retained in this branch:

- `./sam3` runs the current versioned v1/v2 service without Grounding DINO.
- `./sam3-dino` runs the bounded Grounding DINO proposal service. It uses the
  same cached SAM/Qwen models, ZED camera, port `8765`, and output contracts, so
  the two launchers must be run sequentially.

Run the bounded DINO path with:

```bash
./sam3-dino "pick up the orange and grey box"
./sam3-dino stop
```

See `GROUNDING_DINO_PIPELINE.md` for its proposal thresholds, artifacts, and
validation gates.

From this directory, submit an instruction with one command:

```bash
./sam3 "the orange box inside the blue bin"
```

Normal requests use version 2: the client first calls `/v2/interpret`, then sends
the exact sealed command envelope to `/v2/segment`. Qwen interprets every
command. The segmentation call validates hashes and source evidence but never
re-parses the sentence.

Version-2 segmentation is fail-closed during rollout. Production can be enabled
only after the image-backend parity, relation calibration, frozen-scene,
language, and per-relationship live gates are combined into an approved
aggregate report:

```bash
export SAM3_V2_ENABLED=1
export SAM3_SAM_BACKEND=image
export SAM3_IMAGE_PARITY_REPORT="$PWD/evaluation/sam_image_parity_approved.json"
export SAM3_V2_RELATION_THRESHOLD_REPORT="$PWD/evaluation/relation_thresholds_approved.json"
export SAM3_V2_RELEASE_REPORT="$PWD/evaluation/v2_release_approved.json"
./sam3 restart
```

Until those gates are approved, use `./sam3 interpret "..."` for language-only
testing or `./sam3 --v1 "orange and white box"` for the retained version-1
rollback path. `/segment`, `/v1/segment`, and the robot bridge remain unchanged.

To collect the frozen, shadow, and live evidence needed to build the aggregate
release report, use the separate evaluation flag. It opens only the same
perception-only endpoint and still requires the approved image-parity and
relation-calibration reports; it does not require or bypass the final report:

```bash
export SAM3_V2_ENABLED=0
export SAM3_V2_EVALUATION_MODE=1
export SAM3_SAM_BACKEND=image
export SAM3_IMAGE_PARITY_REPORT="$PWD/evaluation/sam_image_parity_approved.json"
export SAM3_V2_RELATION_THRESHOLD_REPORT="$PWD/evaluation/relation_thresholds_approved.json"
./sam3 restart
```

After every release gate passes, unset `SAM3_V2_EVALUATION_MODE`, set
`SAM3_V2_ENABLED=1`, and provide `SAM3_V2_RELEASE_REPORT` as shown above.

If a compatible service was already started manually from the local `.venv`, `./sam3` detects and reuses it instead of trying to start a conflicting Docker container. New managed starts use Docker for consistent lifecycle handling.

Quotes are optional when the instruction contains only ordinary words:

```bash
./sam3 get the rightmost pasta box
```

The standard live ZED crop is `448,360,384,360` at HD720.
The default SAM mask-presence confidence threshold is `0.10`; all later Qwen,
mask-area, depth-quality, and robot-target safety gates still apply.

## Commands worth remembering

```bash
./sam3 "your instruction"  # Normal use; starts automatically
./sam3 interpret "..."     # Qwen + schema/hashes only; no frame capture
./sam3 --v1 "..."          # Explicit version-1 rollback
./sam3 status              # Check health
./sam3 logs                # Follow logs; Ctrl-C stops following only
./sam3 stop                # Release the ZED camera
```

Less common maintenance commands:

```bash
./sam3 restart             # Apply service/configuration changes
./sam3 rebuild             # Only after Dockerfile/dependency changes
./sam3 --help              # Complete command summary
```

After pulling or editing service code, run `./sam3 restart` once so the resident process loads the changes. The status response then reports the active crop and Qwen model.

Use the direct SAM path without the Qwen fallback when deliberately testing it:

```bash
./sam3 ask --v1 --no-agent-fallback orange box
```

## Version-2 identification pipeline

1. Qwen2.5-VL-7B returns one strict command envelope with the raw command,
   exact visual evidence span, allowlisted action, separate destination, target,
   up to three anchors, entity-scoped attributes/selectors, and up to four
   normalized relationships.
2. Deterministic validation checks literal evidence, rejects repeated ambiguous
   spans and unsupported references, normalizes only declared relationship
   operators, and seals both the visual intent and complete envelope with
   SHA-256 hashes. There is one format-only retry and no semantic fallback.
3. Each entity receives at most four open-vocabulary prompts: exact mention,
   attribute-qualified head noun, noun-modifier plus head noun, and head noun.
4. One Qwen visual call proposes boxes for every entity. The parity-approved
   SAM image model refines them through its interactive box interface while
   reusing one image embedding per unique workspace/tile/context view.
5. Target and anchor candidates are generated and deduplicated in independent
   role-scoped pools. Relation-aware crops cover containment, support,
   proximity, directional half-planes, and depth-order context.
6. Deterministic code records every target/anchor candidate-pair measurement for
   `inside`, `on`, `left_of`, `right_of`, `above`, `below`, `near`, `next_to`,
   `in_front_of`, and `behind`. Missing required geometry is unavailable, never
   a pass.
7. Qwen receives the unmodified image, T#/A# overlay, high-resolution entity
   crop sheets, and measurements. It may select one target and matching anchors
   or return `no_match`.
8. A target is accepted only if SAM produced its mask, Qwen approved every
   entity, every deterministic relation and entity selector passed, and the
   existing workspace, mask-area, depth-coverage, and depth-spread gates passed.

Version 2 is perception-only. Responses always contain `robot_target: null` and
`motion_permitted: false`; this path does not publish or write a robot target.
The version-1 structured parser and API are retained unchanged for rollback.

## Why Docker is present

Docker is the service runtime, not the normal user interface. It provides:

- The NVIDIA GPU runtime and required library paths
- ZED camera device and SDK access
- A consistent Python/model environment
- A long-lived process so SAM and Qwen stay loaded
- Automatic restart after a crash or reboot

You should not normally run `docker build`, `docker compose up`, `docker compose logs`, or `docker compose stop` yourself. The `./sam3` command wraps those operations.

Avoid running the Docker service and a standalone camera script simultaneously because only one process can own the ZED camera. Run `./sam3 stop` before manual Task 5 or Task 7 camera diagnostics.

## Output

Each request prints its JSON result and saves the frame, overlay, mask metadata, depth results, and request log beneath:

```text
outputs/service/<timestamp>/       # v1
outputs/service/v2/<timestamp>/    # v2
```

Important artifacts include:

- v2 `command_envelope.json` with both identity hashes;
- `qwen_visual_grounding.json`, `qwen_verification.json`, and
  `relationship_measurements.json`;
- T#/A# candidate overlays, per-entity crop sheets, and independent target and
  anchor mask artifacts;
- v1 `grounding_intent.json` and, in shadow mode, `intent_shadow.json`;
- `candidate_generation.json`, all source-run JSON files, and binary candidate
  masks;
- numbered Qwen candidate/zoom images and its strict decision record;
- `overlay_direct.png`, `final_masks/*_crop.png`, `final_masks/*_full.png`;
- `result.json` containing coordinate spaces, scores, gates, and latency.

## Repeatable validation

Validate the reviewed 120-command v2 corpus without camera or robot motion:

```bash
./sam3 v2-language
```

The checked-in corpus contains reviewed Whisper-style seed variations but no
raw site Whisper capture. Append at least one reviewed real capture with the
`actual_whisper` coverage tag before production; the aggregate gate refuses to
treat the seed wording as actual speech evidence.

Exercise pass/fail/unavailable behavior on the checked-in 60-scene synthetic
geometry safety fixture:

```bash
./sam3 v2-synthetic
```

The synthetic fixture does not measure visual recall. Capture at least 60
human-reviewed frozen camera scenes and label each target and anchor mask.
First create image-backend parity at IoU 0.95 or higher and calibrate ordinary
and safety cases for every supported relationship:

```bash
./sam3 image-parity \
  --manifest evaluation/sam_image_parity_manifest.json \
  --output evaluation/sam_image_parity_approved.json

./sam3 calibrate-relations \
  --manifest evaluation/relation_calibration_manifest.json \
  --output evaluation/relation_thresholds_approved.json
```

Start the explicit evaluation mode described above, run the frozen scenes
through v2 without motion, and score their saved `result.json` files:

```bash
./sam3 score-v2-frozen \
  --manifest evaluation/v2_frozen_results_manifest.json \
  --output outputs/v2_frozen_evaluation.json
```

The strict manifest format is demonstrated by
`evaluation/v2_frozen_results_manifest.example.json`. The report measures raw
target/anchor candidate recall, relationship-pair accuracy, final selection,
IoU/Dice, malformed Qwen responses, safety-set acceptance, and end-to-end p95
latency including Qwen interpretation.

After the frozen and live reports exist, build the aggregate approval:

```bash
./sam3 v2-release-gates \
  --language-report outputs/v2_language_evaluation.json \
  --frozen-report outputs/v2_frozen_evaluation.json \
  --image-parity-report evaluation/sam_image_parity_approved.json \
  --relation-threshold-report evaluation/relation_thresholds_approved.json \
  --live-report outputs/live_inside.json \
  --live-report outputs/live_on.json \
  --live-report outputs/live_left_of.json \
  --live-report outputs/live_right_of.json \
  --live-report outputs/live_above.json \
  --live-report outputs/live_below.json \
  --live-report outputs/live_near.json \
  --live-report outputs/live_next_to.json \
  --live-report outputs/live_in_front_of.json \
  --live-report outputs/live_behind.json \
  --output evaluation/v2_release_approved.json
```

The original 40-command version-1 shadow corpus remains available:

```bash
./sam3 intent-shadow
```

Add `--qwen` for the model-backed comparison when no other process is using the
GPU. For a labeled frozen-frame A/B manifest:

```bash
./sam3 evaluate \
  --manifest evaluation/frozen_manifest.json \
  --variant qwen7-crop \
  --expected-qwen-model Qwen/Qwen2.5-VL-7B-Instruct
```

Run the 20-round live acceptance check without robot motion:

```bash
./sam3 soak --expected present --reference-mask /path/to/full-mask.png \
  --relationship inside \
  "small orange and white box inside the blue bin"
```

The aggregate gate accepts only v2 live reports with exactly 20 rounds, a
reviewed full-frame reference mask, IoU threshold of at least 0.5, and at least
19 correct identity-and-mask rounds for each relationship.

To run the 3B comparison, restart with the alternate cached model, evaluate,
then restore 7B:

```bash
SAM3_AGENT_QWEN_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct ./sam3 restart
SAM3_AGENT_QWEN_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct ./sam3 restart
```

The detailed architecture and development roadmap are documented separately in
`V2_API.md`, `Daemon_plan.md`, `Second_plan.md`,
`Segmentation_Identification_Improvement_Plan.md`, and
`Multiscale_Multiprompt_Candidate_Generation_Plan.md`.
