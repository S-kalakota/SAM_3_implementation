# Frozen-frame segmentation evaluation

Copy `frozen_manifest.example.json`, add representative RGB frames and binary
full-frame ground-truth masks, and keep that manifest fixed across variants.
The resident service exposes `/v1/evaluate`; it uses the loaded SAM/Qwen models
but never opens a new camera frame and never creates a robot target.

Run the four planned variants (start the service with the named Qwen model
before each corresponding command):

```bash
.venv/bin/python scripts/evaluate_segmentation.py \
  --manifest evaluation/frozen_manifest.json \
  --variant qwen3-full \
  --crop none \
  --expected-qwen-model Qwen/Qwen2.5-VL-3B-Instruct

.venv/bin/python scripts/evaluate_segmentation.py \
  --manifest evaluation/frozen_manifest.json \
  --variant qwen7-full \
  --crop none \
  --expected-qwen-model Qwen/Qwen2.5-VL-7B-Instruct

.venv/bin/python scripts/evaluate_segmentation.py \
  --manifest evaluation/frozen_manifest.json \
  --variant qwen3-crop \
  --crop 448,360,384,360 \
  --expected-qwen-model Qwen/Qwen2.5-VL-3B-Instruct

.venv/bin/python scripts/evaluate_segmentation.py \
  --manifest evaluation/frozen_manifest.json \
  --variant qwen7-crop \
  --crop 448,360,384,360 \
  --expected-qwen-model Qwen/Qwen2.5-VL-7B-Instruct
```

Each report records raw candidate recall, correct-object selection accuracy,
absent-target false-positive rate, IoU/Dice, malformed verifier responses, and
end-to-end latency. Reports are written beneath
`outputs/evaluation_reports/<variant>/`.

## Version-2 rollout fixtures

`v2_language_commands.json` contains 120 reviewed command/entity-graph cases,
12 primary cases for each supported relationship. Its transcript-style seeds
do not impersonate raw site Whisper data: append reviewed real captures with an
`actual_whisper` coverage tag before production. Run the corpus with the cached
Qwen 7B model:

```bash
./sam3 v2-language --output outputs/v2_language_evaluation.json
```

`v2_synthetic_frozen_scenes.json` contains 60 deterministic mask/depth scenes
covering valid, missing-target, missing-anchor, relationship-false,
attribute-swapped, and multiple-match cases for all ten relationships:

```bash
./sam3 v2-synthetic
```

Those scenes test geometry and orchestration only. They do not count as the
human-reviewed frozen camera set for target/anchor candidate recall, IoU/Dice,
or SAM backend parity. Before enabling `SAM3_V2_ENABLED`, collect at least 60
real frozen scenes, mark each case as reviewed, and label the full-frame target
and anchor masks. First use:

```bash
./sam3 image-parity \
  --manifest evaluation/sam_image_parity_manifest.json \
  --output evaluation/sam_image_parity_approved.json

./sam3 calibrate-relations \
  --manifest evaluation/relation_calibration_manifest.json \
  --output evaluation/relation_thresholds_approved.json
```

Parity approval requires at least 60 cases at IoU 0.95 or higher. Relation
calibration requires at least 60 labeled cases, with ordinary and safety cases
covering all ten relationships and zero safety false accepts. With those
artifacts, start `SAM3_V2_EVALUATION_MODE=1`, run the fixed scenes through the
no-motion v2 path, and score the saved results:

```bash
./sam3 score-v2-frozen \
  --manifest evaluation/v2_frozen_results_manifest.json \
  --output outputs/v2_frozen_evaluation.json
```

Start from `v2_frozen_results_manifest.example.json`. The service refuses to
start with the image backend unless the parity gate passes. Evaluation-mode v2
additionally requires the calibrated relation report. Production v2 also
requires an aggregate `v2-release-gates` report proving the 98% language, 95%
candidate/final-selection, zero wrong-acceptance, 19/20-per-relationship, and
30-second end-to-end p95 gates.
