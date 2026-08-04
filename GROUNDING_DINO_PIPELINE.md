# Grounding DINO → SAM 3.1 → Qwen Deployment

This branch adds a bounded proposal pipeline to the resident ZED/SAM service.
The default request path is:

```text
deterministic request parser
  -> one Grounding DINO Base multi-phrase pass
  -> at most three deduplicated boxes
  -> one SAM box prompt per proposal on one image state
  -> one fail-closed Qwen2.5-VL-7B verification stage
  -> existing spatial/depth selection
```

The service never downloads a model during startup or a robot request. Robot
motion must remain disabled until the frozen-frame and live perception gates at
the end of this document pass.

## Cache models before offline startup

Grounding DINO uses the official
[`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base)
Transformers checkpoint. Run the cache command while network access is allowed:

```bash
cd /home/team/VLA_Model_Work/SAM3_Projects/SAM3_With_GroundingDINO
./sam3 cache-dino
```

The command downloads only JSON/tokenizer/processor files and safetensors,
then reloads the configuration and processor locally to validate the snapshot.
The existing Qwen snapshot must also be present under the mounted Hugging Face
cache. At service runtime, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`local_files_only=True` are enforced. Missing or invalid DINO weights abort
startup with an explicit cache error.

## Start and inspect the service

The project launcher reuses the virtual environment, SAM source, and SAM
checkpoint from `SAM3_Without_GroundingDINO`, starts the service in the
background, and waits until SAM, Grounding DINO, Qwen, and the ZED camera are
ready:

```bash
cd /home/team/VLA_Model_Work/SAM3_Projects/SAM3_With_GroundingDINO
./sam3 start
```

Only one SAM service can own the ZED camera and port 8765. If the no-DINO SAM
service is running, `./sam3 start` recognizes it and stops it cleanly before
starting DINO. An unknown service occupying the port is never killed. If
another terminal or development session restarts the no-DINO Docker container
during DINO warmup, the launcher stops DINO and reports the conflict instead of
allowing two GPU/camera processes to compete.

Submit a bounded segmentation request with no agent fallback:

```bash
./sam3 "small orange box"
```

Check readiness, follow logs, and release the camera with:

```bash
./sam3 status
./sam3 logs
./sam3 stop
```

Set `SAM3_RUNTIME_PROJECT` only if the shared no-DINO project moves from
`/home/team/VLA_Model_Work/SAM3_Projects/SAM3_Without_GroundingDINO`.
DINO mode always uses the calibrated HD720 crop `448,360,384,360` and offline
model caches. A different resolution, crop, or `--no-crop` is rejected.

`grounding_dino` reports `loaded`, model ID, model class, BF16 dtype, CUDA
devices, evaluation mode, local-only status, thresholds, proposal cap, padding,
and the calibrated crop. The Compose health check also requires warmed Qwen 7B.

## Request modes and safety behavior

The relevant service options are:

```text
--pipeline-mode dino|legacy          default: dino
--dino-model MODEL                   default: IDEA-Research/grounding-dino-base
--dino-box-threshold FLOAT           default: 0.25
--dino-text-threshold FLOAT          default: 0.20
--dino-nms-iou FLOAT                 default: 0.50
--dino-max-proposals INTEGER         default/max: 3
--dino-box-padding FLOAT             default: 0.05
--warm-dino
```

Ordinary commands such as `pick the rightmost small orange box` use the
deterministic parser. The cached Qwen text parser is used only for an unresolved
or ambiguous target. Box/package/carton targets use a controlled phrase family
containing the exact target, category head, `box`, `package`, `carton`, and
`flat rectangular item`, bounded to five phrases in one DINO call.

A DINO request makes at most three SAM box calls and one visual verification
stage. If DINO has no usable proposal or no score/area-gated box mask, the
service makes one legacy SAM text call and verifies that result through the same
Qwen gate. A Qwen `no_match`, low-confidence answer, malformed answer, model
error, semantic mismatch, or selected area over 25% produces no target and does
not trigger another fallback.

The unbounded Meta agent is disabled by default. It is considered only after
the bounded DINO and legacy paths return no candidates and the caller explicitly
sets `use_agent_fallback=true`:

```bash
curl -X POST \
  'http://127.0.0.1:8765/segment?request=small%20orange%20box&use_agent_fallback=false'
```

Use `scripts/mask_client.py --agent-fallback ...` only for a deliberate
diagnostic request.

## Coordinate and response contract

The combined `sam_json` retains the existing crop-local SAM structure:

- `orig_img_w=384`, `orig_img_h=360`
- index-compatible `pred_masks`, mask-derived normalized `pred_boxes`, and
  `pred_scores`
- `presence_gate.kept_indices` indexing those arrays directly

Candidate order follows proposal order even when SAM scores differ. Each kept
candidate carries its DINO phrase/score, original and padded proposal boxes,
SAM prompt/score, mask geometry, and per-prompt latency. For the standard crop,
crop point `(u,v)` maps to full-camera point `(u+448,v+360)`. The response also
provides `selected_mask.center_xy_crop_pixels`,
`selected_mask.center_xy_full_pixels`, and the crop-to-full offset when exactly
one target survives.

All pre-existing robot-consumed fields remain present, including `num_kept`,
`scores`, `presence_gate`, `object_depth`, `zed_frame`, `sam_json`, `selection`,
and `result_json`. New top-level fields include `pipeline_mode`,
`candidate_generation`, `proposal_provenance`, `dino_proposals`,
`combined_candidates`, `selected_mask`, and `stage_timings`.

## Request artifacts

Every request directory under `outputs/service/<timestamp>/` contains the raw
crop and final `result.json`. DINO requests additionally retain:

```text
frame.png
dino_proposals.json
dino_sam/combined_candidates.json
dino_sam/combined_candidates.png
dino_sam/mask_001.png ... mask_003.png
dino_qwen_candidates.png
dino_qwen_candidate_zooms.png
dino_qwen_candidates.json
dino_qwen_verification.json
overlay_dino.png
result.json
```

Legacy fallback artifacts use the `legacy_` prefix. `stage_timings` records
capture, parsing, DINO, every SAM box prompt, legacy SAM if used, Qwen inference,
spatial selection, depth extraction, and total request time.

## Validation before robot handoff

Keep motion disabled and run a frozen set of at least 20 crop frames covering
target-present, target-absent, small-package, and package-inside-bin scenes.
Required gates are:

- proposal recall at least 90% on target-present frames;
- no accepted target on target-absent frames;
- correct final single-mask selection at least 85%;
- median end-to-end latency below 15 seconds and P95 below 20 seconds after
  warmup for deterministic commands;
- no OOM, restart, stale frame, or coordinate mismatch in a 20-request soak.

Then make one live text-only request with motion disabled. Audit the crop mask,
full-frame center, depth/XYZ, DINO provenance, and Qwen decision before enabling
the normal ROS handoff.

## Rollback

To retain the cached Qwen verifier and existing SAM/depth contract while
disabling DINO, start the service with `--pipeline-mode legacy` and omit
`--warm-dino`. No response consumer changes are required. For a code rollback,
return the `GroundingDino` branch to the verifier-only commit immediately before
the DINO integration; do not copy the Daemon branch's multiscale/V2 working-tree
changes into this branch.
