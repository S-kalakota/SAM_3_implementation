# SAM 3.1 Command-to-Segmentation Prototype

This folder is a prototype for turning a typed natural-language command into object segmentation masks on a saved image or a live ZED camera frame.

The near-term goal is simple: type a request such as `green object` or `green water bottle on top of the grey ramp`, run SAM 3.1 against the current scene, filter weak detections, and write an overlay image showing the mask or masks that were kept.

The longer-term goal is to make this the perception front end for a robot workflow: language request in, object mask out, then later depth and grasping can be added on top.

## Pipeline

```text
typed request
    |
    v
saved image or ZED RGB frame
    |
    v
SAM 3.1 direct prompt or Qwen-backed SAM 3.1 agent
    |
    v
masks, scores, boxes
    |
    v
presence gate
    |
    v
overlay image plus JSON summary
```

There are two ways to drive SAM 3.1:

1. Direct prompt path: send a clean phrase straight to SAM 3.1.
2. Agent path: send a more natural request to Meta's SAM 3.1 agent, backed by local Qwen-VL, so the agent can reason about wording and choose which masks to return.

## Project Layout

- `scripts/task2_sam31_image_prompt.py`: direct SAM 3.1 prompt on a saved image. Also owns shared helpers for mask conversion, presence gating, result summaries, and overlays.
- `scripts/task3_sam31_presence_gate.py`: wrapper for Task 3 presence-gate validation.
- `scripts/task4_sam31_overlay.py`: wrapper for Task 4 overlay generation.
- `scripts/task5_zed_live_prompt.py`: grabs a ZED RGB frame, optionally crops it, then runs the direct SAM prompt path.
- `scripts/local_qwen.py`: local Qwen-VL adapter for the agent. It defaults to local-cache-only model loading.
- `scripts/task6_sam31_agent.py`: runs Meta's SAM 3.1 agent on a saved image using local Qwen-VL and an injected SAM service.
- `scripts/task7_zed_live_agent.py`: grabs a live ZED frame and runs the Qwen-backed SAM agent.
- `checkpoints/`: ignored local SAM checkpoint files.
- `outputs/`: ignored generated JSON summaries, live frames, overlays, and agent workspaces.
- `third_party/sam3/`: ignored local checkout of the upstream SAM 3 repository.
- `intial plan.md`: original task breakdown and milestone plan.

## Current Status

As of July 2, 2026, the prototype has working code for Tasks 2 through 7:

- SAM 3.1 checkpoint loading is wired to `checkpoints/sam3.1/sam3.1_multiplex.pt`.
- Saved-image direct prompts work for some clean phrases, including `bottle`.
- Presence gating filters masks by score and area.
- Overlay generation writes PNG images with kept masks and labels.
- ZED live frame capture works, including crop support and preview mode.
- Cropped live-frame prompts have produced good results for `green object`.
- Full-frame prompts are less reliable than cropped prompts in the current artifacts.
- The Qwen-backed SAM agent has succeeded on examples such as `green object` and `green water bottle on top of the grey ramp`.
- Task 7 is implemented, but the available artifacts look more like partial/live agent experiments than a clean final run record.

## Typical Commands

Run direct SAM 3.1 on a saved image:

```bash
.venv/bin/python scripts/task2_sam31_image_prompt.py \
  --image outputs/live_sam_frame_green_object_crop_448_360_384_360.png \
  --prompt "green object" \
  --overlay-output outputs/result.png
```

Run the live ZED direct prompt path:

```bash
.venv/bin/python scripts/task5_zed_live_prompt.py \
  --prompt "green object" \
  --crop 448,360,384,360 \
  --overlay-output outputs/result_live.png
```

Run the Qwen-backed agent on a saved image:

```bash
.venv/bin/python scripts/task6_sam31_agent.py \
  --image outputs/live_sam_frame_green_object_crop_448_360_384_360.png \
  --request "green water bottle on top of the grey ramp" \
  --overlay-output outputs/result_agent.png
```

Run the live ZED agent path:

```bash
.venv/bin/python scripts/task7_zed_live_agent.py \
  --request "green object" \
  --crop 448,360,384,360 \
  --overlay-output outputs/result_live_agent.png
```

## Important Runtime Notes

- The SAM 3.1 checkpoint is large and gated, so it is kept out of git.
- The Qwen adapter defaults to local Hugging Face cache only. Use `--allow-qwen-downloads` only when downloads are expected and allowed.
- FlashAttention 3 is optional and currently left off by default.
- The ZED scripts require the ZED SDK and importable `pyzed`.
- GPU inference is expected for the SAM and Qwen paths.

## Cleanup Targets

- Move shared helpers out of `task2_sam31_image_prompt.py` into a common module.
- Rename `intial plan.md` to `initial_plan.md`.
- Add a small smoke-test or health-check script for dependencies and checkpoint presence.
- Normalize output filenames so prompt names and JSON contents do not drift.
- Decide whether old probe outputs should stay in `outputs/` or be archived elsewhere.
- Produce one clean Task 7 run artifact after the live agent path is tuned.
