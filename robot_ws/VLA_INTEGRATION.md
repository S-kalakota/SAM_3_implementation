# Language request to checked FR5 target

`vla_pick_target.py` connects the text/voice command parser, the resident
Grounding DINO/SAM 3.1/Qwen/ZED service, and the FR5 target-file interface. It
never moves the robot.

## Fast path

For the normal two-terminal workflow, run this in the first terminal:

```bash
./robot_ws/scripts/start_vla_pickup.sh
```

It stops the system-managed ZED RTSP stream when necessary (and may ask for
your `sudo` password), starts the resident DINO service, builds the package,
and launches the real FR5 stack. It stays in the foreground and commands no
trajectory.

In a second terminal, create a target and run the complete no-motion review:

```bash
./robot_ws/scripts/run_vla_pickup.sh \
  "pick up the grey and orange box"
```

The runner allows 45 seconds for a freshly captured frame to complete the
current Qwen verification stage and waits up to 45 seconds for the FR5 driver
to publish live joints; keep the scene still during vision capture. It opens
the audit image and plans the complete pickup sequence.
Physical motion requires an explicit `--execute`; there is no subsequent typed
confirmation, so motion starts immediately after target creation and successful
motion preflight. The pickup still uses an internal 100 mm approach waypoint
but does not pause for a separate hover. A clean empty close retries at 10 mm
deeper increments, for three total default attempts at 5, 15, and 25 mm below
the selected surface. The fingertip TCP also receives a fixed 47 mm downward
correction in `base_link`: `[0, 0, -0.047]` metres. It never adds an X/Y
correction when the wrist is tilted, and the calibrated TCP transform itself
is not changed.
A close that remains more than 95% open is classified as likely side contact,
reopened before retreat, and is not eligible for a deeper retry.

## 1. Start the resident perception service

```bash
cd /path/to/grounded-cobot-vla
./sam3-dino start
```

After pulling or editing perception code, use `./sam3-dino restart` once. The
service owns the ZED camera while it is running. `./sam3-dino status` reports
the active crop, DINO proposal thresholds, SAM presence threshold, Qwen model,
verifier policy, and depth-refinement gates.

## 2. Build and create one no-motion target

```bash
cd /path/to/grounded-cobot-vla/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fr5_bringup
source install/setup.bash

ros2 run fr5_bringup vla_pick_target.py \
  --text "pick up the rightmost yellow box"
```

The bridge separates the robot action/destination from one versioned visual
intent. It sends that intent and its SHA-256 hash to the DINO service's
`/v1/segment` endpoint. The service validates the sealed intent before camera
capture, derives the DINO target phrase without re-parsing the command, and
returns the same source phrase, schema, complete intent, and hash. The bridge
refuses any identity mismatch.

The bounded DINO architecture is the default. The unbounded SAM/Qwen agent
fallback is disabled for robot-target creation unless `--agent-fallback` is
explicitly supplied for a diagnostic run. Source-region and relational intents
currently fail closed because this DINO path does not yet apply those semantics
deterministically; object attributes and one spatial selector are supported.

Accepted requests write:

- `/tmp/fr5_vla_target_audit.png`: selected box, center, depth, and base XYZ;
- `/tmp/fr5_vla_target.json`: schema-1 surface target for the existing B3/D0
  tools;
- the crop-local and full-frame binary masks beneath the SAM result directory.

Rejected requests write no new target. An older `/tmp/fr5_vla_target.json` can
still exist, so never use it after a refusal without checking its timestamp.

## Safety gates

The complete path is fail-closed and requires:

- schema-valid, unambiguous structured intent and matching identity hash;
- at most three Grounding DINO proposals refined by SAM, followed by Qwen
  identity verification on unmodified DINO-box crops;
- deterministic spatial-selector resolution after semantic verification;
- conservative geometry and ZED depth-discontinuity mask refinement;
- workspace membership and a mask score greater than `0.10`;
- at least 80% valid masked depth and at least 20 depth pixels; depth spread is
  recorded as evidence but does not reject a center target;
- a final SAM mask-centroid image target consistent within 40 mm of the
  registered ZED masked-XYZ reference;
- plausible projected object size and calibrated camera/base workspace bounds;
- a calibrated surface-height range and camera-XYZ consistency.

The Qwen verifier threshold (`0.70`) and oversized-mask limit (`0.25` of the
workspace crop) are independent of the SAM score threshold.

For push-to-talk, run the source script with the VLA environment containing
Whisper:

```bash
.venv/bin/python \
  robot_ws/src/fr5_bringup/scripts/vla_pick_target.py \
  --voice --voice-duration 5
```

## 3. Inspect and plan before any motion

```bash
xdg-open /tmp/fr5_vla_target_audit.png
jq . /tmp/fr5_vla_target.json

ros2 run fr5_bringup b3_hover.py \
  --target-file=/tmp/fr5_vla_target.json

ros2 run fr5_bringup d0_point_grab.py \
  --target-file=/tmp/fr5_vla_target.json
```

Both robot commands above are plan-only by default. The language/SAM bridge has
no live-execution flag; real motion remains a separately reviewed D0 operation.
