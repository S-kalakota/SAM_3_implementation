# Language request to FR5 target

`vla_pick_target.py` connects the existing constrained language parser, the
resident SAM 3.1/ZED service, and the FR5 target-file interface. It does not
move the robot.

## 1. Start the resident SAM service

The service code currently lives on the local `Daemon` branch:

```bash
cd ~/VLA_Model_Work/SAM_3_implementation
git switch Daemon
.venv/bin/python scripts/mask_service.py \
  --host 127.0.0.1 \
  --port 8765 \
  --selection-roi 430,380,430,170
```

Wait for `service ready`. The service owns the ZED while it is running.

## 2. Build and make one no-motion target

```bash
cd ~/VLA_Model_Work/robot_ws
colcon build --packages-select fr5_bringup
source install/setup.bash

ros2 run fr5_bringup vla_pick_target.py \
  --text "pick up the rightmost yellow box"
```

The bridge writes:

- `/tmp/fr5_vla_target_audit.png`: selected box, center, depth, and base XYZ;
- `/tmp/fr5_vla_target.json`: schema-1 surface target accepted by the existing
  B3/D0 tools.

It refuses zero or multiple unresolved masks, mismatched object/selector
metadata, stale frames, low-confidence or sparse/noisy depth, the wrong ZED
view/resolution, a large box-center versus mask-XYZ disagreement, or a target
outside the calibrated B1 envelope.

For push-to-talk, run the source script with the VLA environment that contains
Whisper:

```bash
~/VLA_Model_Work/VLA_project/.venv/bin/python \
  ~/VLA_Model_Work/robot_ws/src/fr5_bringup/scripts/vla_pick_target.py \
  --voice --voice-duration 5
```

## 3. Validate before any grasp

Inspect the audit overlay, then run the printed plan-only hover command:

```bash
ros2 run fr5_bringup b3_hover.py \
  --target-file=/tmp/fr5_vla_target.json
```

Only after repeated visual and hover validation, plan the D0 sequence:

```bash
ros2 run fr5_bringup d0_point_grab.py \
  --target-file=/tmp/fr5_vla_target.json
```

Both commands above are plan-only. The language/SAM bridge intentionally has
no execution flag; live motion remains an explicit, separately reviewed D0
operation.
