# Next time — where we are and what's next

Session notes, updated 2026-08-03. Continues `Second_plan.md`. All robot code lives in
`~/VLA_Model_Work/robot_ws` (package `fr5_bringup`), pushed to
`github.com/S-kalakota/cobot_ws` (the current canonical remote; the older
`S-kalakota/robot_ws` repository contains the same pre-update tree).

## Completed

### Milestone A1 — command + read the robot ✅ (verified live)
- Bringup: `ros2 launch fr5_bringup a1_bringup.launch.py sim:=false`
- Moved the arm from code (small j2 dip and back at 0.1 speed) and read the
  live TCP round-trip: standby TCP = `x -0.0384, y +0.1534, z +0.5740`.
- Facts recorded: MoveIt plans in `base_link`, group `fairino5_v6_group`,
  robot at `192.168.58.2` (firmware V3.9.x), ROS 2 Jazzy on the Thor.
- Soft stop that works without the e-stop:
  `ros2 topic pub --once /trajectory_execution_event std_msgs/msg/String "{data: stop}"`
  then re-command `--pose standby --execute` (MoveIt replans from wherever
  the arm is).

### Milestone A2 — gripper + fingertip TCP ✅
- Gripper (DH PGC140) works from code: `a2_gripper.py --open/--close/--pos/--stroke-test`.
  Measured (in `config/gripper.yaml`): max jaw stroke **50 mm**, pads touch at
  0%, pads 20×40 mm. Production grasp settings (from the old Lua program):
  close = 71% @ force 57, open = 100% @ force 49 → boxes are ~35 mm across.
- Fingertip TCP calibrated by pivot touches (`a2_tcp_calibrate.py`):
  offset `x 0.0025, y -0.0034, z 0.2323` m in `config/tcp_offset.yaml`,
  4-touch fit, RMS 2.8 mm. TF `base_link → tcp_link` is now the fingertip.
- The two-touch verify landed at ~6 mm. On 2026-07-16 this was accepted as the
  project-wide A2 TCP verification tolerance; A2 is complete. Dependent safety
  margins must include that 6 mm uncertainty. If Milestone B's hover test
  misses by >1 cm, the first diagnostic remains refitting the TCP with
  `a2_tcp_calibrate.py --samples 6 --write`, large wrist tilts (~40°), a
  bringup restart, and `--verify`.

### Milestone B1/B2 — camera-to-base transform ✅ (2026-07-16)
- B1 retained 8 well-spread ZED-point ↔ fingertip-touch pairs in
  `calib/calib_points.json`.
- B2 Kabsch fit passes: **7.532 mm RMS**, **6.542 mm mean**, **12.358 mm max**.
- Accepted transform is `calib/T_base_cam.json`; `b2_fit_transform.py --write`
  refuses failed fits and records per-point residuals plus the source hash.
- `a1_bringup.launch.py` now publishes `base_link → zed_left_optical` by
  default and rejects failed or stale calibration files. Restart bringup to
  load it, then verify with
  `ros2 run tf2_ros tf2_echo base_link zed_left_optical`.

### Milestone B3 — physical hover validation ✅ (2026-07-17)
- Five camera-selected points were tested across the bin pick area using the
  separated `b3_pick_point.py` → plan-only → `b3_hover.py --execute` workflow.
- The X/Y alignment was visually accurate at all five points and accepted
  within B3's 15 mm tolerance. Execution remained capped at 5% speed.
- The apparent 14–16 cm clearance was not a transform error: the commanded
  hover was 100 mm at the calibrated TCP, while the physical gripper reference
  used for the ruler measurement is about 50 mm away from that TCP reference.
- B3 is complete. The accepted B2 transform is physically validated for the
  current fixed camera/base installation.
- The proposed camera-drift tag check was removed on 2026-07-17. The camera is
  permanently bolted and zip-tied above the cobot. Any impact, maintenance,
  loosening, or repositioning of the camera or robot base requires repeating
  B1–B3 before vision-guided motion resumes.

### Experimental D0 — consistent live grabs at 35 mm depth (2026-08-03)
- The operator chose to defer C1/C2/C3 temporarily and try one narrow physical
  grab using the already validated click-to-base target.
- Initial physical attempts planned correctly but passed too high to secure the
  box. After increasing the depth experimentally, the operator reports that
  **35 mm below the clicked surface picks up the object consistently**.
- `d0_point_grab.py` now leaves the hover at 100 mm above the clicked surface
  and accepts `--grasp-depth-mm` from 0–100 mm. Its software default remains
  5 mm, so use the proven rig setting **`--grasp-depth-mm=35`** explicitly. It
  reads `plans.sqlite` directly, chooses the nearer left/right DB
  grab endpoint, and replays these recorded trajectories point-for-point with
  their saved timing: `standby_to_*grab`, `*grab_to_*lift`, and
  `*lift_to_standby`. MoveIt is used only for the short DB-grab-point ↔ new
  hover connections. Descent and retreat are straight Cartesian paths capped
  at 20 mm/s. The gripper opens to 100%, closes to **71%**, and remains at 71%
  after the final recorded trajectory reaches `standby`.
- Execution is refused unless the live arm is within 0.02 rad of the recorded
  DB standby start. Adjacent saved-trajectory endpoints are validated within
  0.005 rad before any motion; the current DB differs by less than 0.0001 rad.
- Plan-only is the default. Real execution requires both `--execute` and
  `--confirm-ungated-grab`, plus a target clicked within the last 10 minutes.
- Build and both left/right full plans pass. A full left-side mock execution
  replayed all 228 recorded DB points, reached the selected TCP point, retreated,
  and returned through the two saved return trajectories to `standby`. Live
  35 mm-depth trials now pick the object consistently and return to `standby`
  holding it at the commanded 71% gripper position.
- This experiment does not complete C1-C3 or D1/D2. There is still no table
  plane, object-height/width check, bin-wall gate, or environment collision
  scene. The 35 mm setting is an experimentally determined fixed depth, not an
  automatic object-center or support-plane-aware grasp calculation.

### Infrastructure fixed along the way
- `~/fairino5` was renamed to `~/fairino_ros_connector`; re-pointed the
  `robot_ws/src/fairino_description` symlink and fixed broken absolute
  `libfairino.so.2` symlinks in the old install (now relative).
- The host Fairino driver was a stale build with no gripper channel. Replaced
  `~/fairino_ros_connector/install/fairino_hardware/.../libfairino_hardware.so`
  with a fresh build of `fairino_hardware_v3_9_6` (source symlinked into
  `robot_ws/src`, backup kept as `*.stale-20260715`). This driver hosts
  `/fairino_remote_command_service` + `/nonrt_state_data` inside
  ros2_control (one shared RPC session — same as production).
- `leftGrab` / `rightGrab` SRDF poses were applied 2026-07-16 from the DB at
  that time. The trajectories were updated again on 2026-07-31, so D0 does not
  rely on the SRDF copy: it reads the current `plans.sqlite` endpoints and full
  recorded paths directly.

### Hard-won gotchas (read before debugging)
1. **`tcp_offset.yaml` is baked in at LAUNCH time.** Editing it does nothing
   until the bringup is restarted. Two verify runs failed at 31/42 mm purely
   because the old placeholder (z=0.15) was still loaded. Check what's live:
   the running URDF on `/robot_description` must show the yaml's values.
2. Ctrl-C in the launch terminal kills the whole driver stack — the script
   terminal is where Ctrl-C is safe.
3. Only ONE RPC session to the FR5: stop the docker production stack
   (`fairino_plan_executor`) before `sim:=false`, and vice versa.
4. Every new terminal needs `source ~/VLA_Model_Work/robot_ws/install/setup.bash`.
5. `/home/team/fairino_db` is empty; the real DB is
   `~/fairino_ros_connector/fairino_ros_controller/db/plans.sqlite`
   (11 taught trajectories = the full pick choreography, incl. a fixed drop
   pose — answers Second_plan open question #3).

## Milestone A3 — taught positions ✅ (confirmed complete 2026-07-17)

- The required fixed-station positions are already known; no additional
  position-finding or teaching work is needed. The `home`, `hover_bin_left`,
  `hover_bin_right`, and `drop` waypoints, plus the table/bin positions needed
  later by the C3 gate, are accepted as complete for the current layout.
- The workspace-safety code (`a3_measure_workspace.py`,
  `a3_planning_scene.py`, `a3_tcp_watchdog.py`, `config/workspace.yaml`) remains
  intentionally deleted after the 2026-07-16 scope change. It is recoverable
  from git history (`6f65f44`) if the layout changes.
- Transit remains limited to the known taught joint-space waypoints. A3 is
  complete.

## What's next (in order)

1. **Repeat the working D0 live grab:** click the box, run the complete sequence
   at the proven 35 mm depth plan-only, inspect it, then explicitly execute it:

   ```bash
   ros2 launch fr5_bringup a1_bringup.launch.py sim:=false
   ros2 run fr5_bringup b3_pick_point.py
   ros2 run fr5_bringup d0_point_grab.py \
     --target-file=/tmp/fr5_b3_target.json --grasp-depth-mm=35
   ros2 run fr5_bringup d0_point_grab.py \
     --target-file=/tmp/fr5_b3_target.json --grasp-depth-mm=35 \
     --execute --confirm-ungated-grab
   ros2 run fr5_bringup a2_gripper.py --open
   ```

   The arm must start at the recorded DB standby point. Stop unless preflight
   reports three validated DB trajectories plus four planned new segments.
   Keep the full path clear and a hand on the e-stop. The box remains held at
   71% when the saved return choreography reaches `standby`.
2. **Characterize reliability:** record the number of 35 mm-depth attempts and
   successful pickups rather than relying only on the qualitative
   "consistently" result.
3. **Deferred, not complete:** C1 support geometry, C2 `GraspTarget`, and C3
   safety gating remain the path from this experiment to a repeatable picker.

## Also parked
- `mask_service.py` (711-line resident masking daemon, Daemon plan) is
  committed on the `Daemon` branch of SAM_3_implementation — not merged to
  `main` yet.
- The repo's symlinks (`src/fairino_description`, `src/fairino_hardware_v3_9_6`)
  are absolute paths — they dangle on any machine that isn't the Thor.
