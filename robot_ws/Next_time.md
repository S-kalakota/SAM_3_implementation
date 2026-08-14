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

### Experimental D0 — retry choreography live-validated (2026-08-03)

- The operator temporarily deferred C1/C2/C3 and completed constrained physical
  clicked-point grasp trials. `d0_point_grab.py` consumes the accepted
  camera-to-base target, reads `plans.sqlite`, chooses the nearer proven
  left/right grab endpoint, and replays the recorded standby/grab/lift/standby
  trajectories point-for-point. MoveIt plans only the short DB-endpoint ↔ new
  hover links plus straight Cartesian descents and retreats.
- The hover remains 100 mm above the selected surface. Grasp depth accepts
  0–100 mm, but that is an experimental software range rather than a verified
  safe range. Both 35 mm and 45 mm have picked the flexible box intermittently;
  neither is a reliable universal depth. Large retry steps such as 20 mm and a
  final 90 mm descent are not accepted operating defaults.
- Grasp verification uses measured jaw position. A close passes when the final
  position is at least 8 percentage points above the commanded close value:
  60% requires >=68%, 40% requires >=48%, and 0% requires >=8%. The gripper's
  reported current has remained 0% in these trials, so current is logged but is
  not a usable decision signal yet. Because the box is flexible, it can deform
  all the way to the command and be incorrectly classified as an empty close;
  position-only verification is therefore useful but not sufficient.
- A live three-attempt run validated the complete redo choreography at 25, 30,
  and 35 mm. After attempts 1 and 2, the gripper confirmed a 100% reopen, the
  arm retreated vertically to hover, reset only through the selected DB grab
  anchor, re-approached, and descended 5 mm deeper. After the third failure the
  arm returned through the exact DB lift path to standby; its final reopen is
  the unreliable 0%-for-100% case noted below. Grasp-point errors were
  0.12–0.17 mm and final retreat-hover errors were 0.17–0.19 mm. This proves
  the retry motion on real hardware; it does **not** prove successful
  reacquisition.
- Retries currently change only Z. They reuse the same XY and fixed left/right
  DB wrist orientation, so a point that is off-center or a rotated box will
  still be missed at every depth. Segmentation-center targeting is the next fix;
  mask-derived yaw follows immediately after it.
- `--grasp-retries` allows up to three retries and `--retry-step-mm` up to 20
  mm. Every close emits `GRASP_ATTEMPT_RESULT` with schema
  `fr5.grasp_attempt.v1`, which is the feedback contract the VLA executive will
  consume. A final failure returns the arm to standby when recovery feedback
  remains trustworthy.
- Gripper command handling was hardened: `MoveGripper` now waits long enough
  for the server's motion window, and a timeout is treated as an unknown result;
  it is not followed by automatic reconfiguration, activation, or a duplicate
  motion command. Two reliability issues remain before unattended VLA motion:
  frequent direct `fault=1` telemetry after activation, and a live reopen that
  reported position 0% for a commanded 100% while `motion_done` appeared true.
  Reopen success must be based on the measured position near the requested
  value, not the motion-done flag alone, and startup needs a deliberate
  activation/health wait.
- Plan-only remains the default. Execution requires `--execute`,
  `--confirm-ungated-grab`, a fresh target, and the arm at the recorded DB
  standby start. This experiment still has no table plane, object-width/bin-wall
  gate, or environment collision scene.

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

1. **VLA input — voice to constrained intent.** Add speech-to-text, then map
   the transcript to a small command schema such as `action`, `object`, and an
   optional spatial qualifier. Show or speak back the interpreted request
   before motion. The language model selects intent; it never emits joints,
   poses, or raw gripper commands.
2. **VLA target — requested mask to a robust center point.** Connect the
   resident `/segment` service, select the requested object's mask, and start
   from its centroid. If that pixel is outside an irregular mask or has invalid
   depth, use the nearest valid in-mask pixel or the mask's interior
   distance-transform maximum. Take median depth from a small in-mask patch,
   reject sparse/noisy depth, transform through accepted `T_base_cam.json`, and
   write the same target contract that D0 already consumes.
3. **Prove perception before grasping.** First display the transcript, selected
   mask, chosen center, depth, and base-frame XYZ without moving. Then run
   plan-only and 100 mm hover-only trials at varied box positions. Only after
   those pass should a confirmed one-object pickup call the existing D0 motion
   layer and feed `fr5.grasp_attempt.v1` results back to the VLA.
4. **Harden gripper state before unattended execution.** Add an activation
   settle/health check, reject direct `fault=1`, and require measured reopen
   position near the command. Collect successful/empty/flexible-box trials;
   use current only if nonzero readings eventually separate the outcomes.
5. **Add orientation immediately after center-pick works.** Estimate the
   selected mask's major/minor axes, convert the grasp axis through camera 3D
   into base-frame yaw, align the jaws across the short dimension, preview it,
   and test rotated boxes at 0/30/45/60/90 degrees. Keep the top-down approach
   and choose the equivalent reachable wrist yaw nearest a proven DB
   orientation. Low-confidence orientation must fall back to the current DB
   side or refuse autonomous execution.
6. **Then finish the production gates and executive.** Add support-plane,
   width, depth-quality, bin-wall, and reachability checks, followed by the
   complete voice-commanded pick/place loop. FoundationPose remains the later
   upgrade for tilt and full 6-DoF pose.

## Also parked
- `mask_service.py` (711-line resident `/segment` daemon) is committed on the
  `Daemon` branch of `SAM_3_implementation`, not merged to its `main`. It is now
  a required input to the next VLA milestone: merge it or run that branch and
  define a stable request/response adapter before wiring voice commands to D0.
- The repo's symlinks (`src/fairino_description`, `src/fairino_hardware_v3_9_6`)
  are absolute paths — they dangle on any machine that isn't the Thor.
