# Next time — where we are and what's next

Session notes, 2026-07-15. Continues `Second_plan.md`. All robot code lives in
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
- Updated `leftGrab` / `rightGrab` SRDF poses were applied 2026-07-16 from the
  latest taught start points in `db/plans.sqlite` (`leftgrab_to_leftlift` and
  `rightgrab_to_rightlift`). The SRDF now matches the trajectories that will be
  sampled for the A3 keep-in envelope.

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

## Written but not yet run — Milestone A3 (workspace safety)

Three scripts exist, are wired into the bringup, and ship **disabled** via
`config/workspace.yaml` (fail-closed until measured):
- `a3_measure_workspace.py` — interactive: 3 table touches (surface height),
  4 corner hovers (extent), 2 opposite-corner hovers per obstacle
  (gantry/monitor, +5 cm padding), then keep-in samples (visit named poses —
  can be done in sim). Jog-and-press-Enter, same as calibration; never moves
  the robot. Run with `--tcp-calibrated` now that the 6 mm A2 result is
  accepted.
- `a3_planning_scene.py` — publishes the keep-out boxes into MoveIt.
- `a3_tcp_watchdog.py` — 20 Hz keep-in guard: breach → cancel all motion
  goals → latch until `~/acknowledge` service is called. No auto-recovery by
  design. Refuses to arm with an uncalibrated TCP. Its table-floor offset is
  11 mm: 6 mm accepted TCP uncertainty + 5 mm nominal physical clearance.
  It also shrinks every effective keep-in wall inward by 6 mm, while the table
  collision surface is padded 6 mm above the measured plane.

## What's next (in order)

1. **A3 measure:** `ros2 run fr5_bringup a3_measure_workspace.py --tcp-calibrated`
   (~11 arm placements; precision not critical — err on bigger boxes).
2. **Review in RViz** (`rviz:=true`): enable `planning_scene.enabled` + each
   box in `workspace.yaml`, restart bringup, eyeball that boxes sit on the
   real table/gantry/monitor.
3. **Test keep-out:** a plan crossing the gantry must re-route; a target
   inside a box must be rejected.
4. **Test keep-in, in sim** (`sim:=true`): enable `keep_in` + `watchdog`,
   command a pose outside the box, confirm mid-flight cancel + latch +
   acknowledge flow.
5. **Enable for real.** A3 done-when: boxes visible, gantry plans re-route,
   out-of-bounds motion cancelled <100 ms, nothing moves until acknowledged.
6. **Milestone B — hand-eye calibration (the critical path).** B1 touch-point
   capture tool (click a pixel in the ZED view ↔ touch the same point with
   the fingertip; 8–12 pairs at varied heights), B2 solve `T_base←cam`
   (Umeyama fit, ≤8 mm RMS), B3 hover validation (≤15 mm at 5 spots — this is
   where a weak TCP would show), B4 AprilTag drift tripwire.
   After B2, everything downstream of the camera unblocks.
7. Then Milestone C (table plane, `GraspTarget`, safety gate) per
   `Second_plan.md` step list.

## Also parked
- `mask_service.py` (711-line resident masking daemon, Daemon plan) is
  committed on the `Daemon` branch of SAM_3_implementation — not merged to
  `main` yet.
- The repo's symlinks (`src/fairino_description`, `src/fairino_hardware_v3_9_6`)
  are absolute paths — they dangle on any machine that isn't the Thor.
