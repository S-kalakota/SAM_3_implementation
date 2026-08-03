# Second plan — Voice-guided VLA picking with the Fairino

Picks up where `intial plan.md` ended and is current through the 2026-08-03
physical grasp trials. Robot command/readback, gripper/TCP setup, taught
trajectories, camera-to-base calibration, and five-point physical hover
validation are complete. Clicked-point grasping now has measured close feedback
and a live-validated three-attempt retry path. The missing link is no longer the
camera transform; it is turning a spoken object request into a robust point and
orientation on the selected segmentation mask.

Goal of this plan: a person speaks a constrained request, the VLA selects the
correct mask, and the Fairino picks and places that object. MoveIt and the
proven `plans.sqlite` choreography remain the only motion layer. The language
model may interpret the request and respond to outcomes, but it never emits
joint values or bypasses deterministic target, safety, and execution checks.

## Stack decision: MoveIt, FoundationPose, AnyGrasp/Contact-GraspNet

The requested stack, and where each piece lands:

- **MoveIt — yes, core.** It's already the backbone of Milestones A, C and D (it's on the Thor, the Fairino has a `fairino_moveit_config`). Nothing to push back on.
- **FoundationPose — yes, but *after* the first pick, as Milestone F.** For flat boxes grasped top-down it adds nothing that mask + depth + `minAreaRect` don't already give: that's x, y, z, yaw, and the remaining two rotations are fixed by "top-down". It earns its keep the moment boxes can be **tilted, stacked, or replaced by non-box objects** — then you need the full rotation, and only FoundationPose provides it. It also fits our pipeline unusually well: FoundationPose **requires a segmentation mask as input**, and masks are the thing this project is best at (SAM mask → FoundationPose → 6-DoF pose is the intended usage). It stays off the critical path to the first pick because debugging a TensorRT/Isaac ROS graph at the same time as hand-eye calibration doubles the unknowns for zero first-pick benefit.
- **AnyGrasp / Contact-GraspNet — pushback: at most one of them, and probably neither for boxes.** These networks solve the *unknown-object* grasp problem: point cloud in, 6-DoF grasp candidates out, no object model needed. But once FoundationPose gives the full box pose and we know the box dimensions, grasp poses **fall out geometrically** (antipodal grasps across the short faces) — running a grasp net on top of a known pose + known mesh is redundant machinery. They become the right tool when we grasp **irregular objects that have no mesh**. If/when that trigger fires (Milestone G), prefer **Contact-GraspNet** over AnyGrasp: Contact-GraspNet is open research code with a maintained PyTorch port and accepts a segmentation mask to scope grasps to our object; AnyGrasp is a closed SDK with a per-machine license file, a commercial fee, and unverified aarch64/Jetson support — three separate ways to lose a week on the Thor. Check both licenses against company use before committing either.

The through-line that makes all of this cheap to change: **every grasp source produces the same `GraspTarget` object behind one interface** (Task C2). Geometric top-down, FoundationPose-derived, and grasp-net-proposed grasps are then swaps, not rewrites — the executive and the safety gate never care where a grasp came from.

## What we have today (validated on the rig)

- Masks from language via the SAM 3.1 agent loop (task7), with per-object depth
  and a 3D centroid in the **left-camera optical frame**. The resident
  `/segment` service exists on the `Daemon` branch of `SAM_3_implementation`;
  it is not yet merged or connected to this robot workspace.
- The accepted `calib/T_base_cam.json` converts those camera points to
  `base_link`: 7.532 mm RMS fit, followed by five successful physical hover
  checks across the bin. Bringup publishes the static camera TF.
- D0 reaches a clicked base-frame point through the nearest proven left/right
  database pose, performs a measured close, retries at deeper Z through that
  same DB anchor, and returns to standby. A live run completed the 25/30/35 mm
  sequence with 0.12–0.17 mm grasp-point error.
- Every D0 close emits `fr5.grasp_attempt.v1`, giving the future VLA a stable
  action/observation/outcome record without giving it low-level motion control.
- The 3B Qwen once chose the left box for “rightmost”; spatial qualifiers must
  therefore be resolved deterministically among returned masks.
- Grasping is not yet robust. Retries only change Z, the wrist orientation is
  still the chosen DB side, the flexible box can defeat position-only grasp
  verification, current reports 0%, and gripper activation/reopen health still
  needs hardening before unattended voice execution.

## Target data flow

```
 voice ─► speech-to-text ─► constrained intent {action, object, qualifier}
                                      │
                                      ▼
                         SAM `/segment` ─► selected mask
                                      │
                         robust in-mask center + depth
                                      │
                             accepted T_base←cam
                                      │
                 position-only GraspTarget / D0-compatible target
                                      │
                       overlay ─► plan ─► hover ─► confirmed pick
                                      │
                       `fr5.grasp_attempt.v1` outcome ─► VLA response

 after center-pick works:
 selected mask + masked 3D points ─► short-axis grasp direction ─► base-frame yaw
                                      │
                     oriented top-down GraspTarget ─► same motion/safety layer

 later for tilted objects:
 selected mask + mesh + depth ─► FoundationPose 6-DoF ─► same GraspTarget interface
```

The first version deliberately uses the center of the selected segmentation,
not a free-form point invented by the language model. “Center” means a valid,
well-supported pixel inside the mask: start with the mask centroid; if it lies
outside an irregular mask or lacks depth, use the nearest valid in-mask pixel
or the interior distance-transform maximum, then median a small in-mask depth
patch. The system refuses sparse or inconsistent depth.

---

# Milestone A: robot groundwork

Prove we can command the Fairino from code and read back where it is. Nothing vision-related here.

## Task A1: command + read the robot programmatically — DONE (2026-07-15)

**Status:** Complete. The FR5 ROS 2 + MoveIt bringup runs against the real
robot, a named pose can be commanded from code, live robot state/TCP readback
works, and MoveIt's planning frame is confirmed as `base_link`.

**Do:**
- Bring up the Fairino ROS 2 driver + MoveIt stack (or confirm it already runs, and where — Thor or another machine).
- From a script: move to a safe named pose via MoveIt, then read the live TCP pose back (Fairino SDK `GetActualTCPPose()` or TF `base_link → tool0`).
- Record which frame name MoveIt plans in (`base_link`? `world`?) — that exact frame is what we calibrate to in Milestone B.
- Record the **ROS 2 distro and JetPack version** on the Thor — that pair decides which Isaac ROS release Milestone F can use.

**Done when:** one script moves the arm to a safe pose and prints the live TCP pose continuously while you jog it.

## Task A2: pin down TCP + gripper — DONE (2026-07-16)

**Status:** Complete. The DH PGC140 opens/closes from `a2_gripper.py`; its
physical geometry is recorded, and the fingertip TCP is published as
`tcp_link`. The retained four-touch pivot fit has 2.8 mm RMS residual and
offset `[+0.0025, -0.0034, +0.2323]` m in the flange frame. The final
two-orientation check disagreed by approximately 6 mm; on 2026-07-16 that was
accepted as the project-wide TCP verification tolerance.

Measured clamp geometry:

- Maximum open jaw gap: **0.050 m (50 mm)**.
- Closed jaw gap: **0.000 m (0 mm)**.
- Usable jaw stroke (`open gap - closed gap`): **0.050 m (50 mm)**.
- Finger-pad width: **0.020 m (20 mm)**.
- Finger-pad length: **0.040 m (40 mm)**.

**Do:**
- Confirm the gripper model and how it opens/closes from code (ROS action? DIO? Modbus?).
- Measure the **max jaw stroke and finger pad size** — these numbers filter grasp candidates in C2/F5/G2 and decide whether the boxes are even graspable across their short side.
- Set the TCP offset (controller and/or MoveIt end-effector) so the reported TCP is the **gripper fingertip center**, not the flange.
- Sanity check: touch one fixed point on the table from two different wrist orientations; the reported TCP must agree within the accepted **6 mm** tolerance.

**Done when:** the two-orientation touch test agrees ≤ 6 mm, and gripper open/close works from a script. **Passed.**

## Task A3: taught waypoints (zones removed 2026-07-16) — DONE (confirmed 2026-07-17)

**Status:** Complete by operator confirmation. The required fixed-station
positions are already known and accepted for the current layout; no additional
position-finding or teaching work is required.

**Scope change (2026-07-16, final):** the station layout is fixed — the arm picks from one of **two bins (left / right)** and drops in the **same region**. All transit motion runs between hardcoded taught waypoints; vision only fine-positions the grasp inside a bin. Decision: **no planning-scene collision boxes and no keep-in watchdog.** The `a3_*` scripts and `workspace.yaml` were deleted from the repo on 2026-07-16 (recoverable from git history, commit `6f65f44`, if the layout ever changes back). Safety for live runs = taught joint-space waypoints + 10 % speed + dry-run default + hand on the e-stop.

Two notes carried forward into other tasks (numbers in code, not zones):
- The **table Z measurement stays** — one 3-touch measurement via the A1 TCP stream. It's not published anywhere; it becomes the floor clamp in the C3 gate (grasp Z may never go below it), because vision computes the descend target fresh every pick and waypoints can't cover it.
- Teach waypoints as **joint configurations**, not TCP poses — MoveIt samples a fresh path between any two poses, and joint-space goals with close start/goal keep transits consistent.

**Do:**
- Teach and store the fixed waypoints as joint configurations: `home`, `hover_bin_left`, `hover_bin_right`, `drop`.
- Touch the table at 3 spread-out spots; record the averaged Z (with the 6 mm TCP tolerance in mind) for the C3 floor clamp.
- Record each bin's interior extent in base frame (jog fingertip to the bin walls) — these numbers become the C3 "target inside active bin" check.

**Done when:** the required taught positions and gate coordinates are known for
the fixed station. **Passed by operator confirmation.**

---

# Milestone B: hand-eye calibration (the critical path)

One fixed rigid transform `T_base←cam` converts camera points to robot points: `p_base = T @ [x, y, z, 1]`. Both camera and robot base are bolted down, so it's constant until something physically moves. Everything downstream — centroid targets today, FoundationPose poses in Milestone F — rides on this one transform.

## Task B1: touch-point capture tool — DONE (2026-07-16)

**Status:** Complete. Eight well-spread camera-point ↔ fingertip-touch pairs
are retained in `calib/calib_points.json`.

**Do:**
- Small script: shows the live ZED left view; you click a pixel, it records that pixel's `XYZ` from `MEASURE.XYZ` (median of a 5×5 patch); then you jog the Fairino so the fingertip touches the same physical point and the script records the TCP position.
- Collect **8–12 point pairs spread across the whole workspace, including different heights** (put a box or block under some touches — coplanar points make the fit degenerate in Z).

**Done when:** `calib_points.json` holds ≥ 8 well-spread pairs.

## Task B2: solve the transform — DONE (2026-07-16)

**Status:** Complete. The accepted rigid fit is saved in
`calib/T_base_cam.json` with 7.532 mm RMS, 6.542 mm mean, and 12.358 mm maximum
residual. Bringup publishes it as `base_link → zed_left_optical`.

**Do:**
- Fit with Umeyama/Kabsch (no scale). Core of it:

```python
def fit_rigid_transform(cam_pts, base_pts):        # Nx3, Nx3
    cc, cb = cam_pts.mean(0), base_pts.mean(0)
    H = (cam_pts - cc).T @ (base_pts - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ cc
    return R, t                                     # p_base = R @ p_cam + t
```

- Report per-point residuals. Save `T_base_cam.json` (R, t, date, residuals).
- Also publish it as a **static TF** (`base_link → zed_left_optical`) whenever the ROS graph is up — one source of truth for MoveIt, RViz, and later Isaac ROS nodes (F2).

**Done when:** RMS residual ≤ ~8 mm and no single point is a wild outlier (an outlier means a bad touch — redo that point).

## Task B3: hover validation — DONE (2026-07-17)

**Status:** Complete. Five camera-selected points across the bin pick area were
validated with the separated click, plan, and 5%-speed execute workflow. X/Y
alignment was accurate at all five points and accepted within the 15 mm
tolerance. The apparent 14–16 cm physical gap was traced to measuring from a
gripper reference about 50 mm away from the calibrated TCP; the commanded TCP
hover remained 100 mm.

**Do:**
- Pipeline reports a point in base frame → command MoveIt to hover the TCP
  100 mm directly above it → measure the horizontal miss. Repeat at ≥ 5 spots
  spread across the reachable bin pick area.
- Interpret: constant offset everywhere = TCP or frame mistake; error growing toward the edges = poor calibration spread → add points there and refit.

**Done when:** miss ≤ ~15 mm at every test spot. **Passed.**

**Drift-tripwire task removed (2026-07-17):** the camera is permanently
bolted and zip-tied to the fixed station above the cobot, so no AprilTag or
ChArUco startup check will be added. Any impact, maintenance, loosening, or
physical repositioning of the camera or robot base invalidates
`T_base_cam.json`; repeat B1–B3 before enabling vision-guided motion again.

---

# Immediate Milestone V: voice-to-segmentation-center VLA pickup

This is the next work. It reuses the accepted camera transform and D0 motion
layer instead of replacing either with model-generated motion.

## Task V1: voice to constrained object intent

**Do:**

- Add speech-to-text and normalize the transcript into a small command object:
  `action`, `object/category`, and an optional spatial qualifier such as
  `leftmost`, `rightmost`, or `nearest`.
- Echo the transcript and parsed intent on screen and, where practical, speak
  it back. Require explicit confirmation before the first live-motion phase.
- Keep parsing separate from execution. Unsupported actions, uncertain object
  names, or multiple unresolved matches are refusals; neither speech nor the
  VLA may emit joints, arbitrary poses, or gripper RPC calls.
- Resolve spatial qualifiers deterministically over mask/base-frame metadata,
  not by asking Qwen to judge image position.

**Done when:** at least 20 representative spoken requests, including noise and
an intentionally unsupported request, produce the correct constrained intent
or a readable refusal without any robot movement.

## Task V2: selected mask to center target

**Do:**

- Merge or run the `Daemon` branch's resident `/segment` service and define a
  stable request/response adapter for category, masks, confidence, depth, and
  camera timestamp/frame.
- For the selected mask, start with its pixel centroid. If it is outside the
  mask or has invalid depth, select the nearest valid in-mask pixel or the
  distance-transform interior maximum. Median a small patch containing only
  valid masked depth and refuse low valid-pixel count or excessive spread.
- Back-project that pixel into the left-camera optical frame, transform it with
  the accepted `T_base_cam.json`, and write the existing fresh target JSON (or
  equivalent position-only `GraspTarget`) consumed by D0. Preserve frame name,
  timestamp, mask/intent identity, depth evidence, and confidence.
- Render an audit overlay containing transcript, selected mask, center pixel,
  depth patch, and resulting base-frame XYZ.

**Done when:** saved-frame tests and live dry runs select a visibly interior
point on the requested box, invalid depth is refused, and the transformed point
matches the existing clicked-point target convention.

## Task V3: perception dry-run, hover, then controlled pickup

**Do:**

- Stage 1: produce only the transcript, intent, overlay, target, and refusal
  reasons. Stage 2: invoke D0 plan-only. Stage 3: perform 100 mm hover-only
  checks at varied object positions. Do not jump directly from speech to a
  close command.
- After the overlay and hover tests pass, allow one explicitly confirmed,
  low-speed pickup of one clear box using the existing DB choreography and D0
  grasp sequence. Feed `fr5.grasp_attempt.v1` results back to the VLA so it can
  report success, failure, or that operator help is required.
- Before unattended execution, require healthy gripper activation, trustworthy
  measured reopen position, no direct gripper fault, fresh camera/intent data,
  and at least the minimum reachability/depth/bin-interior gates.

**Done when:** a spoken request repeatedly selects the correct box center,
passes 10/10 hover trials at varied positions, and completes controlled picks
without the language layer commanding low-level motion.

---

# Milestone C: grasp geometry + safety

## Experimental D0 override — RETRY MOTION LIVE-VALIDATED (2026-08-03)

The operator deferred C1-C3 temporarily for constrained physical trials.
`d0_point_grab.py` consumes a fresh base-frame target, reads the current
`plans.sqlite`, chooses the nearer left/right DB endpoint, and replays the exact
recorded `standby_to_*grab`, `*grab_to_*lift`, and `*lift_to_standby` paths.
MoveIt plans only the short DB endpoint ↔ target hover links and straight
Cartesian descent/retreat. The original 100 mm hover is unchanged.

The grasp check commands a close value and passes only if measured final
position remains at least 8 percentage points more open: 60% requires >=68%,
40% requires >=48%, and 0% requires >=8%. Current is sampled, but every trial
has reported 0%, so it is not a decision signal. A clean miss reopens, retreats
to hover, resets through the chosen proven DB grab pose, and approaches again at
the same XY/orientation with a deeper Z. Retry count and depth step are bounded
by the CLI, and every close emits a structured `fr5.grasp_attempt.v1` record.

A real-hardware run completed all three 25/30/35 mm attempts. Attempts 1 and 2
closed empty to exactly 60%, reopened to 100%, retreated vertically, reset
through the verified right-grab DB anchor, and descended 5 mm deeper. Attempt 3
also closed empty; the arm then retreated and returned through the exact DB
lift/standby paths. Grasp-point error was 0.12–0.17 mm and final retreat-hover
error 0.17–0.19 mm. That completes live validation of the retry motion and
standby recovery, but not a successful retry pickup: all attempts retained the
same XY and fixed DB wrist orientation.

Two observations keep this experimental. First, the flexible box can compress
to the commanded position, so a real grasp may be mislabeled `empty_close` by
position alone. Second, the final reopen in that run reported position 0% for a
100% command while `motion_done` appeared true. The gripper client now gives
`MoveGripper` the server's full motion window and treats timeout as “result
unknown” without automatic reactivation or duplicate commands, but startup and
completion still need stronger state checks: wait after activation, reject
direct `fault=1`, and require measured final position near the requested open
value. This must be fixed before unattended VLA execution.

This does not complete the skipped safety work. There is no table/floor plane,
object-height or jaw-width check, bin-wall gate, or environment collision
scene. A 0–100 mm configured depth range is not a claim that every value is
safe; 20 mm retry steps leading to a 90 mm descent are not accepted defaults.
At `standby`, release with `ros2 run fr5_bringup a2_gripper.py --open`.

## Task C1: table plane
**Do:**
- RANSAC-fit the table plane from the ZED point cloud once (store in camera frame + transformed to base frame). Object height = plane Z − top-face Z. Also a sanity filter: any detected "object" whose centroid isn't between the plane and ~40 cm above it is a segmentation ghost — reject.

**Done when:** reported box height matches a ruler within ~1 cm.

## Task C2: mask-derived box orientation — NEXT AFTER VLA CENTER PICK

Keep the first orientation upgrade top-down: V2 supplies position; C2 adds only
yaw and required jaw width. Full roll/pitch stays fixed until FoundationPose.

**Do:**

- Estimate the selected mask's major/minor axes with PCA or
  `cv2.minAreaRect`. The gripper's closing direction must span the object's
  short dimension; render the proposed jaw line and approach center on the
  image before planning.
- Do not apply the raw image angle directly because the ZED views the box at an
  angle. Back-project mask points (or at least two axis endpoints) with valid
  depth, transform them through `T_base_cam.json`, project the resulting axis
  onto the base XY plane, and compute base-frame wrist yaw there.
- Resolve the 180-degree symmetry and any equivalent tool-yaw solutions by
  selecting the one closest to a proven, reachable left/right DB wrist
  orientation. Plan-only must succeed before execution. If the mask is nearly
  square, too small, noisy, or has poor 3D support, fall back to the current DB
  orientation for supervised trials and refuse an unattended oriented pick.
- Estimate the physical short-side width from the masked 3D points. Refuse a
  grasp that cannot fit inside the measured 50 mm jaw stroke with clearance.
- Keep Z on the selected top surface and the approach vertical. Preview and
  plan boxes rotated at 0, 30, 45, 60, and 90 degrees, then execute at low speed
  only after all overlays and plans agree with the physical box.
- Emit a standard `GraspTarget` so the existing executive and later
  FoundationPose provider use the same contract. For example:

```python
@dataclass
class GraspTarget:
    position: np.ndarray      # (3,) base frame, meters
    quaternion: np.ndarray    # (4,) base frame gripper orientation
    approach: np.ndarray      # (3,) unit vector, direction of final descent
    width: float              # required jaw opening, meters
    source: str               # "geometric" | "foundationpose" | "graspnet"
    confidence: float
```

- The executive, the safety gate, and MoveIt only ever consume `GraspTarget`.
  Milestones F and G plug in behind it.

**Done when:** the rendered center and jaw axis are correct at all five test
angles, width estimates agree with a ruler closely enough to reject oversized
objects, MoveIt finds a reachable top-down plan, and rotated-box picks succeed
without manually changing the wrist orientation.

## Task C3: safety gate (code, not vibes)
**Do:**
- Refuse to produce a target unless ALL pass: inside the **active bin's box** (base frame — this replaces the generic workspace-AABB check now that picks come from two fixed bins), reachable, `valid_fraction ≥ 0.8`, `p90 − p10` spread below threshold, consistent with table plane, `width ≤` gripper max stroke.
- Wire it as one function `gate(target: GraspTarget, evidence) -> (ok, reasons)`; the executive refuses to move on any failure. The `VLA_project/src/co_bot_vlm/safety.py` scaffold is a sensible home for this logic.
- The gate is grasp-source-agnostic — FoundationPose and grasp-net targets pass through the **same** checks later.

**Done when:** deliberately bad inputs (target outside the active bin,
occluded mask, inconsistent support-plane depth, box wider than the gripper)
are each refused with a readable reason.

## Task C4: MoveIt planning scene — REMOVED (2026-07-16)
Zones were removed with the A3 scope change: no collision boxes, no keep-in watchdog. With an empty scene there is nothing for carry motions to plan around, so attached-object handling is dropped too. What survives from this task: **keep speed scaling ~10 % for all first live runs**, and transit between taught joint-space waypoints only (no free-space pose goals across the cage).

---

# Milestone D: look-then-move pick executive

Static scene assumption: capture → compute → move. No visual servoing yet.

## Task D1: hover-only executive
**Do:**
- End-to-end: spoken request → constrained intent → selected mask center →
  depth → base-frame `GraspTarget` → safety gate → MoveIt hover 100 mm above
  the object → standby. Keep dry-run/plan-only as the default.

**Done when:** 10/10 hovers over the correct box, varied positions, no manual help.

## Task D2: full pick-and-place
**Do:**
- Sequence: approach (above) → descend to grasp Z along `GraspTarget.approach` → close gripper → verify grasp (gripper feedback or re-segment: box gone from table) → lift → move to fixed drop zone → release → home.
- Use a MoveIt Cartesian path for descend/lift (straight-line, no elbow surprises near the table); pose goals for the free-space moves.
- Median the target over ~5 frames before moving; re-check depth right before descending.

**Done when:** ≥ 8/10 successful picks of a yellow box placed anywhere reachable.

## Task D3: failure handling
**Do:**
- Timeouts and aborts at every stage. On grasp-verify failure, consume the
  structured D0 outcome, retreat safely, request a fresh segmentation, and
  choose a bounded retry policy. Log transcript, intent, target, gate results,
  gripper observations, and outcome together.

**Done when:** yanking the box away mid-sequence produces a clean abort + retry, never a crash or a blind grasp.

---

# Milestone E: harden selection + close the VLA loop

## Task E1: deterministic spatial selection
The "rightmost" failure was the MLLM's job to get right, and it didn't. Spatial superlatives should not be LLM judgment calls when we have metric coordinates.
**Do:**
- Extend V1's constrained parser for spatial qualifiers
  (rightmost/leftmost/nearest/largest/…) and select among kept masks using
  base-frame measurements. Do not use raw image x or leave the comparison to
  the language model.

**Done when:** "rightmost yellow box" selects the correct box 10/10 with both boxes visible — the exact case that failed in run_009.

## Task E2: brain upgrade
**Do:**
- Default the agent to `Qwen/Qwen2.5-VL-7B-Instruct` (already in the HF cache; Thor has the memory) and keep `SAM3_AGENT_QWEN_REPETITION_PENALTY=1.15` for insurance. Revisit the initial plan's Qwen3-VL note only after the 7B misbehaves.

**Done when:** a 20-request soak run completes with zero malformed-output retries in the logs.

## Task E3: demo loop
**Do:**
- One spoken command: voice → intent → segmentation → oriented pick → place →
  spoken/printed report (for example, “picked the rightmost yellow box and
  placed it at the drop zone”). Keep per-round structured logging as the
  metrics source.

**Done when:** a naive visitor can speak supported requests and watch correct
picks without operator intervention; ambiguous or unsafe requests are refused.

**Milestone E3 passing = the core project works.** F and G below are the requested perception upgrades that finish it.

---

# Milestone F: FoundationPose — full 6-DoF object pose

Turns "a centroid and a yaw" into a full pose (x, y, z + rotation), which is what unlocks tilted boxes, stacked boxes, and eventually non-box objects. Runs via **Isaac ROS FoundationPose** (TensorRT-accelerated, built for Jetson). Its required inputs are RGB + aligned depth + **object mask** + object mesh — the mask comes from our SAM service, which is the whole reason this integrates cleanly.

## Task F1: compatibility check + install
**Do:**
- Match the Thor's JetPack + ROS 2 distro (recorded in A1) against the Isaac ROS release support matrix; install `isaac_ros_foundationpose` from the matching release.
- Build/download the TensorRT engines and run NVIDIA's shipped sample (rosbag) end-to-end. Do this **before** touching our data — engine builds and version mismatches are where the time goes, so isolate them.

**Done when:** the stock Isaac ROS sample produces pose estimates on the Thor.

## Task F2: bridge our pipeline into ROS
The ZED can only be opened by one process, and today the mask daemon owns it. Decide the single camera owner:
- **(a)** the daemon keeps the ZED and *also publishes* RGB + depth + `camera_info` to ROS topics via `rclpy` — smallest change, keeps the Daemon plan intact; or
- **(b)** switch to `zed-ros2-wrapper` as the owner and make the mask service subscribe to its topics — more standard, but reworks the daemon's capture path.

Default to (a) unless the wrapper is already running for other reasons.
**Do:**
- Publish synced RGB/depth/camera_info from the chosen owner; publish the selected SAM mask on a topic per request.
- Publish `T_base_cam.json` as a static TF (B2) so FoundationPose output composes into the base frame with zero new math.

**Done when:** `ros2 topic echo` shows synced RGB/depth/mask, and RViz shows the camera frame correctly placed relative to `base_link`.

## Task F3: object model registry
FoundationPose's model-based mode needs a mesh per object.
**Do:**
- Measure each demo box with calipers/ruler; generate cuboid meshes (a 10-line trimesh script). Create `objects.yaml`: category name → mesh path, dimensions, graspable faces, required jaw width.
- Note the escape hatch for later: FoundationPose's **model-free mode** (reference images instead of a mesh) is the path for objects we can't measure — don't build it now, just don't design the registry in a way that excludes it.

**Done when:** every demo object has a mesh + registry entry.

## Task F4: validate pose against the geometric baseline
**Do:**
- Run FoundationPose on the yellow box; transform its pose to base frame via the static TF; compare translation to the C2 centroid (expect ≤ ~1–2 cm agreement) and yaw to `minAreaRect`.
- Now **tilt the box ~20°** on a wedge: FoundationPose should report the tilt; the mask method can't. This is the capability we're buying — verify we actually got it.
- Log per-frame pose jitter over 100 frames of a static box; jitter feeds the safety gate threshold.

**Done when:** flat-box agreement holds, tilt is correctly reported, and jitter is characterized.

## Task F5: pose-derived grasp provider
**Do:**
- From full pose + registry dims, compute grasp candidates analytically: antipodal grasps across the short faces, approach along the box's top-face normal (no longer assumed vertical). Rank by verticality + reachability; emit the best as a `GraspTarget(source="foundationpose")`.
- Executive gains `--grasp-source {geometric,foundationpose}`; both flow through the same C3 gate and D2 sequence.

**Done when:** the tilted-box pick succeeds — the case the geometric top-down grasp physically cannot do — and flat-box success rate is no worse than D2's.

---

# Milestone G: learned grasp synthesis — conditional, honest pushback

**Trigger:** objects that are irregular, deformable, or have no mesh — i.e., F3's registry can't describe them. **Until that trigger fires, skip this milestone**: for known boxes, F5's analytic grasps from a known pose beat a network's guesses, and one fewer model on the GPU is one fewer failure mode.

If/when triggered:

## Task G1: choose and clear the engine
**Do:**
- Default choice: **Contact-GraspNet** (PyTorch port) — open code, consumes a depth image/point cloud + our segmentation mask to scope grasps to the requested object, outputs ranked 6-DoF grasps + widths.
- Before writing any code, clear two gates: **license** (Contact-GraspNet ships under an NVIDIA research/non-commercial license — check it against company use; AnyGrasp needs a purchased license) and **platform** (verify aarch64/Jetson builds exist; AnyGrasp's closed SDK has historically been x86-only). If both fail for both engines, open alternatives (e.g. GR-ConvNet) or FoundationPose model-free mode are the fallback.

**Done when:** an engine runs on the Thor on a recorded point cloud, with licensing signed off.

## Task G2: grasp-net provider
**Do:**
- Feed the masked point cloud (camera frame) → get grasp candidates → transform to base frame via the static TF → filter: jaw width ≤ gripper stroke (A2), approach reachable, passes the C3 gate → emit best as `GraspTarget(source="graspnet")`.
- Add to the executive's `--grasp-source` switch.

**Done when:** an object with no registry mesh (crumpled bag, odd toy) is picked ≥ 6/10.

---

# Step-by-step: finishing the project

This order intentionally starts VLA integration now, proves center targeting,
then adds orientation. Safety and gripper-state work are explicit gates before
unattended execution, not reasons to let a model bypass checks.

1. **A1 — DONE (2026-07-15):** FR5 command/readback and MoveIt planning frame.
2. **A2 — DONE (2026-07-16):** DH PGC140 I/O/geometry and fingertip TCP; 6 mm
   project tolerance accepted.
3. **A3 — DONE (2026-07-17):** fixed station and DB choreography accepted.
4. **B1 — DONE (2026-07-16):** eight camera ↔ fingertip point pairs.
5. **B2 — DONE (2026-07-16):** `T_base_cam.json`, 7.532 mm RMS, static TF.
6. **B3 — DONE (2026-07-17):** five physical hover points passed.
7. **D0 RETRY MOTION — LIVE-VALIDATED (2026-08-03):** the real arm completed
   the 25/30/35 mm miss sequence, DB-anchor resets, retreats, and standby
   return. Successful reacquisition is not yet proven; retry changes only Z.
8. **V1 — NEXT:** voice-to-text plus constrained action/object/qualifier
   intent, transcript confirmation, and deterministic refusals.
9. **V2:** connect `/segment`, choose a robust valid point near the selected
   mask center, validate its depth, transform to base, and render the audit
   overlay.
10. **V3 perception proof:** recorded-frame tests → live no-motion output → D0
    plan-only → 10/10 center hover tests at varied positions.
11. **Gripper live-execution gate (parallel with 8–10):** activation settle and
    health check, direct-fault handling, measured open-position confirmation,
    and flexible-box grasp evidence. Do not enable unattended voice motion
    until this passes.
12. **V3 controlled center pickup:** one object, explicit confirmation, low
    speed, existing DB/D0 motion, structured outcome returned to the VLA.
13. **C2 ORIENTATION — immediately after center pickup:** derive base-frame yaw
    and jaw width from the selected mask's masked 3D short axis; preview, plan,
    and test boxes at 0/30/45/60/90 degrees.
14. **C1 + C3:** support plane and deterministic reachability, depth-quality,
    object-width, and bin-interior gates. C4 remains removed; waypoint-only
    transit and low-speed first runs remain policy.
15. **D1–D3:** production hover/pick/place executive, fresh observations,
    bounded failure handling, and >=8/10 successful reachable-box picks.
16. **E1–E3:** deterministic multi-object qualifiers, model soak testing, and
    the complete spoken-command demo loop. **Core project complete.**
17. **F1–F5:** FoundationPose through the same `GraspTarget` and safety layer;
    demonstrate a tilted-box pick. **Full 6-DoF upgrade complete.**
18. **G1–G2:** only if an irregular/no-mesh object requires learned grasp
    synthesis; otherwise close it as not required.

# Definition of done

- **First VLA pickup (step 12):** a supported spoken request selects the correct
  segmentation, shows the center/depth/base target, passes the hover checks,
  and performs a confirmed pickup through the existing motion layer.
- **Orientation complete (step 13):** the mask-derived jaw axis matches the
  physical box at all five test rotations and enables rotated-box pickup.
- **Core (step 16):** a naive visitor speaks requests; the correct reachable box
  is picked and placed >=8/10 with no manual targeting. Unsafe, ambiguous, or
  unsupported requests are refused with a clear reason; spatial qualifiers are
  deterministic and every attempt has a structured outcome.
- **Full 6-DoF upgrade (step 17):** FoundationPose flows through the same gate
  and executive and enables a successful approximately 20-degree tilted-box
  pick. Grasp providers remain swappable behind `GraspTarget`.
- **Milestone G:** delivered only if its trigger fires, or consciously closed as
  not required. Both are valid completion states.

# Open questions (answer these early, they shape A/B/F)

1. **Answered:** FR5 on ROS 2 Jazzy; the robot workspace and ZED/SAM work run on
   the Thor.
2. **Answered:** DH PGC140, commanded through the Fairino remote-command service (`SetGripperConfig` / `ActGripper` / `MoveGripper`); 50 mm usable jaw stroke, 0 mm closed gap, and 20 × 40 mm finger pads. Any grasped box dimension between the pads must be < 50 mm with practical clearance.
3. **Answered (2026-07-16):** drop zone is fixed, in the same region as the two pick bins; taught as the `drop` joint-space waypoint in A3.
4. **Answered (2026-07-17):** the camera mount is final, bolted and zip-tied
   above the cobot. Any physical change to the camera or robot base invalidates
   Milestone B and requires repeating B1–B3.
5. Which JetPack is the Thor on, and which Isaac ROS release supports it? (Decides F1 versions.)
6. If G triggers: who signs off the grasp-net license for company use?
7. For V1, should the first speech interface be push-to-talk or a wake word,
   and must speech recognition stay fully on the Thor? Start with push-to-talk
   unless deployment requirements say otherwise.

# Risks → mitigations

- **Camera/base mount is altered after calibration** → treat
  `T_base_cam.json` as invalid and repeat B1–B3 before vision-guided motion.
- **Speech is misheard or intent is ambiguous** → display/speak back transcript
  and constrained intent; require confirmation during commissioning and refuse
  low-confidence/unsupported commands.
- **MLLM selects the wrong instance** → segment the category, then resolve
  spatial qualifiers deterministically from mask/base-frame measurements.
- **Mask centroid is outside the object or has bad depth** → snap to a valid
  interior mask point, use an in-mask median patch, and gate valid count/spread.
- **Raw image angle produces the wrong wrist yaw** → compute the short axis from
  masked 3D points in `base_link`, preview it, and refuse low confidence.
- **Depth degrades with distance squared** → keep the pickable workspace under
  approximately 2 m and gate valid fraction and spread.
- **Gripper reports fault/stale completion** → activation settle and health
  check; require measured final position near each command; unknown results
  stop retries without automatic reactivation or duplicate motion.
- **Flexible box closes to the target while held** → do not trust position alone;
  retain the attempt record and add repeatable secondary evidence such as
  post-lift segmentation or a proven nonzero current signal.
- **First live motion hits something** → exact taught DB transit, low speed,
  plan-only/hover stages first, fresh target, explicit confirmation, and a hand
  on the e-stop. No collision scene or watchdog is currently present.
- **Descend goes too deep or clips a bin wall** → C3 clamps grasp Z to the
  support plane and requires target/finger clearance inside the active bin.
- **Two processes fight over the ZED** (mask daemon vs ROS camera node) → F2 picks exactly one owner before any Isaac ROS work starts.
- **Isaac ROS / TensorRT version hell on Jetson** → F1 proves the stock sample first, in isolation from our pipeline.
- **Grasp-net licensing blocks a commercial demo** → G1 clears license + aarch64 *before* integration effort; FoundationPose model-free mode is the open fallback.
- **GPU memory pile-up** (SAM 3.1 + Qwen 7B + FoundationPose engines resident together) → Thor's unified memory is large, but measure in F4; if tight, lazy-load Qwen (the daemon already does) and keep grasp nets out of memory unless G triggered.
