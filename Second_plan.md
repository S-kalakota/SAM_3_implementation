# Second plan — From masks + depth to a Fairino pick

Picks up where `intial plan.md` ended. Milestones 1–3 are done and extended: the agent masks objects from a typed request on live ZED frames, and every kept mask now carries metric depth and a 3D centroid (`object_depth` in the result JSON). What nothing in the system knows yet is **where anything is relative to the robot** — the coordinates are in the camera's frame, and the Fairino has never heard of it.

Goal of this plan: type a request, and the Fairino picks up the right box. MoveIt does the motion planning; our job is to hand it a **correct, safe target pose in the robot's base frame** and command the grasp sequence around it.

## Stack decision: MoveIt, FoundationPose, AnyGrasp/Contact-GraspNet

The requested stack, and where each piece lands:

- **MoveIt — yes, core.** It's already the backbone of Milestones A, C and D (it's on the Thor, the Fairino has a `fairino_moveit_config`). Nothing to push back on.
- **FoundationPose — yes, but *after* the first pick, as Milestone F.** For flat boxes grasped top-down it adds nothing that mask + depth + `minAreaRect` don't already give: that's x, y, z, yaw, and the remaining two rotations are fixed by "top-down". It earns its keep the moment boxes can be **tilted, stacked, or replaced by non-box objects** — then you need the full rotation, and only FoundationPose provides it. It also fits our pipeline unusually well: FoundationPose **requires a segmentation mask as input**, and masks are the thing this project is best at (SAM mask → FoundationPose → 6-DoF pose is the intended usage). It stays off the critical path to the first pick because debugging a TensorRT/Isaac ROS graph at the same time as hand-eye calibration doubles the unknowns for zero first-pick benefit.
- **AnyGrasp / Contact-GraspNet — pushback: at most one of them, and probably neither for boxes.** These networks solve the *unknown-object* grasp problem: point cloud in, 6-DoF grasp candidates out, no object model needed. But once FoundationPose gives the full box pose and we know the box dimensions, grasp poses **fall out geometrically** (antipodal grasps across the short faces) — running a grasp net on top of a known pose + known mesh is redundant machinery. They become the right tool when we grasp **irregular objects that have no mesh**. If/when that trigger fires (Milestone G), prefer **Contact-GraspNet** over AnyGrasp: Contact-GraspNet is open research code with a maintained PyTorch port and accepts a segmentation mask to scope grasps to our object; AnyGrasp is a closed SDK with a per-machine license file, a commercial fee, and unverified aarch64/Jetson support — three separate ways to lose a week on the Thor. Check both licenses against company use before committing either.

The through-line that makes all of this cheap to change: **every grasp source produces the same `GraspTarget` object behind one interface** (Task C2). Geometric top-down, FoundationPose-derived, and grasp-net-proposed grasps are then swaps, not rewrites — the executive and the safety gate never care where a grasp came from.

## What we have today (validated on the rig)

- Masks from language via the SAM 3.1 agent loop (task7), robust to bad Qwen output.
- Per-object depth: median + XYZ centroid in the **left-camera optical frame** (X right, Y down, Z out of the lens, meters). Example measured: yellow box at `[-0.280, 0.474, 1.382]`.
- Camera intrinsics/extrinsics measured: fx = 521.5 px, baseline = 0.1199 m (HD720). Accuracy at ~1.4 m: ≈ ±1 cm per axis (grows with distance²).
- Known weak spot: the 3B Qwen picked the **left** box when asked for the **rightmost** (Milestone E fixes this deterministically).
- The Daemon plan's resident mask service (`/segment`) is the masking entry point the executive will call.

## Target data flow

```
 typed request ─► SAM 3.1 agent ─► masks ─► gate ─► XYZ centroid (camera frame)
                                     │                  │
                                     │ (Milestone F)    │
                                     ▼                  │
                            FoundationPose ─► full 6-DoF│pose (camera frame)
                                     │                  │
                                     └───────┬──────────┘
                                       T_base←cam  (hand-eye calibration, Milestone B)
                                             │
                              GraspTarget in Fairino base frame
                      (C2 geometric │ F5 pose-derived │ G2 grasp-net — same interface)
                                             │
                          safety gate (bounds, table plane, depth quality)
                                             │
                      MoveIt (Fairino) plans & executes: hover ─► pick ─► place
```

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

## Task A3: safety box — planning scene + keep-in watchdog (C4 pulled forward)

**Why now, not in C:** the old `fairino_plan_saver` stack replayed trajectories a human had approved once in RViz; our pipeline asks MoveIt for a *fresh* path every run, so no human has ever vetted the path about to execute. And Milestone B means hours of jogging and commanded moves near the table before any of Milestone C exists. Two layers, opposite jobs:

1. **Keep-out (plan time):** table, camera gantry, and monitor published as collision boxes in the MoveIt planning scene. Every plan then routes around them automatically, and a target inside one fails loudly instead of executing.
2. **Keep-in (run time):** a watchdog node that watches `base_link → TCP` TF at ~20 Hz. The moment the TCP exits the allowed workspace AABB (or dips below the table plane), it cancels the active MoveGroup / FollowJointTrajectory goals and the arm holds. This catches what planning can't: wrong scene numbers, execution deviating from the plan, or a bad target that slipped through.

**No automatic return-to-home on breach.** After a violation the arm's state is by definition unexpected; an automatic recovery motion is a blind move at the worst possible moment. Breach → stop + hold + log the breach pose → human acknowledges → only then a commanded move back to `standby`.

**Measure (with the robot itself, not a tape measure):** jog the arm and read positions off the A1 TCP stream (`a1_move_and_read.py --watch-only`) — the readings are already in `base_link`, so there is no frame conversion to get wrong.
- Table plane: touch the fingertip to the table surface at 3 spread-out spots; average the z values and interpret their spread in light of the accepted 6 mm TCP uncertainty. Place the planning-scene table surface 6 mm above that measured plane.
- Table extent: touch (or hover over) the reachable corners/edges; record x, y.
- Gantry and monitor: jog the fingertip to the closest faces the arm could plausibly hit; record xyz per face; build boxes with ≥ 5 cm padding.
- Keep-in AABB: in **sim**, drive through every taught SRDF pose and record the TCP min/max per axis; add ~10 cm margin. Shift every effective AABB wall inward by the 6 mm TCP uncertainty. Floor = table z + **11 mm** (6 mm accepted TCP uncertainty + 5 mm nominal physical clearance); ceiling and walls from the workspace edges.
- Caveat: before A2 the reported TCP is the flange, so every touch is offset by the gripper length — acceptable while boxes carry ≥ 5 cm padding; re-measure the table plane after A2 sets the fingertip TCP.

**Do:**
- `a3_measure_workspace.py` — interactive: "jog fingertip to table corner 1, press Enter", records TCP from TF, writes `workspace.yaml` (table plane, keep-out boxes, keep-in AABB).
- `a3_planning_scene.py` — reads `workspace.yaml`, publishes the CollisionObjects into the planning scene at startup.
- `a3_tcp_watchdog.py` — TF watcher; on breach cancels active motion goals, logs the breach pose, and refuses further motion until explicit operator acknowledgment.
- Wire the scene publisher + watchdog into `a1_bringup.launch.py` so they are always on — nobody has to remember to start the safety layer.

**Done when:**
- RViz shows the boxes; a plan whose direct path would cross the gantry visibly routes around it; a pose target inside a keep-out box is rejected at planning time.
- In sim, sending the arm toward a pose outside the keep-in AABB gets the trajectory cancelled mid-flight (< 100 ms after crossing) and the arm holds; nothing moves again until the operator acknowledges.

---

# Milestone B: hand-eye calibration (the critical path)

One fixed rigid transform `T_base←cam` converts camera points to robot points: `p_base = T @ [x, y, z, 1]`. Both camera and robot base are bolted down, so it's constant until something physically moves. Everything downstream — centroid targets today, FoundationPose poses in Milestone F — rides on this one transform.

## Task B1: touch-point capture tool
**Do:**
- Small script: shows the live ZED left view; you click a pixel, it records that pixel's `XYZ` from `MEASURE.XYZ` (median of a 5×5 patch); then you jog the Fairino so the fingertip touches the same physical point and the script records the TCP position.
- Collect **8–12 point pairs spread across the whole workspace, including different heights** (put a box or block under some touches — coplanar points make the fit degenerate in Z).

**Done when:** `calib_points.json` holds ≥ 8 well-spread pairs.

## Task B2: solve the transform
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

## Task B3: hover validation
**Do:**
- Pipeline reports a box in base frame → command MoveIt to hover the TCP 100 mm directly above it → measure the horizontal miss with a ruler. Repeat at ≥ 5 spots spread over the table.
- Interpret: constant offset everywhere = TCP or frame mistake; error growing toward the edges = poor calibration spread → add points there and refit.

**Done when:** miss ≤ ~15 mm at every test spot.

## Task B4: drift tripwire
**Do:**
- Glue an AprilTag/ChArUco tag to a table corner. At pipeline startup, detect it and compare its camera-frame pose to the pose recorded at calibration time; warn loudly if translation moved beyond **6 mm** or rotation moved beyond the chosen angular threshold (camera got bumped → recalibrate).

**Done when:** nudging the camera mount on purpose triggers the warning.

---

# Milestone C: grasp geometry + safety

## Task C1: table plane
**Do:**
- RANSAC-fit the table plane from the ZED point cloud once (store in camera frame + transformed to base frame). Object height = plane Z − top-face Z. Also a sanity filter: any detected "object" whose centroid isn't between the plane and ~40 cm above it is a segmentation ghost — reject.

**Done when:** reported box height matches a ruler within ~1 cm.

## Task C2: grasp pose from mask + depth — and the `GraspTarget` interface
**Do:**
- Top-down grasp: XY = centroid in base frame; yaw = mask principal axis via `cv2.minAreaRect` (align gripper jaws across the box's short side); Z = top face (use `p10` of mask depth, not median — median can include side pixels) plus gripper-specific offset; approach waypoint 100 mm above.
- Define the interface every grasp source must emit, e.g.:

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

- The executive, the safety gate, and MoveIt only ever consume `GraspTarget`. Milestones F and G plug in behind it.

**Done when:** rendered grasp axis on the overlay looks right for boxes in several orientations, and the hover executive (D1) consumes a `GraspTarget` rather than raw centroids.

## Task C3: safety gate (code, not vibes)
**Do:**
- Refuse to produce a target unless ALL pass: inside workspace AABB (define it in base frame), reachable, `valid_fraction ≥ 0.8`, `p90 − p10` spread below threshold, consistent with table plane, drift tripwire (B4) quiet, `width ≤` gripper max stroke.
- Wire it as one function `gate(target: GraspTarget, evidence) -> (ok, reasons)`; the executive refuses to move on any failure. The `VLA_project/src/co_bot_vlm/safety.py` scaffold is a sensible home for this logic.
- The gate is grasp-source-agnostic — FoundationPose and grasp-net targets pass through the **same** checks later.

**Done when:** deliberately bad inputs (object off-table, occluded mask, tag moved, box wider than the gripper) are each refused with a readable reason.

## Task C4: MoveIt planning scene
Mostly pulled forward to **Task A3** (collision boxes, keep-in watchdog, `workspace.yaml`). What remains here:
**Do:**
- Re-verify the A3 scene numbers after A2 sets the fingertip TCP (the original touches were flange-offset). Keep speed scaling ~10 % for all first live runs.
- After a successful grasp, **attach the box as a collision object** to the gripper (and detach on release) so carry motions plan around what's in the jaws.

**Done when:** RViz shows the collision objects and a plan that would clip the gantry gets rejected/re-routed, with the fingertip TCP active.

---

# Milestone D: look-then-move pick executive

Static scene assumption: capture → compute → move. No visual servoing yet.

## Task D1: hover-only executive
**Do:**
- End-to-end: request → agent → depth → base-frame `GraspTarget` → safety gate → MoveIt hover 100 mm above the object → home. Add `--dry-run` (default ON) that plans and visualizes but does not execute.

**Done when:** 10/10 hovers over the correct box, varied positions, no manual help.

## Task D2: full pick-and-place
**Do:**
- Sequence: approach (above) → descend to grasp Z along `GraspTarget.approach` → close gripper → verify grasp (gripper feedback or re-segment: box gone from table) → lift → move to fixed drop zone → release → home.
- Use a MoveIt Cartesian path for descend/lift (straight-line, no elbow surprises near the table); pose goals for the free-space moves.
- Median the target over ~5 frames before moving; re-check depth right before descending.

**Done when:** ≥ 8/10 successful picks of a yellow box placed anywhere reachable.

## Task D3: failure handling
**Do:**
- Timeouts and aborts at every stage; on grasp-verify failure, retreat and retry once with a fresh capture; log every attempt (target, gate results, outcome) next to the round JSON.

**Done when:** yanking the box away mid-sequence produces a clean abort + retry, never a crash or a blind grasp.

---

# Milestone E: fix selection + close the VLA loop

## Task E1: deterministic spatial selection
The "rightmost" failure was the MLLM's job to get right, and it didn't. Spatial superlatives should not be LLM judgment calls when we have metric coordinates.
**Do:**
- Parse spatial qualifiers (rightmost/leftmost/nearest/largest/…) in code. Ask the agent for the *category* ("yellow box" → all instances), then select among kept masks by base-frame coordinate (rightmost = max base-frame Y-or-X, fixed once in A1; don't use image x, which flips with camera orientation).

**Done when:** "rightmost yellow box" selects the correct box 10/10 with both boxes visible — the exact case that failed in run_009.

## Task E2: brain upgrade
**Do:**
- Default the agent to `Qwen/Qwen2.5-VL-7B-Instruct` (already in the HF cache; Thor has the memory) and keep `SAM3_AGENT_QWEN_REPETITION_PENALTY=1.15` for insurance. Revisit the initial plan's Qwen3-VL note only after the 7B misbehaves.

**Done when:** a 20-request soak run completes with zero malformed-output retries in the logs.

## Task E3: demo loop
**Do:**
- One command: typed request → pick → place → report ("picked the rightmost yellow box, 1.38 m away, placed at drop zone; 14 s"). Keep per-round JSON logging as the metrics source.

**Done when:** a naive visitor can type requests and watch correct picks without you touching anything.

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

The single ordered path from today to done. Each step is a task above; don't start a step before its predecessor's **Done when** holds (parallel tracks marked).

1. **A1 — DONE (2026-07-15)** — command the Fairino from code, read TCP back; record frames + ROS/JetPack versions.
2. **A2 — DONE (2026-07-16)** — gripper I/O and geometry recorded; fingertip TCP calibrated; approximately 6 mm two-orientation disagreement accepted as the project tolerance.
3. **A3 — IN PROGRESS** — measure the rig, review the collision scene, and pass the simulated keep-out/keep-in tests before enabling it on the real arm.
4. **B1** — capture 8–12 touch-point pairs across the workspace, varied heights.
5. **B2** — solve `T_base←cam`, residuals ≤ 8 mm RMS; save JSON.
6. **B3** — hover validation at 5+ spots, miss ≤ 15 mm.
7. **B4** — AprilTag drift tripwire at startup.
8. **C1** — table plane fit + ghost filter.
9. **C2** — geometric grasp + the `GraspTarget` interface.
10. **C3** — safety gate as one function with readable refusals.
11. **C4** — collision scene, 10 % speed, attached-object handling.
12. **D1** — hover-only executive, dry-run default, 10/10.
13. *(parallel with 14–15, software-only)* **E1** deterministic spatial selection + **E2** Qwen 7B upgrade.
14. **D2** — full pick-and-place, ≥ 8/10.
15. **D3** — failure handling; yank-the-box test passes.
16. **E3** — one-command demo loop. **← core project complete.**
17. **F1–F5** — FoundationPose: install → ROS bridge → mesh registry → validate vs baseline → 6-DoF grasp provider; tilted-box pick passes. **← requested stack integrated, project finished.**
18. **G1–G2** — *only if* the no-mesh-object trigger fires; otherwise explicitly closed as "not required".

The single highest-value day of work is still **Milestone B** — everything after it is unblocked the moment `T_base_cam.json` exists and the hover test passes.

# Definition of done

- **Core (step 16):** a naive visitor types requests; the correct box is picked and placed ≥ 8/10 with zero operator help; every unsafe/ambiguous request is *refused with a printed reason* rather than attempted; spatial superlatives resolve deterministically.
- **Finished with requested stack (step 17):** everything above, plus FoundationPose poses flowing through the same gate and executive, demonstrated by a successful pick of a ~20°-tilted box; MoveIt planning throughout; grasp sources swappable by flag.
- **Milestone G:** delivered *or* consciously closed with the trigger documented as never having fired. Both count as finished.

# Open questions (answer these early, they shape A/B/F)

1. Which Fairino model (FR3/FR5/…), and which ROS 2 distro is its driver running on? Same machine as the ZED/SAM stack (the Thor) or a separate PC?
2. **Answered:** DH PGC140, commanded through the Fairino remote-command service (`SetGripperConfig` / `ActGripper` / `MoveGripper`); 50 mm usable jaw stroke, 0 mm closed gap, and 20 × 40 mm finger pads. Any grasped box dimension between the pads must be < 50 mm with practical clearance.
3. Where is the drop zone, and is it fixed?
4. Is the camera mount final? Every physical change to it invalidates Milestone B (the B4 tripwire catches this, but recalibration still costs an hour).
5. Which JetPack is the Thor on, and which Isaac ROS release supports it? (Decides F1 versions.)
6. If G triggers: who signs off the grasp-net license for company use?

# Risks → mitigations

- **Camera bump silently ruins calibration** → B4 tag tripwire at every startup.
- **Depth degrades with distance²** → keep pickable workspace under ~2 m from the lens; gate on `valid_fraction` and spread.
- **MLLM misselects the object** → E1 makes spatial selection deterministic code.
- **First live motion hits something** → C4 collision scene + 10 % speed + dry-run default + hand on the e-stop.
- **Grasping on median depth grabs a side face** → C2 uses top-face (`p10`) depth for grasp Z.
- **Two processes fight over the ZED** (mask daemon vs ROS camera node) → F2 picks exactly one owner before any Isaac ROS work starts.
- **Isaac ROS / TensorRT version hell on Jetson** → F1 proves the stock sample first, in isolation from our pipeline.
- **Grasp-net licensing blocks a commercial demo** → G1 clears license + aarch64 *before* integration effort; FoundationPose model-free mode is the open fallback.
- **GPU memory pile-up** (SAM 3.1 + Qwen 7B + FoundationPose engines resident together) → Thor's unified memory is large, but measure in F4; if tight, lazy-load Qwen (the daemon already does) and keep grasp nets out of memory unless G triggered.
