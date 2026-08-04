# Segmentation Identification Improvement Plan

Date: 2026-08-04

Implementation status: the accepted `448,360,384,360` HD720 crop is now the centralized default for every live ZED entry point. Use `--no-crop` only for full-frame diagnostics, or `--crop X,Y,W,H` to override it deliberately.

## Goal

Improve instruction-based object identification first, then improve final mask quality. The immediate target is reliable identification of small packages and boxes in the ZED workspace without accepting a confidently segmented but incorrect object.

## What the baseline system did before these changes

The resident service currently has two paths:

1. The fast path optionally converts some natural-language commands into a short noun phrase and sends that phrase directly to SAM 3.1.
2. The visual Qwen agent is used only when the fast path returns zero accepted masks.

This creates several limitations:

- Normal short prompts such as `orange package` bypass Qwen entirely.
- Qwen does not visually verify a non-empty direct result, even when SAM selected the wrong object.
- The fallback condition checks only whether the mask count is zero.
- The presence gate checks SAM score and pixel area, but not whether a candidate semantically matches the request.
- Spatial selectors cover only a limited set of relationships.
- The current service processes the complete 1280 x 720 frame, where the requested package may occupy only a thin, small region.

Recent saved runs support this diagnosis:

- `orange and white box`, `cardboard box`, `air filter box`, `orange package`, `rectangular package`, and `orange label` produced zero raw candidates.
- Large, common objects still worked: `robot arm` produced one mask and `shelf` produced three masks.
- A visual Qwen 3B agent run incorrectly concluded that the scene contained a robot but no car-filter box, even though the small package was visible on the cart.

## Crop and ROI values already saved

Crop coordinates use `x,y,width,height`, with `(0,0)` at the top-left of the 1280 x 720 frame.

### Standard true inference crop

```text
448,360,384,360
```

This means:

- Left edge: `x = 448`
- Top edge: `y = 360`
- Right edge: `x = 832`
- Bottom edge: `y = 720`
- Cropped image: `384 x 360`

This crop was used by earlier Task 5/Task 7 runs and the July 2 cropped test. In that cropped test, one `right most pasta box` round produced one mask with score `0.9375`; two later rounds produced no candidates. It is now the centralized default in `task5_zed_live_prompt.py` and is inherited by Task 7 and the resident mask service.

Evidence:

- `outputs/task5_live_green_object_crop_448_360_384_360.json`
- `outputs/06/02/2026-1_test/cropped/run_001/summary.json`

### Current service selection ROI

```text
430,380,430,170
```

This means:

- Left edge: `x = 430`
- Top edge: `y = 380`
- Right edge: `x = 860`
- Bottom edge: `y = 550`

This was previously configured in `docker-compose.mask-service.yml`, but it was only a post-segmentation selection ROI. It has been removed from the default service command because those full-frame coordinates are invalid after applying the standard true crop. A future selection ROI must use crop-local coordinates.

### Current service capture

The saved August 4 service results from before standardization record cropping as disabled:

```text
enabled: false
input/output: 1280 x 720
```

New service processes default to the standard crop after they are restarted. Historical result files remain unchanged.

## Recommended target pipeline

```text
Natural-language instruction
        |
        v
Structured intent
  - target category
  - visual attributes
  - relation/anchor
  - spatial selector
        |
        v
Candidate generation
  - full frame
  - workspace crop
  - optional overlapping/multiscale crops
  - several concise category synonyms
        |
        v
Merge and deduplicate candidate masks
        |
        v
Qwen-VL visual verification
  - inspect numbered masks
  - choose matching candidate(s)
  - or return no match
        |
        v
Deterministic geometry/depth selection
        |
        v
Full-frame mask + ZED depth/XYZ
```

## Implementation order

### 1. Confirm the workspace crop (complete)

Open the latest full frame and choose a crop that contains:

- The complete pickable workspace
- Enough surrounding context to recognize objects
- Minimal robot, wall, monitor, shelving, and other irrelevant scene content

The saved `448,360,384,360` crop has been accepted as the standard HD720 workspace crop. Reconfirm it if the camera or workspace moves.

### 2. Make the ROI a real inference crop

- Apply the crop to RGB before both SAM and Qwen.
- Apply the identical crop to depth and XYZ, or expand the final crop mask back into a full-frame zero mask before depth lookup.
- Record both crop-local and full-frame coordinates in every result.
- Keep the existing selection ROI as a separate concept if it remains useful.

### 3. Add multiscale candidate generation

- Run the direct prompt on the confirmed workspace crop.
- Optionally retain a full-frame pass for large/contextual objects.
- For weak prompts, try a small controlled prompt family such as `box`, `package`, `carton`, and the requested visual noun phrase.
- Merge overlapping masks instead of accepting each prompt result independently.

### 4. Use Qwen as a candidate verifier (complete)

Run visual verification when any of these conditions holds:

- No candidate was found.
- More than one candidate was found.
- Candidate scores are close.
- Confidence is below the validated threshold.
- The instruction contains relationships, product attributes, or printed-label references.
- A candidate is outside the expected workspace.
- The result will be sent to the robot.

The resident service now sends every score/area-gated direct or agent candidate
through Qwen before spatial selection, depth extraction, or robot-coordinate
generation. Qwen receives the raw crop, a full-scene numbered overlay, and
enlarged candidate views. It must return strict JSON containing selected IDs or
`no_match`; malformed output gets one format-only retry and then fails closed.

The verifier records its raw response, decision, candidate-to-SAM index mapping,
confidence, reason, and image artifacts in each result. A verifier rejection or
error cannot trigger the agent fallback and therefore cannot bypass the gate.
Relative selectors are applied deterministically only after semantic approval.

Live paired control on 2026-08-04:

- A tight package mask covering `0.021021` of the crop was approved by Qwen at
  `0.84` confidence and produced one depth object.
- The erroneous full-bin `box` mask covering `0.331554` of the crop was refused
  by the final `0.25` pick-target area guard and produced zero depth objects.

The maximum area fraction and minimum Qwen confidence are configurable with
`SAM3_QWEN_VERIFIER_MAX_AREA_FRACTION` and
`SAM3_QWEN_VERIFIER_MIN_CONFIDENCE`.

### 5. Upgrade Qwen 3B to Qwen2.5-VL-7B (complete)

The resident Docker service now defaults to the locally cached
`Qwen/Qwen2.5-VL-7B-Instruct` snapshot. Docker runs Transformers and the
Hugging Face Hub in offline mode, mounts the persistent model cache, warms the
7B BF16 model onto `cuda:0` during service startup, and reports the loaded
runtime through `/health`. Agent generation is capped at 512 new tokens with a
1.15 repetition penalty.

An in-container inference check completed successfully. Step 4 now ensures that
non-empty direct and fallback SAM masks are verified before they can reach the
robot-target path.

Use the locally cached `Qwen/Qwen2.5-VL-7B-Instruct` before considering a much larger model.

Recommended generation controls:

- Deterministic generation for tool calls
- Maximum output around 256-512 tokens
- `SAM3_AGENT_QWEN_REPETITION_PENALTY=1.15`
- Strict JSON validation with one controlled retry

The current adapter is hard-coded for Qwen2.5-VL. Qwen2.5-VL-7B is therefore the low-risk upgrade. Qwen3-VL requires an adapter/model-class update rather than only changing the model ID.

The 7B model should improve visual verification and tool use, but it cannot improve direct requests in which Qwen is never called. It also cannot recover detail that is lost because the target is too small in the full frame.

### 6. Make relationships deterministic where possible

- Use coordinates for leftmost/rightmost/topmost/bottommost.
- Use ZED depth for nearest/farthest.
- Use mask area for largest/smallest.
- Eventually perform robot-relative spatial selection in the robot base frame rather than image coordinates.
- Use OCR or VLM text recognition when the request depends on a brand, model, or printed label.

### 7. Add semantic and safety checks

Do not accept a mask based only on SAM score and area. Require:

- Semantic verifier approval when the result is ambiguous
- Workspace/ROI membership
- Sufficient valid depth coverage
- Reasonable physical size and table height
- A rejection/no-match option

### 8. Build a repeatable A/B evaluation

Freeze representative frames and expected targets, then compare:

1. Qwen 3B + full frame
2. Qwen 7B + full frame
3. Qwen 3B + workspace crop
4. Qwen 7B + workspace crop

Measure:

- Raw candidate recall
- Correct-object selection accuracy
- No-object false-positive rate
- Final mask IoU/Dice where labels exist
- Malformed agent responses
- End-to-end latency

This will show whether crop scale, Qwen size, or SAM candidate generation is the actual limiting factor.

## Initial acceptance criteria

- Correct target selected in at least 19 of 20 repeated static-scene requests.
- No wrong object accepted when the target is absent.
- Correct behavior for multiple similar boxes and spatial requests.
- No malformed Qwen tool calls in a 20-request soak test.
- Crop masks remain correctly aligned with ZED depth and XYZ.
- Every result records crop, prompt candidates, selected mask, verifier decision, scores, and latency.

## Image inspection commands

Open the latest full 1280 x 720 service frame:

```bash
xdg-open /home/team/VLA_Model_Work/SAM_3_implementation/outputs/service/20260804_103012_681818/frame.png
```

Open an earlier frame that already used the saved `448,360,384,360` crop:

```bash
xdg-open /home/team/VLA_Model_Work/SAM_3_implementation/outputs/06/02/2026-1_test/cropped/run_001/round_001/frame.png
```

When reporting a new crop, provide either `x,y,width,height` or the number of pixels to remove from the left, top, right, and bottom. Convert the latter using:

```text
x = left removal
y = top removal
width = 1280 - left removal - right removal
height = 720 - top removal - bottom removal
```
