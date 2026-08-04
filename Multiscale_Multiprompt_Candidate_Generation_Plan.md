# Multiscale, Multi-Prompt Candidate Generation

Status: implemented in the resident mask service on 2026-08-04.

## Goal

Improve target recall before Qwen verification. A request such as `pick the
small orange and white box in the upper bin` should let SAM find the target
under several safe descriptions and at several image scales, without changing
the calibrated crop/depth coordinate contract.

## Request-to-candidate flow

1. Parse the request into the versioned grounding intent: category, typed
   visual attributes, optional selector, and optional source region.
2. Generate at most five deterministic SAM prompts. For example:
   `small orange and white box`, `orange and white package`, `small carton`,
   `flat rectangular item`, and `box`.
3. Run the whole prompt family on the canonical workspace crop.
4. When `source_region` is present, remove its directional modifier and run one
   SAM localization pass for the region noun (`upper bin` becomes `bin`). Use
   the removed `upper`, `lower`, `left`, `right`, or `center` modifier to choose
   the matching returned region; otherwise choose the highest-scoring valid
   region. Pad it by 15 percent on every side, clamp it to the workspace, and
   run the primary and category prompts on that enlarged view.
5. Run the primary and category prompts on four overlapping workspace tiles at
   scale `0.72`.
6. Project every mask back into canonical workspace-crop coordinates and merge
   near-identical masks at `0.80` mask IoU. Keep the highest-scoring mask while
   preserving every prompt/view provenance record.
7. Apply score, area, workspace-retention, and 12-candidate limits, then send
   the merged candidates to Qwen2.5-VL-7B in one numbered verification round.
8. Apply depth-quality and deterministic spatial-selector gates to Qwen's
   selected candidates. Never pass a Qwen-rejected candidate to the robot.

Full-frame candidate search remains available through
`--candidate-full-frame`, but is disabled by default. The uncropped frame is
still saved for auditing.

## Coordinate and depth contract

The workspace crop is the only canonical segmentation space. With the standard
crop `[448,360,384,360]`:

- workspace masks, boxes, centers, depth, and XYZ use a `384 x 360` array;
- tile and localized-region masks are projected into that array before
  deduplication, Qwen verification, or depth lookup;
- crop point `(u, v)` reads depth and XYZ at `(u, v)`;
- the corresponding full-camera point is `(u + 448, v + 360)`;
- a full-size audit mask is produced by inserting the crop mask into an empty
  `1280 x 720` mask at `[448:832, 360:720]`.

`merged_candidates.json` stays compatible with the robot bridge: its image
size and normalized boxes are crop-local, and `presence_gate.kept_indices`
indexes the same merged candidate order.

## Artifacts and failures

Each request writes `candidate_generation.json` containing the prompt family,
view ROIs, localized-region decision, source SAM runs, failures, timings,
deduplication settings, candidate provenance, and crop/full-frame boxes.

An individual prompt, localized-region pass, or tile may fail without losing
successful candidates from other views. If every target candidate-generation
run errors, the service writes the audit record and refuses the request. A
successful run with no candidates can still use the existing agent fallback;
Qwen or geometry rejection cannot.

## Configuration

- `--candidate-max-prompts 5`
- `--candidate-multiscale` / `--no-candidate-multiscale`
- `--candidate-tile-scale 0.72`
- `--candidate-region-padding-fraction 0.15`
- `--candidate-dedup-iou 0.80`
- `--candidate-max-count 12`
- `--candidate-full-frame` for opt-in full-frame candidates

## Acceptance tests

- Prompt/view matrix: full prompt family on the workspace, two prompts on the
  localized region, and two prompts on each of four tiles.
- Region localization: directional selection overrides a higher-scoring region
  in the wrong direction; padding clamps safely to image boundaries.
- Canonical projection: workspace, tile, and localized masks resolve to the
  same crop pixel.
- Crop/depth regression: crop point `(100, 50)` reads depth at `(100, 50)` and
  becomes full-frame point `(548, 410)` under the standard crop.
- Deduplication: prompt/view duplicates merge, while a package mask nested in a
  larger bin mask remains a separate candidate.
- Qwen integration: one verification call receives only merged candidates.
- Failure handling: partial SAM failures are recorded and tolerated; total SAM
  target-run failure is audited and refused.

## Operating check

After restarting the resident service, inspect `/health`. Candidate generation
should report multiscale enabled, region padding `0.15`, tile scale `0.72`, and
full-frame context disabled. Run a frozen-frame evaluation before live robot
motion, then inspect `candidate_generation.json`, the numbered Qwen overlay,
the selected crop mask, and its expanded full-frame mask.
