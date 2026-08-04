#!/usr/bin/env python3
"""Version-2 entity grounding, role-isolated candidates, and verification."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

import candidate_generation
import grounding_v2
import local_qwen
import mask_depth
import relation_geometry


GROUNDING_RESPONSE_KEYS = {"entities"}
GROUNDING_ENTITY_KEYS = {"entity_id", "boxes"}
VERIFICATION_KEYS = {"decision", "target", "anchors", "confidence", "reason"}


def _strict_json(text: str, label: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise grounding_v2.GroundingV2Error(
            f"Qwen returned an empty {label}",
            code=f"invalid_{label}_response",
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise grounding_v2.GroundingV2Error(
            f"Qwen returned invalid {label} JSON: {exc.msg}",
            code=f"invalid_{label}_response",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(value, dict):
        raise grounding_v2.GroundingV2Error(
            f"Qwen {label} response must be an object",
            code=f"invalid_{label}_response",
        )
    return value


def visual_grounding_messages(envelope: dict[str, Any]) -> list[dict[str, str]]:
    canonical = grounding_v2.validate_command_envelope(envelope)
    entities = [canonical["target"], *canonical["anchors"]]
    entity_spec = [
        {
            "entity_id": entity["id"],
            "mention": entity["mention"],
            "head_noun": entity["head_noun"],
            "attributes": entity["attributes"],
            "selector": entity["selector"],
        }
        for entity in entities
    ]
    system = f"""Ground every listed entity in the supplied unmodified image.
Return exactly {{"entities":[{{"entity_id":"target","boxes":[[x0,y0,x1,y1]]}}]}} and no markdown.
Include every requested entity exactly once and no others, in the supplied order. Coordinates are integers in [0,1000], relative to the complete image, in XYXY order with positive area. Give zero to {grounding_v2.MAX_BOXES_PER_ENTITY} plausible boxes per entity. Boxes are candidate proposals only. Preserve entity roles; never put an anchor instance under target or a target instance under an anchor. If an entity is absent or its printed marking is unreadable, use an empty boxes list. Do not select a final answer."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Entities:\n" + json.dumps(entity_spec, ensure_ascii=False),
        },
    ]


def validate_visual_grounding_response(
    value: Any,
    *,
    expected_entity_ids: Iterable[str],
) -> dict[str, list[list[int]]]:
    if not isinstance(value, dict) or set(value) != GROUNDING_RESPONSE_KEYS:
        raise grounding_v2.GroundingV2Error(
            "visual-grounding response keys are invalid",
            code="invalid_visual_grounding_response",
        )
    raw_entities = value["entities"]
    if not isinstance(raw_entities, list):
        raise grounding_v2.GroundingV2Error(
            "visual-grounding entities must be a list",
            code="invalid_visual_grounding_response",
        )
    expected = list(expected_entity_ids)
    actual: list[str] = []
    result: dict[str, list[list[int]]] = {}
    for raw_entity in raw_entities:
        if not isinstance(raw_entity, dict) or set(raw_entity) != GROUNDING_ENTITY_KEYS:
            raise grounding_v2.GroundingV2Error(
                "each visual-grounding entity requires entity_id and boxes",
                code="invalid_visual_grounding_response",
            )
        entity_id = raw_entity["entity_id"]
        if not isinstance(entity_id, str) or entity_id in result:
            raise grounding_v2.GroundingV2Error(
                "visual-grounding entity ids are invalid or duplicated",
                code="invalid_visual_grounding_response",
            )
        boxes = raw_entity["boxes"]
        if not isinstance(boxes, list) or len(boxes) > grounding_v2.MAX_BOXES_PER_ENTITY:
            raise grounding_v2.GroundingV2Error(
                "visual-grounding box limit exceeded",
                code="request_complexity_limit",
            )
        canonical_boxes: list[list[int]] = []
        for box in boxes:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in box)
            ):
                raise grounding_v2.GroundingV2Error(
                    "each visual-grounding box must contain four numbers",
                    code="invalid_visual_grounding_response",
                )
            rounded = [int(round(float(item))) for item in box]
            if (
                any(item < 0 or item > 1000 for item in rounded)
                or rounded[2] <= rounded[0]
                or rounded[3] <= rounded[1]
            ):
                raise grounding_v2.GroundingV2Error(
                    "visual-grounding box is outside normalized XYXY bounds",
                    code="invalid_visual_grounding_response",
                    details={"box": box},
                )
            canonical_boxes.append(rounded)
        actual.append(entity_id)
        result[entity_id] = canonical_boxes
    if actual != expected:
        raise grounding_v2.GroundingV2Error(
            "visual-grounding entities do not exactly match the command envelope",
            code="grounding_identity_mismatch",
            details={"expected": expected, "actual": actual},
        )
    return result


def qwen_visual_boxes(
    envelope: dict[str, Any],
    image_path: Path,
    *,
    model_id: str,
    max_new_tokens: int,
    local_files_only: bool,
    device_map: str,
    generate: Callable[..., str] = local_qwen.qwen_generate,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    entities = list(grounding_v2.all_entities(envelope))
    started = time.monotonic()
    raw: str | None = None
    try:
        raw = generate(
            visual_grounding_messages(envelope),
            images=[image_path],
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
            device_map=device_map,
            do_sample=False,
        )
        parsed = validate_visual_grounding_response(
            _strict_json(raw, "visual_grounding"),
            expected_entity_ids=[entity["id"] for entity in entities],
        )
        return parsed, {
            "status": "accepted",
            "raw_response": raw,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except grounding_v2.GroundingV2Error as exc:
        exc.details = {
            **exc.details,
            "raw_response": raw,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        raise
    except Exception as exc:
        raise grounding_v2.GroundingV2Error(
            "Qwen visual grounding is unavailable",
            code="visual_grounding_unavailable",
            details={"error": repr(exc)},
        ) from exc


def _normalized_box_to_workspace(
    box: list[int],
    *,
    full_shape_hw: tuple[int, int],
    crop_info: dict[str, Any],
) -> tuple[list[float], float] | None:
    full_height, full_width = full_shape_hw
    x0 = box[0] * full_width / 1000.0
    y0 = box[1] * full_height / 1000.0
    x1 = box[2] * full_width / 1000.0
    y1 = box[3] * full_height / 1000.0
    if crop_info.get("enabled", False):
        crop_x0, crop_y0, crop_x1, crop_y1 = [
            float(item) for item in crop_info["applied_xyxy"]
        ]
    else:
        crop_x0, crop_y0, crop_x1, crop_y1 = 0.0, 0.0, full_width, full_height
    clipped_x0 = max(x0, crop_x0)
    clipped_y0 = max(y0, crop_y0)
    clipped_x1 = min(x1, crop_x1)
    clipped_y1 = min(y1, crop_y1)
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None
    original_area = max(1.0, (x1 - x0) * (y1 - y0))
    retained_fraction = (
        (clipped_x1 - clipped_x0) * (clipped_y1 - clipped_y0) / original_area
    )
    return (
        [
            clipped_x0 - crop_x0,
            clipped_y0 - crop_y0,
            clipped_x1 - crop_x0,
            clipped_y1 - crop_y0,
        ],
        float(retained_fraction),
    )


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write image {path}")


def _project_view_mask(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    canonical_shape: tuple[int, int],
) -> np.ndarray:
    return candidate_generation.project_mask_to_canonical(
        mask,
        source_roi_xyxy=(0, 0, mask.shape[1], mask.shape[0]),
        canonical_roi_xyxy=roi,
        canonical_shape_hw=canonical_shape,
    )


def _raw_candidate(
    *,
    entity_id: str,
    role: str,
    mask: np.ndarray,
    score: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "role": role,
        "mask": np.asarray(mask, dtype=bool),
        "score": float(score),
        "provenance": [source],
    }


def deduplicate_entity_candidates(
    raw_candidates: Iterable[dict[str, Any]],
    *,
    iou_threshold: float,
    conf_threshold: float,
    min_area: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Deduplicate only within one entity+role pool."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    materialized = list(raw_candidates)
    identities = {(item["entity_id"], item["role"]) for item in materialized}
    if len(identities) > 1:
        raise ValueError("candidate deduplication cannot cross entity or role boundaries")
    ordered = sorted(
        materialized,
        key=lambda item: (-float(item["score"]), -int(np.count_nonzero(item["mask"]))),
    )
    merged: list[dict[str, Any]] = []
    for raw in ordered:
        mask = np.asarray(raw["mask"], dtype=bool)
        area = int(np.count_nonzero(mask))
        if mask.ndim != 2 or area <= min_area or float(raw["score"]) <= conf_threshold:
            continue
        duplicate = next(
            (
                item
                for item in merged
                if candidate_generation.mask_iou(mask, item["mask"]) >= iou_threshold
            ),
            None,
        )
        if duplicate is not None:
            duplicate["provenance"].extend(raw.get("provenance", []))
            duplicate["duplicate_count"] = len(duplicate["provenance"])
            continue
        merged.append(
            {
                "entity_id": raw["entity_id"],
                "role": raw["role"],
                "mask": mask,
                "score": float(raw["score"]),
                "area_pixels": area,
                "bbox_xywh_crop_pixels": candidate_generation.mask_bbox_xywh(mask),
                "provenance": list(raw.get("provenance", [])),
                "duplicate_count": len(raw.get("provenance", [])) or 1,
            }
        )
        if len(merged) == max_candidates:
            break
    return merged


def _base_views(
    rgb: np.ndarray,
    full_rgb: np.ndarray,
    frame_info: dict[str, Any],
    frame_path: Path,
    req_dir: Path,
    *,
    multiscale: bool,
    tile_scale: float,
    full_context: bool,
) -> list[dict[str, Any]]:
    height, width = rgb.shape[:2]
    views = [
        {
            "view_id": "workspace",
            "view_kind": "workspace_crop",
            "image_path": frame_path,
            "roi": (0, 0, width, height),
            "prompt_mode": "all",
        }
    ]
    if multiscale:
        for index, roi in enumerate(
            candidate_generation.overlapping_tile_rois(width, height, scale=tile_scale),
            start=1,
        ):
            x0, y0, x1, y1 = roi
            path = req_dir / "v2_views" / f"tile_{index}.png"
            _write_rgb(path, rgb[y0:y1, x0:x1])
            views.append(
                {
                    "view_id": f"tile_{index}",
                    "view_kind": "overlapping_tile",
                    "image_path": path,
                    "roi": roi,
                    "prompt_mode": "compact",
                }
            )
    crop = frame_info["crop"]
    if full_context and crop.get("enabled", False):
        views.append(
            {
                "view_id": "full_frame",
                "view_kind": "full_frame_context",
                "image_path": Path(frame_info["saved_full_frame"]),
                "roi": (0, 0, width, height),
                "source_roi": tuple(int(item) for item in crop["applied_xyxy"]),
                "prompt_mode": "exact",
                "full_shape": full_rgb.shape[:2],
            }
        )
    return views


def _text_candidates_for_entity(
    entity: dict[str, Any],
    *,
    role: str,
    views: list[dict[str, Any]],
    canonical_shape: tuple[int, int],
    sam_service: Any,
    run_records: list[dict[str, Any]],
    min_workspace_retained_fraction: float,
) -> list[dict[str, Any]]:
    prompts = grounding_v2.build_entity_prompts(entity)
    raw: list[dict[str, Any]] = []
    for view in views:
        if view["prompt_mode"] == "all":
            selected_prompts = prompts
        elif view["prompt_mode"] == "compact":
            selected_prompts = [prompts[0], prompts[-1]] if len(prompts) > 1 else prompts
        else:
            selected_prompts = prompts[:1]
        for prompt in selected_prompts:
            started = time.monotonic()
            try:
                output = sam_service.predict_text(view["image_path"], prompt)
                masks = np.asarray(output["masks"], dtype=bool)
                scores = np.asarray(output["scores"], dtype=np.float32).reshape(-1)
                admitted_count = 0
                for source_index, mask in enumerate(masks):
                    retained_fraction = 1.0
                    if view["view_kind"] == "full_frame_context":
                        sx0, sy0, sx1, sy1 = view["source_roi"]
                        total_area = int(np.count_nonzero(mask))
                        retained_area = int(
                            np.count_nonzero(mask[sy0:sy1, sx0:sx1])
                        )
                        retained_fraction = (
                            retained_area / total_area if total_area else 0.0
                        )
                        if retained_fraction < min_workspace_retained_fraction:
                            continue
                        canonical = candidate_generation.project_mask_to_canonical(
                            mask,
                            source_roi_xyxy=(sx0, sy0, sx1, sy1),
                            canonical_roi_xyxy=view["roi"],
                            canonical_shape_hw=canonical_shape,
                        )
                    else:
                        canonical = _project_view_mask(mask, view["roi"], canonical_shape)
                    score = float(scores[source_index]) if source_index < scores.size else 0.0
                    raw.append(
                        _raw_candidate(
                            entity_id=entity["id"],
                            role=role,
                            mask=canonical,
                            score=score,
                            source={
                                "source": "sam_text",
                                "view_id": view["view_id"],
                                "view_kind": view["view_kind"],
                                "prompt": prompt,
                                "source_candidate_index": source_index,
                                "workspace_retained_fraction": retained_fraction,
                            },
                        )
                    )
                    admitted_count += 1
                run_records.append(
                    {
                        "entity_id": entity["id"],
                        "role": role,
                        "view_id": view["view_id"],
                        "prompt": prompt,
                        "status": "ok",
                        "candidate_count": int(masks.shape[0]),
                        "workspace_admitted_count": admitted_count,
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
            except Exception as exc:
                run_records.append(
                    {
                        "entity_id": entity["id"],
                        "role": role,
                        "view_id": view["view_id"],
                        "prompt": prompt,
                        "status": "error",
                        "error": repr(exc),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
    return raw


def _box_candidates_for_entity(
    entity: dict[str, Any],
    boxes: list[list[int]],
    *,
    role: str,
    rgb_shape: tuple[int, int],
    full_shape: tuple[int, int],
    crop_info: dict[str, Any],
    frame_path: Path,
    sam_service: Any,
    min_workspace_retained_fraction: float,
) -> list[dict[str, Any]]:
    converted_boxes = [
        converted
        for box in boxes
        if (
            converted := _normalized_box_to_workspace(
                box,
                full_shape_hw=full_shape,
                crop_info=crop_info,
            )
        )
        is not None
        and converted[1] >= min_workspace_retained_fraction
    ]
    workspace_boxes = [box for box, _fraction in converted_boxes]
    refined = sam_service.predict_boxes(frame_path, workspace_boxes)
    if len(refined) != len(converted_boxes):
        raise RuntimeError("interactive SAM did not return one mask per admitted box")
    raw: list[dict[str, Any]] = []
    for index, (item, (_box, retained_fraction)) in enumerate(
        zip(refined, converted_boxes)
    ):
        mask = np.asarray(item["mask"], dtype=bool)
        if mask.shape != rgb_shape:
            raise RuntimeError(
                f"box-refined mask shape {mask.shape} does not match workspace {rgb_shape}"
            )
        raw.append(
            _raw_candidate(
                entity_id=entity["id"],
                role=role,
                mask=mask,
                score=float(item["score"]),
                source={
                    "source": "qwen_box_sam_interactive",
                    "view_id": "workspace",
                    "view_kind": "workspace_crop",
                    "prompt": None,
                    "source_candidate_index": index,
                    "box_xyxy_crop_pixels": item["box_xyxy"],
                    "workspace_retained_fraction": retained_fraction,
                },
            )
        )
    return raw


def _relation_crop_candidates(
    target: dict[str, Any],
    relationships: list[dict[str, str]],
    pools: dict[str, list[dict[str, Any]]],
    *,
    rgb: np.ndarray,
    req_dir: Path,
    sam_service: Any,
    max_anchor_candidates: int = 2,
) -> list[dict[str, Any]]:
    prompts = grounding_v2.build_entity_prompts(target)
    compact_prompts = [prompts[0], prompts[-1]] if len(prompts) > 1 else prompts
    raw: list[dict[str, Any]] = []
    for relationship_index, relationship in enumerate(relationships, start=1):
        for anchor_index, anchor_candidate in enumerate(
            pools.get(relationship["anchor_id"], [])[:max_anchor_candidates],
            start=1,
        ):
            roi = relation_geometry.relation_aware_roi(
                anchor_candidate["mask"],
                relationship["type"],
            )
            x0, y0, x1, y1 = roi
            crop_path = (
                req_dir
                / "v2_views"
                / f"relation_{relationship_index}_{anchor_index}_{relationship['type']}.png"
            )
            _write_rgb(crop_path, rgb[y0:y1, x0:x1])
            for prompt in compact_prompts:
                output = sam_service.predict_text(crop_path, prompt)
                masks = np.asarray(output["masks"], dtype=bool)
                scores = np.asarray(output["scores"], dtype=np.float32).reshape(-1)
                for source_index, mask in enumerate(masks):
                    projected = _project_view_mask(mask, roi, rgb.shape[:2])
                    raw.append(
                        _raw_candidate(
                            entity_id="target",
                            role="target",
                            mask=projected,
                            score=float(scores[source_index])
                            if source_index < scores.size
                            else 0.0,
                            source={
                                "source": "relation_anchor_crop",
                                "view_id": crop_path.stem,
                                "view_kind": "relation_anchor_crop",
                                "prompt": prompt,
                                "source_candidate_index": source_index,
                                "relationship": relationship["type"],
                                "anchor_id": relationship["anchor_id"],
                                "anchor_candidate_index": anchor_index,
                                "roi_xyxy_crop_pixels": list(roi),
                            },
                        )
                    )
    return raw


def _candidate_center(candidate: dict[str, Any]) -> tuple[float, float]:
    ys, xs = np.where(candidate["mask"])
    return float(xs.mean()), float(ys.mean())


def selector_winner(
    candidates: list[dict[str, Any]],
    selector: dict[str, str] | None,
    *,
    depth: np.ndarray | None,
    min_valid_depth_pixels: int,
) -> str | None:
    if selector is None or not candidates:
        return None
    selector_type = selector["type"]
    if selector_type in {"nearest", "farthest"}:
        ranked: list[tuple[float, str]] = []
        for candidate in candidates:
            stats = mask_depth.mask_depth_stats(candidate["mask"], depth)
            if stats["median"] is not None and stats["valid_depth_pixels"] >= min_valid_depth_pixels:
                ranked.append((float(stats["median"]), candidate["label"]))
        if not ranked:
            raise grounding_v2.GroundingV2Error(
                "entity selector needs reliable depth",
                code="selector_geometry_unavailable",
                details={"selector": selector_type},
            )
        return (min if selector_type == "nearest" else max)(ranked)[1]
    if selector_type in {"largest", "smallest"}:
        ranked_area = [(int(item["area_pixels"]), item["label"]) for item in candidates]
        return (max if selector_type == "largest" else min)(ranked_area)[1]
    centers = [(*_candidate_center(item), item["label"]) for item in candidates]
    if selector_type == "leftmost":
        return min(centers, key=lambda item: item[0])[2]
    if selector_type == "rightmost":
        return max(centers, key=lambda item: item[0])[2]
    if selector_type == "topmost":
        return min(centers, key=lambda item: item[1])[2]
    if selector_type == "bottommost":
        return max(centers, key=lambda item: item[1])[2]
    raise ValueError(f"unsupported selector {selector_type!r}")


def _apply_target_safety_gates(
    candidates: list[dict[str, Any]],
    *,
    depth: np.ndarray,
    args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = {
            "area_fraction": float(candidate["mask"].mean()),
            "reject_reasons": [],
        }
        center_x, center_y = _candidate_center(candidate)
        selection_roi = getattr(args, "selection_roi", None)
        if selection_roi is not None:
            rx, ry, rw, rh = selection_roi
            if not (rx <= center_x < rx + rw and ry <= center_y < ry + rh):
                record["reject_reasons"].append("outside_workspace_roi")
        max_area = float(getattr(args, "verifier_max_area_fraction", 0.25))
        if record["area_fraction"] > max_area:
            record["reject_reasons"].append("excessive_mask_area")
        stats = mask_depth.mask_depth_stats(candidate["mask"], depth)
        record["depth_stats_m"] = stats
        spread = None
        if stats["p10"] is not None and stats["p90"] is not None:
            spread = float(stats["p90"] - stats["p10"])
        record["depth_spread_m"] = spread
        if stats["valid_fraction"] < float(
            getattr(args, "selection_min_valid_depth_fraction", 0.8)
        ):
            record["reject_reasons"].append("low_valid_depth_fraction")
        if stats["valid_depth_pixels"] < int(
            getattr(args, "candidate_min_valid_depth_pixels", 20)
        ):
            record["reject_reasons"].append("insufficient_valid_depth_pixels")
        if spread is None or spread > float(
            getattr(args, "candidate_max_depth_spread_mm", 75.0)
        ) / 1000.0:
            record["reject_reasons"].append("excessive_depth_spread")
        record["accepted"] = not record["reject_reasons"]
        record["label"] = candidate["label"]
        records.append(record)
        if record["accepted"]:
            accepted.append(candidate)
    return accepted, records


def _assign_labels(pools: dict[str, list[dict[str, Any]]]) -> None:
    for index, candidate in enumerate(pools.get("target", []), start=1):
        candidate["label"] = f"T{index}"
    anchor_number = 1
    for entity_id in sorted(key for key in pools if key != "target"):
        for candidate in pools[entity_id]:
            candidate["label"] = f"A{anchor_number}"
            anchor_number += 1


def _write_candidate_mask_artifacts(
    pools: dict[str, list[dict[str, Any]]],
    req_dir: Path,
) -> None:
    root = req_dir / "v2_candidate_masks"
    for entity_id, candidates in pools.items():
        entity_dir = root / entity_id
        entity_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            path = entity_dir / f"{candidate['label']}.png"
            if not cv2.imwrite(
                str(path), np.asarray(candidate["mask"], dtype=np.uint8) * 255
            ):
                raise OSError(f"cannot write v2 candidate mask {path}")
            candidate["mask_path_crop"] = str(path)


def compute_relationship_matrix(
    envelope: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    *,
    depth: np.ndarray | None,
    xyz: np.ndarray | None,
    thresholds: dict[str, float | int] | None = None,
) -> list[dict[str, Any]]:
    canonical = grounding_v2.validate_command_envelope(envelope)
    matrix: list[dict[str, Any]] = []
    for relationship_index, relationship in enumerate(canonical["relationships"]):
        for target in pools.get("target", []):
            for anchor in pools.get(relationship["anchor_id"], []):
                measurement = relation_geometry.evaluate_relationship(
                    relationship["type"],
                    target["mask"],
                    anchor["mask"],
                    depth=depth,
                    xyz=xyz,
                    thresholds=thresholds,
                )
                matrix.append(
                    {
                        "relationship_index": relationship_index,
                        "relationship": relationship["type"],
                        "anchor_id": relationship["anchor_id"],
                        "target_candidate": target["label"],
                        "anchor_candidate": anchor["label"],
                        **measurement,
                    }
                )
    return matrix


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key != "mask"
    }


def _draw_candidate_overlay(
    rgb: np.ndarray,
    pools: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    palette = {
        "target": (50, 230, 70),
        "anchor": (50, 150, 255),
    }
    for entity_id, candidates in pools.items():
        color = palette["target" if entity_id == "target" else "anchor"]
        for candidate in candidates:
            mask = np.asarray(candidate["mask"], dtype=bool)
            canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
            x, y, width, height = candidate["bbox_xywh_crop_pixels"]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                canvas,
                candidate["label"],
                (x, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    _write_rgb(path, canvas)


def _write_entity_crop_sheet(
    rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    path: Path,
) -> None:
    tile_size = 384
    columns = 3
    rows = max(1, math.ceil(len(candidates) / columns))
    sheet = np.zeros((rows * tile_size, columns * tile_size, 3), dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        x, y, width, height = candidate["bbox_xywh_crop_pixels"]
        pad_x = max(4, int(width * 0.15))
        pad_y = max(4, int(height * 0.15))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1 = min(rgb.shape[1], x + width + pad_x)
        y1 = min(rgb.shape[0], y + height + pad_y)
        crop = rgb[y0:y1, x0:x1]
        if crop.size:
            resized = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
            row, column = divmod(index, columns)
            sheet[
                row * tile_size : (row + 1) * tile_size,
                column * tile_size : (column + 1) * tile_size,
            ] = resized
            cv2.putText(
                sheet,
                candidate["label"],
                (column * tile_size + 8, row * tile_size + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    _write_rgb(path, sheet)


def verification_messages(
    envelope: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    relationship_matrix: list[dict[str, Any]],
    selector_winners: dict[str, str | None],
) -> list[dict[str, str]]:
    canonical = grounding_v2.validate_command_envelope(envelope)
    candidate_records = {
        entity_id: [_candidate_public(item) for item in candidates]
        for entity_id, candidates in pools.items()
    }
    compact_matrix = [
        {
            "relationship_index": item["relationship_index"],
            "relationship": item["relationship"],
            "anchor_id": item["anchor_id"],
            "target_candidate": item["target_candidate"],
            "anchor_candidate": item["anchor_candidate"],
            "status": item["status"],
            "reason": item["reason"],
            "measurements": item["measurements"],
        }
        for item in relationship_matrix
    ]
    system = """Select entities only from the supplied labeled candidates and images. Return exactly five keys: decision, target, anchors, confidence, reason. decision is "select" or "no_match". For select, target is one T label; anchors is an object mapping every anchor entity id to one A label; confidence is 0..1. For no_match, target is null and anchors is {}. Verify exact entity identity, attribute ownership, readable markings, and every declared relationship. A deterministic relationship status of fail or unavailable cannot be overridden. Entity selector winners cannot be overridden. Return no_match if any entity is absent, ambiguous, unreadable, role-swapped, or relationship-false. Return JSON only."""
    payload = {
        "command_envelope": canonical,
        "candidates": candidate_records,
        "selector_winners": selector_winners,
        "relationship_measurements": compact_matrix,
        "image_order": [
            "unmodified complete camera image",
            "T#/A# overlay",
            "one high-resolution crop sheet per entity",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def validate_verification_response(
    value: Any,
    *,
    pools: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != VERIFICATION_KEYS:
        raise grounding_v2.GroundingV2Error(
            "verification response keys are invalid",
            code="invalid_verification_response",
        )
    decision = value["decision"]
    confidence = value["confidence"]
    reason = value["reason"]
    if decision not in {"select", "no_match"}:
        raise grounding_v2.GroundingV2Error(
            "verification decision must be select or no_match",
            code="invalid_verification_response",
        )
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise grounding_v2.GroundingV2Error(
            "verification confidence must be in [0,1]",
            code="invalid_verification_response",
        )
    if not isinstance(reason, str) or not reason.strip():
        raise grounding_v2.GroundingV2Error(
            "verification reason must be non-empty",
            code="invalid_verification_response",
        )
    if decision == "no_match":
        if value["target"] is not None or value["anchors"] != {}:
            raise grounding_v2.GroundingV2Error(
                "no_match must have null target and empty anchors",
                code="invalid_verification_response",
            )
        return {
            "decision": decision,
            "target": None,
            "anchors": {},
            "confidence": float(confidence),
            "reason": reason,
        }
    target_labels = {item["label"] for item in pools.get("target", [])}
    if value["target"] not in target_labels:
        raise grounding_v2.GroundingV2Error(
            "verification selected an unknown target candidate",
            code="invalid_verification_response",
        )
    anchors = value["anchors"]
    expected_anchor_ids = {key for key in pools if key != "target"}
    if not isinstance(anchors, dict) or set(anchors) != expected_anchor_ids:
        raise grounding_v2.GroundingV2Error(
            "verification anchor selection does not match requested anchors",
            code="invalid_verification_response",
        )
    for entity_id, label in anchors.items():
        allowed = {item["label"] for item in pools[entity_id]}
        if label not in allowed:
            raise grounding_v2.GroundingV2Error(
                f"verification selected an unknown candidate for {entity_id}",
                code="invalid_verification_response",
            )
    return {
        "decision": decision,
        "target": value["target"],
        "anchors": dict(anchors),
        "confidence": float(confidence),
        "reason": reason,
    }


def _selected_candidate(pools: dict[str, list[dict[str, Any]]], entity_id: str, label: str) -> dict[str, Any]:
    return next(item for item in pools[entity_id] if item["label"] == label)


def segment_frame(
    envelope: dict[str, Any],
    *,
    rgb: np.ndarray,
    full_rgb: np.ndarray,
    depth: np.ndarray,
    xyz: np.ndarray,
    frame_info: dict[str, Any],
    frame_path: Path,
    req_dir: Path,
    sam_service: Any,
    args: Any,
    generate: Callable[..., str] = local_qwen.qwen_generate,
) -> dict[str, Any]:
    """Run the complete perception-only v2 path for one already-captured frame."""

    started = time.monotonic()
    canonical = grounding_v2.validate_command_envelope(envelope)
    req_dir.mkdir(parents=True, exist_ok=True)
    envelope_path = req_dir / "command_envelope.json"
    envelope_path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    full_image_path = Path(frame_info["saved_full_frame"])
    entities = [canonical["target"], *canonical["anchors"]]
    visual_boxes, visual_record = qwen_visual_boxes(
        canonical,
        full_image_path,
        model_id=args.qwen_model,
        max_new_tokens=int(getattr(args, "v2_visual_max_new_tokens", 512)),
        local_files_only=not args.allow_qwen_downloads,
        device_map=args.qwen_device_map,
        generate=generate,
    )
    (req_dir / "qwen_visual_grounding.json").write_text(
        json.dumps({"boxes": visual_boxes, **visual_record}, indent=2) + "\n",
        encoding="utf-8",
    )

    views = _base_views(
        rgb,
        full_rgb,
        frame_info,
        frame_path,
        req_dir,
        multiscale=bool(getattr(args, "candidate_multiscale", True)),
        tile_scale=float(getattr(args, "candidate_tile_scale", 0.72)),
        full_context=bool(getattr(args, "v2_full_context", True)),
    )
    run_records: list[dict[str, Any]] = []
    raw_by_entity: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        role = "target" if entity["id"] == "target" else "anchor"
        raw = _text_candidates_for_entity(
            entity,
            role=role,
            views=views,
            canonical_shape=rgb.shape[:2],
            sam_service=sam_service,
            run_records=run_records,
            min_workspace_retained_fraction=float(
                getattr(args, "candidate_min_workspace_retained_fraction", 0.90)
            ),
        )
        raw.extend(
            _box_candidates_for_entity(
                entity,
                visual_boxes[entity["id"]],
                role=role,
                rgb_shape=rgb.shape[:2],
                full_shape=full_rgb.shape[:2],
                crop_info=frame_info["crop"],
                frame_path=frame_path,
                sam_service=sam_service,
                min_workspace_retained_fraction=float(
                    getattr(args, "candidate_min_workspace_retained_fraction", 0.90)
                ),
            )
        )
        raw_by_entity[entity["id"]] = raw

    def dedup(entity_id: str) -> list[dict[str, Any]]:
        return deduplicate_entity_candidates(
            raw_by_entity.get(entity_id, []),
            iou_threshold=float(getattr(args, "candidate_dedup_iou", 0.80)),
            conf_threshold=float(getattr(args, "presence_conf_threshold", 0.10)),
            min_area=int(getattr(args, "min_area", 64)),
            max_candidates=int(getattr(args, "candidate_max_count", 12)),
        )

    pools = {entity["id"]: dedup(entity["id"]) for entity in entities}
    # Relation crops are target-search views derived from independently found
    # anchors.  Their masks return only to the target pool.
    raw_by_entity["target"].extend(
        _relation_crop_candidates(
            canonical["target"],
            canonical["relationships"],
            pools,
            rgb=rgb,
            req_dir=req_dir,
            sam_service=sam_service,
        )
    )
    pools["target"] = dedup("target")
    _assign_labels(pools)
    _write_candidate_mask_artifacts(pools, req_dir)
    raw_candidate_pool_records = {
        key: [_candidate_public(item) for item in value]
        for key, value in pools.items()
    }
    raw_candidate_pools_path = req_dir / "raw_candidate_pools.json"
    raw_candidate_pools_path.write_text(
        json.dumps(raw_candidate_pool_records, indent=2) + "\n",
        encoding="utf-8",
    )

    if any(not pools[entity["id"]] for entity in entities):
        missing = [entity["id"] for entity in entities if not pools[entity["id"]]]
        return {
            "schema_version": 2,
            "status": "no_match",
            "accepted": False,
            "reason": "missing_entity_candidates",
            "missing_entities": missing,
            "command_envelope": canonical,
            "candidate_pools": raw_candidate_pool_records,
            "raw_candidate_pools": raw_candidate_pool_records,
            "visual_grounding": visual_record,
            "candidate_runs": run_records,
            "robot_target": None,
            "motion_permitted": False,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    safe_targets, target_gate_records = _apply_target_safety_gates(
        pools["target"], depth=depth, args=args
    )
    pools["target"] = safe_targets
    if not pools["target"]:
        return {
            "schema_version": 2,
            "status": "no_match",
            "accepted": False,
            "reason": "target_safety_gates_rejected_all_candidates",
            "command_envelope": canonical,
            "target_safety_gates": target_gate_records,
            "raw_candidate_pools": raw_candidate_pool_records,
            "candidate_pools": {
                key: [_candidate_public(item) for item in value]
                for key, value in pools.items()
            },
            "candidate_runs": run_records,
            "robot_target": None,
            "motion_permitted": False,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    selector_winners = {
        entity["id"]: selector_winner(
            pools[entity["id"]],
            entity["selector"],
            depth=depth,
            min_valid_depth_pixels=int(
                getattr(args, "candidate_min_valid_depth_pixels", 20)
            ),
        )
        for entity in entities
    }
    matrix = compute_relationship_matrix(
        canonical,
        pools,
        depth=depth,
        xyz=xyz,
        thresholds=getattr(args, "v2_relation_thresholds", None),
    )
    matrix_path = req_dir / "relationship_measurements.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    overlay_path = req_dir / "v2_candidates_overlay.png"
    _draw_candidate_overlay(rgb, pools, overlay_path)
    crop_paths: list[Path] = []
    for entity in entities:
        crop_path = req_dir / "v2_entity_crops" / f"{entity['id']}.png"
        _write_entity_crop_sheet(rgb, pools[entity["id"]], crop_path)
        crop_paths.append(crop_path)
    verification_started = time.monotonic()
    try:
        raw_verification = generate(
            verification_messages(canonical, pools, matrix, selector_winners),
            images=[full_image_path, overlay_path, *crop_paths],
            model_id=args.qwen_model,
            max_new_tokens=int(getattr(args, "verifier_max_new_tokens", 256)),
            local_files_only=not args.allow_qwen_downloads,
            device_map=args.qwen_device_map,
            do_sample=False,
        )
    except Exception as exc:
        raise grounding_v2.GroundingV2Error(
            "Qwen candidate verification is unavailable",
            code="candidate_verification_unavailable",
            details={"error": repr(exc)},
        ) from exc
    verification_raw_path = req_dir / "qwen_verification_raw.txt"
    verification_raw_path.write_text(raw_verification, encoding="utf-8")
    verification = validate_verification_response(
        _strict_json(raw_verification, "verification"),
        pools=pools,
    )
    verification.update(
        {
            "raw_response": raw_verification,
            "elapsed_s": round(time.monotonic() - verification_started, 3),
            "candidate_overlay": str(overlay_path),
            "entity_crop_sheets": [str(path) for path in crop_paths],
            "raw_response_artifact": str(verification_raw_path),
        }
    )
    (req_dir / "qwen_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    if (
        verification["decision"] == "no_match"
        or verification["confidence"]
        < float(getattr(args, "verifier_min_confidence", 0.70))
    ):
        return {
            "schema_version": 2,
            "status": "no_match",
            "accepted": False,
            "reason": "qwen_no_match_or_low_confidence",
            "command_envelope": canonical,
            "verification": verification,
            "relationship_measurements": matrix,
            "selector_winners": selector_winners,
            "target_safety_gates": target_gate_records,
            "candidate_pools": {
                key: [_candidate_public(item) for item in value] for key, value in pools.items()
            },
            "raw_candidate_pools": raw_candidate_pool_records,
            "robot_target": None,
            "motion_permitted": False,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    selected_labels = {"target": verification["target"], **verification["anchors"]}
    for entity_id, winner in selector_winners.items():
        if winner is not None and selected_labels[entity_id] != winner:
            return {
                "schema_version": 2,
                "status": "no_match",
                "accepted": False,
                "reason": "entity_selector_failed",
                "selector_winners": selector_winners,
                "selected_entities": selected_labels,
                "command_envelope": canonical,
                "verification": verification,
                "candidate_pools": {
                    key: [_candidate_public(item) for item in value]
                    for key, value in pools.items()
                },
                "raw_candidate_pools": raw_candidate_pool_records,
                "robot_target": None,
                "motion_permitted": False,
                "elapsed_s": round(time.monotonic() - started, 3),
            }

    selected_relation_records: list[dict[str, Any]] = []
    for index, relationship in enumerate(canonical["relationships"]):
        match = next(
            item
            for item in matrix
            if item["relationship_index"] == index
            and item["target_candidate"] == selected_labels["target"]
            and item["anchor_candidate"] == selected_labels[relationship["anchor_id"]]
        )
        selected_relation_records.append(match)
    unavailable = [item for item in selected_relation_records if item["status"] == "unavailable"]
    if unavailable:
        raise grounding_v2.GroundingV2Error(
            "reliable geometry is unavailable for a selected relationship",
            code="relation_geometry_unavailable",
            details={
                "relationships": [item["relationship"] for item in unavailable],
            },
        )
    if any(item["status"] != "pass" for item in selected_relation_records):
        return {
            "schema_version": 2,
            "status": "no_match",
            "accepted": False,
            "reason": "deterministic_relationship_failed",
            "command_envelope": canonical,
            "verification": verification,
            "selected_relationships": selected_relation_records,
            "relationship_measurements": matrix,
            "candidate_pools": {
                key: [_candidate_public(item) for item in value]
                for key, value in pools.items()
            },
            "raw_candidate_pools": raw_candidate_pool_records,
            "robot_target": None,
            "motion_permitted": False,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    target_candidate = _selected_candidate(
        pools, "target", selected_labels["target"]
    )
    anchor_candidates = {
        entity_id: _selected_candidate(pools, entity_id, label)
        for entity_id, label in verification["anchors"].items()
    }
    return {
        "schema_version": 2,
        "status": "accepted",
        "accepted": True,
        "reason": "all_acceptance_gates_passed",
        "command_envelope": canonical,
        "selected_entities": selected_labels,
        "selected_relationships": selected_relation_records,
        "relationship_measurements": matrix,
        "selector_winners": selector_winners,
        "verification": verification,
        "visual_grounding": visual_record,
        "candidate_runs": run_records,
        "candidate_pools": {
            key: [_candidate_public(item) for item in value] for key, value in pools.items()
        },
        "raw_candidate_pools": raw_candidate_pool_records,
        "target_safety_gates": target_gate_records,
        "audit_artifacts": {
            "command_envelope": str(envelope_path),
            "raw_candidate_pools": str(raw_candidate_pools_path),
            "candidate_overlay": str(overlay_path),
            "entity_crop_sheets": [str(path) for path in crop_paths],
            "relationship_measurements": str(matrix_path),
        },
        "robot_target": None,
        "motion_permitted": False,
        "elapsed_s": round(time.monotonic() - started, 3),
        "_target_mask": target_candidate["mask"],
        "_target_score": target_candidate["score"],
        "_anchor_masks": {
            key: item["mask"] for key, item in anchor_candidates.items()
        },
        "_anchor_scores": {
            key: item["score"] for key, item in anchor_candidates.items()
        },
    }
