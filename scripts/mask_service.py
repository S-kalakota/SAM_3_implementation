#!/usr/bin/env python3
"""Resident SAM 3.1 + ZED masking service.

Loads SAM once, holds the ZED camera open, and serves one fresh segmentation
request at a time over localhost.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError:  # pragma: no cover - exercised only before deps are installed.
    FastAPI = None
    HTTPException = None
    uvicorn = None

import candidate_generation
import check_v2_release_gates
import grounding_intent
import grounding_v2
import local_qwen
import mask_depth
import qwen_candidate_verifier as candidate_verifier
import relation_geometry
import sam3_image_service
import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6
import v2_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "service"
DEFAULT_SELECTION_ROI = os.environ.get("MASK_SERVICE_SELECTION_ROI")
DEFAULT_VERIFIER_MAX_NEW_TOKENS = int(
    os.environ.get("SAM3_QWEN_VERIFIER_MAX_NEW_TOKENS", "256")
)
DEFAULT_VERIFIER_MIN_CONFIDENCE = float(
    os.environ.get("SAM3_QWEN_VERIFIER_MIN_CONFIDENCE", "0.70")
)
DEFAULT_VERIFIER_MAX_AREA_FRACTION = float(
    os.environ.get("SAM3_QWEN_VERIFIER_MAX_AREA_FRACTION", "0.25")
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

LOCK = threading.Lock()
STATE: dict[str, Any] = {}

if FastAPI is not None:
    app = FastAPI(title="SAM 3.1 Mask Service")
else:
    app = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve SAM 3.1 + ZED segmentation requests from one warm process."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--checkpoint", default=task6.DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument(
        "--presence-conf-threshold",
        default=task2.DEFAULT_PRESENCE_CONF_THRESH,
        type=float,
    )
    parser.add_argument("--min-area", default=task2.DEFAULT_MIN_AREA, type=int)
    parser.add_argument("--det-threshold", default=None, type=float)
    parser.add_argument("--max-generations", default=8, type=int)
    parser.add_argument(
        "--intent-parser-mode",
        choices=("deterministic", "qwen", "shadow"),
        default=os.environ.get("SAM3_INTENT_PARSER_MODE", "shadow"),
        help=(
            "Structured-intent parser. Shadow uses deterministic intent for "
            "runtime and records the Qwen comparison."
        ),
    )
    parser.add_argument("--intent-max-new-tokens", default=256, type=int)
    parser.add_argument(
        "--v2-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SAM3_V2_ENABLED", False),
        help=(
            "Enable perception-only /v2/segment. /v2/interpret remains available "
            "for rollout corpus testing even when segmentation is disabled."
        ),
    )
    parser.add_argument(
        "--v2-evaluation-mode",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SAM3_V2_EVALUATION_MODE", False),
        help=(
            "Allow perception-only v2 segmentation while collecting frozen, "
            "shadow, and live release evidence. This still requires approved "
            "image parity and relation calibration, but not the aggregate "
            "production release report."
        ),
    )
    parser.add_argument(
        "--v2-interpret-max-new-tokens",
        default=int(os.environ.get("SAM3_V2_INTERPRET_MAX_NEW_TOKENS", "1024")),
        type=int,
    )
    parser.add_argument(
        "--v2-visual-max-new-tokens",
        default=int(os.environ.get("SAM3_V2_VISUAL_MAX_NEW_TOKENS", "512")),
        type=int,
    )
    parser.add_argument(
        "--v2-full-context",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SAM3_V2_FULL_CONTEXT", True),
        help="Include a projected full-camera context pass for every v2 entity.",
    )
    parser.add_argument(
        "--sam-backend",
        choices=("multiplex", "image"),
        default=os.environ.get("SAM3_SAM_BACKEND", "multiplex"),
        help=(
            "SAM runtime. The image backend reuses embeddings and is required "
            "for v2 box refinement."
        ),
    )
    parity_default = os.environ.get("SAM3_IMAGE_PARITY_REPORT")
    parser.add_argument(
        "--sam-image-parity-report",
        default=Path(parity_default) if parity_default else None,
        type=Path,
        help=(
            "Frozen-frame parity report with approved=true; required before "
            "switching the service to the image backend."
        ),
    )
    relation_report_default = os.environ.get("SAM3_V2_RELATION_THRESHOLD_REPORT")
    parser.add_argument(
        "--v2-relation-threshold-report",
        default=Path(relation_report_default) if relation_report_default else None,
        type=Path,
        help=(
            "Calibration report from at least 60 labeled scenes with zero "
            "safety false accepts; required when v2 segmentation is enabled."
        ),
    )
    release_report_default = os.environ.get("SAM3_V2_RELEASE_REPORT")
    parser.add_argument(
        "--v2-release-report",
        default=Path(release_report_default) if release_report_default else None,
        type=Path,
        help="Approved aggregate release-gate report required to enable v2.",
    )
    parser.add_argument("--candidate-max-prompts", default=5, type=int)
    parser.add_argument("--candidate-dedup-iou", default=0.80, type=float)
    parser.add_argument("--candidate-max-count", default=12, type=int)
    parser.add_argument(
        "--candidate-min-workspace-retained-fraction",
        default=0.90,
        type=float,
        help="Reject full-frame masks that mostly lie outside the workspace crop.",
    )
    parser.add_argument(
        "--candidate-multiscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a localized source-region crop, when requested, and four "
            "overlapping workspace tiles in addition to the full crop."
        ),
    )
    parser.add_argument("--candidate-tile-scale", default=0.72, type=float)
    parser.add_argument(
        "--candidate-region-padding-fraction",
        default=0.15,
        type=float,
        help="Padding added on each side of a localized source-region box.",
    )
    parser.add_argument(
        "--candidate-full-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also admit primary-prompt candidates from the full camera frame. "
            "Disabled by default because robot depth is workspace-crop aligned."
        ),
    )
    parser.add_argument("--qwen-model", default=local_qwen.DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--qwen-max-new-tokens",
        default=local_qwen.DEFAULT_MAX_NEW_TOKENS,
        type=int,
    )
    parser.add_argument("--qwen-device-map", default=local_qwen.DEFAULT_DEVICE_MAP)
    parser.add_argument(
        "--verifier-max-new-tokens",
        default=DEFAULT_VERIFIER_MAX_NEW_TOKENS,
        type=int,
        help="Maximum Qwen output tokens for candidate verification.",
    )
    parser.add_argument(
        "--verifier-min-confidence",
        default=DEFAULT_VERIFIER_MIN_CONFIDENCE,
        type=float,
        help="Reject Qwen candidate selections below this confidence.",
    )
    parser.add_argument(
        "--verifier-max-area-fraction",
        default=DEFAULT_VERIFIER_MAX_AREA_FRACTION,
        type=float,
        help=(
            "Reject a Qwen-selected mask covering more than this fraction of "
            "the cropped image."
        ),
    )
    parser.add_argument(
        "--allow-qwen-downloads",
        action="store_true",
        help="Allow Transformers to fetch missing Qwen files. Default is local cache only.",
    )
    parser.add_argument(
        "--warm-qwen",
        action="store_true",
        help="Load cached Qwen weights during startup instead of on the first request.",
    )
    task5.add_crop_arguments(parser)
    parser.add_argument("--resolution", default="HD720", choices=task5.RESOLUTION_NAMES)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--view", default="LEFT", choices=task5.VIEW_NAMES)
    parser.add_argument(
        "--warmup-frames",
        default=2,
        type=int,
        help="Frames to flush before each served request.",
    )
    parser.add_argument(
        "--selection-min-valid-depth-fraction",
        default=0.8,
        type=float,
        help=(
            "When selecting among multiple masks, prefer candidates whose ZED "
            "depth coverage is at least this fraction. Use 0 to disable."
        ),
    )
    parser.add_argument("--candidate-min-valid-depth-pixels", default=20, type=int)
    parser.add_argument("--candidate-max-depth-spread-mm", default=75.0, type=float)
    parser.add_argument(
        "--selection-roi",
        default=task5.parse_crop(DEFAULT_SELECTION_ROI)
        if DEFAULT_SELECTION_ROI
        else None,
        type=task5.parse_crop,
        help=(
            "Optional crop-local workspace ROI as x,y,w,h. Spatial selection is "
            "performed inside this region when any candidates fall inside it."
        ),
    )
    parser.add_argument("--grab-timeout", default=5.0, type=float)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.verifier_max_new_tokens <= 0:
        parser.error("--verifier-max-new-tokens must be greater than zero")
    if args.intent_max_new_tokens <= 0:
        parser.error("--intent-max-new-tokens must be greater than zero")
    if args.v2_interpret_max_new_tokens <= 0:
        parser.error("--v2-interpret-max-new-tokens must be greater than zero")
    if args.v2_visual_max_new_tokens <= 0:
        parser.error("--v2-visual-max-new-tokens must be greater than zero")
    if v2_segmentation_available(args) and args.sam_backend != "image":
        parser.error(
            "--v2-enabled/--v2-evaluation-mode requires --sam-backend image"
        )
    if args.candidate_max_prompts < 1 or args.candidate_max_prompts > 5:
        parser.error("--candidate-max-prompts must be in [1, 5]")
    if not 0.0 < args.candidate_dedup_iou <= 1.0:
        parser.error("--candidate-dedup-iou must be in (0, 1]")
    if args.candidate_max_count < 1:
        parser.error("--candidate-max-count must be positive")
    if not 0.0 <= args.candidate_min_workspace_retained_fraction <= 1.0:
        parser.error(
            "--candidate-min-workspace-retained-fraction must be in [0, 1]"
        )
    if not 0.5 <= args.candidate_tile_scale < 1.0:
        parser.error("--candidate-tile-scale must be in [0.5, 1.0)")
    if not 0.0 <= args.candidate_region_padding_fraction <= 1.0:
        parser.error("--candidate-region-padding-fraction must be in [0, 1]")
    if args.candidate_min_valid_depth_pixels < 1:
        parser.error("--candidate-min-valid-depth-pixels must be positive")
    if args.candidate_max_depth_spread_mm <= 0.0:
        parser.error("--candidate-max-depth-spread-mm must be positive")
    if not 0.0 <= args.verifier_min_confidence <= 1.0:
        parser.error("--verifier-min-confidence must be in [0, 1]")
    if not 0.0 < args.verifier_max_area_fraction <= 1.0:
        parser.error("--verifier-max-area-fraction must be in (0, 1]")
    return args


def v2_segmentation_available(args: argparse.Namespace) -> bool:
    """Whether the no-motion v2 endpoint may execute in this process."""

    return bool(
        getattr(args, "v2_enabled", False)
        or getattr(args, "v2_evaluation_mode", False)
    )


def require_server_deps() -> None:
    if FastAPI is not None and uvicorn is not None:
        return
    raise RuntimeError(
        "fastapi and uvicorn are required. Install them with "
        "`.venv/bin/pip install fastapi uvicorn`."
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def qwen_grounding_intent(
    source_phrase: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Parse one structured intent with one format-only retry."""

    started = time.monotonic()
    messages: list[dict[str, Any]] = list(
        grounding_intent.qwen_parser_messages(source_phrase)
    )
    attempts: list[dict[str, Any]] = []
    for attempt_number in (1, 2):
        try:
            raw = local_qwen.qwen_generate(
                messages,
                model_id=args.qwen_model,
                max_new_tokens=args.intent_max_new_tokens,
                local_files_only=not args.allow_qwen_downloads,
                device_map=args.qwen_device_map,
                do_sample=False,
                response_prefix='{"schema_version":',
            )
            parsed = grounding_intent.parse_qwen_grounding_intent(
                raw,
                source_phrase,
            )
            attempts.append(
                {"attempt": attempt_number, "raw_response": raw, "error": None}
            )
            return parsed, {
                "status": "accepted",
                "attempts": attempts,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": locals().get("raw"),
                    "error": repr(exc),
                }
            )
            if attempt_number == 1:
                messages.extend(
                    [
                        {"role": "assistant", "content": str(locals().get("raw", ""))[:2000]},
                        {
                            "role": "user",
                            "content": (
                                "The response was schema-invalid. Retry once with exactly "
                                "the eight required keys, source_phrase copied exactly, no "
                                "invented evidence, and no markdown. The response is already "
                                "prefixed with {\"schema_version\":; continue with 1 and the "
                                "remaining JSON fields."
                            ),
                        },
                    ]
                )
    return None, {
        "status": "error",
        "attempts": attempts,
        "error": "Qwen returned no schema-valid grounding intent after two attempts",
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def qwen_command_envelope(
    raw_command: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Interpret every v2 command with constrained Qwen and bounded repairs."""

    if not isinstance(raw_command, str) or not raw_command.strip():
        raise grounding_v2.GroundingV2Error("raw_command must be non-empty")
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    correction: dict[str, Any] | None = None
    schema = grounding_v2.qwen_semantic_json_schema()
    for attempt_number in (1, 2, 3):
        raw: str | None = None
        try:
            messages = grounding_v2.qwen_interpretation_messages(
                raw_command,
                correction=correction,
            )
            raw = local_qwen.qwen_generate(
                messages,
                model_id=args.qwen_model,
                max_new_tokens=args.v2_interpret_max_new_tokens,
                local_files_only=not args.allow_qwen_downloads,
                device_map=args.qwen_device_map,
                do_sample=False,
                repetition_penalty=1.0,
                json_schema=schema,
            )
            envelope = grounding_v2.parse_qwen_semantic_interpretation(
                raw,
                raw_command,
            )
            attempts.append(
                {"attempt": attempt_number, "raw_response": raw, "error": None}
            )
            return envelope, {
                "status": "accepted",
                "model": args.qwen_model,
                "deterministic_generation": True,
                "schema_constrained_generation": True,
                "semantic_contract_version": (
                    grounding_v2.QWEN_SEMANTIC_CONTRACT_VERSION
                ),
                "attempts": attempts,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except grounding_v2.GroundingV2Error as exc:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": raw,
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                    },
                }
            )
            can_retry = grounding_v2.qwen_interpretation_error_is_retryable(exc)
            if not can_retry or attempt_number == 3:
                exc.details = {
                    **exc.details,
                    "attempt_count": len(attempts),
                    "attempt_errors": [item["error"] for item in attempts],
                }
                raise
            correction = {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
            continue
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": raw,
                    "error": {"code": "qwen_unavailable", "message": repr(exc)},
                }
            )
            # Model/runtime failures are not format errors and must not be retried
            # into a deterministic or alternative semantic interpretation.
            raise grounding_v2.GroundingV2Error(
                "Qwen interpretation is unavailable",
                code="grounding_parser_unavailable",
                details={"attempts": attempts},
            ) from exc
    raise RuntimeError("unreachable Qwen interpretation loop")


def parse_request_intent(
    request: str,
    args: argparse.Namespace,
    *,
    supplied_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one canonical intent and retain parser provenance/shadow data."""

    source_phrase = grounding_intent.collapse_space(request)
    if supplied_intent is not None:
        active = grounding_intent.validate_grounding_intent(
            supplied_intent,
            expected_source_phrase=source_phrase,
        )
        qwen_intent = None
        qwen_record = None
        if args.intent_parser_mode in {"qwen", "shadow"}:
            qwen_intent, qwen_record = qwen_grounding_intent(source_phrase, args)
        if args.intent_parser_mode == "qwen":
            if qwen_intent is None:
                raise grounding_intent.GroundingIntentError(
                    "Qwen could not validate the supplied structured intent",
                    code="grounding_parser_unavailable",
                    details=qwen_record,
                )
            if qwen_intent != active:
                raise grounding_intent.GroundingIntentError(
                    "Qwen and the supplied structured intent disagree",
                    code="ambiguous_grounding_intent",
                    details=grounding_intent.compare_intents(active, qwen_intent),
                )
        parser_record = {
            "mode": "supplied_v1",
            "active_parser": "validated_request_intent",
            "configured_parser_mode": args.intent_parser_mode,
            "qwen": qwen_record,
        }
        shadow_comparison = (
            grounding_intent.compare_intents(
                active,
                qwen_intent,
                shadow_error=None
                if qwen_intent is not None
                else str((qwen_record or {}).get("error")),
            )
            if args.intent_parser_mode == "shadow"
            else None
        )
    else:
        deterministic = grounding_intent.parse_grounding_intent(source_phrase)
        qwen_intent = None
        qwen_record = None
        if args.intent_parser_mode in {"qwen", "shadow"}:
            qwen_intent, qwen_record = qwen_grounding_intent(source_phrase, args)

        if args.intent_parser_mode == "qwen":
            if qwen_intent is not None:
                active = qwen_intent
                active_parser = "qwen_structured"
            else:
                simple = (
                    deterministic["source_region"] is None
                    and not deterministic["relations"]
                    and len(deterministic["attributes"]) <= 2
                )
                if not simple:
                    raise grounding_intent.GroundingIntentError(
                        "Qwen structured parser failed for a complex request",
                        code="grounding_parser_unavailable",
                        details=qwen_record,
                    )
                active = deterministic
                active_parser = "deterministic_simple_fallback"
        else:
            active = deterministic
            active_parser = "deterministic"

        shadow_comparison = None
        if args.intent_parser_mode == "shadow":
            shadow_comparison = grounding_intent.compare_intents(
                deterministic,
                qwen_intent,
                shadow_error=None
                if qwen_intent is not None
                else str((qwen_record or {}).get("error")),
            )
        parser_record = {
            "mode": args.intent_parser_mode,
            "active_parser": active_parser,
            "qwen": qwen_record,
        }

    return {
        "grounding_intent": active,
        "intent_hash": grounding_intent.intent_hash(active),
        "primary_sam_phrase": grounding_intent.construct_primary_prompt(active),
        "prompt_family": grounding_intent.build_prompt_family(
            active,
            max_prompts=args.candidate_max_prompts,
        ),
        "parser": parser_record,
        "shadow_comparison": shadow_comparison,
    }


def mask_center_xy(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def select_spatial_mask(
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    selector: str | None,
    depth_np: np.ndarray | None = None,
    min_valid_depth_fraction: float = 0.0,
    selection_roi: tuple[int, int, int, int] | None = None,
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]], dict[str, Any] | None]:
    if selector is None or len(kept) <= 1:
        return kept, candidates, None

    geometries = []
    for kept_pos, (mask, score) in enumerate(kept):
        center_x, center_y = mask_center_xy(mask)
        area = int(np.count_nonzero(mask))
        depth_stats = None
        if depth_np is not None:
            depth_stats = mask_depth.mask_depth_stats(mask, depth_np)
        inside_roi = None
        if selection_roi is not None:
            roi_x, roi_y, roi_w, roi_h = selection_roi
            inside_roi = (
                roi_x <= center_x < roi_x + roi_w
                and roi_y <= center_y < roi_y + roi_h
            )
        geometries.append(
            {
                "kept_position": kept_pos,
                "score": float(score),
                "center_xy_pixels": [center_x, center_y],
                "area_pixels": area,
                "median_depth_m": None if depth_stats is None else depth_stats["median"],
                "valid_depth_fraction": (
                    None if depth_stats is None else depth_stats["valid_fraction"]
                ),
                "inside_selection_roi": inside_roi,
            }
        )

    eligible_geometries = geometries
    if selection_roi is not None:
        roi_eligible = [
            item for item in eligible_geometries if item["inside_selection_roi"]
        ]
        if roi_eligible:
            eligible_geometries = roi_eligible

    if depth_np is not None and min_valid_depth_fraction > 0:
        depth_eligible = [
            item
            for item in eligible_geometries
            if item["valid_depth_fraction"] is not None
            and item["valid_depth_fraction"] >= min_valid_depth_fraction
        ]
        if depth_eligible:
            eligible_geometries = depth_eligible

    if selector == "rightmost":
        selected = max(eligible_geometries, key=lambda item: item["center_xy_pixels"][0])
    elif selector == "leftmost":
        selected = min(eligible_geometries, key=lambda item: item["center_xy_pixels"][0])
    elif selector == "topmost":
        selected = min(eligible_geometries, key=lambda item: item["center_xy_pixels"][1])
    elif selector == "bottommost":
        selected = max(eligible_geometries, key=lambda item: item["center_xy_pixels"][1])
    elif selector == "largest":
        selected = max(eligible_geometries, key=lambda item: item["area_pixels"])
    elif selector == "smallest":
        selected = min(eligible_geometries, key=lambda item: item["area_pixels"])
    elif selector in {"nearest", "farthest"}:
        depth_geometries = [
            item for item in eligible_geometries if item["median_depth_m"] is not None
        ]
        if not depth_geometries:
            return kept, candidates, None
        if selector == "nearest":
            selected = min(depth_geometries, key=lambda item: item["median_depth_m"])
        else:
            selected = max(depth_geometries, key=lambda item: item["median_depth_m"])
    else:
        return kept, candidates, None

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    selected_candidate_index = kept_candidates[selected["kept_position"]]["index"]
    selected_kept = [kept[selected["kept_position"]]]
    eligible_positions = {item["kept_position"] for item in eligible_geometries}
    kept_position_by_candidate_index = {
        candidate["index"]: kept_pos
        for kept_pos, candidate in enumerate(kept_candidates)
    }

    updated_candidates = []
    for candidate in candidates:
        candidate = dict(candidate)
        if candidate["kept"] and candidate["index"] != selected_candidate_index:
            kept_pos = kept_position_by_candidate_index.get(candidate["index"])
            if kept_pos is not None and kept_pos not in eligible_positions:
                extra_reasons = []
                geometry = geometries[kept_pos]
                if geometry["inside_selection_roi"] is False:
                    extra_reasons.append("outside_selection_roi")
                if (
                    geometry["valid_depth_fraction"] is not None
                    and geometry["valid_depth_fraction"] < min_valid_depth_fraction
                ):
                    extra_reasons.append("low_valid_depth_for_selection")
                candidate["reject_reasons"] = (
                    list(candidate["reject_reasons"]) + extra_reasons
                )
            candidate["kept"] = False
            candidate["reject_reasons"] = list(candidate["reject_reasons"]) + [
                f"not_{selector}"
            ]
        updated_candidates.append(candidate)

    selection = {
        "selector": selector,
        "selected_candidate_index": int(selected_candidate_index),
        "selected_kept_position": int(selected["kept_position"]),
        "candidate_geometries": geometries,
        "eligible_kept_positions": sorted(int(item) for item in eligible_positions),
        "min_valid_depth_fraction": float(min_valid_depth_fraction),
        "selection_roi_xywh": None
        if selection_roi is None
        else [int(value) for value in selection_roi],
    }
    return selected_kept, updated_candidates, selection


def load_image_parity_approval(path: Path | None) -> dict[str, Any]:
    """Require an explicit frozen-frame parity artifact before backend switch."""

    if path is None:
        raise RuntimeError(
            "the SAM image backend requires --sam-image-parity-report from a "
            "completed frozen-frame parity run"
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"SAM image parity report does not exist: {resolved}")
    report = read_json(resolved)
    if report.get("schema_version") != 1:
        raise RuntimeError("SAM image parity report schema_version must be 1")
    if report.get("approved") is not True:
        raise RuntimeError(
            f"SAM image parity report is not approved: {resolved}"
        )
    if not isinstance(report.get("cases"), int) or report["cases"] < 60:
        raise RuntimeError("SAM image parity report must record at least 60 cases")
    if not isinstance(report.get("min_iou"), (int, float)) or float(
        report["min_iou"]
    ) < 0.95:
        raise RuntimeError("SAM image parity approval requires min_iou >= 0.95")
    return {"path": str(resolved), **report}


def load_relation_threshold_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise RuntimeError(
            "v2 segmentation requires --v2-relation-threshold-report from "
            "the labeled safety-set grid search"
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"relation threshold report does not exist: {resolved}")
    report = read_json(resolved)
    if report.get("schema_version") != 1:
        raise RuntimeError("relation threshold report schema_version must be 1")
    if not isinstance(report.get("cases"), int) or report["cases"] < 60:
        raise RuntimeError("relation threshold calibration requires at least 60 cases")
    if report.get("safety_false_accepts") != 0:
        raise RuntimeError("relation threshold calibration has safety false accepts")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RuntimeError("relation threshold calibration has no selected thresholds")
    unknown = set(thresholds) - set(relation_geometry.DEFAULT_THRESHOLDS)
    if unknown:
        raise RuntimeError(f"relation threshold report has unknown keys: {sorted(unknown)}")
    case_counts = report.get("relationship_case_counts")
    safety_counts = report.get("relationship_safety_case_counts")
    expected_relationships = relation_geometry.SUPPORTED_RELATIONSHIPS
    if (
        not isinstance(case_counts, dict)
        or set(case_counts) != expected_relationships
        or any(not isinstance(value, int) or value < 1 for value in case_counts.values())
    ):
        raise RuntimeError(
            "relation threshold report must cover every supported relationship"
        )
    if (
        not isinstance(safety_counts, dict)
        or set(safety_counts) != expected_relationships
        or any(not isinstance(value, int) or value < 1 for value in safety_counts.values())
    ):
        raise RuntimeError(
            "relation threshold report needs safety cases for every relationship"
        )
    return {"path": str(resolved), **report}


def load_v2_release_approval(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise RuntimeError(
            "v2 segmentation requires --v2-release-report after every release gate passes"
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"v2 release report does not exist: {resolved}")
    report = read_json(resolved)
    gates = report.get("gates")
    if (
        report.get("schema_version") != 2
        or report.get("approved") is not True
        or not isinstance(gates, dict)
        or set(gates) != check_v2_release_gates.RELEASE_GATE_NAMES
        or not all(value is True for value in gates.values())
    ):
        raise RuntimeError("v2 release report is not fully approved")
    return {"path": str(resolved), **report}


def startup(args: argparse.Namespace) -> None:
    STATE["t0"] = time.monotonic()
    STATE["args"] = args
    STATE["run_dir"] = args.run_dir.expanduser().resolve()
    STATE["run_dir"].mkdir(parents=True, exist_ok=True)

    STATE["v2_relation_calibration"] = None
    STATE["v2_release_approval"] = None
    args.v2_relation_thresholds = None
    if args.v2_enabled:
        STATE["v2_release_approval"] = load_v2_release_approval(
            args.v2_release_report
        )
    if v2_segmentation_available(args):
        STATE["v2_relation_calibration"] = load_relation_threshold_calibration(
            args.v2_relation_threshold_report
        )
        args.v2_relation_thresholds = dict(
            STATE["v2_relation_calibration"]["thresholds"]
        )

    print(f"loading SAM 3.1 once ({args.sam_backend} backend)...")
    STATE["sam_image_parity"] = None
    if args.sam_backend == "image":
        STATE["sam_image_parity"] = load_image_parity_approval(
            args.sam_image_parity_report
        )
        STATE["sam"] = sam3_image_service.Sam3ImageService(
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
            det_threshold=args.det_threshold,
            use_fa3=args.use_fa3,
            verbose_load=args.verbose_load,
        )
    else:
        STATE["sam"] = task6.MultiplexSam3AgentService(
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
            det_threshold=args.det_threshold,
            use_fa3=args.use_fa3,
            verbose_load=args.verbose_load,
        )

    STATE["qwen_runtime"] = None
    if args.warm_qwen:
        print(f"loading cached Qwen once: {args.qwen_model}...")
        STATE["qwen_runtime"] = local_qwen.preload_qwen(
            args.qwen_model,
            local_files_only=not args.allow_qwen_downloads,
            device_map=args.qwen_device_map,
        )
        print(f"Qwen ready: {STATE['qwen_runtime']}")

    print("opening ZED camera once...")
    STATE["sl"], STATE["zed"] = task5.open_zed(args)
    print("service ready")


def shutdown() -> None:
    sam = STATE.pop("sam", None)
    if sam is not None:
        sam.close()
    zed = STATE.pop("zed", None)
    if zed is not None:
        zed.close()


atexit.register(shutdown)


def capture_request_frame(
    req_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    args = STATE["args"]
    sl = STATE["sl"]
    zed = STATE["zed"]

    full_rgb_np = task5.grab_zed_rgb(
        sl,
        zed,
        view_name=args.view,
        warmup_frames=args.warmup_frames,
        grab_timeout=args.grab_timeout,
    )
    rgb_np, crop_info = task5.apply_crop(full_rgb_np, args.crop)
    depth_np, xyz_np, depth_info = task5.retrieve_zed_depth_and_xyz(sl, zed, args.view)
    depth_np, _ = task5.apply_crop(depth_np, args.crop)
    xyz_np, _ = task5.apply_crop(xyz_np, args.crop)

    frame_path = req_dir / "frame.png"
    task5.write_rgb_image(frame_path, rgb_np)
    full_frame_path = req_dir / "full_frame.png"
    if crop_info.get("enabled", False):
        task5.write_rgb_image(full_frame_path, full_rgb_np)
    else:
        full_frame_path = frame_path
    return rgb_np, full_rgb_np, depth_np, xyz_np, {
        "view": args.view,
        "resolution": args.resolution,
        "camera_fps": int(args.camera_fps),
        "crop": crop_info,
        **depth_info,
        "saved_frame": str(frame_path),
        "saved_full_frame": str(full_frame_path),
    }


def _unique_prompts(*prompts: str) -> list[str]:
    unique: list[str] = []
    normalized: set[str] = set()
    for prompt in prompts:
        key = grounding_intent.normalize_phrase(prompt)
        if key and key not in normalized:
            normalized.add(key)
            unique.append(prompt)
    return unique


def localize_source_region(
    *,
    rgb_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    intent_bundle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build one padded canonical-crop view around a named source region."""

    source_region = intent_bundle["grounding_intent"].get("source_region")
    region_prompt = (
        candidate_generation.source_region_prompt(str(source_region))
        if source_region
        else None
    )
    record: dict[str, Any] = {
        "requested": bool(args.candidate_multiscale and source_region),
        "source_region": source_region,
        "prompt": region_prompt,
        "status": "not_requested",
        "sam_json": None,
        "raw_candidate_count": 0,
        "eligible_candidate_count": 0,
        "selected_source_candidate_index": None,
        "selected_score": None,
        "source_bbox_xywh_crop_pixels": None,
        "padded_roi_xyxy_crop_pixels": None,
        "padding_fraction_each_side": float(
            args.candidate_region_padding_fraction
        ),
        "image_path": str(frame_path),
        "localized_image_path": None,
        "error": None,
        "elapsed_s": 0.0,
    }
    if not record["requested"]:
        return None, record

    started = time.monotonic()
    try:
        sam_json = STATE["sam"](
            image_path=str(frame_path),
            text_prompt=str(region_prompt),
            output_folder_path=str(req_dir / "candidate_runs" / "region_localizer"),
        )
        record["sam_json"] = sam_json
        outputs = read_json(sam_json)
        masks = task6.decode_agent_masks(outputs)
        scores = np.asarray(outputs.get("pred_scores", []), dtype=np.float32)
        record["raw_candidate_count"] = int(masks.shape[0])
        if masks.shape[1:] != rgb_np.shape[:2]:
            raise ValueError(
                "source-region mask shape does not match the canonical workspace crop: "
                f"{masks.shape[1:]} != {rgb_np.shape[:2]}"
            )

        eligible_count = 0
        for index, mask in enumerate(masks):
            score = float(scores[index]) if index < scores.size else 0.0
            if score > args.presence_conf_threshold and int(mask.sum()) > args.min_area:
                eligible_count += 1
        record["eligible_candidate_count"] = eligible_count
        selected_index = candidate_generation.select_source_region_candidate(
            masks,
            scores,
            source_region=str(source_region),
            conf_threshold=args.presence_conf_threshold,
            min_area=args.min_area,
        )
        if selected_index is None:
            record["status"] = "no_eligible_region"
            return None, record

        selected_mask = np.asarray(masks[selected_index], dtype=bool)
        x0, y0, x1, y1 = candidate_generation.padded_mask_roi(
            selected_mask,
            padding_fraction=args.candidate_region_padding_fraction,
        )
        localized_image = rgb_np[y0:y1, x0:x1].copy()
        localized_path = req_dir / "candidate_views" / "localized_region.png"
        task5.write_rgb_image(localized_path, localized_image)
        selected_score = (
            float(scores[selected_index]) if selected_index < scores.size else 0.0
        )
        source_bbox = candidate_generation.mask_bbox_xywh(selected_mask)
        record.update(
            {
                "status": "localized",
                "selected_source_candidate_index": int(selected_index),
                "selected_score": selected_score,
                "source_bbox_xywh_crop_pixels": source_bbox,
                "padded_roi_xyxy_crop_pixels": [x0, y0, x1, y1],
                "localized_image_path": str(localized_path),
            }
        )
        primary = intent_bundle["primary_sam_phrase"]
        category = intent_bundle["grounding_intent"]["category"]
        return {
            "view_id": "localized_region",
            "view_kind": "localized_region",
            "image": localized_image,
            "image_path": localized_path,
            "source_roi_xyxy": (
                0,
                0,
                int(localized_image.shape[1]),
                int(localized_image.shape[0]),
            ),
            "canonical_roi_xyxy": (x0, y0, x1, y1),
            "prompts": _unique_prompts(primary, category),
            "source_region": str(source_region),
            "region_source_bbox_xywh": source_bbox,
            "region_padding_fraction": float(
                args.candidate_region_padding_fraction
            ),
        }, record
    except Exception as exc:
        record.update({"status": "error", "error": repr(exc)})
        return None, record
    finally:
        record["elapsed_s"] = round(time.monotonic() - started, 3)


def build_candidate_views(
    *,
    rgb_np: np.ndarray,
    full_rgb_np: np.ndarray,
    frame_info: dict[str, Any],
    frame_path: Path,
    req_dir: Path,
    intent_bundle: dict[str, Any],
    args: argparse.Namespace,
    localized_region_view: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build workspace, localized-region, overlapping-tile, and context views."""

    height, width = rgb_np.shape[:2]
    all_prompts = list(intent_bundle["prompt_family"])
    primary = intent_bundle["primary_sam_phrase"]
    category = intent_bundle["grounding_intent"]["category"]
    views: list[dict[str, Any]] = [
        {
            "view_id": "workspace",
            "view_kind": "workspace_crop",
            "image": rgb_np,
            "image_path": frame_path,
            "source_roi_xyxy": (0, 0, width, height),
            "canonical_roi_xyxy": (0, 0, width, height),
            "prompts": all_prompts,
        }
    ]
    if localized_region_view is not None:
        views.append(localized_region_view)

    if args.candidate_multiscale:
        tile_prompts = _unique_prompts(primary, category)
        tile_dir = req_dir / "candidate_views"
        for index, (x0, y0, x1, y1) in enumerate(
            candidate_generation.overlapping_tile_rois(
                width,
                height,
                scale=args.candidate_tile_scale,
            ),
            start=1,
        ):
            tile = rgb_np[y0:y1, x0:x1].copy()
            tile_path = tile_dir / f"tile_{index}.png"
            task5.write_rgb_image(tile_path, tile)
            views.append(
                {
                    "view_id": f"tile_{index}",
                    "view_kind": "overlapping_tile",
                    "image": tile,
                    "image_path": tile_path,
                    "source_roi_xyxy": (0, 0, tile.shape[1], tile.shape[0]),
                    "canonical_roi_xyxy": (x0, y0, x1, y1),
                    "prompts": tile_prompts,
                }
            )

    crop_info = frame_info["crop"]
    if args.candidate_full_frame and crop_info.get("enabled", False):
        crop_x0, crop_y0, crop_x1, crop_y1 = [
            int(value) for value in crop_info["applied_xyxy"]
        ]
        views.append(
            {
                "view_id": "full_frame",
                "view_kind": "full_frame_context",
                "image": full_rgb_np,
                "image_path": Path(frame_info["saved_full_frame"]),
                "source_roi_xyxy": (crop_x0, crop_y0, crop_x1, crop_y1),
                "canonical_roi_xyxy": (0, 0, width, height),
                "prompts": [primary],
            }
        )
    return views


def generate_direct_candidates(
    *,
    rgb_np: np.ndarray,
    full_rgb_np: np.ndarray,
    frame_info: dict[str, Any],
    frame_path: Path,
    req_dir: Path,
    intent_bundle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    list[tuple[np.ndarray, float]],
    list[dict[str, Any]],
    Path,
    dict[str, Any],
]:
    """Run the bounded prompt/view family and merge canonical crop masks."""

    localized_region_view, region_localization = localize_source_region(
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        intent_bundle=intent_bundle,
        args=args,
    )
    views = build_candidate_views(
        rgb_np=rgb_np,
        full_rgb_np=full_rgb_np,
        frame_info=frame_info,
        frame_path=frame_path,
        req_dir=req_dir,
        intent_bundle=intent_bundle,
        args=args,
        localized_region_view=localized_region_view,
    )
    raw_candidates: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    canonical_shape = rgb_np.shape[:2]
    generation_start = time.monotonic()
    for view in views:
        for prompt in view["prompts"]:
            run_start = time.monotonic()
            source_json = None
            error = None
            raw_count = 0
            try:
                source_json = STATE["sam"](
                    image_path=str(view["image_path"]),
                    text_prompt=prompt,
                    output_folder_path=str(
                        req_dir
                        / "candidate_runs"
                        / str(view["view_id"])
                        / task6.safe_name(prompt)
                    ),
                )
                outputs = read_json(source_json)
                masks = task6.decode_agent_masks(outputs)
                scores = np.asarray(outputs.get("pred_scores", []), dtype=np.float32)
                raw_count = int(masks.shape[0])
                for source_index, mask in enumerate(masks):
                    score = float(scores[source_index]) if source_index < scores.size else 0.0
                    projected = candidate_generation.project_mask_to_canonical(
                        mask,
                        source_roi_xyxy=view["source_roi_xyxy"],
                        canonical_roi_xyxy=view["canonical_roi_xyxy"],
                        canonical_shape_hw=canonical_shape,
                    )
                    if projected.any():
                        source_area = int(np.count_nonzero(mask))
                        raw_candidates.append(
                            {
                                "mask": projected,
                                "score": score,
                                "prompt": prompt,
                                "view_id": view["view_id"],
                                "view_kind": view["view_kind"],
                                "source_candidate_index": source_index,
                                "source_json": source_json,
                                "workspace_retained_fraction": float(
                                    np.count_nonzero(projected) / max(source_area, 1)
                                ),
                            }
                        )
            except Exception as exc:
                error = repr(exc)
            source_runs.append(
                {
                    "run_role": "target_candidate_generation",
                    "view_id": view["view_id"],
                    "view_kind": view["view_kind"],
                    "image_path": str(view["image_path"]),
                    "source_roi_xyxy": list(view["source_roi_xyxy"]),
                    "canonical_roi_xyxy": list(view["canonical_roi_xyxy"]),
                    "prompt": prompt,
                    "sam_json": source_json,
                    "raw_candidate_count": raw_count,
                    "elapsed_s": round(time.monotonic() - run_start, 3),
                    "error": error,
                }
            )

    successful_run_count = sum(run["error"] is None for run in source_runs)
    all_target_runs_failed = bool(source_runs) and successful_run_count == 0

    merged = candidate_generation.deduplicate_candidates(
        raw_candidates,
        iou_threshold=args.candidate_dedup_iou,
    )
    kept, candidates = candidate_generation.gate_merged_candidates(
        merged,
        conf_threshold=args.presence_conf_threshold,
        min_area=args.min_area,
        max_candidates=args.candidate_max_count,
        min_workspace_retained_fraction=(
            args.candidate_min_workspace_retained_fraction
        ),
    )
    mask_dir = req_dir / "candidate_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    width = int(rgb_np.shape[1])
    height = int(rgb_np.shape[0])
    pred_boxes: list[list[float]] = []
    manifest_candidates: list[dict[str, Any]] = []
    for index, (merged_candidate, candidate) in enumerate(zip(merged, candidates)):
        mask_path = mask_dir / f"candidate_{index + 1:02d}.png"
        if not cv2.imwrite(str(mask_path), merged_candidate["mask"].astype(np.uint8) * 255):
            raise OSError(f"cannot write candidate mask {mask_path}")
        bbox = list(merged_candidate["bbox_xywh_crop_pixels"])
        candidate["mask_path_crop"] = str(mask_path)
        candidate["bbox_xywh_full_pixels"] = candidate_generation.crop_bbox_to_full(
            bbox,
            frame_info["crop"],
        )
        pred_boxes.append(
            [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            ]
        )
        manifest_candidates.append(
            {
                **candidate,
                "mask_path_crop": str(mask_path),
            }
        )

    merged_json_path = req_dir / "merged_candidates.json"
    merged_output = {
        "schema_version": 1,
        "orig_img_h": height,
        "orig_img_w": width,
        "pred_boxes": pred_boxes,
        "pred_scores": [float(item["score"]) for item in merged],
        "candidate_masks": [item["mask_path_crop"] for item in manifest_candidates],
        "source_runs": source_runs,
        "dedup_iou_threshold": float(args.candidate_dedup_iou),
        "raw_candidate_count": len(raw_candidates),
        "merged_candidate_count": len(merged),
    }
    merged_json_path.write_text(
        json.dumps(merged_output, indent=2) + "\n",
        encoding="utf-8",
    )
    generation_record = {
        "prompt_family": list(intent_bundle["prompt_family"]),
        "views": [
            {
                "view_id": view["view_id"],
                "view_kind": view["view_kind"],
                "image_path": str(view["image_path"]),
                "source_roi_xyxy": list(view["source_roi_xyxy"]),
                "canonical_roi_xyxy": list(view["canonical_roi_xyxy"]),
                "prompts": list(view["prompts"]),
                **(
                    {
                        "source_region": view["source_region"],
                        "region_source_bbox_xywh": list(
                            view["region_source_bbox_xywh"]
                        ),
                        "region_padding_fraction": float(
                            view["region_padding_fraction"]
                        ),
                    }
                    if view["view_kind"] == "localized_region"
                    else {}
                ),
            }
            for view in views
        ],
        "region_localization": region_localization,
        "source_runs": source_runs,
        "successful_source_run_count": successful_run_count,
        "failed_source_run_count": len(source_runs) - successful_run_count,
        "all_target_runs_failed": all_target_runs_failed,
        "raw_candidate_count": len(raw_candidates),
        "merged_candidate_count": len(merged),
        "dedup_iou_threshold": float(args.candidate_dedup_iou),
        "max_candidates": int(args.candidate_max_count),
        "min_workspace_retained_fraction": float(
            args.candidate_min_workspace_retained_fraction
        ),
        "candidate_manifest": manifest_candidates,
        "merged_json": str(merged_json_path),
        "elapsed_s": round(time.monotonic() - generation_start, 3),
    }
    generation_path = req_dir / "candidate_generation.json"
    generation_path.write_text(
        json.dumps(generation_record, indent=2) + "\n",
        encoding="utf-8",
    )
    generation_record["output"] = str(generation_path)
    if all_target_runs_failed:
        raise RuntimeError(
            "all target candidate-generation runs failed; details were saved to "
            f"{generation_path}"
        )
    return kept, candidates, merged_json_path, generation_record


def summarize_kept(
    *,
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    return {
        "conf_threshold": float(args.presence_conf_threshold),
        "min_area_pixels": int(args.min_area),
        "num_candidates": len(candidates),
        "num_kept": len(kept),
        "kept_indices": [candidate["index"] for candidate in kept_candidates],
        "kept_scores": [candidate["score"] for candidate in kept_candidates],
        "kept_areas_pixels": [
            candidate["area_pixels"] for candidate in kept_candidates
        ],
        "rejected": [candidate for candidate in candidates if not candidate["kept"]],
    }


def apply_geometry_safety_gates(
    *,
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    depth_np: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on workspace membership and unusable/noisy mask depth."""

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    if len(kept_candidates) != len(kept):
        raise ValueError("kept masks and candidate metadata are inconsistent")
    retained: list[tuple[np.ndarray, float]] = []
    updated_by_index: dict[int, dict[str, Any]] = {}
    rejection_count = 0
    for (mask, score), original in zip(kept, kept_candidates):
        candidate = dict(original)
        center_x, center_y = mask_center_xy(mask)
        inside_workspace = True
        if args.selection_roi is not None:
            roi_x, roi_y, roi_width, roi_height = args.selection_roi
            inside_workspace = (
                roi_x <= center_x < roi_x + roi_width
                and roi_y <= center_y < roi_y + roi_height
            )
        stats = mask_depth.mask_depth_stats(mask, depth_np)
        spread_m = None
        if stats["p10"] is not None and stats["p90"] is not None:
            spread_m = float(stats["p90"] - stats["p10"])
        rejection_reasons: list[str] = []
        if not inside_workspace:
            rejection_reasons.append("outside_workspace_roi")
        if stats["valid_fraction"] < args.selection_min_valid_depth_fraction:
            rejection_reasons.append("low_valid_depth_fraction")
        if stats["valid_depth_pixels"] < args.candidate_min_valid_depth_pixels:
            rejection_reasons.append("insufficient_valid_depth_pixels")
        if (
            spread_m is None
            or spread_m > args.candidate_max_depth_spread_mm / 1000.0
        ):
            rejection_reasons.append("excessive_depth_spread")
        candidate.update(
            {
                "center_xy_crop_pixels": [round(center_x, 2), round(center_y, 2)],
                "inside_workspace_roi": inside_workspace,
                "depth_stats_m": stats,
                "depth_spread_m": spread_m,
            }
        )
        if rejection_reasons:
            candidate["kept"] = False
            candidate["reject_reasons"] = list(candidate["reject_reasons"]) + rejection_reasons
            rejection_count += 1
        else:
            retained.append((mask, score))
        updated_by_index[int(candidate["index"])] = candidate

    updated = [
        updated_by_index.get(int(candidate["index"]), dict(candidate))
        for candidate in candidates
    ]
    report = {
        "workspace_roi_xywh": None
        if args.selection_roi is None
        else [int(value) for value in args.selection_roi],
        "min_valid_depth_fraction": float(args.selection_min_valid_depth_fraction),
        "min_valid_depth_pixels": int(args.candidate_min_valid_depth_pixels),
        "max_depth_spread_mm": float(args.candidate_max_depth_spread_mm),
        "input_candidate_count": len(kept),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": rejection_count,
    }
    return retained, updated, report


def build_verifier_candidate_records(
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe score/area-gated candidates using one-based verifier IDs."""

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    if len(kept_candidates) != len(kept):
        raise ValueError("kept masks and candidate metadata are inconsistent")

    records = []
    for candidate_id, ((mask, score), candidate) in enumerate(
        zip(kept, kept_candidates), start=1
    ):
        ys, xs = np.where(mask)
        if xs.size == 0:
            raise ValueError(f"Candidate {candidate_id} has an empty mask")
        height, width = mask.shape
        records.append(
            {
                "candidate_id": candidate_id,
                "sam_index": int(candidate["index"]),
                "sam_score": float(score),
                "area_pixels": int(candidate["area_pixels"]),
                "area_fraction": round(float(mask.mean()), 6),
                "center_xy_crop_pixels": [
                    round(float(xs.mean()), 2),
                    round(float(ys.mean()), 2),
                ],
                "bbox_xywh_crop_pixels": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ],
                "bbox_xywh_full_pixels": candidate.get("bbox_xywh_full_pixels"),
                "crop_size_wh_pixels": [int(width), int(height)],
                "duplicate_count": int(candidate.get("duplicate_count", 1)),
                "prompt_view_provenance": candidate.get("provenance", []),
                "mask_path_crop": candidate.get("mask_path_crop"),
            }
        )
    return records


def verify_candidates_with_qwen(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    grounding_intent_value: dict[str, Any],
    rgb_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    artifact_stem: str,
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[
    list[tuple[np.ndarray, float]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run the fail-closed Qwen visual gate before depth or spatial selection."""

    verification_start = time.monotonic()
    verification: dict[str, Any] = {
        "status": "not_run",
        "decision": None,
        "selected_candidate_ids": [],
        "model_selected_candidate_ids": [],
        "selected_sam_indices": [],
        "confidence": None,
        "reason": "no_score_area_gated_candidates",
        "model": args.qwen_model,
        "target_phrase": target_phrase,
        "grounding_intent": grounding_intent_value,
        "intent_hash": grounding_intent.intent_hash(grounding_intent_value),
        "selector_deferred": selector,
        "min_select_confidence": float(args.verifier_min_confidence),
        "max_area_fraction": float(args.verifier_max_area_fraction),
        "candidate_count": len(kept),
        "candidate_records": [],
        "candidate_overlay": None,
        "candidate_zoom": None,
        "candidate_manifest": None,
        "attempts": [],
        "elapsed_s": 0.0,
    }
    if not kept:
        return kept, candidates, verification

    try:
        candidate_records = build_verifier_candidate_records(kept, candidates)
        candidate_overlay_path = req_dir / f"{artifact_stem}_qwen_candidates.png"
        candidate_overlay = candidate_verifier.render_numbered_candidates(
            rgb_np,
            kept,
            candidate_overlay_path,
        )
        candidate_zoom_path = req_dir / f"{artifact_stem}_qwen_candidate_zooms.png"
        candidate_zoom = candidate_verifier.render_candidate_zooms(
            rgb_np,
            kept,
            candidate_zoom_path,
        )
        manifest_path = req_dir / f"{artifact_stem}_qwen_candidates.json"
        manifest_path.write_text(
            json.dumps(candidate_records, indent=2) + "\n",
            encoding="utf-8",
        )

        def send_generate_request(messages: list[dict[str, Any]]) -> str:
            return local_qwen.qwen_generate(
                messages,
                model_id=args.qwen_model,
                max_new_tokens=args.verifier_max_new_tokens,
                local_files_only=not args.allow_qwen_downloads,
                device_map=args.qwen_device_map,
                do_sample=False,
                response_prefix='{"decision":',
            )

        verification = candidate_verifier.run_visual_verifier(
            request=request,
            target_phrase=target_phrase,
            selector=selector,
            frame_path=frame_path,
            candidate_overlay_path=candidate_overlay_path,
            candidate_zoom_path=candidate_zoom_path,
            candidate_records=candidate_records,
            send_generate_request=send_generate_request,
            min_select_confidence=args.verifier_min_confidence,
            grounding_intent_value=grounding_intent_value,
        )
        verification = candidate_verifier.apply_max_area_fraction_policy(
            verification,
            candidate_records,
            args.verifier_max_area_fraction,
        )
        verification.update(
            {
                "model": args.qwen_model,
                "target_phrase": target_phrase,
                "grounding_intent": grounding_intent_value,
                "intent_hash": grounding_intent.intent_hash(grounding_intent_value),
                "selector_deferred": selector,
                "min_select_confidence": float(args.verifier_min_confidence),
                "max_area_fraction": float(args.verifier_max_area_fraction),
                "candidate_count": len(candidate_records),
                "candidate_records": candidate_records,
                "candidate_overlay": candidate_overlay["output"],
                "candidate_zoom": candidate_zoom["output"],
                "candidate_manifest": str(manifest_path),
            }
        )
    except Exception as exc:
        verification.update(
            {
                "status": "error",
                "decision": None,
                "selected_candidate_ids": [],
                "model_selected_candidate_ids": [],
                "confidence": None,
                "reason": f"Qwen verifier setup failed: {exc!r}",
                "attempts": [],
            }
        )

    kept, candidates, selected_sam_indices = (
        candidate_verifier.apply_verifier_selection(
            kept,
            candidates,
            verification,
        )
    )
    verification["selected_sam_indices"] = selected_sam_indices
    verification["elapsed_s"] = round(time.monotonic() - verification_start, 3)
    return kept, candidates, verification


def direct_segment(
    *,
    request: str,
    rgb_np: np.ndarray,
    full_rgb_np: np.ndarray,
    depth_np: np.ndarray,
    frame_path: Path,
    frame_info: dict[str, Any],
    req_dir: Path,
    args: argparse.Namespace,
    intent_bundle: dict[str, Any],
) -> dict[str, Any]:
    canonical_intent = intent_bundle["grounding_intent"]
    sam_prompt = intent_bundle["primary_sam_phrase"]
    selector = canonical_intent.get("selector")
    kept, candidates, sam_json_path, generation = generate_direct_candidates(
        rgb_np=rgb_np,
        full_rgb_np=full_rgb_np,
        frame_info=frame_info,
        frame_path=frame_path,
        req_dir=req_dir,
        intent_bundle=intent_bundle,
        args=args,
    )
    fallback_eligible = len(kept) == 0
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request,
        target_phrase=sam_prompt,
        selector=selector,
        grounding_intent_value=canonical_intent,
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem="direct",
        kept=kept,
        candidates=candidates,
        args=args,
    )
    kept, candidates, geometry_gate = apply_geometry_safety_gates(
        kept=kept,
        candidates=candidates,
        depth_np=depth_np,
        args=args,
    )
    kept, candidates, selection = select_spatial_mask(
        kept,
        candidates,
        selector,
        depth_np=depth_np,
        min_valid_depth_fraction=args.selection_min_valid_depth_fraction,
        selection_roi=args.selection_roi,
    )
    overlay = task2.overlay_masks(
        rgb_np,
        kept,
        request,
        req_dir / "overlay_direct.png",
    )
    return {
        "path": "direct_verified_spatial"
        if selection is not None
        else "direct_verified",
        "kept": kept,
        "presence_gate": summarize_kept(kept=kept, candidates=candidates, args=args),
        "sam_json": str(sam_json_path),
        "overlay": overlay,
        "candidate_overlay": verification.get("candidate_overlay"),
        "candidate_zoom": verification.get("candidate_zoom"),
        "agent_render_output": None,
        "sam_prompt": sam_prompt,
        "sam_prompts": list(intent_bundle["prompt_family"]),
        "candidate_generation": generation,
        "geometry_gate": geometry_gate,
        "selection": selection,
        "intent": {
            "target_phrase": sam_prompt,
            "selector": selector,
            "parser": "structured_v1",
        },
        "verification": verification,
        "fallback_eligible": fallback_eligible,
    }


def agent_fallback_segment(
    *,
    request: str,
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    args: argparse.Namespace,
    intent_bundle: dict[str, Any],
) -> dict[str, Any]:
    agent_dir = req_dir / "agent_workspace"
    agent_dir.mkdir(parents=True, exist_ok=True)
    history, final_outputs, rendered_final = task6.agent_inference(
        str(frame_path),
        request,
        debug=args.debug,
        send_generate_request=task6.build_qwen_sender(args),
        call_sam_service=STATE["sam"],
        max_generations=args.max_generations,
        output_dir=str(agent_dir),
    )
    agent_render_output = req_dir / "agent_render.png"
    rendered_final.save(agent_render_output)

    masks = task6.decode_agent_masks(final_outputs)
    scores = np.asarray(final_outputs.get("pred_scores", []), dtype=np.float32)
    kept, candidates = task2.gate_masks(
        masks,
        scores,
        conf_thresh=args.presence_conf_threshold,
        min_area=args.min_area,
    )
    canonical_intent = intent_bundle["grounding_intent"]
    target_phrase = intent_bundle["primary_sam_phrase"]
    selector = canonical_intent.get("selector")
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request,
        target_phrase=target_phrase,
        selector=selector,
        grounding_intent_value=canonical_intent,
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem="agent",
        kept=kept,
        candidates=candidates,
        args=args,
    )
    kept, candidates, geometry_gate = apply_geometry_safety_gates(
        kept=kept,
        candidates=candidates,
        depth_np=depth_np,
        args=args,
    )
    kept, candidates, selection = select_spatial_mask(
        kept,
        candidates,
        selector,
        depth_np=depth_np,
        min_valid_depth_fraction=args.selection_min_valid_depth_fraction,
        selection_roi=args.selection_roi,
    )
    overlay = task2.overlay_masks(
        rgb_np,
        kept,
        request,
        req_dir / "overlay_agent.png",
    )
    final_outputs_path = req_dir / "agent_final_outputs.json"
    final_outputs_path.write_text(
        json.dumps(final_outputs, indent=2) + "\n",
        encoding="utf-8",
    )
    history_path = req_dir / "agent_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    return {
        "path": "agent_verified_spatial"
        if selection is not None
        else "agent_verified",
        "kept": kept,
        "presence_gate": summarize_kept(kept=kept, candidates=candidates, args=args),
        "sam_json": str(final_outputs_path),
        "overlay": overlay,
        "candidate_overlay": verification.get("candidate_overlay"),
        "candidate_zoom": verification.get("candidate_zoom"),
        "agent_render_output": str(agent_render_output),
        "agent_history": str(history_path),
        "sam_prompt": target_phrase,
        "sam_prompts": [target_phrase],
        "candidate_generation": {
            "path": "qwen_agent_fallback",
            "prompt_family": [target_phrase],
            "agent_history": str(history_path),
            "agent_outputs": str(final_outputs_path),
        },
        "geometry_gate": geometry_gate,
        "selection": selection,
        "intent": {
            "target_phrase": target_phrase,
            "selector": selector,
            "parser": "structured_v1",
            "fallback": "agent",
        },
        "verification": verification,
        "fallback_eligible": False,
    }


def write_final_mask_artifacts(
    *,
    kept: list[tuple[np.ndarray, float]],
    kept_indices: list[int],
    crop_info: dict[str, Any],
    req_dir: Path,
) -> list[dict[str, Any]]:
    """Persist final masks in crop-local and full-camera coordinates."""

    if len(kept) != len(kept_indices):
        raise ValueError("final masks and kept indices are inconsistent")
    output_dir = req_dir / "final_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, ((mask, score), candidate_index) in enumerate(
        zip(kept, kept_indices),
        start=1,
    ):
        crop_path = output_dir / f"mask_{position:02d}_crop.png"
        full_path = output_dir / f"mask_{position:02d}_full.png"
        crop_mask = np.asarray(mask, dtype=bool)
        full_mask = candidate_generation.expand_crop_mask_to_full(crop_mask, crop_info)
        if not cv2.imwrite(str(crop_path), crop_mask.astype(np.uint8) * 255):
            raise OSError(f"cannot write final crop mask {crop_path}")
        if not cv2.imwrite(str(full_path), full_mask.astype(np.uint8) * 255):
            raise OSError(f"cannot write final full-frame mask {full_path}")
        bbox_crop = candidate_generation.mask_bbox_xywh(crop_mask)
        bbox_full = candidate_generation.crop_bbox_to_full(bbox_crop, crop_info)
        center_x, center_y = mask_center_xy(crop_mask)
        offset_x = offset_y = 0
        if crop_info.get("enabled", False):
            offset_x, offset_y = [int(value) for value in crop_info["applied_xyxy"][:2]]
        records.append(
            {
                "candidate_index": int(candidate_index),
                "score": float(score),
                "mask_crop": str(crop_path),
                "mask_full": str(full_path),
                "area_pixels": int(np.count_nonzero(crop_mask)),
                "bbox_xywh_crop_pixels": bbox_crop,
                "bbox_xywh_full_pixels": bbox_full,
                "center_xy_crop_pixels": [round(center_x, 2), round(center_y, 2)],
                "center_xy_full_pixels": [
                    round(center_x + offset_x, 2),
                    round(center_y + offset_y, 2),
                ],
            }
        )
    return records


def segment_once(
    request: str,
    *,
    use_agent_fallback: bool = True,
    supplied_intent: dict[str, Any] | None = None,
    supplied_intent_hash: str | None = None,
) -> dict[str, Any]:
    args = STATE["args"]
    if hasattr(STATE.get("sam"), "clear_embedding_cache"):
        STATE["sam"].clear_embedding_cache()
    req_dir = STATE["run_dir"] / timestamp_slug()
    req_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    intent_bundle = parse_request_intent(
        request,
        args,
        supplied_intent=supplied_intent,
    )
    if supplied_intent_hash is not None and supplied_intent_hash != intent_bundle["intent_hash"]:
        raise grounding_intent.GroundingIntentError(
            "supplied intent hash does not match the canonical intent",
            code="grounding_identity_mismatch",
            details={
                "supplied": supplied_intent_hash,
                "computed": intent_bundle["intent_hash"],
            },
        )
    intent_path = req_dir / "grounding_intent.json"
    intent_path.write_text(
        json.dumps(intent_bundle, indent=2) + "\n",
        encoding="utf-8",
    )
    if intent_bundle.get("shadow_comparison") is not None:
        shadow_path = req_dir / "intent_shadow.json"
        shadow_path.write_text(
            json.dumps(intent_bundle["shadow_comparison"], indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        shadow_path = None

    rgb_np, full_rgb_np, depth_np, xyz_np, frame_info = capture_request_frame(req_dir)
    frame_path = Path(frame_info["saved_frame"])

    result = direct_segment(
        request=request,
        rgb_np=rgb_np,
        full_rgb_np=full_rgb_np,
        depth_np=depth_np,
        frame_path=frame_path,
        frame_info=frame_info,
        req_dir=req_dir,
        args=args,
        intent_bundle=intent_bundle,
    )
    if result["fallback_eligible"] and use_agent_fallback:
        result = agent_fallback_segment(
            request=request,
            rgb_np=rgb_np,
            depth_np=depth_np,
            frame_path=frame_path,
            req_dir=req_dir,
            args=args,
            intent_bundle=intent_bundle,
        )

    kept = result.pop("kept")
    result.pop("fallback_eligible", None)
    object_depth = task5.object_depth_report(
        kept,
        depth_np=depth_np,
        xyz_np=xyz_np,
        view_name=args.view,
        depth_measure=frame_info["depth_measure"],
        xyz_measure=frame_info["xyz_measure"],
    )
    final_masks = write_final_mask_artifacts(
        kept=kept,
        kept_indices=result["presence_gate"]["kept_indices"],
        crop_info=frame_info["crop"],
        req_dir=req_dir,
    )
    response = {
        "schema_version": 1,
        "request": request,
        "grounding_intent": intent_bundle["grounding_intent"],
        "intent_hash": intent_bundle["intent_hash"],
        "intent_parser": intent_bundle["parser"],
        "intent_shadow": intent_bundle["shadow_comparison"],
        "intent_record": str(intent_path),
        "intent_shadow_record": None if shadow_path is None else str(shadow_path),
        "primary_sam_phrase": intent_bundle["primary_sam_phrase"],
        "prompt_family": intent_bundle["prompt_family"],
        "path": result["path"],
        "num_kept": result["presence_gate"]["num_kept"],
        "scores": [float(score) for _, score in kept],
        "presence_gate": result["presence_gate"],
        "object_depth": object_depth,
        "frame": frame_info["saved_frame"],
        "full_frame": frame_info["saved_full_frame"],
        "zed_frame": frame_info,
        "coordinate_spaces": {
            "segmentation": "workspace_crop",
            "depth_xyz": "workspace_crop_aligned",
            "robot_pixel": "full_camera_frame",
        },
        "final_masks": final_masks,
        "selected_mask": final_masks[0] if len(final_masks) == 1 else None,
        "overlay": result["overlay"],
        "candidate_overlay": result.get("candidate_overlay"),
        "candidate_zoom": result.get("candidate_zoom"),
        "sam_json": result["sam_json"],
        "agent_render_output": result.get("agent_render_output"),
        "sam_prompt": result.get("sam_prompt", request),
        "sam_prompts": result.get("sam_prompts", []),
        "candidate_generation": result.get("candidate_generation"),
        "geometry_gate": result.get("geometry_gate"),
        "selection": result.get("selection"),
        "intent": result.get("intent"),
        "verification": result.get("verification"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "output_dir": str(req_dir),
    }
    if result.get("agent_history") is not None:
        response["agent_history"] = result["agent_history"]

    result_path = req_dir / "result.json"
    response["result_json"] = str(result_path)
    result_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    return response


V1_SEGMENT_REQUEST_KEYS = {
    "schema_version",
    "source_phrase",
    "grounding_intent",
    "intent_hash",
    "use_agent_fallback",
}
V1_EVALUATE_REQUEST_KEYS = {
    "schema_version",
    "case_id",
    "source_phrase",
    "grounding_intent",
    "intent_hash",
    "image_path",
    "crop_xywh",
    "use_multiscale",
    "use_full_frame",
}


def validate_v1_segment_request(payload: Any) -> dict[str, Any]:
    """Validate the exact JSON request contract before camera capture."""

    if not isinstance(payload, dict):
        raise grounding_intent.GroundingIntentError("v1 request must be a JSON object")
    keys = set(payload)
    if keys != V1_SEGMENT_REQUEST_KEYS:
        raise grounding_intent.GroundingIntentError(
            "v1 request keys are invalid",
            details={
                "missing": sorted(V1_SEGMENT_REQUEST_KEYS - keys),
                "extra": sorted(keys - V1_SEGMENT_REQUEST_KEYS),
            },
        )
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != 1:
        raise grounding_intent.GroundingIntentError("v1 request schema_version must be 1")
    source_phrase = payload["source_phrase"]
    if not isinstance(source_phrase, str) or not source_phrase.strip():
        raise grounding_intent.GroundingIntentError("source_phrase must be non-empty")
    if not isinstance(payload["use_agent_fallback"], bool):
        raise grounding_intent.GroundingIntentError("use_agent_fallback must be boolean")
    canonical = grounding_intent.validate_grounding_intent(
        payload["grounding_intent"],
        expected_source_phrase=source_phrase,
    )
    supplied_hash = payload["intent_hash"]
    if not isinstance(supplied_hash, str) or supplied_hash != grounding_intent.intent_hash(
        canonical
    ):
        raise grounding_intent.GroundingIntentError(
            "intent_hash does not match grounding_intent",
            code="grounding_identity_mismatch",
        )
    return {
        "source_phrase": grounding_intent.collapse_space(source_phrase),
        "grounding_intent": canonical,
        "intent_hash": supplied_hash,
        "use_agent_fallback": payload["use_agent_fallback"],
    }


def validate_v1_evaluate_request(payload: Any) -> dict[str, Any]:
    """Validate one frozen-frame evaluation request."""

    if not isinstance(payload, dict):
        raise grounding_intent.GroundingIntentError(
            "evaluation request must be a JSON object"
        )
    keys = set(payload)
    if keys != V1_EVALUATE_REQUEST_KEYS:
        raise grounding_intent.GroundingIntentError(
            "evaluation request keys are invalid",
            details={
                "missing": sorted(V1_EVALUATE_REQUEST_KEYS - keys),
                "extra": sorted(keys - V1_EVALUATE_REQUEST_KEYS),
            },
        )
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != 1:
        raise grounding_intent.GroundingIntentError(
            "evaluation schema_version must be 1"
        )
    case_id = payload["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise grounding_intent.GroundingIntentError("case_id must be non-empty")
    image_path = payload["image_path"]
    if not isinstance(image_path, str) or not image_path.strip():
        raise grounding_intent.GroundingIntentError("image_path must be non-empty")
    crop = payload["crop_xywh"]
    if crop is not None:
        if (
            not isinstance(crop, list)
            or len(crop) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in crop)
        ):
            raise grounding_intent.GroundingIntentError(
                "crop_xywh must be null or four integers"
            )
        try:
            crop = task5.parse_crop(",".join(str(item) for item in crop))
        except argparse.ArgumentTypeError as exc:
            raise grounding_intent.GroundingIntentError(str(exc)) from exc
    if not isinstance(payload["use_multiscale"], bool) or not isinstance(
        payload["use_full_frame"], bool
    ):
        raise grounding_intent.GroundingIntentError(
            "use_multiscale and use_full_frame must be boolean"
        )
    segment_part = validate_v1_segment_request(
        {
            "schema_version": payload["schema_version"],
            "source_phrase": payload["source_phrase"],
            "grounding_intent": payload["grounding_intent"],
            "intent_hash": payload["intent_hash"],
            "use_agent_fallback": False,
        }
    )
    return {
        **segment_part,
        "case_id": case_id.strip(),
        "image_path": str(Path(image_path).expanduser().resolve()),
        "crop_xywh": crop,
        "use_multiscale": payload["use_multiscale"],
        "use_full_frame": payload["use_full_frame"],
    }


def evaluate_saved_frame(request: dict[str, Any]) -> dict[str, Any]:
    """Run candidate generation/verification on a frozen RGB image only."""

    args = STATE["args"]
    if hasattr(STATE.get("sam"), "clear_embedding_cache"):
        STATE["sam"].clear_embedding_cache()
    eval_args = argparse.Namespace(**vars(args))
    eval_args.crop = request["crop_xywh"]
    eval_args.candidate_multiscale = request["use_multiscale"]
    eval_args.candidate_full_frame = request["use_full_frame"]
    image_path = Path(request["image_path"])
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise grounding_intent.GroundingIntentError(
            f"cannot read frozen evaluation image {image_path}"
        )
    full_rgb_np = bgr[:, :, ::-1]
    rgb_np, crop_info = task5.apply_crop(full_rgb_np, eval_args.crop)
    req_dir = STATE["run_dir"] / "evaluation" / timestamp_slug()
    req_dir.mkdir(parents=True, exist_ok=True)
    frame_path = req_dir / "frame.png"
    full_frame_path = req_dir / "full_frame.png"
    task5.write_rgb_image(frame_path, rgb_np)
    if crop_info.get("enabled", False):
        task5.write_rgb_image(full_frame_path, full_rgb_np)
    else:
        full_frame_path = frame_path
    frame_info = {
        "view": "FROZEN_RGB",
        "resolution": f"{full_rgb_np.shape[1]}x{full_rgb_np.shape[0]}",
        "camera_fps": 0,
        "crop": crop_info,
        "saved_frame": str(frame_path),
        "saved_full_frame": str(full_frame_path),
        "depth_measure": None,
        "xyz_measure": None,
    }
    intent_bundle = parse_request_intent(
        request["source_phrase"],
        eval_args,
        supplied_intent=request["grounding_intent"],
    )
    if intent_bundle["intent_hash"] != request["intent_hash"]:
        raise grounding_intent.GroundingIntentError(
            "evaluation intent hash changed",
            code="grounding_identity_mismatch",
        )

    started = time.monotonic()
    kept, candidates, sam_json_path, generation = generate_direct_candidates(
        rgb_np=rgb_np,
        full_rgb_np=full_rgb_np,
        frame_info=frame_info,
        frame_path=frame_path,
        req_dir=req_dir,
        intent_bundle=intent_bundle,
        args=eval_args,
    )
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request["source_phrase"],
        target_phrase=intent_bundle["primary_sam_phrase"],
        selector=intent_bundle["grounding_intent"]["selector"],
        grounding_intent_value=intent_bundle["grounding_intent"],
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem="evaluation",
        kept=kept,
        candidates=candidates,
        args=eval_args,
    )
    kept, candidates, selection = select_spatial_mask(
        kept,
        candidates,
        intent_bundle["grounding_intent"]["selector"],
        depth_np=None,
        min_valid_depth_fraction=0.0,
        selection_roi=eval_args.selection_roi,
    )
    presence = summarize_kept(kept=kept, candidates=candidates, args=eval_args)
    final_masks = write_final_mask_artifacts(
        kept=kept,
        kept_indices=presence["kept_indices"],
        crop_info=crop_info,
        req_dir=req_dir,
    )
    overlay = task2.overlay_masks(
        rgb_np,
        kept,
        request["source_phrase"],
        req_dir / "overlay_evaluation.png",
    )
    result_path = req_dir / "result.json"
    response = {
        "schema_version": 1,
        "evaluation_only": True,
        "case_id": request["case_id"],
        "source_image": str(image_path),
        "request": request["source_phrase"],
        "grounding_intent": intent_bundle["grounding_intent"],
        "intent_hash": intent_bundle["intent_hash"],
        "intent_parser": intent_bundle["parser"],
        "intent_shadow": intent_bundle["shadow_comparison"],
        "primary_sam_phrase": intent_bundle["primary_sam_phrase"],
        "prompt_family": intent_bundle["prompt_family"],
        "num_kept": len(kept),
        "scores": [float(score) for _mask, score in kept],
        "presence_gate": presence,
        "candidate_generation": generation,
        "verification": verification,
        "selection": selection,
        "frame": str(frame_path),
        "full_frame": str(full_frame_path),
        "zed_frame": frame_info,
        "final_masks": final_masks,
        "selected_mask": final_masks[0] if len(final_masks) == 1 else None,
        "overlay": overlay,
        "sam_json": str(sam_json_path),
        "elapsed_s": round(time.monotonic() - started, 3),
        "output_dir": str(req_dir),
        "result_json": str(result_path),
        "limitations": [
            "frozen RGB evaluation has no ZED depth or XYZ",
            "nearest/farthest selectors require a live depth evaluation",
            "no robot target is created",
        ],
    }
    result_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    return response


V2_INTERPRET_REQUEST_KEYS = {"raw_command"}


def validate_v2_interpret_request(payload: Any) -> str:
    if not isinstance(payload, dict) or set(payload) != V2_INTERPRET_REQUEST_KEYS:
        actual = set(payload) if isinstance(payload, dict) else set()
        raise grounding_v2.GroundingV2Error(
            "v2 interpret request must contain exactly raw_command",
            details={
                "missing": sorted(V2_INTERPRET_REQUEST_KEYS - actual),
                "extra": sorted(actual - V2_INTERPRET_REQUEST_KEYS),
            },
        )
    raw_command = payload["raw_command"]
    if not isinstance(raw_command, str) or not raw_command.strip():
        raise grounding_v2.GroundingV2Error("raw_command must be non-empty")
    if raw_command != raw_command.strip():
        raise grounding_v2.GroundingV2Error(
            "raw_command must not contain leading or trailing whitespace"
        )
    return raw_command


def validate_v2_segment_request(payload: Any) -> dict[str, Any]:
    """Authenticate the exact sealed envelope without reading its sentence."""

    return grounding_v2.validate_command_envelope(payload, require_hashes=True)


def cache_v2_interpretation(
    envelope: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Retain a bounded Qwen attestation for the subsequent segment call."""

    canonical = grounding_v2.validate_command_envelope(envelope, require_hashes=True)
    cache = STATE.setdefault("v2_interpretations", {})
    cache[canonical["envelope_hash"]] = {
        "command_envelope": canonical,
        "qwen_interpretation": record,
    }
    while len(cache) > 128:
        cache.pop(next(iter(cache)))


def require_v2_interpretation_attestation(
    envelope: dict[str, Any],
) -> dict[str, Any]:
    """Prove that this process obtained the exact envelope from Qwen."""

    canonical = grounding_v2.validate_command_envelope(envelope, require_hashes=True)
    attestation = STATE.get("v2_interpretations", {}).get(
        canonical["envelope_hash"]
    )
    if (
        not isinstance(attestation, dict)
        or attestation.get("command_envelope") != canonical
        or not isinstance(attestation.get("qwen_interpretation"), dict)
    ):
        raise grounding_v2.GroundingV2Error(
            "segment envelope has no matching Qwen interpretation attestation",
            code="interpretation_attestation_missing",
        )
    return attestation


def _write_v2_anchor_artifacts(
    *,
    anchor_masks: dict[str, np.ndarray],
    anchor_scores: dict[str, float],
    crop_info: dict[str, Any],
    req_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir = req_dir / "final_anchor_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for entity_id, raw_mask in anchor_masks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        full_mask = candidate_generation.expand_crop_mask_to_full(mask, crop_info)
        crop_path = output_dir / f"{entity_id}_crop.png"
        full_path = output_dir / f"{entity_id}_full.png"
        if not cv2.imwrite(str(crop_path), mask.astype(np.uint8) * 255):
            raise OSError(f"cannot write v2 anchor mask {crop_path}")
        if not cv2.imwrite(str(full_path), full_mask.astype(np.uint8) * 255):
            raise OSError(f"cannot write v2 anchor mask {full_path}")
        records[entity_id] = {
            "score": float(anchor_scores[entity_id]),
            "mask_crop": str(crop_path),
            "mask_full": str(full_path),
            "area_pixels": int(np.count_nonzero(mask)),
            "bbox_xywh_crop_pixels": candidate_generation.mask_bbox_xywh(mask),
            "bbox_xywh_full_pixels": candidate_generation.crop_bbox_to_full(
                candidate_generation.mask_bbox_xywh(mask), crop_info
            ),
        }
    return records


def segment_v2_once(envelope: dict[str, Any]) -> dict[str, Any]:
    """Capture and execute v2 without reinterpreting the sealed command."""

    request_started = time.monotonic()
    args = STATE["args"]
    if not v2_segmentation_available(args):
        raise grounding_v2.GroundingV2Error(
            "v2 segmentation is disabled during staged rollout",
            code="v2_disabled",
        )
    if args.sam_backend != "image" or not hasattr(STATE.get("sam"), "predict_boxes"):
        raise grounding_v2.GroundingV2Error(
            "v2 segmentation requires the parity-approved SAM image backend",
            code="v2_image_backend_required",
        )
    canonical = grounding_v2.validate_command_envelope(envelope, require_hashes=True)
    attestation = require_v2_interpretation_attestation(canonical)
    STATE["sam"].clear_embedding_cache()
    req_dir = STATE["run_dir"] / "v2" / timestamp_slug()
    req_dir.mkdir(parents=True, exist_ok=True)
    interpretation_record = attestation["qwen_interpretation"]
    interpretation_record_path = None
    if interpretation_record is not None:
        interpretation_record_path = req_dir / "qwen_interpretation.json"
        interpretation_record_path.write_text(
            json.dumps(interpretation_record, indent=2) + "\n",
            encoding="utf-8",
        )
    rgb_np, full_rgb_np, depth_np, xyz_np, frame_info = capture_request_frame(req_dir)
    frame_path = Path(frame_info["saved_frame"])
    sam_image_cache = None
    pipeline_error: grounding_v2.GroundingV2Error | None = None
    result: dict[str, Any] | None = None
    try:
        result = v2_pipeline.segment_frame(
            canonical,
            rgb=rgb_np,
            full_rgb=full_rgb_np,
            depth=depth_np,
            xyz=xyz_np,
            frame_info=frame_info,
            frame_path=frame_path,
            req_dir=req_dir,
            sam_service=STATE["sam"],
            args=args,
        )
    except grounding_v2.GroundingV2Error as exc:
        pipeline_error = exc
    finally:
        sam_image_cache = STATE["sam"].cache_stats()
        STATE["sam"].clear_embedding_cache()
    interpretation_elapsed_s = interpretation_record.get("elapsed_s", 0.0)
    if not isinstance(interpretation_elapsed_s, (int, float)):
        interpretation_elapsed_s = 0.0
    if pipeline_error is not None:
        relationship_path = req_dir / "relationship_measurements.json"
        raw_pools_path = req_dir / "raw_candidate_pools.json"
        failure = {
            "schema_version": 2,
            "status": "error",
            "accepted": False,
            "reason": pipeline_error.code,
            "error": {
                "code": pipeline_error.code,
                "message": str(pipeline_error),
                "details": pipeline_error.details,
            },
            "command_envelope": canonical,
            "frame": frame_info["saved_frame"],
            "full_frame": frame_info["saved_full_frame"],
            "zed_frame": frame_info,
            "coordinate_spaces": {
                "segmentation": "workspace_crop",
                "depth_xyz": "workspace_crop_aligned",
                "saved_full_masks": "full_camera_frame",
            },
            "raw_candidate_pools": (
                read_json(raw_pools_path) if raw_pools_path.is_file() else {}
            ),
            "relationship_measurements": (
                read_json(relationship_path) if relationship_path.is_file() else []
            ),
            "num_kept": 0,
            "scores": [],
            "target_mask": None,
            "target_depth_xyz": None,
            "anchor_masks": {},
            "anchor_depth_xyz": {},
            "output_dir": str(req_dir),
            "sam_backend": args.sam_backend,
            "sam_image_parity": STATE.get("sam_image_parity"),
            "sam_image_cache": sam_image_cache,
            "qwen_interpretation_record": str(interpretation_record_path),
            "audit_artifacts": {
                "qwen_interpretation": str(interpretation_record_path),
                "command_envelope": str(req_dir / "command_envelope.json"),
                "qwen_visual_grounding": (
                    str(req_dir / "qwen_visual_grounding.json")
                    if (req_dir / "qwen_visual_grounding.json").is_file()
                    else None
                ),
                "qwen_verification_raw": (
                    str(req_dir / "qwen_verification_raw.txt")
                    if (req_dir / "qwen_verification_raw.txt").is_file()
                    else None
                ),
                "raw_candidate_pools": (
                    str(raw_pools_path) if raw_pools_path.is_file() else None
                ),
                "relationship_measurements": (
                    str(relationship_path) if relationship_path.is_file() else None
                ),
            },
            "robot_target": None,
            "motion_permitted": False,
        }
        segment_elapsed_s = time.monotonic() - request_started
        failure["latency_breakdown_s"] = {
            "interpretation": round(float(interpretation_elapsed_s), 3),
            "segmentation_request": round(segment_elapsed_s, 3),
        }
        failure["elapsed_s"] = round(
            float(interpretation_elapsed_s) + segment_elapsed_s,
            3,
        )
        failure_path = req_dir / "result.json"
        failure["result_json"] = str(failure_path)
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        pipeline_error.details = {
            **pipeline_error.details,
            "result_json": str(failure_path),
            "output_dir": str(req_dir),
        }
        raise pipeline_error
    if result is None:  # pragma: no cover - defensive invariant.
        raise RuntimeError("v2 pipeline returned no result")
    pipeline_core_elapsed_s = result.get("elapsed_s")
    target_mask = result.pop("_target_mask", None)
    target_score = result.pop("_target_score", None)
    anchor_masks = result.pop("_anchor_masks", {})
    anchor_scores = result.pop("_anchor_scores", {})
    result.update(
        {
            "frame": frame_info["saved_frame"],
            "full_frame": frame_info["saved_full_frame"],
            "zed_frame": frame_info,
            "coordinate_spaces": {
                "segmentation": "workspace_crop",
                "depth_xyz": "workspace_crop_aligned",
                "saved_full_masks": "full_camera_frame",
            },
            "output_dir": str(req_dir),
            "sam_backend": args.sam_backend,
            "sam_image_parity": STATE.get("sam_image_parity"),
            "sam_image_cache": sam_image_cache,
            "qwen_interpretation_record": None
            if interpretation_record_path is None
            else str(interpretation_record_path),
            # This path is contractually perception-only.
            "robot_target": None,
            "motion_permitted": False,
        }
    )
    result.setdefault("audit_artifacts", {})["qwen_interpretation"] = (
        None
        if interpretation_record_path is None
        else str(interpretation_record_path)
    )
    if result["accepted"]:
        if target_mask is None or target_score is None:
            raise RuntimeError("accepted v2 result did not retain its target mask")
        kept = [(np.asarray(target_mask, dtype=bool), float(target_score))]
        target_depth = task5.object_depth_report(
            kept,
            depth_np=depth_np,
            xyz_np=xyz_np,
            view_name=args.view,
            depth_measure=frame_info["depth_measure"],
            xyz_measure=frame_info["xyz_measure"],
        )
        target_artifacts = write_final_mask_artifacts(
            kept=kept,
            kept_indices=[0],
            crop_info=frame_info["crop"],
            req_dir=req_dir,
        )
        anchor_artifacts = _write_v2_anchor_artifacts(
            anchor_masks=anchor_masks,
            anchor_scores=anchor_scores,
            crop_info=frame_info["crop"],
            req_dir=req_dir,
        )
        anchor_depth: dict[str, Any] = {}
        for entity_id, mask in anchor_masks.items():
            report = task5.object_depth_report(
                [(np.asarray(mask, dtype=bool), float(anchor_scores[entity_id]))],
                depth_np=depth_np,
                xyz_np=xyz_np,
                view_name=args.view,
                depth_measure=frame_info["depth_measure"],
                xyz_measure=frame_info["xyz_measure"],
            )
            anchor_depth[entity_id] = report["objects"][0]
        result.update(
            {
                "num_kept": 1,
                "scores": [float(target_score)],
                "target_mask": target_artifacts[0],
                "target_depth_xyz": target_depth["objects"][0],
                "anchor_masks": anchor_artifacts,
                "anchor_depth_xyz": anchor_depth,
            }
        )
    else:
        result.update(
            {
                "num_kept": 0,
                "scores": [],
                "target_mask": None,
                "target_depth_xyz": None,
                "anchor_masks": {},
                "anchor_depth_xyz": {},
            }
        )
    result_path = req_dir / "result.json"
    segment_elapsed_s = time.monotonic() - request_started
    result["pipeline_core_elapsed_s"] = pipeline_core_elapsed_s
    result["latency_breakdown_s"] = {
        "interpretation": round(float(interpretation_elapsed_s), 3),
        "segmentation_request": round(segment_elapsed_s, 3),
    }
    result["elapsed_s"] = round(
        float(interpretation_elapsed_s) + segment_elapsed_s,
        3,
    )
    result["result_json"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if app is not None:

    @app.get("/health")
    def health() -> dict[str, Any]:
        args = STATE.get("args")
        return {
            "sam_loaded": "sam" in STATE,
            "camera_open": "zed" in STATE,
            "busy": LOCK.locked(),
            "uptime_s": round(time.monotonic() - STATE.get("t0", time.monotonic()), 1),
            "crop_xywh": None
            if args is None or args.crop is None
            else [int(value) for value in args.crop],
            "presence_conf_threshold": None
            if args is None
            else float(args.presence_conf_threshold),
            "qwen_model": None if args is None else args.qwen_model,
            "qwen_loaded": STATE.get("qwen_runtime") is not None,
            "qwen_runtime": STATE.get("qwen_runtime"),
            "v2": None
            if args is None
            else {
                "schema_version": grounding_v2.SCHEMA_VERSION,
                "interpret_available": True,
                "segment_enabled": v2_segmentation_available(args),
                "production_enabled": bool(getattr(args, "v2_enabled", False)),
                "evaluation_mode": bool(
                    getattr(args, "v2_evaluation_mode", False)
                ),
                "sam_backend": getattr(args, "sam_backend", "multiplex"),
                "sam_image_parity": STATE.get("sam_image_parity"),
                "relation_threshold_calibration": STATE.get(
                    "v2_relation_calibration"
                ),
                "release_approval": STATE.get("v2_release_approval"),
                "qwen_interprets_every_request": True,
                "semantic_fallback": False,
                "semantic_contract_version": (
                    grounding_v2.QWEN_SEMANTIC_CONTRACT_VERSION
                ),
                "schema_constrained_generation": True,
                "max_interpretation_attempts": 3,
                "max_anchors": grounding_v2.MAX_ANCHORS,
                "max_relationships": grounding_v2.MAX_RELATIONSHIPS,
                "max_prompts_per_entity": grounding_v2.MAX_PROMPTS_PER_ENTITY,
                "full_frame_context": bool(
                    getattr(args, "v2_full_context", True)
                ),
                "perception_only": True,
            },
            "structured_intent": None
            if args is None
            else {
                "schema_version": grounding_intent.SCHEMA_VERSION,
                "parser_mode": args.intent_parser_mode,
                "max_new_tokens": int(args.intent_max_new_tokens),
                "fail_closed": True,
            },
            "candidate_generation": None
            if args is None
            else {
                "max_prompts": int(args.candidate_max_prompts),
                "multiscale": bool(args.candidate_multiscale),
                "tile_scale": float(args.candidate_tile_scale),
                "localized_region": True,
                "region_padding_fraction": float(
                    args.candidate_region_padding_fraction
                ),
                "full_frame_context": bool(args.candidate_full_frame),
                "dedup_iou": float(args.candidate_dedup_iou),
                "max_candidates": int(args.candidate_max_count),
                "min_workspace_retained_fraction": float(
                    args.candidate_min_workspace_retained_fraction
                ),
            },
            "geometry_gate": None
            if args is None
            else {
                "min_valid_depth_fraction": float(
                    args.selection_min_valid_depth_fraction
                ),
                "min_valid_depth_pixels": int(args.candidate_min_valid_depth_pixels),
                "max_depth_spread_mm": float(args.candidate_max_depth_spread_mm),
                "workspace_roi_xywh": None
                if args.selection_roi is None
                else [int(value) for value in args.selection_roi],
            },
            "qwen_verifier": None
            if args is None
            else {
                "min_select_confidence": float(args.verifier_min_confidence),
                "max_area_fraction": float(args.verifier_max_area_fraction),
                "max_new_tokens": int(args.verifier_max_new_tokens),
                "fail_closed": True,
            },
        }

    @app.post("/v2/interpret")
    def interpret_v2(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_command = validate_v2_interpret_request(payload)
        except grounding_v2.GroundingV2Error as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                    "robot_target": None,
                    "motion_permitted": False,
                },
            ) from exc
        with LOCK:
            try:
                envelope, record = qwen_command_envelope(raw_command, STATE["args"])
                cache_v2_interpretation(envelope, record)
                return envelope
            except grounding_v2.GroundingV2Error as exc:
                status = 503 if exc.code == "grounding_parser_unavailable" else 422
                raise HTTPException(
                    status_code=status,
                    detail={
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                        "robot_target": None,
                        "motion_permitted": False,
                    },
                ) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.post("/v2/segment")
    def segment_v2(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            envelope = validate_v2_segment_request(payload)
        except grounding_v2.GroundingV2Error as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                    "robot_target": None,
                    "motion_permitted": False,
                },
            ) from exc
        with LOCK:
            try:
                return segment_v2_once(envelope)
            except grounding_v2.GroundingV2Error as exc:
                unavailable = {
                    "v2_disabled",
                    "v2_image_backend_required",
                    "visual_grounding_unavailable",
                    "candidate_verification_unavailable",
                }
                raise HTTPException(
                    status_code=503 if exc.code in unavailable else 422,
                    detail={
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                        "robot_target": None,
                        "motion_permitted": False,
                    },
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "internal_v2_error",
                        "message": repr(exc),
                        "robot_target": None,
                        "motion_permitted": False,
                    },
                ) from exc

    @app.post("/segment")
    def segment(request: str, use_agent_fallback: bool = True) -> dict[str, Any]:
        if not request.strip():
            raise HTTPException(status_code=400, detail="request must not be empty")
        with LOCK:
            try:
                return segment_once(
                    request.strip(),
                    use_agent_fallback=use_agent_fallback,
                )
            except grounding_intent.GroundingIntentError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": exc.code, "message": str(exc), "details": exc.details},
                ) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.post("/v1/segment")
    def segment_v1(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = validate_v1_segment_request(payload)
        except grounding_intent.GroundingIntentError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code, "message": str(exc), "details": exc.details},
            ) from exc
        with LOCK:
            try:
                return segment_once(
                    request["source_phrase"],
                    use_agent_fallback=request["use_agent_fallback"],
                    supplied_intent=request["grounding_intent"],
                    supplied_intent_hash=request["intent_hash"],
                )
            except grounding_intent.GroundingIntentError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": exc.code, "message": str(exc), "details": exc.details},
                ) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.post("/v1/evaluate")
    def evaluate_v1(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = validate_v1_evaluate_request(payload)
        except grounding_intent.GroundingIntentError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code, "message": str(exc), "details": exc.details},
            ) from exc
        with LOCK:
            try:
                return evaluate_saved_frame(request)
            except grounding_intent.GroundingIntentError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": exc.code, "message": str(exc), "details": exc.details},
                ) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        shutdown()


def main() -> None:
    require_server_deps()
    args = parse_args()
    startup(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
