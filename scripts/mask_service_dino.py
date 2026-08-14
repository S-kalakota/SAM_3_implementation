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
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError:  # pragma: no cover - exercised only before deps are installed.
    FastAPI = None
    HTTPException = None
    uvicorn = None

import local_qwen
import grounding_dino
import grounding_intent
import mask_depth
import qwen_candidate_verifier as candidate_verifier
import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "service"
DEFAULT_SELECTION_ROI = os.environ.get("MASK_SERVICE_SELECTION_ROI")
DEFAULT_VERIFIER_MAX_NEW_TOKENS = int(
    os.environ.get("SAM3_QWEN_VERIFIER_MAX_NEW_TOKENS", "384")
)
DEFAULT_VERIFIER_MIN_CONFIDENCE = float(
    os.environ.get("SAM3_QWEN_VERIFIER_MIN_CONFIDENCE", "0.70")
)
DEFAULT_VERIFIER_MAX_AREA_FRACTION = float(
    os.environ.get("SAM3_QWEN_VERIFIER_MAX_AREA_FRACTION", "0.25")
)

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
    parser.add_argument(
        "--pipeline-mode",
        choices=("dino", "legacy"),
        default="dino",
        help="Use bounded Grounding DINO box proposals or the legacy SAM text path.",
    )
    parser.add_argument("--dino-model", default=grounding_dino.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--dino-box-threshold",
        default=grounding_dino.DEFAULT_BOX_THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--dino-text-threshold",
        default=grounding_dino.DEFAULT_TEXT_THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--dino-nms-iou",
        default=grounding_dino.DEFAULT_NMS_IOU,
        type=float,
    )
    parser.add_argument(
        "--dino-max-proposals",
        default=grounding_dino.DEFAULT_MAX_PROPOSALS,
        type=int,
    )
    parser.add_argument(
        "--dino-box-padding",
        default=grounding_dino.DEFAULT_BOX_PADDING,
        type=float,
    )
    parser.add_argument(
        "--warm-dino",
        action="store_true",
        help=(
            "Warm cached DINO weights at startup. DINO mode always warms; this "
            "flag can also preflight the model while running legacy mode."
        ),
    )
    parser.add_argument("--checkpoint", default=task6.DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument(
        "--presence-conf-threshold",
        default=0.25,
        type=float,
        help="DINO/SAM presence threshold retained from the validated DINO branch.",
    )
    parser.add_argument("--min-area", default=task2.DEFAULT_MIN_AREA, type=int)
    parser.add_argument("--det-threshold", default=None, type=float)
    parser.add_argument("--max-generations", default=8, type=int)
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
    parser.add_argument(
        "--depth-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim verified DINO/SAM masks at ZED depth discontinuities. "
            "Use --no-depth-refinement for an A/B comparison."
        ),
    )
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
    if not 0.0 <= args.verifier_min_confidence <= 1.0:
        parser.error("--verifier-min-confidence must be in [0, 1]")
    if not 0.0 < args.verifier_max_area_fraction <= 1.0:
        parser.error("--verifier-max-area-fraction must be in (0, 1]")
    if not 0.0 <= args.dino_box_threshold <= 1.0:
        parser.error("--dino-box-threshold must be in [0, 1]")
    if not 0.0 <= args.dino_text_threshold <= 1.0:
        parser.error("--dino-text-threshold must be in [0, 1]")
    if not 0.0 <= args.dino_nms_iou <= 1.0:
        parser.error("--dino-nms-iou must be in [0, 1]")
    if not 1 <= args.dino_max_proposals <= grounding_dino.DEFAULT_MAX_PROPOSALS:
        parser.error("--dino-max-proposals must be in [1, 3]")
    if not 0.0 <= args.dino_box_padding <= 1.0:
        parser.error("--dino-box-padding must be in [0, 1]")
    return args


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


VALID_SELECTORS = {
    "rightmost",
    "leftmost",
    "topmost",
    "bottommost",
    "nearest",
    "farthest",
    "largest",
    "smallest",
}

V1_SEGMENT_REQUEST_KEYS = {
    "schema_version",
    "source_phrase",
    "grounding_intent",
    "intent_hash",
    "use_agent_fallback",
}

SELECTOR_PATTERNS = (
    ("rightmost", r"\bright\s*most\b|\brightmost\b|\bon the right\b|\bright side\b"),
    ("leftmost", r"\bleft\s*most\b|\bleftmost\b|\bon the left\b|\bleft side\b"),
    ("topmost", r"\btop\s*most\b|\btopmost\b|\bat the top\b"),
    ("bottommost", r"\bbottom\s*most\b|\bbottommost\b|\bat the bottom\b"),
    ("nearest", r"\bnearest\b|\bclosest\b"),
    ("farthest", r"\bfarthest\b|\bfurthest\b"),
    ("largest", r"\blargest\b|\bbiggest\b"),
    ("smallest", r"\bsmallest\b"),
)


def first_json_object(text: str) -> dict[str, Any] | None:
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_v1_segment_request(payload: Any) -> dict[str, Any]:
    """Validate the robot bridge's exact request before camera capture."""

    if not isinstance(payload, dict):
        raise grounding_intent.GroundingIntentError(
            "v1 request must be a JSON object"
        )
    keys = set(payload)
    if keys != V1_SEGMENT_REQUEST_KEYS:
        raise grounding_intent.GroundingIntentError(
            "v1 request keys are invalid",
            details={
                "missing": sorted(V1_SEGMENT_REQUEST_KEYS - keys),
                "extra": sorted(keys - V1_SEGMENT_REQUEST_KEYS),
            },
        )
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != grounding_intent.SCHEMA_VERSION
    ):
        raise grounding_intent.GroundingIntentError(
            "v1 request schema_version must be 1"
        )
    source_phrase = payload["source_phrase"]
    if not isinstance(source_phrase, str) or not source_phrase.strip():
        raise grounding_intent.GroundingIntentError(
            "source_phrase must be non-empty"
        )
    if not isinstance(payload["use_agent_fallback"], bool):
        raise grounding_intent.GroundingIntentError(
            "use_agent_fallback must be boolean"
        )
    canonical = grounding_intent.validate_grounding_intent(
        payload["grounding_intent"],
        expected_source_phrase=source_phrase,
    )
    supplied_hash = payload["intent_hash"]
    canonical_hash = grounding_intent.intent_hash(canonical)
    if not isinstance(supplied_hash, str) or supplied_hash != canonical_hash:
        raise grounding_intent.GroundingIntentError(
            "intent_hash does not match grounding_intent",
            code="grounding_identity_mismatch",
            details={"supplied": supplied_hash, "computed": canonical_hash},
        )
    return {
        "source_phrase": grounding_intent.collapse_space(source_phrase),
        "grounding_intent": canonical,
        "intent_hash": supplied_hash,
        "use_agent_fallback": payload["use_agent_fallback"],
    }


def build_v1_pipeline_intent(canonical_intent: dict[str, Any]) -> dict[str, Any]:
    """Adapt a sealed v1 intent to the bounded DINO/SAM/Qwen pipeline."""

    canonical = grounding_intent.validate_grounding_intent(canonical_intent)
    unsupported: dict[str, Any] = {}
    if canonical["source_region"] is not None:
        unsupported["source_region"] = canonical["source_region"]
    if canonical["relations"]:
        unsupported["relations"] = canonical["relations"]
    if unsupported:
        raise grounding_intent.GroundingIntentError(
            "the Grounding DINO v1 path does not yet implement source-region "
            "or relational selection",
            code="unsupported_grounding_intent",
            details=unsupported,
        )
    return {
        "target_phrase": grounding_intent.construct_primary_prompt(canonical),
        "selector": canonical["selector"],
        "parser": "validated_grounding_intent_v1",
        "ambiguity_reasons": [],
        "relation_context": None,
        "grounding_intent": canonical,
        "intent_hash": grounding_intent.intent_hash(canonical),
    }


def should_parse_natural_request(request: str) -> bool:
    lowered = request.lower()
    return bool(
        re.search(
            r"\b(get|grab|pick|select|find|choose|show|right|left|top|bottom|"
            r"rightmost|leftmost|nearest|closest|farthest|largest|biggest|"
            r"smallest)\b",
            lowered,
        )
    )


def deterministic_intent(request: str) -> dict[str, Any]:
    """Parse ordinary pick commands without invoking a language model."""

    lowered = request.lower()
    matched_selectors: list[str] = []
    cleaned = lowered
    for name, pattern in SELECTOR_PATTERNS:
        if re.search(pattern, lowered):
            matched_selectors.append(name)
        cleaned = re.sub(pattern, " ", cleaned)

    cleaned = re.sub(r"[^a-z0-9_\- ]+", " ", cleaned)
    relation_context = None
    relation_match = re.search(
        r"\b(inside|within|in\s+front\s+of|in|on|under|below|above|beside|"
        r"next\s+to|near)\b",
        cleaned,
    )
    if relation_match is not None:
        relation_context = " ".join(cleaned[relation_match.start() :].split())
        cleaned = cleaned[: relation_match.start()]
    cleaned = re.sub(
        r"\b(can|could|would|you|get|grab|pick|select|find|choose|show|"
        r"give|bring|take|fetch|retrieve|locate|identify|detect|segment|"
        r"grasp|lift|point|move|up|the|a|an|please|for|me|most)\b",
        " ",
        cleaned,
    )
    target_phrase = " ".join(cleaned.split())
    unique_selectors = list(dict.fromkeys(matched_selectors))
    ambiguity_reasons = []
    if len(unique_selectors) > 1:
        ambiguity_reasons.append("multiple_spatial_selectors")
    if re.search(r"\bor\b", lowered):
        ambiguity_reasons.append("disjunctive_target")
    if not target_phrase:
        ambiguity_reasons.append("missing_target_phrase")
    if target_phrase in {"it", "this", "that", "this one", "that one", "one"}:
        ambiguity_reasons.append("unresolved_reference")
    if target_phrase in {"object", "item", "thing"}:
        ambiguity_reasons.append("generic_target")
    category_mentions = set(
        re.findall(
            r"\b(box|package|carton|bin|cup|mug|bottle|can|filter|tool)\b",
            target_phrase,
        )
    )
    if " and " in f" {target_phrase} " and len(category_mentions) > 1:
        ambiguity_reasons.append("multiple_target_categories")
    return {
        "target_phrase": target_phrase or request.strip(),
        "selector": unique_selectors[0] if len(unique_selectors) == 1 else None,
        "parser": (
            "deterministic_command"
            if should_parse_natural_request(request)
            else "deterministic_direct"
        ),
        "ambiguity_reasons": ambiguity_reasons,
        "relation_context": relation_context,
    }


def fallback_intent(request: str) -> dict[str, Any]:
    """Backward-compatible name for the deterministic parser."""

    return deterministic_intent(request)


def parse_request_intent(request: str, args: argparse.Namespace) -> dict[str, Any]:
    deterministic = deterministic_intent(request)
    if not deterministic["ambiguity_reasons"]:
        return deterministic

    messages = [
        {
            "role": "system",
            "content": (
                "You convert natural-language visual grounding requests into "
                "one compact JSON object. Return JSON only, no markdown. "
                "Schema: {\"target_phrase\": string, \"selector\": null or one "
                "of rightmost,leftmost,topmost,bottommost,nearest,farthest,"
                "largest,smallest}. target_phrase must be a short noun phrase "
                "for a segmentation model and must not include command verbs "
                "or spatial selector words."
            ),
        },
        {
            "role": "user",
            "content": (
                "Examples:\n"
                "Request: get the rightmost yellow box\n"
                "{\"target_phrase\":\"yellow box\",\"selector\":\"rightmost\"}\n"
                "Request: pick the closest red cup\n"
                "{\"target_phrase\":\"red cup\",\"selector\":\"nearest\"}\n"
                "Request: show the blue tape roll\n"
                "{\"target_phrase\":\"blue tape roll\",\"selector\":null}\n"
                f"Request: {request}\n"
            ),
        },
    ]
    try:
        parsed_text = local_qwen.qwen_generate(
            messages,
            model_id=args.qwen_model,
            max_new_tokens=128,
            local_files_only=not args.allow_qwen_downloads,
            device_map=args.qwen_device_map,
            do_sample=False,
        )
        parsed = first_json_object(parsed_text)
        if parsed is None:
            raise ValueError(f"no JSON object in parser output: {parsed_text!r}")
        target_phrase = str(parsed.get("target_phrase", "")).strip()
        selector = parsed.get("selector")
        if selector is not None:
            selector = str(selector).strip().lower()
            if selector in {"closest"}:
                selector = "nearest"
            if selector in {"furthest"}:
                selector = "farthest"
            if selector not in VALID_SELECTORS:
                selector = None
        if not target_phrase:
            raise ValueError(f"missing target_phrase in parser output: {parsed!r}")
        if target_phrase.lower() in {"object", "item", "thing", "it", "this", "that"}:
            raise ValueError(f"ambiguous target_phrase in parser output: {parsed!r}")
        return {
            "target_phrase": target_phrase,
            "selector": selector,
            "parser": "qwen_json_ambiguity_fallback",
            "deterministic_ambiguity_reasons": deterministic[
                "ambiguity_reasons"
            ],
            "raw_parser_output": parsed_text,
        }
    except Exception as exc:
        raise ValueError(
            "Deterministic parsing was ambiguous and the cached Qwen text parser "
            f"failed closed: {exc!r}"
        ) from exc


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


def startup(args: argparse.Namespace) -> None:
    STATE["t0"] = time.monotonic()
    STATE["args"] = args
    STATE["run_dir"] = args.run_dir.expanduser().resolve()
    STATE["run_dir"].mkdir(parents=True, exist_ok=True)

    if args.pipeline_mode == "dino":
        if args.resolution != "HD720" or args.crop != task5.DEFAULT_CROP:
            raise ValueError(
                "DINO mode requires the calibrated HD720 crop "
                f"{task5.DEFAULT_CROP}; use --pipeline-mode legacy for other views"
            )
    STATE["dino"] = None
    STATE["dino_runtime"] = {
        "loaded": False,
        "model_id": args.dino_model,
        "dtype": "torch.bfloat16",
        "devices": [grounding_dino.DEFAULT_DEVICE],
        "configured_device": grounding_dino.DEFAULT_DEVICE,
        "local_files_only": True,
        "evaluation_mode": True,
    }
    if args.pipeline_mode == "dino" or args.warm_dino:
        print(f"loading cached Grounding DINO once: {args.dino_model}...")
        STATE["dino"] = grounding_dino.GroundingDinoAdapter(args.dino_model)
        STATE["dino_runtime"] = STATE["dino"].runtime_metadata()
        print(f"Grounding DINO ready: {STATE['dino_runtime']}")

    print("loading SAM 3.1 once...")
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
    dino = STATE.pop("dino", None)
    if dino is not None:
        dino.close()
    sam = STATE.pop("sam", None)
    if sam is not None:
        sam.close()
    zed = STATE.pop("zed", None)
    if zed is not None:
        zed.close()


atexit.register(shutdown)


def capture_request_frame(req_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    args = STATE["args"]
    sl = STATE["sl"]
    zed = STATE["zed"]

    rgb_np, frame_info = task5.capture_processed_frame(
        sl,
        zed,
        args,
        warmup_frames=args.warmup_frames,
    )
    depth_np, xyz_np, depth_info = task5.retrieve_zed_depth_and_xyz(sl, zed, args.view)
    depth_np, _ = task5.apply_crop(depth_np, args.crop)
    xyz_np, _ = task5.apply_crop(xyz_np, args.crop)

    frame_path = req_dir / "frame.png"
    task5.write_rgb_image(frame_path, rgb_np)
    return rgb_np, depth_np, xyz_np, {
        **frame_info,
        **depth_info,
        "saved_frame": str(frame_path),
    }


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


def build_verifier_candidate_records(
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    depth_np: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Describe candidates with measured pixels/depth for Qwen ranking."""

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
        record = {
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
            "crop_size_wh_pixels": [int(width), int(height)],
        }
        if depth_np is not None:
            stats = mask_depth.mask_depth_stats(mask, depth_np)
            record.update(
                {
                    "median_depth_m": stats["median"],
                    "valid_depth_fraction": stats["valid_fraction"],
                }
            )
        provenance = candidate.get("proposal_provenance")
        if isinstance(provenance, dict):
            record.update(
                {
                    "dino_phrase": provenance.get("dino_phrase"),
                    "dino_score": provenance.get("dino_score"),
                    "proposal_box_xyxy_crop_pixels": provenance.get(
                        "sam_prompt_box_xyxy_crop_pixels"
                    ),
                    "dino_box_xyxy_crop_pixels": provenance.get(
                        "dino_original_box_xyxy_crop_pixels"
                    ),
                    "proposal_id": provenance.get("proposal_id"),
                }
            )
        records.append(record)
    return records


def verify_candidates_with_qwen(
    *,
    request: str,
    target_phrase: str,
    selector: str | None,
    rgb_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    artifact_stem: str,
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    depth_np: np.ndarray | None = None,
    grounding_intent_value: dict[str, Any] | None = None,
) -> tuple[
    list[tuple[np.ndarray, float]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Ask Qwen for exactly one full-intent, mask-aware candidate."""

    verification_started = time.monotonic()
    qwen_call_timings: list[float] = []
    verification_path = req_dir / f"{artifact_stem}_qwen_verification.json"
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
        "selector_deferred": selector,
        "min_select_confidence": float(args.verifier_min_confidence),
        "max_area_fraction": float(args.verifier_max_area_fraction),
        "candidate_count": len(kept),
        "candidate_records": [],
        "candidate_overlay": None,
        "candidate_zoom": None,
        "candidate_clean_crops": None,
        "candidate_manifest": None,
        "verifier_input_mode": None,
        "json_schema_constrained": False,
        "attempts": [],
    }
    if not kept:
        verification["inference_s"] = 0.0
        verification["elapsed_s"] = float(time.monotonic() - verification_started)
        verification["artifact"] = str(verification_path)
        verification_path.write_text(
            json.dumps(verification, indent=2) + "\n",
            encoding="utf-8",
        )
        return kept, candidates, verification

    try:
        candidate_records = build_verifier_candidate_records(
            kept,
            candidates,
            depth_np=depth_np,
        )
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
        has_dino_boxes = all(
            isinstance(record.get("dino_box_xyxy_crop_pixels"), (list, tuple))
            and len(record["dino_box_xyxy_crop_pixels"]) == 4
            for record in candidate_records
        )
        candidate_clean_crops = None
        if has_dino_boxes:
            candidate_clean_crop_path = (
                req_dir / f"{artifact_stem}_qwen_clean_dino_crops.png"
            )
            candidate_clean_crops = (
                candidate_verifier.render_clean_dino_candidate_crops(
                    rgb_np,
                    candidate_records,
                    candidate_clean_crop_path,
                )
            )
        manifest_path = req_dir / f"{artifact_stem}_qwen_candidates.json"
        manifest_path.write_text(
            json.dumps(candidate_records, indent=2) + "\n",
            encoding="utf-8",
        )

        ranked_schema = candidate_verifier.ranked_mask_verifier_json_schema(
            len(candidate_records)
        )

        def send_generate_request(messages: list[dict[str, Any]]) -> str:
            qwen_started = time.monotonic()
            try:
                return local_qwen.qwen_generate(
                    messages,
                    model_id=args.qwen_model,
                    max_new_tokens=args.verifier_max_new_tokens,
                    local_files_only=not args.allow_qwen_downloads,
                    device_map=args.qwen_device_map,
                    do_sample=False,
                    response_prefix=None,
                    json_schema=ranked_schema,
                )
            finally:
                qwen_call_timings.append(time.monotonic() - qwen_started)

        verification = candidate_verifier.run_ranked_mask_verifier(
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
            candidate_crop_path=None
            if candidate_clean_crops is None
            else Path(candidate_clean_crops["output"]),
        )
        verifier_input_mode = "numbered_masks_ranked_single_choice"
        verification = candidate_verifier.apply_max_area_fraction_policy(
            verification,
            candidate_records,
            args.verifier_max_area_fraction,
        )
        verification.update(
            {
                "model": args.qwen_model,
                "target_phrase": target_phrase,
                "selector_deferred": None,
                "selector_requested": selector,
                "min_select_confidence": float(args.verifier_min_confidence),
                "max_area_fraction": float(args.verifier_max_area_fraction),
                "candidate_count": len(candidate_records),
                "candidate_records": candidate_records,
                "candidate_overlay": candidate_overlay["output"],
                "candidate_zoom": candidate_zoom["output"],
                "candidate_clean_crops": None
                if candidate_clean_crops is None
                else candidate_clean_crops["output"],
                "candidate_manifest": str(manifest_path),
                "verifier_input_mode": verifier_input_mode,
                "json_schema_constrained": True,
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
    verification["inference_s"] = float(sum(qwen_call_timings))
    verification["call_timings_s"] = [float(value) for value in qwen_call_timings]
    verification["elapsed_s"] = float(time.monotonic() - verification_started)
    verification["artifact"] = str(verification_path)
    verification_path.write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    return kept, candidates, verification


def decode_and_gate_sam_outputs(
    outputs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]]]:
    """Decode SAM JSON and preserve optional DINO provenance by array index."""

    masks = task6.decode_agent_masks(outputs)
    scores = np.asarray(outputs.get("pred_scores", []), dtype=np.float32)
    kept, candidates = task2.gate_masks(
        masks,
        scores,
        conf_thresh=args.presence_conf_threshold,
        min_area=args.min_area,
    )
    provenance = outputs.get("proposal_provenance", [])
    if provenance:
        if len(provenance) != len(candidates):
            raise ValueError(
                "SAM proposal provenance is not index-compatible with masks"
            )
        candidates = [
            {**candidate, "proposal_provenance": dict(provenance[index])}
            for index, candidate in enumerate(candidates)
        ]
    return kept, candidates


def cap_verifier_candidates(
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    maximum: int = grounding_dino.DEFAULT_MAX_PROPOSALS,
) -> tuple[list[tuple[np.ndarray, float]], list[dict[str, Any]]]:
    """Bound Qwen input while preserving SAM indices and existing order."""

    if len(kept) <= maximum:
        return kept, candidates
    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    retained_indices = {
        int(candidate["index"]) for candidate in kept_candidates[:maximum]
    }
    updated = []
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        if candidate["kept"] and int(candidate["index"]) not in retained_indices:
            candidate["kept"] = False
            candidate["reject_reasons"] = list(candidate["reject_reasons"]) + [
                "verifier_candidate_limit"
            ]
        updated.append(candidate)
    return kept[:maximum], updated


def _render_depth_refinement(
    rgb_np: np.ndarray,
    original_mask: np.ndarray,
    refined_mask: np.ndarray,
    output_path: Path,
) -> str:
    """Save a diagnostic: retained pixels green and removed pixels red."""

    if rgb_np.shape[:2] != original_mask.shape:
        raise ValueError("RGB and mask shapes do not match")
    rendered = np.asarray(rgb_np, dtype=np.float32).copy()
    retained = np.asarray(refined_mask, dtype=bool)
    removed = np.asarray(original_mask, dtype=bool) & ~retained
    rendered[retained] = 0.55 * rendered[retained] + 0.45 * np.asarray(
        [0.0, 255.0, 0.0]
    )
    rendered[removed] = 0.35 * rendered[removed] + 0.65 * np.asarray(
        [255.0, 0.0, 0.0]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8)).save(output_path)
    return str(output_path)


def refine_verified_dino_masks_with_depth(
    *,
    kept: list[tuple[np.ndarray, float]],
    candidates: list[dict[str, Any]],
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    req_dir: Path,
    args: argparse.Namespace,
    enabled: bool,
) -> tuple[
    list[tuple[np.ndarray, float]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Apply conservative ZED-depth cuts to Qwen-approved DINO masks."""

    manifest_path = req_dir / "dino_depth_refinement.json"
    manifest: dict[str, Any] = {
        "enabled": bool(enabled),
        "algorithm": "dino_center_depth_connectivity_v1",
        "status": "disabled" if not enabled else "not_run",
        "adds_pixels": False,
        "candidate_count": len(kept),
        "applied_count": 0,
        "candidates": [],
        "depth_units": "meters",
        "depth_artifact": None,
        "artifact": str(manifest_path),
    }
    if not enabled:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return kept, candidates, manifest

    depth_artifact = req_dir / "depth_crop_m.npy"
    np.save(
        depth_artifact,
        np.asarray(depth_np, dtype=np.float32),
        allow_pickle=False,
    )
    manifest["depth_artifact"] = str(depth_artifact)
    manifest["depth_shape_hw"] = [int(value) for value in depth_np.shape]

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    if len(kept_candidates) != len(kept):
        raise ValueError("kept masks and candidate metadata are inconsistent")
    if not kept:
        manifest["status"] = "no_verified_candidates"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return kept, candidates, manifest

    updated_kept: list[tuple[np.ndarray, float]] = []
    updated_candidates_by_index: dict[int, dict[str, Any]] = {}
    for (original_mask, score), candidate_value in zip(kept, kept_candidates):
        candidate = dict(candidate_value)
        sam_index = int(candidate["index"])
        original_mask = np.asarray(original_mask, dtype=bool)
        provenance_value = candidate.get("proposal_provenance")
        provenance = (
            dict(provenance_value)
            if isinstance(provenance_value, dict)
            else {}
        )
        provenance.setdefault("candidate_index", sam_index)
        original_box = provenance.get("dino_original_box_xyxy_crop_pixels")
        support_box = provenance.get("sam_prompt_box_xyxy_crop_pixels")
        before_stats = mask_depth.mask_depth_stats(original_mask, depth_np)
        final_mask = original_mask.copy()

        if not isinstance(original_box, (list, tuple)) or len(original_box) != 4:
            depth_report: dict[str, Any] = {
                "algorithm": "dino_center_depth_connectivity_v1",
                "status": "skipped_missing_dino_box",
                "reason": "candidate provenance has no original DINO box",
                "applied": False,
                "adds_pixels": False,
                "original_area_pixels": int(original_mask.sum()),
                "refined_area_pixels": int(original_mask.sum()),
                "removed_pixels": 0,
                "before_depth_stats_m": before_stats,
                "after_depth_stats_m": before_stats,
            }
        else:
            try:
                depth_mask, depth_report = (
                    mask_depth.refine_mask_at_depth_discontinuities(
                        original_mask,
                        depth_np,
                        anchor_box_xyxy=original_box,
                    )
                )
                if depth_report["applied"]:
                    geometry_box = (
                        support_box
                        if isinstance(support_box, (list, tuple))
                        and len(support_box) == 4
                        else original_box
                    )
                    geometry_mask, geometry_report = (
                        grounding_dino.refine_mask_with_dino_box(
                            depth_mask,
                            original_box_xyxy=original_box,
                            support_box_xyxy=geometry_box,
                        )
                    )
                    proposed_final_area = int(geometry_mask.sum())
                    retained_fraction = float(
                        proposed_final_area / max(int(original_mask.sum()), 1)
                    )
                    depth_report["post_depth_geometry_refinement"] = geometry_report
                    depth_report["proposed_final_area_pixels"] = proposed_final_area
                    depth_report["proposed_final_retained_fraction"] = retained_fraction
                    if (
                        proposed_final_area <= int(args.min_area)
                        or retained_fraction
                        < mask_depth.DEFAULT_REFINEMENT_MIN_RETAINED_FRACTION
                    ):
                        depth_report.update(
                            depth_cut_status="applied",
                            status="skipped_post_refinement_guardrail",
                            reason=(
                                "depth plus component cleanup would leave an "
                                "unsafe-small mask"
                            ),
                            applied=False,
                            refined_area_pixels=int(original_mask.sum()),
                            removed_pixels=0,
                            after_depth_stats_m=before_stats,
                        )
                    else:
                        final_mask = geometry_mask
                        depth_report.update(
                            refined_area_pixels=proposed_final_area,
                            removed_pixels=int(original_mask.sum())
                            - proposed_final_area,
                            after_depth_stats_m=mask_depth.mask_depth_stats(
                                final_mask,
                                depth_np,
                            ),
                        )
            except Exception as exc:
                depth_report = {
                    "algorithm": "dino_center_depth_connectivity_v1",
                    "status": "skipped_error",
                    "reason": f"depth refinement failed safely: {exc!r}",
                    "applied": False,
                    "adds_pixels": False,
                    "original_area_pixels": int(original_mask.sum()),
                    "refined_area_pixels": int(original_mask.sum()),
                    "removed_pixels": 0,
                    "before_depth_stats_m": before_stats,
                    "after_depth_stats_m": before_stats,
                }

        mask_artifact = req_dir / f"depth_refined_mask_{sam_index + 1:03d}.png"
        Image.fromarray(final_mask.astype(np.uint8) * 255).save(mask_artifact)
        diagnostic_artifact = req_dir / (
            f"depth_refinement_{sam_index + 1:03d}.png"
        )
        _render_depth_refinement(
            rgb_np,
            original_mask,
            final_mask,
            diagnostic_artifact,
        )

        final_area = int(final_mask.sum())
        geometry_mask_artifact = provenance.get("mask_artifact")
        provenance.update(
            {
                "pre_depth_mask_area_pixels": int(original_mask.sum()),
                "geometry_mask_artifact": geometry_mask_artifact,
                "mask_area_pixels": final_area,
                "mask_artifact": str(mask_artifact),
                "depth_refined_mask_artifact": str(mask_artifact),
                "depth_refinement_overlay_artifact": str(diagnostic_artifact),
                "depth_refinement": depth_report,
                "mask_center_xy_crop_pixels": [
                    float(value) for value in mask_center_xy(final_mask)
                ],
                "mask_box_xywh_normalized": (
                    grounding_dino.mask_bbox_xywh_normalized(final_mask)
                ),
            }
        )
        candidate.update(
            {
                "area_pixels": final_area,
                "proposal_provenance": provenance,
                "depth_refinement": depth_report,
            }
        )
        updated_candidates_by_index[sam_index] = candidate
        updated_kept.append((final_mask, float(score)))
        if depth_report.get("applied"):
            manifest["applied_count"] += 1
        manifest["candidates"].append(
            {
                "sam_index": sam_index,
                "score": float(score),
                "mask_artifact": str(mask_artifact),
                "diagnostic_artifact": str(diagnostic_artifact),
                "report": depth_report,
            }
        )

    updated_candidates = [
        updated_candidates_by_index.get(int(candidate["index"]), dict(candidate))
        for candidate in candidates
    ]
    manifest["status"] = "completed"
    manifest["skipped_count"] = len(kept) - int(manifest["applied_count"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated_kept, updated_candidates, manifest


def legacy_candidate_generation(
    *,
    sam_prompt: str,
    frame_path: Path,
    req_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    sam_json = STATE["sam"](
        image_path=str(frame_path),
        text_prompt=sam_prompt,
        output_folder_path=str(req_dir / "legacy_sam"),
    )
    elapsed_s = time.monotonic() - started
    outputs = read_json(sam_json)
    kept, candidates = decode_and_gate_sam_outputs(outputs, args)
    return {
        "kept": kept,
        "candidates": candidates,
        "sam_json": sam_json,
        "outputs": outputs,
        "elapsed_s": float(elapsed_s),
    }


def dino_candidate_generation(
    *,
    target_phrase: str,
    rgb_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run exactly one DINO pass and at most three SAM box prompts."""

    expected_height = int(task5.DEFAULT_CROP[3])
    expected_width = int(task5.DEFAULT_CROP[2])
    if rgb_np.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "DINO received an image outside the calibrated 384x360 crop: "
            f"{rgb_np.shape[:2]}"
        )
    detector = STATE.get("dino")
    if detector is None:
        raise RuntimeError("Grounding DINO is not loaded")

    phrases = grounding_dino.build_phrase_family(target_phrase)
    dino_started = time.monotonic()
    raw_result = detector.detect(
        rgb_np,
        phrases,
        box_threshold=args.dino_box_threshold,
        text_threshold=args.dino_text_threshold,
    )
    dino_elapsed_s = time.monotonic() - dino_started
    if not isinstance(raw_result, dict) or not isinstance(
        raw_result.get("detections"), list
    ):
        raise grounding_dino.GroundingDinoOutputError(
            "DINO adapter returned malformed detections"
        )
    proposals, rejected = grounding_dino.prepare_proposals(
        raw_result["detections"],
        image_width=expected_width,
        image_height=expected_height,
        box_threshold=args.dino_box_threshold,
        nms_iou=args.dino_nms_iou,
        max_proposals=args.dino_max_proposals,
        padding_fraction=args.dino_box_padding,
        dino_inference_s=raw_result.get("inference_s", dino_elapsed_s),
    )
    proposal_path = req_dir / "dino_proposals.json"
    proposal_manifest = {
        "model_id": args.dino_model,
        "local_files_only": True,
        "crop_xywh_full_pixels": [int(value) for value in task5.DEFAULT_CROP],
        "crop_size_wh_pixels": [expected_width, expected_height],
        "phrases": phrases,
        "thresholds": {
            "box": float(args.dino_box_threshold),
            "text": float(args.dino_text_threshold),
            "cross_phrase_nms_iou": float(args.dino_nms_iou),
            "box_padding_fraction": float(args.dino_box_padding),
            "max_proposals": int(args.dino_max_proposals),
            "mask_min_component_pixels": int(
                grounding_dino.DEFAULT_MASK_MIN_COMPONENT_PIXELS
            ),
            "mask_min_component_fraction": float(
                grounding_dino.DEFAULT_MASK_MIN_COMPONENT_FRACTION
            ),
            "mask_min_original_overlap": float(
                grounding_dino.DEFAULT_MASK_MIN_ORIGINAL_OVERLAP
            ),
            "mask_geometry_weight": float(
                grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT
            ),
            "mask_sam_score_weight": float(
                1.0 - grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT
            ),
            "depth_refinement_enabled": bool(
                getattr(args, "depth_refinement", True)
            ),
            "depth_min_valid_pixels": int(
                mask_depth.DEFAULT_REFINEMENT_MIN_VALID_PIXELS
            ),
            "depth_min_valid_fraction": float(
                mask_depth.DEFAULT_REFINEMENT_MIN_VALID_FRACTION
            ),
            "depth_min_jump_m": float(
                mask_depth.DEFAULT_REFINEMENT_MIN_JUMP_M
            ),
            "depth_relative_jump": float(
                mask_depth.DEFAULT_REFINEMENT_RELATIVE_JUMP
            ),
            "depth_max_local_jump_m": float(
                mask_depth.DEFAULT_REFINEMENT_MAX_LOCAL_JUMP_M
            ),
            "depth_max_global_drift_m": float(
                mask_depth.DEFAULT_REFINEMENT_MAX_GLOBAL_DRIFT_M
            ),
        },
        "timing_s": {
            "adapter_total": float(raw_result.get("total_s", dino_elapsed_s)),
            "model_inference": float(
                raw_result.get("inference_s", dino_elapsed_s)
            ),
            "service_stage": float(dino_elapsed_s),
        },
        "raw_detections": raw_result["detections"],
        "proposals": proposals,
        "rejected": rejected,
    }
    proposal_path.write_text(
        json.dumps(proposal_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if not proposals:
        return {
            "kept": [],
            "candidates": [],
            "sam_json": None,
            "combined_candidates": None,
            "outputs": None,
            "phrases": phrases,
            "proposals": proposals,
            "proposal_manifest": str(proposal_path),
            "dino_elapsed_s": float(dino_elapsed_s),
            "sam_box_prompt_timings": [],
            "sam_box_elapsed_s": 0.0,
        }

    sam_started = time.monotonic()
    sam_json = STATE["sam"].segment_boxes(
        image_path=str(frame_path),
        proposals=proposals,
        output_folder_path=str(req_dir / "dino_sam"),
    )
    sam_elapsed_s = time.monotonic() - sam_started
    outputs = read_json(sam_json)
    if (
        int(outputs.get("orig_img_w", -1)) != expected_width
        or int(outputs.get("orig_img_h", -1)) != expected_height
    ):
        raise ValueError("Combined SAM candidates are not crop-local 384x360 data")
    kept, candidates = decode_and_gate_sam_outputs(outputs, args)
    return {
        "kept": kept,
        "candidates": candidates,
        "sam_json": sam_json,
        "combined_candidates": sam_json,
        "outputs": outputs,
        "phrases": phrases,
        "proposals": proposals,
        "proposal_manifest": str(proposal_path),
        "dino_elapsed_s": float(dino_elapsed_s),
        "sam_box_prompt_timings": list(
            outputs.get("sam_box_prompt_timings", [])
        ),
        "sam_box_elapsed_s": float(sam_elapsed_s),
    }


def direct_segment(
    *,
    request: str,
    intent: dict[str, Any],
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sam_prompt = intent["target_phrase"]
    selector = intent.get("selector")
    stage_timings: dict[str, Any] = {
        "grounding_dino_s": 0.0,
        "sam_box_prompts_s": 0.0,
        "sam_box_prompt_timings": [],
        "legacy_sam_s": 0.0,
        "depth_refinement_s": 0.0,
    }
    dino_generation = None
    legacy_fallback_reason = None

    if args.pipeline_mode == "dino":
        dino_generation = dino_candidate_generation(
            target_phrase=sam_prompt,
            rgb_np=rgb_np,
            frame_path=frame_path,
            req_dir=req_dir,
            args=args,
        )
        stage_timings["grounding_dino_s"] = dino_generation[
            "dino_elapsed_s"
        ]
        stage_timings["sam_box_prompts_s"] = dino_generation[
            "sam_box_elapsed_s"
        ]
        stage_timings["sam_box_prompt_timings"] = dino_generation[
            "sam_box_prompt_timings"
        ]
        generation = dino_generation
        source_used = "dino_sam_boxes"
        if not generation["kept"]:
            legacy_fallback_reason = (
                "no_valid_dino_proposals"
                if not generation["proposals"]
                else "no_score_area_gated_sam_box_masks"
            )
            generation = legacy_candidate_generation(
                sam_prompt=sam_prompt,
                frame_path=frame_path,
                req_dir=req_dir,
                args=args,
            )
            stage_timings["legacy_sam_s"] = generation["elapsed_s"]
            source_used = "legacy_text_fallback"
    else:
        generation = legacy_candidate_generation(
            sam_prompt=sam_prompt,
            frame_path=frame_path,
            req_dir=req_dir,
            args=args,
        )
        stage_timings["legacy_sam_s"] = generation["elapsed_s"]
        source_used = "legacy_text"

    kept = generation["kept"]
    candidates = generation["candidates"]
    if source_used == "dino_sam_boxes" and generation.get("outputs"):
        provenance_by_index = {
            int(item.get("candidate_index", index)): dict(item)
            for index, item in enumerate(
                generation["outputs"].get("proposal_provenance", [])
            )
        }
        candidates = [
            {
                **candidate,
                "proposal_provenance": {
                    **provenance_by_index.get(int(candidate["index"]), {}),
                    **(
                        candidate.get("proposal_provenance", {})
                        if isinstance(candidate.get("proposal_provenance"), dict)
                        else {}
                    ),
                },
            }
            for candidate in candidates
        ]
    kept, candidates = cap_verifier_candidates(kept, candidates)
    fallback_eligible = len(kept) == 0
    artifact_stem = "dino" if source_used == "dino_sam_boxes" else "legacy"
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request,
        target_phrase=sam_prompt,
        selector=selector,
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem=artifact_stem,
        kept=kept,
        candidates=candidates,
        args=args,
        depth_np=depth_np,
        grounding_intent_value=intent.get("grounding_intent"),
    )
    stage_timings["qwen_s"] = float(verification.get("inference_s", 0.0))
    stage_timings["qwen_stage_s"] = float(verification.get("elapsed_s", 0.0))
    depth_refinement: dict[str, Any] = {
        "enabled": False,
        "algorithm": "dino_center_depth_connectivity_v1",
        "status": "not_applicable_without_dino_boxes",
        "adds_pixels": False,
        "candidate_count": len(kept),
        "applied_count": 0,
        "candidates": [],
        "artifact": None,
    }
    if source_used == "dino_sam_boxes":
        depth_refinement_started = time.monotonic()
        kept, candidates, depth_refinement = (
            refine_verified_dino_masks_with_depth(
                kept=kept,
                candidates=candidates,
                rgb_np=rgb_np,
                depth_np=depth_np,
                req_dir=req_dir,
                args=args,
                enabled=bool(getattr(args, "depth_refinement", True)),
            )
        )
        stage_timings["depth_refinement_s"] = float(
            time.monotonic() - depth_refinement_started
        )
    selection_started = time.monotonic()
    kept, candidates, selection = select_spatial_mask(
        kept,
        candidates,
        selector,
        depth_np=depth_np,
        min_valid_depth_fraction=args.selection_min_valid_depth_fraction,
        selection_roi=args.selection_roi,
    )
    stage_timings["spatial_selection_s"] = float(
        time.monotonic() - selection_started
    )
    overlay = task2.overlay_masks(
        rgb_np,
        kept,
        request,
        req_dir / f"overlay_{artifact_stem}.png",
    )
    proposal_provenance = []
    if source_used == "dino_sam_boxes" and generation.get("outputs"):
        candidate_provenance = {
            int(candidate["index"]): dict(candidate["proposal_provenance"])
            for candidate in candidates
            if isinstance(candidate.get("proposal_provenance"), dict)
        }
        proposal_provenance = [
            candidate_provenance.get(
                int(item.get("candidate_index", index)),
                dict(item),
            )
            for index, item in enumerate(
                generation["outputs"].get("proposal_provenance", [])
            )
        ]
    candidate_generation = {
        "configured_pipeline_mode": args.pipeline_mode,
        "source_used": source_used,
        "legacy_fallback_reason": legacy_fallback_reason,
        "dino_proposals_json": None
        if dino_generation is None
        else dino_generation["proposal_manifest"],
        "combined_candidates_json": None
        if dino_generation is None
        else dino_generation["combined_candidates"],
        "phrases": [sam_prompt]
        if dino_generation is None
        else dino_generation["phrases"],
        "proposals": []
        if dino_generation is None
        else dino_generation["proposals"],
        "depth_refinement_json": depth_refinement.get("artifact"),
    }
    path_prefix = {
        "dino_sam_boxes": "dino_sam",
        "legacy_text_fallback": "dino_legacy_fallback",
        "legacy_text": "legacy",
    }[source_used]
    return {
        "path": f"{path_prefix}_verified_spatial"
        if selection is not None
        else f"{path_prefix}_verified",
        "kept": kept,
        "presence_gate": summarize_kept(kept=kept, candidates=candidates, args=args),
        "sam_json": generation["sam_json"],
        "combined_candidates": candidate_generation["combined_candidates_json"],
        "dino_proposals": candidate_generation["dino_proposals_json"],
        "overlay": overlay,
        "candidate_overlay": verification.get("candidate_overlay"),
        "candidate_zoom": verification.get("candidate_zoom"),
        "candidate_clean_crops": verification.get("candidate_clean_crops"),
        "agent_render_output": None,
        "sam_prompt": sam_prompt,
        "sam_prompts": candidate_generation["phrases"],
        "selection": selection,
        "intent": intent,
        "verification": verification,
        "candidate_generation": candidate_generation,
        "depth_refinement": depth_refinement,
        "proposal_provenance": proposal_provenance,
        "stage_timings": stage_timings,
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
    intent: dict[str, Any],
) -> dict[str, Any]:
    agent_dir = req_dir / "agent_workspace"
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_started = time.monotonic()
    history, final_outputs, rendered_final = task6.agent_inference(
        str(frame_path),
        request,
        debug=args.debug,
        send_generate_request=task6.build_qwen_sender(args),
        call_sam_service=STATE["sam"],
        max_generations=args.max_generations,
        output_dir=str(agent_dir),
    )
    agent_elapsed_s = time.monotonic() - agent_started
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
    kept, candidates = cap_verifier_candidates(kept, candidates)
    target_phrase = intent["target_phrase"]
    selector = intent.get("selector")
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request,
        target_phrase=target_phrase,
        selector=selector,
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem="agent",
        kept=kept,
        candidates=candidates,
        args=args,
        depth_np=depth_np,
        grounding_intent_value=intent.get("grounding_intent"),
    )
    selection_started = time.monotonic()
    kept, candidates, selection = select_spatial_mask(
        kept,
        candidates,
        selector,
        depth_np=depth_np,
        min_valid_depth_fraction=args.selection_min_valid_depth_fraction,
        selection_roi=args.selection_roi,
    )
    selection_elapsed_s = time.monotonic() - selection_started
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
        "combined_candidates": None,
        "dino_proposals": None,
        "overlay": overlay,
        "candidate_overlay": verification.get("candidate_overlay"),
        "candidate_zoom": verification.get("candidate_zoom"),
        "candidate_clean_crops": verification.get("candidate_clean_crops"),
        "agent_render_output": str(agent_render_output),
        "agent_history": str(history_path),
        "sam_prompt": target_phrase,
        "sam_prompts": [target_phrase],
        "selection": selection,
        "intent": {**intent, "fallback": "agent"},
        "verification": verification,
        "candidate_generation": {
            "configured_pipeline_mode": args.pipeline_mode,
            "source_used": "agent_fallback",
            "legacy_fallback_reason": None,
            "dino_proposals_json": None,
            "combined_candidates_json": None,
            "phrases": [target_phrase],
            "proposals": [],
        },
        "depth_refinement": {
            "enabled": False,
            "algorithm": "dino_center_depth_connectivity_v1",
            "status": "not_applicable_agent_fallback",
            "adds_pixels": False,
            "candidate_count": len(kept),
            "applied_count": 0,
            "candidates": [],
            "artifact": None,
        },
        "proposal_provenance": [],
        "stage_timings": {
            "agent_s": float(agent_elapsed_s),
            "qwen_s": float(verification.get("inference_s", 0.0)),
            "qwen_stage_s": float(verification.get("elapsed_s", 0.0)),
            "spatial_selection_s": float(selection_elapsed_s),
        },
        "fallback_eligible": False,
    }


def build_selected_mask_record(
    kept: list[tuple[np.ndarray, float]],
    presence_gate: dict[str, Any],
    frame_info: dict[str, Any],
    proposal_provenance: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Describe the single selected crop mask and its full-camera translation."""

    kept_indices = presence_gate.get("kept_indices", [])
    if len(kept) != 1 or len(kept_indices) != 1:
        return None
    mask, score = kept[0]
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
    # Match the historical ROS fallback, which derives the target pixel from
    # the normalized SAM bounding-box center rather than the mask centroid.
    center = [
        float(bbox[0] + bbox[2] / 2.0),
        float(bbox[1] + bbox[3] / 2.0),
    ]
    crop_info = frame_info.get("crop", {})
    if crop_info.get("enabled"):
        applied = crop_info.get("applied_xyxy")
        if not isinstance(applied, list) or len(applied) != 4:
            raise ValueError("Enabled crop is missing applied_xyxy metadata")
        crop_xywh = [
            int(applied[0]),
            int(applied[1]),
            int(applied[2]) - int(applied[0]),
            int(applied[3]) - int(applied[1]),
        ]
    else:
        crop_xywh = None
    full_center = grounding_dino.crop_point_to_full(center, crop_xywh)
    offset_x = 0 if crop_xywh is None else crop_xywh[0]
    offset_y = 0 if crop_xywh is None else crop_xywh[1]
    sam_index = int(kept_indices[0])
    provenance = next(
        (
            item
            for item in proposal_provenance
            if int(item.get("candidate_index", -1)) == sam_index
        ),
        None,
    )
    return {
        "sam_index": sam_index,
        "score": float(score),
        "bbox_xywh_crop_pixels": bbox,
        "bbox_xywh_full_pixels": [
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2],
            bbox[3],
        ],
        "bbox_xywh_normalized": grounding_dino.mask_bbox_xywh_normalized(mask),
        "center_xy_crop_pixels": center,
        "center_xy_full_pixels": full_center,
        "crop_to_full_offset_xy_pixels": [offset_x, offset_y],
        "mask_artifact": None
        if provenance is None
        else provenance.get("mask_artifact"),
        "depth_refinement": None
        if provenance is None
        else provenance.get("depth_refinement"),
        "proposal_provenance": provenance,
    }


def segment_once(
    request: str,
    *,
    use_agent_fallback: bool = False,
    supplied_intent: dict[str, Any] | None = None,
    supplied_intent_hash: str | None = None,
) -> dict[str, Any]:
    args = STATE["args"]
    canonical_intent = None
    pipeline_intent = None
    canonical_hash = None
    if supplied_intent is not None:
        canonical_intent = grounding_intent.validate_grounding_intent(
            supplied_intent,
            expected_source_phrase=request,
        )
        canonical_hash = grounding_intent.intent_hash(canonical_intent)
        if supplied_intent_hash != canonical_hash:
            raise grounding_intent.GroundingIntentError(
                "supplied intent hash does not match the canonical intent",
                code="grounding_identity_mismatch",
                details={
                    "supplied": supplied_intent_hash,
                    "computed": canonical_hash,
                },
            )
        # Build this before touching the camera so unsupported semantics fail
        # without producing a fresh frame or a partially trusted target.
        pipeline_intent = build_v1_pipeline_intent(canonical_intent)
    elif supplied_intent_hash is not None:
        raise grounding_intent.GroundingIntentError(
            "an intent hash was supplied without a grounding intent",
            code="grounding_identity_mismatch",
        )

    req_dir = STATE["run_dir"] / timestamp_slug()
    req_dir.mkdir(parents=True, exist_ok=True)

    intent_path = None
    if canonical_intent is not None:
        intent_path = req_dir / "grounding_intent.json"
        intent_path.write_text(
            json.dumps(
                {
                    "schema_version": grounding_intent.SCHEMA_VERSION,
                    "grounding_intent": canonical_intent,
                    "intent_hash": canonical_hash,
                    "pipeline_intent": pipeline_intent,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    t0 = time.monotonic()
    capture_started = time.monotonic()
    rgb_np, depth_np, xyz_np, frame_info = capture_request_frame(req_dir)
    capture_elapsed_s = time.monotonic() - capture_started
    frame_path = Path(frame_info["saved_frame"])

    parsing_started = time.monotonic()
    intent = (
        pipeline_intent
        if pipeline_intent is not None
        else parse_request_intent(request, args)
    )
    parsing_elapsed_s = time.monotonic() - parsing_started
    result = direct_segment(
        request=request,
        intent=intent,
        rgb_np=rgb_np,
        depth_np=depth_np,
        frame_path=frame_path,
        req_dir=req_dir,
        args=args,
    )
    if result["fallback_eligible"] and use_agent_fallback:
        bounded_candidate_generation = result.get("candidate_generation")
        bounded_stage_timings = dict(result.get("stage_timings", {}))
        bounded_dino_proposals = result.get("dino_proposals")
        bounded_combined_candidates = result.get("combined_candidates")
        result = agent_fallback_segment(
            request=request,
            rgb_np=rgb_np,
            depth_np=depth_np,
            frame_path=frame_path,
            req_dir=req_dir,
            args=args,
            intent=intent,
        )
        result["candidate_generation"]["bounded_attempt"] = (
            bounded_candidate_generation
        )
        result["candidate_generation"]["dino_proposals_json"] = (
            bounded_dino_proposals
        )
        result["candidate_generation"]["combined_candidates_json"] = (
            bounded_combined_candidates
        )
        result["dino_proposals"] = bounded_dino_proposals
        result["combined_candidates"] = bounded_combined_candidates
        result["stage_timings"] = {
            **bounded_stage_timings,
            **result.get("stage_timings", {}),
        }

    kept = result.pop("kept")
    result.pop("fallback_eligible", None)
    depth_started = time.monotonic()
    object_depth = task5.object_depth_report(
        kept,
        depth_np=depth_np,
        xyz_np=xyz_np,
        view_name=args.view,
        depth_measure=frame_info["depth_measure"],
        xyz_measure=frame_info["xyz_measure"],
    )
    depth_elapsed_s = time.monotonic() - depth_started
    stage_timings = {
        "capture_s": float(capture_elapsed_s),
        "parsing_s": float(parsing_elapsed_s),
        **result.get("stage_timings", {}),
        "depth_s": float(depth_elapsed_s),
    }
    proposal_provenance = list(result.get("proposal_provenance", []))
    selected_mask = build_selected_mask_record(
        kept,
        result["presence_gate"],
        frame_info,
        proposal_provenance,
    )
    total_elapsed_s = time.monotonic() - t0
    stage_timings["total_s"] = float(total_elapsed_s)
    response = {
        "request": request,
        "pipeline_mode": args.pipeline_mode,
        "path": result["path"],
        "num_kept": result["presence_gate"]["num_kept"],
        "scores": [float(score) for _, score in kept],
        "presence_gate": result["presence_gate"],
        "object_depth": object_depth,
        "frame": frame_info["saved_frame"],
        "zed_frame": frame_info,
        "overlay": result["overlay"],
        "candidate_overlay": result.get("candidate_overlay"),
        "candidate_zoom": result.get("candidate_zoom"),
        "candidate_clean_crops": result.get("candidate_clean_crops"),
        "sam_json": result["sam_json"],
        "combined_candidates": result.get("combined_candidates"),
        "dino_proposals": result.get("dino_proposals"),
        "agent_render_output": result.get("agent_render_output"),
        "sam_prompt": result.get("sam_prompt", request),
        "sam_prompts": result.get("sam_prompts", [result.get("sam_prompt", request)]),
        "selection": result.get("selection"),
        "intent": result.get("intent"),
        "verification": result.get("verification"),
        "candidate_generation": result.get("candidate_generation"),
        "depth_refinement": result.get("depth_refinement"),
        "proposal_provenance": proposal_provenance,
        "selected_mask": selected_mask,
        "stage_timings": stage_timings,
        "agent_fallback_explicitly_requested": bool(use_agent_fallback),
        "elapsed_s": round(total_elapsed_s, 2),
        "output_dir": str(req_dir),
    }
    if canonical_intent is not None:
        response.update(
            {
                "schema_version": grounding_intent.SCHEMA_VERSION,
                "grounding_intent": canonical_intent,
                "intent_hash": canonical_hash,
                "intent_parser": {
                    "mode": "supplied_v1",
                    "active_parser": "validated_request_intent",
                },
                "intent_record": str(intent_path),
                "primary_sam_phrase": pipeline_intent["target_phrase"],
                "prompt_family": grounding_dino.build_phrase_family(
                    pipeline_intent["target_phrase"]
                ),
            }
        )
    if result.get("agent_history") is not None:
        response["agent_history"] = result["agent_history"]

    result_path = req_dir / "result.json"
    response["result_json"] = str(result_path)
    result_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    return response


if app is not None:

    @app.get("/health")
    def health() -> dict[str, Any]:
        args = STATE.get("args")
        return {
            "sam_loaded": "sam" in STATE,
            "camera_open": "zed" in STATE,
            "busy": LOCK.locked(),
            "uptime_s": round(time.monotonic() - STATE.get("t0", time.monotonic()), 1),
            "pipeline_mode": None if args is None else args.pipeline_mode,
            "crop_xywh": None
            if args is None or args.crop is None
            else [int(value) for value in args.crop],
            "qwen_model": None if args is None else args.qwen_model,
            "qwen_loaded": STATE.get("qwen_runtime") is not None,
            "qwen_runtime": STATE.get("qwen_runtime"),
            "grounding_dino": None
            if args is None
            else {
                **STATE.get(
                    "dino_runtime",
                    {
                        "loaded": False,
                        "model_id": args.dino_model,
                        "dtype": "torch.bfloat16",
                        "devices": [grounding_dino.DEFAULT_DEVICE],
                        "local_files_only": True,
                    },
                ),
                "box_threshold": float(args.dino_box_threshold),
                "text_threshold": float(args.dino_text_threshold),
                "nms_iou": float(args.dino_nms_iou),
                "max_proposals": int(args.dino_max_proposals),
                "box_padding_fraction": float(args.dino_box_padding),
                "mask_refinement": {
                    "min_component_pixels": int(
                        grounding_dino.DEFAULT_MASK_MIN_COMPONENT_PIXELS
                    ),
                    "min_component_fraction": float(
                        grounding_dino.DEFAULT_MASK_MIN_COMPONENT_FRACTION
                    ),
                    "min_original_overlap": float(
                        grounding_dino.DEFAULT_MASK_MIN_ORIGINAL_OVERLAP
                    ),
                    "geometry_weight": float(
                        grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT
                    ),
                    "sam_score_weight": float(
                        1.0 - grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT
                    ),
                    "adds_pixels": False,
                },
                "depth_refinement": {
                    "enabled": bool(args.depth_refinement),
                    "algorithm": "dino_center_depth_connectivity_v1",
                    "anchor": "original_dino_box_center_patch",
                    "min_valid_pixels": int(
                        mask_depth.DEFAULT_REFINEMENT_MIN_VALID_PIXELS
                    ),
                    "min_valid_fraction": float(
                        mask_depth.DEFAULT_REFINEMENT_MIN_VALID_FRACTION
                    ),
                    "min_jump_m": float(
                        mask_depth.DEFAULT_REFINEMENT_MIN_JUMP_M
                    ),
                    "relative_jump": float(
                        mask_depth.DEFAULT_REFINEMENT_RELATIVE_JUMP
                    ),
                    "max_local_jump_m": float(
                        mask_depth.DEFAULT_REFINEMENT_MAX_LOCAL_JUMP_M
                    ),
                    "max_global_drift_m": float(
                        mask_depth.DEFAULT_REFINEMENT_MAX_GLOBAL_DRIFT_M
                    ),
                    "min_retained_fraction": float(
                        mask_depth.DEFAULT_REFINEMENT_MIN_RETAINED_FRACTION
                    ),
                    "adds_pixels": False,
                },
                "calibrated_crop_xywh": [
                    int(value) for value in task5.DEFAULT_CROP
                ],
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

    @app.post("/segment")
    def segment(request: str, use_agent_fallback: bool = False) -> dict[str, Any]:
        if not request.strip():
            raise HTTPException(status_code=400, detail="request must not be empty")
        with LOCK:
            try:
                return segment_once(
                    request.strip(),
                    use_agent_fallback=use_agent_fallback,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.post("/v1/segment")
    def segment_v1(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            validated = validate_v1_segment_request(payload)
        except grounding_intent.GroundingIntentError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                },
            ) from exc
        with LOCK:
            try:
                return segment_once(
                    validated["source_phrase"],
                    use_agent_fallback=validated["use_agent_fallback"],
                    supplied_intent=validated["grounding_intent"],
                    supplied_intent_hash=validated["intent_hash"],
                )
            except grounding_intent.GroundingIntentError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                    },
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
