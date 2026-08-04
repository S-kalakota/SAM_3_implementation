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

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError:  # pragma: no cover - exercised only before deps are installed.
    FastAPI = None
    HTTPException = None
    uvicorn = None

import local_qwen
import mask_depth
import qwen_candidate_verifier as candidate_verifier
import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6


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


def first_json_object(text: str) -> dict[str, Any] | None:
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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


def fallback_intent(request: str) -> dict[str, Any]:
    """Generic local fallback if the language parser is unavailable."""
    lowered = request.lower()
    selector_patterns = (
        ("rightmost", r"\bright\s*most\b|\brightmost\b|\bon the right\b|\bright side\b"),
        ("leftmost", r"\bleft\s*most\b|\bleftmost\b|\bon the left\b|\bleft side\b"),
        ("topmost", r"\btop\s*most\b|\btopmost\b|\bat the top\b"),
        ("bottommost", r"\bbottom\s*most\b|\bbottommost\b|\bat the bottom\b"),
        ("nearest", r"\bnearest\b|\bclosest\b"),
        ("farthest", r"\bfarthest\b|\bfurthest\b"),
        ("largest", r"\blargest\b|\bbiggest\b"),
        ("smallest", r"\bsmallest\b"),
    )
    selector = None
    cleaned = lowered
    for name, pattern in selector_patterns:
        if selector is None and re.search(pattern, lowered):
            selector = name
        cleaned = re.sub(pattern, " ", cleaned)

    cleaned = re.sub(r"[^a-z0-9_\- ]+", " ", cleaned)
    cleaned = re.sub(
        r"\b(get|grab|pick|select|find|choose|show|the|a|an|please|me|object|item|one|most)\b",
        " ",
        cleaned,
    )
    target_phrase = " ".join(cleaned.split()) or request
    return {
        "target_phrase": target_phrase,
        "selector": selector,
        "parser": "fallback_rules",
    }


def parse_request_intent(request: str, args: argparse.Namespace) -> dict[str, Any]:
    if not should_parse_natural_request(request):
        return {
            "target_phrase": request,
            "selector": None,
            "parser": "direct_prompt",
        }

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
        return {
            "target_phrase": target_phrase,
            "selector": selector,
            "parser": "qwen_json",
            "raw_parser_output": parsed_text,
        }
    except Exception as exc:
        intent = fallback_intent(request)
        intent["parser_error"] = repr(exc)
        return intent


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
                "crop_size_wh_pixels": [int(width), int(height)],
            }
        )
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
) -> tuple[
    list[tuple[np.ndarray, float]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run the fail-closed Qwen visual gate before depth or spatial selection."""

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
        "candidate_manifest": None,
        "attempts": [],
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
    return kept, candidates, verification


def direct_segment(
    *,
    request: str,
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    frame_path: Path,
    req_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    intent = parse_request_intent(request, args)
    sam_prompt = intent["target_phrase"]
    selector = intent.get("selector")
    sam_json = STATE["sam"](
        image_path=str(frame_path),
        text_prompt=sam_prompt,
        output_folder_path=str(req_dir / "direct"),
    )
    outputs = read_json(sam_json)
    masks = task6.decode_agent_masks(outputs)
    scores = np.asarray(outputs.get("pred_scores", []), dtype=np.float32)
    kept, candidates = task2.gate_masks(
        masks,
        scores,
        conf_thresh=args.presence_conf_threshold,
        min_area=args.min_area,
    )
    fallback_eligible = len(kept) == 0
    kept, candidates, verification = verify_candidates_with_qwen(
        request=request,
        target_phrase=sam_prompt,
        selector=selector,
        rgb_np=rgb_np,
        frame_path=frame_path,
        req_dir=req_dir,
        artifact_stem="direct",
        kept=kept,
        candidates=candidates,
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
        "sam_json": sam_json,
        "overlay": overlay,
        "candidate_overlay": verification.get("candidate_overlay"),
        "candidate_zoom": verification.get("candidate_zoom"),
        "agent_render_output": None,
        "sam_prompt": sam_prompt,
        "selection": selection,
        "intent": intent,
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
    intent: dict[str, Any],
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
        "selection": selection,
        "intent": {**intent, "fallback": "agent"},
        "verification": verification,
        "fallback_eligible": False,
    }


def segment_once(request: str, *, use_agent_fallback: bool = True) -> dict[str, Any]:
    args = STATE["args"]
    req_dir = STATE["run_dir"] / timestamp_slug()
    req_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    rgb_np, depth_np, xyz_np, frame_info = capture_request_frame(req_dir)
    frame_path = Path(frame_info["saved_frame"])

    result = direct_segment(
        request=request,
        rgb_np=rgb_np,
        depth_np=depth_np,
        frame_path=frame_path,
        req_dir=req_dir,
        args=args,
    )
    if result["fallback_eligible"] and use_agent_fallback:
        direct_intent = result["intent"]
        result = agent_fallback_segment(
            request=request,
            rgb_np=rgb_np,
            depth_np=depth_np,
            frame_path=frame_path,
            req_dir=req_dir,
            args=args,
            intent=direct_intent,
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
    response = {
        "request": request,
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
        "sam_json": result["sam_json"],
        "agent_render_output": result.get("agent_render_output"),
        "sam_prompt": result.get("sam_prompt", request),
        "selection": result.get("selection"),
        "intent": result.get("intent"),
        "verification": result.get("verification"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "output_dir": str(req_dir),
    }
    if result.get("agent_history") is not None:
        response["agent_history"] = result["agent_history"]

    result_path = req_dir / "result.json"
    result_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    response["result_json"] = str(result_path)
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
            "crop_xywh": None
            if args is None or args.crop is None
            else [int(value) for value in args.crop],
            "qwen_model": None if args is None else args.qwen_model,
            "qwen_loaded": STATE.get("qwen_runtime") is not None,
            "qwen_runtime": STATE.get("qwen_runtime"),
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
    def segment(request: str, use_agent_fallback: bool = True) -> dict[str, Any]:
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
