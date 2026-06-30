#!/usr/bin/env python3
"""Task 2/3/4 smoke test: prompt SAM 3.1, gate detections, and draw overlays."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task2_sam31_image_prompt.json"
DEFAULT_PRESENCE_CONF_THRESH = 0.25
DEFAULT_MIN_AREA = 250


def parse_args(
    description: str = "Run SAM 3.1 on a saved image with a typed text prompt.",
    default_output: Path = DEFAULT_OUTPUT,
    default_overlay_output: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the saved RGB image to segment.",
    )
    parser.add_argument(
        "--prompt",
        default="green water bottle",
        help="Text prompt to send to SAM 3.1.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        type=Path,
        help="Path to sam3.1_multiplex.pt.",
    )
    parser.add_argument(
        "--threshold",
        default=0.05,
        type=float,
        help="SAM output probability threshold.",
    )
    parser.add_argument(
        "--output-json",
        default=default_output,
        type=Path,
        help="Where to write the compact result summary.",
    )
    parser.add_argument(
        "--presence-conf-threshold",
        default=DEFAULT_PRESENCE_CONF_THRESH,
        type=float,
        help="Task 3 gate: drop masks with scores at or below this value.",
    )
    parser.add_argument(
        "--min-area",
        default=DEFAULT_MIN_AREA,
        type=int,
        help="Task 3 gate: drop masks with pixel area at or below this value.",
    )
    parser.add_argument(
        "--overlay-output",
        default=default_overlay_output,
        type=Path,
        help="Task 4: where to write the overlay image. Omit to skip.",
    )
    parser.add_argument(
        "--det-threshold",
        default=None,
        type=float,
        help="Optional internal detector threshold override for probing weak prompts.",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3. Leave off on this CUDA 13 Thor setup.",
    )
    parser.add_argument(
        "--verbose-load",
        action="store_true",
        help="Print the full checkpoint load log from SAM 3.1.",
    )
    return parser.parse_args()


def to_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def masks_to_bool_array(masks: object, threshold: float = 0.5) -> np.ndarray:
    masks_np = to_numpy_array(masks)
    if masks_np.size == 0:
        if masks_np.ndim >= 3:
            shape = (masks_np.shape[0], masks_np.shape[-2], masks_np.shape[-1])
            return np.zeros(shape, dtype=bool)
        return np.zeros((0, 0, 0), dtype=bool)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    elif masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim > 3:
        masks_np = np.squeeze(masks_np)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks shaped as NxHxW, got {masks_np.shape}")

    return masks_np > threshold


def gate_masks(
    masks: object,
    scores: object,
    conf_thresh: float = DEFAULT_PRESENCE_CONF_THRESH,
    min_area: int = DEFAULT_MIN_AREA,
) -> tuple[list[tuple[np.ndarray, float]], list[dict]]:
    bool_masks = masks_to_bool_array(masks)
    scores_np = to_numpy_array(scores).astype(np.float32, copy=False).reshape(-1)

    kept = []
    candidates = []
    for idx, mask in enumerate(bool_masks):
        score = float(scores_np[idx]) if idx < scores_np.size else 0.0
        area = int(mask.sum())
        reject_reasons = []
        if score <= conf_thresh:
            reject_reasons.append("low_score")
        if area <= min_area:
            reject_reasons.append("small_area")

        candidate = {
            "index": idx,
            "score": score,
            "area_pixels": area,
            "kept": len(reject_reasons) == 0,
            "reject_reasons": reject_reasons,
        }
        candidates.append(candidate)
        if candidate["kept"]:
            kept.append((mask, score))

    return kept, candidates


def summarize_presence_gate(
    outputs: dict,
    conf_thresh: float = DEFAULT_PRESENCE_CONF_THRESH,
    min_area: int = DEFAULT_MIN_AREA,
) -> dict:
    kept, candidates = gate_masks(
        outputs["out_binary_masks"],
        outputs.get("out_probs", []),
        conf_thresh=conf_thresh,
        min_area=min_area,
    )
    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    rejected_candidates = [
        candidate for candidate in candidates if not candidate["kept"]
    ]
    return {
        "conf_threshold": float(conf_thresh),
        "min_area_pixels": int(min_area),
        "num_candidates": len(candidates),
        "num_kept": len(kept),
        "kept_indices": [candidate["index"] for candidate in kept_candidates],
        "kept_scores": [candidate["score"] for candidate in kept_candidates],
        "kept_areas_pixels": [
            candidate["area_pixels"] for candidate in kept_candidates
        ],
        "rejected": rejected_candidates,
    }


def summarize_outputs(outputs: dict) -> dict:
    obj_ids = to_numpy_array(outputs["out_obj_ids"])
    scores = to_numpy_array(outputs.get("out_probs", [])).astype(np.float32, copy=False)
    boxes_xywh = to_numpy_array(outputs["out_boxes_xywh"]).astype(np.float32, copy=False)
    masks = masks_to_bool_array(outputs["out_binary_masks"])

    height, width = masks.shape[-2:] if masks.ndim == 3 else (0, 0)
    if masks.shape[0] == 0:
        areas = []
    else:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(int).tolist()

    return {
        "num_masks": int(masks.shape[0]),
        "image_height": int(height),
        "image_width": int(width),
        "object_ids": obj_ids.astype(int).tolist(),
        "scores": [float(score) for score in scores.tolist()],
        "boxes_xywh_normalized": boxes_xywh.tolist(),
        "mask_areas_pixels": areas,
    }


def overlay_masks(
    rgb_np: np.ndarray,
    kept: list[tuple[np.ndarray, float]],
    label: str,
    output_path: Path,
) -> dict:
    out = rgb_np.copy()
    overlay_color = np.array([0, 255, 0], dtype=np.float32)

    for mask, score in kept:
        if mask.shape != out.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {out.shape[:2]}"
            )
        if not mask.any():
            continue

        out[mask] = (0.5 * out[mask] + 0.5 * overlay_color).astype(np.uint8)
        ys, xs = np.where(mask)
        origin = (int(xs.min()), max(int(ys.min()) - 8, 14))
        text = f"{label} {score:.2f}"
        cv2.putText(
            out,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), out[:, :, ::-1]):
        raise OSError(f"Failed to write overlay image: {output_path}")

    return {
        "output": str(output_path),
        "num_masks_drawn": int(len(kept)),
    }


def run_once(args: argparse.Namespace) -> dict:
    image_path = args.image.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"image={image_path}")
    print(f"image_size={image.size[0]}x{image.size[1]}")
    print(f"prompt={args.prompt!r}")
    print(f"checkpoint={checkpoint_path}")
    print(f"threshold={args.threshold:.3f}")
    print(f"presence_conf_threshold={args.presence_conf_threshold:.3f}")
    print(f"min_area={args.min_area}")
    if args.det_threshold is not None:
        print(f"det_threshold={args.det_threshold:.3f}")

    torch.set_float32_matmul_precision("high")
    build_kwargs = dict(
        checkpoint_path=str(checkpoint_path),
        use_fa3=args.use_fa3,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        default_output_prob_thresh=args.threshold,
    )
    if args.verbose_load:
        predictor = build_sam3_multiplex_video_predictor(**build_kwargs)
    else:
        load_log = io.StringIO()
        with contextlib.redirect_stdout(load_log):
            predictor = build_sam3_multiplex_video_predictor(**build_kwargs)
        if load_log.tell():
            print(f"suppressed_model_load_log_chars={load_log.tell()}")

    try:
        model = predictor.model
        if args.det_threshold is not None and hasattr(model, "score_threshold_detection"):
            model.score_threshold_detection = args.det_threshold
        inference_state = model.init_state(
            resource_path=str(image_path),
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, outputs = model.add_prompt(
                inference_state=inference_state,
                frame_idx=0,
                text_str=args.prompt,
                output_prob_thresh=args.threshold,
            )
        summary = summarize_outputs(outputs)
        presence_gate = summarize_presence_gate(
            outputs,
            conf_thresh=args.presence_conf_threshold,
            min_area=args.min_area,
        )
        overlay = None
        if args.overlay_output is not None:
            kept, _ = gate_masks(
                outputs["out_binary_masks"],
                outputs.get("out_probs", []),
                conf_thresh=args.presence_conf_threshold,
                min_area=args.min_area,
            )
            overlay = overlay_masks(
                np.asarray(image, dtype=np.uint8)[:, :, :3],
                kept,
                args.prompt,
                args.overlay_output.expanduser().resolve(),
            )
    finally:
        if "inference_state" in locals():
            del inference_state
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "image": str(image_path),
        "prompt": args.prompt,
        "checkpoint": str(checkpoint_path),
        "threshold": args.threshold,
        "det_threshold": args.det_threshold,
        "presence_gate": presence_gate,
        "overlay": overlay,
        **summary,
    }
    return result


def main(
    description: str = "Run SAM 3.1 on a saved image with a typed text prompt.",
    default_output: Path = DEFAULT_OUTPUT,
    default_overlay_output: Path | None = None,
) -> None:
    args = parse_args(
        description=description,
        default_output=default_output,
        default_overlay_output=default_overlay_output,
    )
    result = run_once(args)

    print(f"num_masks={result['num_masks']}")
    print(f"scores={result['scores']}")
    print(f"mask_areas_pixels={result['mask_areas_pixels']}")
    print(f"boxes_xywh_normalized={result['boxes_xywh_normalized']}")
    gate_summary = result["presence_gate"]
    print(f"kept={gate_summary['num_kept']}")
    print(f"kept_scores={gate_summary['kept_scores']}")
    print(f"kept_areas_pixels={gate_summary['kept_areas_pixels']}")
    print(f"rejected={gate_summary['rejected']}")
    if result["overlay"] is not None:
        print(f"overlay_masks_drawn={result['overlay']['num_masks_drawn']}")
        print(f"wrote_overlay={result['overlay']['output']}")

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
