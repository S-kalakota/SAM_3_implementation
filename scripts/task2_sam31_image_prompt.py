#!/usr/bin/env python3
"""Task 2 smoke test: prompt SAM 3.1 on one saved image."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task2_sam31_image_prompt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 3.1 on a saved image with a typed text prompt."
    )
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
        default=0.5,
        type=float,
        help="SAM output probability threshold.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Where to write the compact result summary.",
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


def summarize_outputs(outputs: dict) -> dict:
    obj_ids = np.asarray(outputs["out_obj_ids"])
    scores = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    boxes_xywh = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)

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
        **summary,
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_once(args)

    print(f"num_masks={result['num_masks']}")
    print(f"scores={result['scores']}")
    print(f"mask_areas_pixels={result['mask_areas_pixels']}")
    print(f"boxes_xywh_normalized={result['boxes_xywh_normalized']}")

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
