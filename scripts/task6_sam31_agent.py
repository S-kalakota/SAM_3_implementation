#!/usr/bin/env python3
"""Task 6: run Meta's SAM 3.1 agent with local cached Qwen-VL."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

# Meta agent imports its default OpenAI client even when callers inject a local LLM.
# Keep Task 6 local-only by providing a tiny import stub when openai is absent.
try:
    import openai  # noqa: F401
except ImportError:
    openai_stub = types.ModuleType("openai")

    class _UnavailableOpenAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OpenAI client is unavailable; Task 6 uses local Qwen instead.")

    openai_stub.OpenAI = _UnavailableOpenAI
    sys.modules["openai"] = openai_stub

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from sam3.agent.agent_core import agent_inference
from sam3.agent.viz import visualize
from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.train.masks_ops import rle_encode

import local_qwen
import task2_sam31_image_prompt as task2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task6_sam31_agent.json"
DEFAULT_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_agent.png"
DEFAULT_AGENT_RENDER_OUTPUT = PROJECT_ROOT / "outputs/result_agent_meta.png"
DEFAULT_AGENT_OUTPUT_DIR = PROJECT_ROOT / "outputs/task6_agent_workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SAM 3.1 Meta agent on one saved RGB image using local cached Qwen-VL."
    )
    parser.add_argument("--image", required=True, type=Path, help="Saved RGB image to ground.")
    parser.add_argument(
        "--request",
        required=True,
        help="Natural-language grounding request, e.g. 'the green object'.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--overlay-output", default=DEFAULT_OVERLAY_OUTPUT, type=Path)
    parser.add_argument(
        "--agent-render-output",
        default=DEFAULT_AGENT_RENDER_OUTPUT,
        type=Path,
        help="Where to save Meta agent's numbered final render.",
    )
    parser.add_argument(
        "--agent-output-dir",
        default=DEFAULT_AGENT_OUTPUT_DIR,
        type=Path,
        help="Workspace for intermediate agent SAM renders and histories.",
    )
    parser.add_argument(
        "--qwen-model",
        default=local_qwen.DEFAULT_QWEN_MODEL,
        help="Local cached Qwen-VL model id or path. No download is attempted by default.",
    )
    parser.add_argument("--qwen-max-new-tokens", default=2048, type=int)
    parser.add_argument(
        "--allow-qwen-downloads",
        action="store_true",
        help="Allow Transformers to fetch missing Qwen files. Leave off for local-cache-only mode.",
    )
    parser.add_argument("--qwen-device-map", default=local_qwen.DEFAULT_DEVICE_MAP)
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument("--presence-conf-threshold", default=task2.DEFAULT_PRESENCE_CONF_THRESH, type=float)
    parser.add_argument("--min-area", default=task2.DEFAULT_MIN_AREA, type=int)
    parser.add_argument("--det-threshold", default=None, type=float)
    parser.add_argument("--max-generations", default=8, type=int)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return safe.strip("_") or "prompt"


class MultiplexSam3AgentService:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        threshold: float,
        det_threshold: float | None,
        use_fa3: bool,
        verbose_load: bool,
    ) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        torch.set_float32_matmul_precision("high")
        build_kwargs = dict(
            checkpoint_path=str(checkpoint_path),
            use_fa3=use_fa3,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            default_output_prob_thresh=threshold,
        )
        if verbose_load:
            self.predictor = build_sam3_multiplex_video_predictor(**build_kwargs)
        else:
            load_log = io.StringIO()
            with contextlib.redirect_stdout(load_log):
                self.predictor = build_sam3_multiplex_video_predictor(**build_kwargs)
            if load_log.tell():
                print(f"suppressed_model_load_log_chars={load_log.tell()}")

        self.model = self.predictor.model
        if det_threshold is not None and hasattr(self.model, "score_threshold_detection"):
            self.model.score_threshold_detection = det_threshold
        self.threshold = threshold

    def close(self) -> None:
        if hasattr(self, "predictor"):
            del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(
        self,
        *,
        image_path: str,
        text_prompt: str,
        output_folder_path: str,
    ) -> str:
        image_path_obj = Path(image_path).expanduser().resolve()
        image = Image.open(image_path_obj).convert("RGB")
        width, height = image.size
        output_dir = Path(output_folder_path).expanduser().resolve() / safe_name(image_path_obj.stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_name = safe_name(text_prompt)
        output_json_path = output_dir / f"{prompt_name}.json"
        output_image_path = output_dir / f"{prompt_name}.png"

        inference_state = None
        try:
            inference_state = self.model.init_state(
                resource_path=str(image_path_obj),
                offload_video_to_cpu=False,
                async_loading_frames=False,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, raw_outputs = self.model.add_prompt(
                    inference_state=inference_state,
                    frame_idx=0,
                    text_str=text_prompt,
                    output_prob_thresh=self.threshold,
                )
            masks = task2.masks_to_bool_array(raw_outputs["out_binary_masks"])
            scores = task2.to_numpy_array(raw_outputs.get("out_probs", [])).astype(np.float32, copy=False).reshape(-1)
            boxes_xywh = task2.to_numpy_array(raw_outputs["out_boxes_xywh"]).astype(np.float32, copy=False)
        finally:
            if inference_state is not None:
                del inference_state

        if masks.shape[0] == 0:
            pred_masks: list[str] = []
            pred_boxes: list[list[float]] = []
            pred_scores: list[float] = []
        else:
            valid = []
            for index, mask in enumerate(masks):
                if index < len(scores) and int(mask.sum()) > 0:
                    valid.append(index)
            pred_scores = [float(scores[index]) for index in valid]
            pred_boxes = [boxes_xywh[index].astype(float).tolist() for index in valid]
            if valid:
                rles = rle_encode(torch.as_tensor(masks[valid], dtype=torch.bool))
                pred_masks = [rle["counts"] for rle in rles]
            else:
                pred_masks = []

            order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
            pred_scores = [pred_scores[i] for i in order]
            pred_boxes = [pred_boxes[i] for i in order]
            pred_masks = [pred_masks[i] for i in order]

        agent_outputs = {
            "original_image_path": str(image_path_obj),
            "output_image_path": str(output_image_path),
            "orig_img_h": int(height),
            "orig_img_w": int(width),
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }
        output_json_path.write_text(json.dumps(agent_outputs, indent=2) + "\n", encoding="utf-8")
        if pred_masks:
            visualize(agent_outputs).save(output_image_path)
        else:
            image.save(output_image_path)
        return str(output_json_path)


def decode_agent_masks(agent_outputs: dict[str, Any]) -> np.ndarray:
    height = int(agent_outputs["orig_img_h"])
    width = int(agent_outputs["orig_img_w"])
    decoded = []
    for counts in agent_outputs.get("pred_masks", []):
        mask = mask_utils.decode({"size": [height, width], "counts": counts})
        decoded.append(mask.astype(bool))
    if not decoded:
        return np.zeros((0, height, width), dtype=bool)
    return np.stack(decoded, axis=0)


def build_qwen_sender(args: argparse.Namespace):
    def send_generate_request(messages: list[dict[str, Any]]) -> str:
        return local_qwen.qwen_generate(
            messages,
            model_id=args.qwen_model,
            max_new_tokens=args.qwen_max_new_tokens,
            local_files_only=not args.allow_qwen_downloads,
            device_map=args.qwen_device_map,
        )

    return send_generate_request


def run_agent(args: argparse.Namespace) -> dict[str, Any]:
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.overlay_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.agent_render_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.agent_output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    sam_service = MultiplexSam3AgentService(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        det_threshold=args.det_threshold,
        use_fa3=args.use_fa3,
        verbose_load=args.verbose_load,
    )
    try:
        history, final_outputs, rendered_final = agent_inference(
            str(image_path),
            args.request,
            debug=args.debug,
            send_generate_request=build_qwen_sender(args),
            call_sam_service=sam_service,
            max_generations=args.max_generations,
            output_dir=str(args.agent_output_dir.expanduser().resolve()),
        )
    finally:
        sam_service.close()

    rendered_final.save(args.agent_render_output.expanduser().resolve())
    masks = decode_agent_masks(final_outputs)
    scores = np.asarray(final_outputs.get("pred_scores", []), dtype=np.float32)
    kept, candidates = task2.gate_masks(
        masks,
        scores,
        conf_thresh=args.presence_conf_threshold,
        min_area=args.min_area,
    )
    rgb_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    overlay = task2.overlay_masks(
        rgb_np,
        kept,
        args.request,
        args.overlay_output.expanduser().resolve(),
    )

    kept_candidates = [candidate for candidate in candidates if candidate["kept"]]
    result = {
        "image": str(image_path),
        "request": args.request,
        "qwen_model": args.qwen_model,
        "qwen_local_files_only": not args.allow_qwen_downloads,
        "threshold": args.threshold,
        "presence_gate": {
            "conf_threshold": float(args.presence_conf_threshold),
            "min_area_pixels": int(args.min_area),
            "num_candidates": len(candidates),
            "num_kept": len(kept),
            "kept_indices": [candidate["index"] for candidate in kept_candidates],
            "kept_scores": [candidate["score"] for candidate in kept_candidates],
            "kept_areas_pixels": [candidate["area_pixels"] for candidate in kept_candidates],
            "rejected": [candidate for candidate in candidates if not candidate["kept"]],
        },
        "agent_final_outputs": final_outputs,
        "agent_history": history,
        "agent_render_output": str(args.agent_render_output.expanduser().resolve()),
        "overlay": overlay,
    }
    args.output_json.expanduser().resolve().write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    result = run_agent(args)
    print(f"request={result['request']!r}")
    print(f"qwen_model={result['qwen_model']}")
    print(f"kept={result['presence_gate']['num_kept']}")
    print(f"kept_scores={result['presence_gate']['kept_scores']}")
    print(f"wrote={args.output_json.expanduser().resolve()}")
    print(f"wrote_agent_render={result['agent_render_output']}")
    print(f"wrote_overlay={result['overlay']['output']}")


if __name__ == "__main__":
    main()
