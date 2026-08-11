#!/usr/bin/env python3
"""Task 6: run Meta's SAM 3.1 agent with local cached Qwen-VL."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
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
import grounding_dino
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
    parser.add_argument(
        "--qwen-max-new-tokens",
        default=local_qwen.DEFAULT_MAX_NEW_TOKENS,
        type=int,
    )
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

    def segment_boxes(
        self,
        *,
        image_path: str,
        proposals: list[dict[str, Any]],
        output_folder_path: str,
    ) -> str:
        """Run one positive SAM box prompt per proposal on one image state.

        At most one mask is retained for each proposal.  Output arrays remain in
        proposal order so ``pred_masks``, ``pred_boxes``, verifier IDs, and the
        service presence-gate indices describe the same candidate.
        """

        if not 1 <= len(proposals) <= grounding_dino.DEFAULT_MAX_PROPOSALS:
            raise ValueError(
                "SAM box prompting requires between one and three proposals"
            )
        image_path_obj = Path(image_path).expanduser().resolve()
        image = Image.open(image_path_obj).convert("RGB")
        width, height = image.size
        output_dir = Path(output_folder_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / "combined_candidates.json"
        output_image_path = output_dir / "combined_candidates.png"

        selected_masks: list[np.ndarray] = []
        selected_scores: list[float] = []
        selected_boxes: list[list[float]] = []
        provenance: list[dict[str, Any]] = []
        prompt_timings: list[dict[str, Any]] = []
        inference_state = None
        try:
            inference_state = self.model.init_state(
                resource_path=str(image_path_obj),
                offload_video_to_cpu=False,
                async_loading_frames=False,
            )
            for proposal in proposals:
                proposal_id = int(proposal["proposal_id"])
                padded_box = [
                    float(value)
                    for value in proposal["padded_box_xyxy_crop_pixels"]
                ]
                normalized_box = grounding_dino.pixel_xyxy_to_normalized_sam_xywh(
                    padded_box,
                    width,
                    height,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prompt_started = time.monotonic()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, raw_outputs = self.model.add_prompt(
                        inference_state=inference_state,
                        frame_idx=0,
                        text_str="visual",
                        boxes_xywh=[normalized_box],
                        box_labels=[1],
                        output_prob_thresh=self.threshold,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_s = time.monotonic() - prompt_started

                masks = task2.masks_to_bool_array(
                    raw_outputs["out_binary_masks"]
                )
                scores = task2.to_numpy_array(
                    raw_outputs.get("out_probs", [])
                ).astype(np.float32, copy=False).reshape(-1)
                eligible: list[dict[str, Any]] = []
                raw_candidate_records: list[dict[str, Any]] = []
                for raw_index, mask in enumerate(masks):
                    candidate_record: dict[str, Any] = {
                        "raw_index": int(raw_index),
                        "sam_score": None,
                        "eligible": False,
                        "reject_reason": None,
                    }
                    if raw_index >= len(scores):
                        candidate_record["reject_reason"] = "missing_sam_score"
                        raw_candidate_records.append(candidate_record)
                        continue
                    score = float(scores[raw_index])
                    candidate_record["sam_score"] = score
                    if not mask.any():
                        candidate_record["reject_reason"] = "empty_raw_mask"
                        raw_candidate_records.append(candidate_record)
                        continue
                    refined_mask, refinement = (
                        grounding_dino.refine_mask_with_dino_box(
                            mask,
                            original_box_xyxy=proposal[
                                "original_box_xyxy_crop_pixels"
                            ],
                            support_box_xyxy=padded_box,
                        )
                    )
                    candidate_record["mask_refinement"] = refinement
                    if not refined_mask.any():
                        candidate_record["reject_reason"] = (
                            "empty_after_geometric_refinement"
                        )
                        raw_candidate_records.append(candidate_record)
                        continue
                    refined_geometry = refinement["refined_geometry"]
                    if refined_geometry["inside_dino_pixels"] <= 0:
                        candidate_record["reject_reason"] = (
                            "no_overlap_with_original_dino_box"
                        )
                        raw_candidate_records.append(candidate_record)
                        continue
                    center = grounding_dino.mask_center_xy(refined_mask)
                    candidate_record["refined_center_xy_crop_pixels"] = [
                        float(center[0]),
                        float(center[1]),
                    ]
                    if not grounding_dino.box_contains_point(padded_box, center):
                        candidate_record["reject_reason"] = (
                            "refined_center_outside_prompt_box"
                        )
                        raw_candidate_records.append(candidate_record)
                        continue
                    selection_score = (
                        grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT
                        * refined_geometry["geometry_score"]
                        + (1.0 - grounding_dino.DEFAULT_MASK_GEOMETRY_WEIGHT)
                        * score
                    )
                    candidate_record["eligible"] = True
                    candidate_record["selection_score"] = float(selection_score)
                    raw_candidate_records.append(candidate_record)
                    eligible.append(
                        {
                            "sam_score": score,
                            "raw_index": int(raw_index),
                            "raw_mask": mask,
                            "refined_mask": refined_mask,
                            "center": center,
                            "refinement": refinement,
                            "selection_score": float(selection_score),
                        }
                    )

                timing_record: dict[str, Any] = {
                    "proposal_id": proposal_id,
                    "elapsed_s": float(elapsed_s),
                    "raw_candidate_count": int(len(masks)),
                    "center_valid_candidate_count": int(len(eligible)),
                    "selected_raw_sam_index": None,
                    "selected_geometry_score": None,
                    "selected_mask_score": None,
                    "raw_candidates": raw_candidate_records,
                }
                if not eligible:
                    prompt_timings.append(timing_record)
                    continue

                selected = max(
                    eligible,
                    key=lambda item: (
                        item["selection_score"],
                        item["refinement"]["refined_geometry"]["geometry_score"],
                        item["sam_score"],
                        -item["raw_index"],
                    ),
                )
                score = float(selected["sam_score"])
                raw_index = int(selected["raw_index"])
                raw_mask = selected["raw_mask"]
                mask = selected["refined_mask"]
                center = selected["center"]
                refinement = selected["refinement"]
                timing_record["selected_raw_sam_index"] = int(raw_index)
                timing_record["selected_geometry_score"] = float(
                    refinement["refined_geometry"]["geometry_score"]
                )
                timing_record["selected_mask_score"] = float(
                    selected["selection_score"]
                )
                prompt_timings.append(timing_record)
                mask_box = grounding_dino.mask_bbox_xywh_normalized(mask)
                artifact_index = len(selected_masks) + 1
                raw_mask_artifact = output_dir / f"raw_mask_{artifact_index:03d}.png"
                mask_artifact = output_dir / f"mask_{artifact_index:03d}.png"
                Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L").save(
                    raw_mask_artifact
                )
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                    mask_artifact
                )
                selected_masks.append(mask)
                selected_scores.append(float(score))
                selected_boxes.append(mask_box)
                provenance.append(
                    {
                        "candidate_index": len(selected_masks) - 1,
                        "proposal_id": proposal_id,
                        "dino_index": int(proposal["dino_index"]),
                        "dino_phrase": str(proposal["phrase"]),
                        "dino_text_label": str(proposal.get("text_label", "")),
                        "dino_score": float(proposal["dino_score"]),
                        "dino_original_box_xyxy_crop_pixels": list(
                            proposal["original_box_xyxy_crop_pixels"]
                        ),
                        "sam_prompt_box_xyxy_crop_pixels": padded_box,
                        "sam_prompt_box_xywh_normalized": normalized_box,
                        "sam_prompt": "visual",
                        "sam_box_label": 1,
                        "sam_raw_index": int(raw_index),
                        "sam_score": float(score),
                        "mask_geometry_score": float(
                            refinement["refined_geometry"]["geometry_score"]
                        ),
                        "mask_selection_score": float(
                            selected["selection_score"]
                        ),
                        "sam_prompt_elapsed_s": float(elapsed_s),
                        "raw_mask_area_pixels": int(raw_mask.sum()),
                        "raw_mask_artifact": str(raw_mask_artifact),
                        "mask_center_xy_crop_pixels": [
                            float(center[0]),
                            float(center[1]),
                        ],
                        "mask_box_xywh_normalized": mask_box,
                        "mask_area_pixels": int(mask.sum()),
                        "mask_artifact": str(mask_artifact),
                        "mask_refinement": refinement,
                    }
                )
        finally:
            if inference_state is not None:
                del inference_state

        if selected_masks:
            rles = rle_encode(
                torch.as_tensor(np.stack(selected_masks), dtype=torch.bool)
            )
            pred_masks = [rle["counts"] for rle in rles]
        else:
            pred_masks = []
        combined_outputs = {
            "original_image_path": str(image_path_obj),
            "output_image_path": str(output_image_path),
            "orig_img_h": int(height),
            "orig_img_w": int(width),
            "pred_boxes": selected_boxes,
            "pred_masks": pred_masks,
            "pred_scores": selected_scores,
            "proposal_provenance": provenance,
            "sam_box_prompt_timings": prompt_timings,
            "sam_image_state_count": 1,
        }
        output_json_path.write_text(
            json.dumps(combined_outputs, indent=2) + "\n",
            encoding="utf-8",
        )
        if pred_masks:
            visualize(combined_outputs).save(output_image_path)
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


AGENT_TOOL_NAMES = (
    "segment_phrase",
    "examine_each_mask",
    "select_masks_and_return",
    "report_no_mask",
)


def _decode_json_object_at(text: str, start: int) -> tuple[dict[str, Any], int] | None:
    """Decode a JSON object embedded in a larger model response."""

    suffix = text[start:]
    stripped = suffix.lstrip()
    offset = len(suffix) - len(stripped)
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value, start + offset + end


def _first_json_object(text: str) -> dict[str, Any] | None:
    for match in re.finditer(r"\{", text):
        decoded = _decode_json_object_at(text, match.start())
        if decoded is not None:
            return decoded[0]
    return None


def _coerce_tool_call(value: dict[str, Any]) -> dict[str, Any] | None:
    name = value.get("name")
    if isinstance(name, str) and name in AGENT_TOOL_NAMES:
        parameters = value.get("parameters", {})
        if parameters is None:
            parameters = {}
        if isinstance(parameters, dict):
            return {"name": name, "parameters": parameters}
        return None

    if set(value) == {"text_prompt"}:
        return {"name": "segment_phrase", "parameters": value}
    if set(value) == {"final_answer_masks"}:
        return {"name": "select_masks_and_return", "parameters": value}
    return None


def _extract_tool_call_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        decoded = _decode_json_object_at(stripped, 0)
        if decoded is not None:
            direct_call = _coerce_tool_call(decoded[0])
            if direct_call is not None:
                return direct_call

    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    direct_json = _first_json_object(cleaned)
    if direct_json is not None:
        direct_call = _coerce_tool_call(direct_json)
        if direct_call is not None:
            return direct_call

    tool_name_pattern = "|".join(re.escape(name) for name in AGENT_TOOL_NAMES)
    match = re.search(
        rf"<?\s*({tool_name_pattern})(?=\s|\{{|>|/|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    tool_name = next(
        name for name in AGENT_TOOL_NAMES if name.lower() == match.group(1).lower()
    )
    json_start = cleaned.find("{", match.end())
    if json_start == -1:
        return {"name": tool_name, "parameters": {}}

    parameters = _first_json_object(cleaned[json_start:])
    if parameters is None:
        return None
    nested_call = _coerce_tool_call(parameters)
    if nested_call is not None:
        return nested_call
    return {"name": tool_name, "parameters": parameters}


def normalize_agent_tool_response(text: str) -> str:
    """Normalize Qwen's occasional XML-ish tool call into SAM agent JSON."""

    tool_match = re.search(
        r"<tool\b[^>]*>(.*?)</tool>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    tool_body = tool_match.group(1) if tool_match is not None else text
    tool_call = _extract_tool_call_from_text(tool_body)
    if tool_call is None:
        return text

    tool_json = json.dumps(tool_call, ensure_ascii=False)
    normalized_tool = f"<tool> {tool_json} </tool>"
    if tool_match is None:
        return normalized_tool
    return text[: tool_match.start()] + normalized_tool + text[tool_match.end() :]


def build_qwen_sender(args: argparse.Namespace):
    def send_generate_request(messages: list[dict[str, Any]]) -> str:
        generated_text = local_qwen.qwen_generate(
            messages,
            model_id=args.qwen_model,
            max_new_tokens=args.qwen_max_new_tokens,
            local_files_only=not args.allow_qwen_downloads,
            device_map=args.qwen_device_map,
        )
        normalized_text = normalize_agent_tool_response(generated_text)
        if normalized_text != generated_text:
            print("normalized_qwen_tool_call=true")
        return normalized_text

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
