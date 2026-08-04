#!/usr/bin/env python3
"""Warm SAM 3 image-model adapter with reusable per-view embeddings."""

from __future__ import annotations

import contextlib
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class Sam3ImageService:
    """Serve text and box prompts without recomputing an image embedding."""

    backend_name = "sam3_image_processor"

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        threshold: float,
        det_threshold: float | None = None,
        use_fa3: bool = False,
        verbose_load: bool = False,
        device: str | None = None,
        max_cached_views: int = 24,
    ) -> None:
        del use_fa3  # The image builder selects its own supported attention path.
        checkpoint = checkpoint_path.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if max_cached_views < 1:
            raise ValueError("max_cached_views must be positive")

        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("high")
        if resolved_device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        build_kwargs = {
            "checkpoint_path": str(checkpoint),
            "load_from_HF": False,
            "device": resolved_device,
            "eval_mode": True,
            "enable_segmentation": True,
            "enable_inst_interactivity": True,
            "compile": False,
        }
        if verbose_load:
            model = build_sam3_image_model(**build_kwargs)
        else:
            load_log = io.StringIO()
            with contextlib.redirect_stdout(load_log):
                model = build_sam3_image_model(**build_kwargs)
            if load_log.tell():
                print(f"suppressed_image_model_load_log_chars={load_log.tell()}")
        if det_threshold is not None and hasattr(model, "score_threshold_detection"):
            model.score_threshold_detection = det_threshold
        self.model = model
        self.processor = Sam3Processor(
            model,
            device=resolved_device,
            confidence_threshold=threshold,
        )
        self.device = resolved_device
        self.threshold = float(threshold)
        self.max_cached_views = int(max_cached_views)
        self._states: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
        self._embedding_computations = 0
        self._embedding_cache_hits = 0

    @staticmethod
    def _cache_key(image_path: str | Path) -> tuple[str, int, int]:
        path = Path(image_path).expanduser().resolve()
        stat = path.stat()
        return str(path), int(stat.st_mtime_ns), int(stat.st_size)

    def prepare_image(self, image_path: str | Path) -> dict[str, Any]:
        import torch

        key = self._cache_key(image_path)
        if key in self._states:
            state = self._states.pop(key)
            self._states[key] = state
            self._embedding_cache_hits += 1
            return state
        with Image.open(key[0]) as source_image:
            image = source_image.convert("RGB")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast:
            state = self.processor.set_image(image)
        state["_sam3_image_cache_key"] = key
        self._states[key] = state
        self._embedding_computations += 1
        while len(self._states) > self.max_cached_views:
            self._states.popitem(last=False)
        return state

    def clear_embedding_cache(self) -> None:
        self._states.clear()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "cached_views": len(self._states),
            "max_cached_views": self.max_cached_views,
            "embedding_computations": self._embedding_computations,
            "embedding_cache_hits": self._embedding_cache_hits,
            "instance_interactivity_enabled": self.model.inst_interactive_predictor
            is not None,
        }

    def predict_text(self, image_path: str | Path, prompt: str) -> dict[str, Any]:
        import torch

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("text prompt must be non-empty")
        state = self.prepare_image(image_path)
        self.processor.reset_all_prompts(state)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast:
            output = self.processor.set_text_prompt(prompt=prompt, state=state)
        masks = _to_numpy(output.get("masks", [])).astype(bool, copy=False)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if masks.size == 0:
            height = int(state["original_height"])
            width = int(state["original_width"])
            masks = np.zeros((0, height, width), dtype=bool)
        scores = _to_numpy(output.get("scores", [])).astype(np.float32).reshape(-1)
        boxes = _to_numpy(output.get("boxes", [])).astype(np.float32).reshape(-1, 4)
        return {"masks": masks, "scores": scores, "boxes_xyxy": boxes}

    def predict_boxes(
        self,
        image_path: str | Path,
        boxes_xyxy: Iterable[Iterable[float]],
    ) -> list[dict[str, Any]]:
        """Refine multiple candidate boxes in one interactive decoder call."""

        import torch

        state = self.prepare_image(image_path)
        boxes = np.asarray(list(boxes_xyxy), dtype=np.float32)
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape Nx4")
        height = int(state["original_height"])
        width = int(state["original_width"])
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError("every box must have positive width and height")

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast:
            masks, qualities, _logits = self.model.predict_inst(
                state,
                box=boxes,
                multimask_output=False,
            )
        masks = np.asarray(masks, dtype=bool)
        qualities = np.asarray(qualities, dtype=np.float32)
        if boxes.shape[0] == 1:
            if masks.ndim == 2:
                masks = masks[None, None]
            elif masks.ndim == 3:
                masks = masks[None]
            if qualities.ndim == 1:
                qualities = qualities[None]
        elif masks.ndim == 3:
            masks = masks[:, None]
        if masks.shape[0] != boxes.shape[0]:
            raise RuntimeError(
                f"interactive SAM returned {masks.shape[0]} batches for {len(boxes)} boxes"
            )
        records: list[dict[str, Any]] = []
        for index, box in enumerate(boxes):
            candidate_masks = masks[index]
            candidate_scores = np.asarray(qualities[index]).reshape(-1)
            if candidate_masks.ndim == 2:
                candidate_masks = candidate_masks[None]
            best = int(np.argmax(candidate_scores)) if candidate_scores.size else 0
            score = float(candidate_scores[best]) if candidate_scores.size else 0.0
            records.append(
                {
                    "box_xyxy": box.astype(float).tolist(),
                    "mask": np.asarray(candidate_masks[best], dtype=bool),
                    "score": score,
                }
            )
        return records

    def __call__(
        self,
        *,
        image_path: str,
        text_prompt: str,
        output_folder_path: str,
    ) -> str:
        """Compatibility adapter for the existing candidate/agent call shape."""

        import torch
        from sam3.train.masks_ops import rle_encode

        result = self.predict_text(image_path, text_prompt)
        masks = result["masks"]
        scores = result["scores"]
        source = Path(image_path).expanduser().resolve()
        output_dir = Path(output_folder_path).expanduser().resolve() / _safe_name(source.stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(text_prompt)
        json_path = output_dir / f"{stem}.json"
        render_path = output_dir / f"{stem}.png"

        valid = [
            index
            for index, mask in enumerate(masks)
            if int(np.count_nonzero(mask)) > 0 and index < len(scores)
        ]
        order = sorted(valid, key=lambda index: float(scores[index]), reverse=True)
        pred_scores = [float(scores[index]) for index in order]
        pred_boxes: list[list[float]] = []
        height, width = masks.shape[1:] if masks.ndim == 3 else Image.open(source).size[::-1]
        for index in order:
            x0, y0, x1, y1 = _mask_bbox_xyxy(masks[index])
            pred_boxes.append(
                [
                    x0 / width,
                    y0 / height,
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ]
            )
        if order:
            rles = rle_encode(torch.as_tensor(masks[order], dtype=torch.bool))
            pred_masks = [item["counts"] for item in rles]
        else:
            pred_masks = []
        payload = {
            "original_image_path": str(source),
            "output_image_path": str(render_path),
            "orig_img_h": int(height),
            "orig_img_w": int(width),
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
            "sam_backend": self.backend_name,
            "embedding_cache": self.cache_stats(),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if pred_masks:
            from sam3.agent.viz import visualize

            visualize(payload).save(render_path)
        else:
            Image.open(source).convert("RGB").save(render_path)
        return str(json_path)

    def close(self) -> None:
        import torch

        self.clear_embedding_cache()
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("_") or "prompt"


def mask_iou(first: Any, second: Any) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    if first_mask.shape != second_mask.shape:
        raise ValueError("mask shapes must match")
    union = np.count_nonzero(first_mask | second_mask)
    return 1.0 if union == 0 else float(np.count_nonzero(first_mask & second_mask) / union)


def _threshold_bipartite_matching(
    similarities: np.ndarray,
    minimum: float,
) -> dict[int, int]:
    """Find a maximum-cardinality one-to-one matching above a threshold."""

    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarities must be a 2-D matrix")
    image_to_legacy: dict[int, int] = {}

    def augment(legacy_index: int, visited: set[int]) -> bool:
        ranked = sorted(
            (
                (float(matrix[legacy_index, image_index]), image_index)
                for image_index in range(matrix.shape[1])
                if matrix[legacy_index, image_index] >= minimum
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _similarity, image_index in ranked:
            if image_index in visited:
                continue
            visited.add(image_index)
            prior = image_to_legacy.get(image_index)
            if prior is None or augment(prior, visited):
                image_to_legacy[image_index] = legacy_index
                return True
        return False

    for legacy_index in range(matrix.shape[0]):
        augment(legacy_index, set())
    return {legacy_index: image_index for image_index, legacy_index in image_to_legacy.items()}


def frozen_frame_parity_report(
    legacy_masks: Iterable[Any],
    image_model_masks: Iterable[Any],
    *,
    min_iou: float = 0.95,
) -> dict[str, Any]:
    """Match mask sets greedily and report whether the backend-switch gate passes."""

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be in [0, 1]")
    legacy = [np.asarray(mask, dtype=bool) for mask in legacy_masks]
    image = [np.asarray(mask, dtype=bool) for mask in image_model_masks]
    similarities = np.zeros((len(legacy), len(image)), dtype=np.float64)
    for legacy_index, legacy_mask in enumerate(legacy):
        for image_index, image_mask in enumerate(image):
            similarities[legacy_index, image_index] = mask_iou(legacy_mask, image_mask)
    matching = _threshold_bipartite_matching(similarities, min_iou)
    matches: list[dict[str, Any]] = []
    for legacy_index in range(len(legacy)):
        image_index = matching.get(legacy_index)
        if image_index is None:
            matches.append(
                {"legacy_index": legacy_index, "image_index": None, "iou": 0.0}
            )
            continue
        matches.append(
            {
                "legacy_index": legacy_index,
                "image_index": image_index,
                "iou": float(similarities[legacy_index, image_index]),
            }
        )
    unmatched = set(range(len(image))) - set(matching.values())
    passed = (
        len(legacy) == len(image)
        and len(matching) == len(legacy)
        and not unmatched
    )
    return {
        "approved": passed,
        "min_iou": float(min_iou),
        "legacy_mask_count": len(legacy),
        "image_model_mask_count": len(image),
        "matches": matches,
        "unmatched_image_indices": sorted(unmatched),
    }
