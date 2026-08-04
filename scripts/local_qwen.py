#!/usr/bin/env python3
"""Local Qwen-VL adapter for the SAM 3.1 agent.

This module intentionally loads only from the local Hugging Face cache by default.
It does not download model weights.
"""

from __future__ import annotations

import contextlib
import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_QWEN_MODEL = os.environ.get(
    "SAM3_AGENT_QWEN_MODEL_ID",
    os.environ.get("CO_BOT_VLM_QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
)
DEFAULT_DEVICE_MAP = os.environ.get("SAM3_AGENT_QWEN_DEVICE_MAP", "auto")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("SAM3_AGENT_QWEN_MAX_NEW_TOKENS", "512"))


def _repetition_penalty_default() -> float | None:
    # The model's shipped generation_config already applies 1.05; set this env
    # var (e.g. 1.15) to damp the repetition loops small Qwen models fall into.
    raw = os.environ.get("SAM3_AGENT_QWEN_REPETITION_PENALTY", "").strip()
    return float(raw) if raw else None


def _local_only_default() -> bool:
    raw = os.environ.get("SAM3_AGENT_QWEN_LOCAL_ONLY", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _image_to_qwen_ref(image: Any) -> str:
    if isinstance(image, Path):
        return str(image.expanduser().resolve())
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return image
        if image.startswith("file://"):
            return image[7:]
        return str(Path(image).expanduser().resolve())
    raise TypeError(f"Unsupported image reference type for Qwen: {type(image)!r}")


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    normalized = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        if item.get("type") == "image":
            normalized.append({"type": "image", "image": _image_to_qwen_ref(item["image"])})
        elif item.get("type") == "image_url":
            normalized.append(item)
        elif item.get("type") == "text":
            normalized.append({"type": "text", "text": str(item.get("text", ""))})
        else:
            normalized.append(item)
    return normalized


def _normalize_messages(messages: list[dict[str, Any]], images: list[Any] | None = None) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": message["role"],
                "content": _normalize_content(message.get("content", "")),
            }
        )

    if images:
        image_items = [{"type": "image", "image": _image_to_qwen_ref(image)} for image in images]
        for message in reversed(normalized):
            if message["role"] == "user":
                content = message.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                message["content"] = image_items + list(content)
                break
        else:
            normalized.append({"role": "user", "content": image_items})

    return normalized


@lru_cache(maxsize=2)
def _load_qwen(model_id: str, local_files_only: bool, device_map: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device_map,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, processor


def preload_qwen(
    model_id: str = DEFAULT_QWEN_MODEL,
    *,
    local_files_only: bool | None = None,
    device_map: str = DEFAULT_DEVICE_MAP,
) -> dict[str, Any]:
    """Load the configured model once and return compact runtime metadata."""

    local_only = _local_only_default() if local_files_only is None else local_files_only
    model, _processor = _load_qwen(model_id, local_only, device_map)
    constraint_data = _lmfe_tokenizer_data(model_id, local_only, device_map)
    hf_device_map = getattr(model, "hf_device_map", {})
    devices = sorted({str(value) for value in hf_device_map.values()})
    if not devices and hasattr(model, "device"):
        devices = [str(model.device)]
    first_parameter = next(model.parameters(), None)
    dtype = None if first_parameter is None else str(first_parameter.dtype)
    return {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "dtype": dtype,
        "devices": devices,
        "local_files_only": bool(local_only),
        "json_schema_constraint_ready": True,
        "constraint_vocab_size": int(constraint_data.vocab_size),
    }


@lru_cache(maxsize=2)
def _lmfe_tokenizer_data(model_id: str, local_files_only: bool, device_map: str):
    """Build reusable LM Format Enforcer tokenizer data for Transformers 5.x."""

    try:
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Schema-constrained Qwen generation requires lm-format-enforcer. "
            "Install it with `.venv/bin/python -m pip install -r "
            "requirements.mask-service.txt`."
        ) from exc

    _model, processor = _load_qwen(model_id, local_files_only, device_map)
    return build_token_enforcer_tokenizer_data(processor.tokenizer)


def _json_schema_prefix_allowed_tokens_fn(
    schema: dict[str, Any],
    *,
    model_id: str,
    local_files_only: bool,
    device_map: str,
):
    """Build one request-scoped JSON grammar over cached tokenizer metadata."""

    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Schema-constrained Qwen generation requires lm-format-enforcer. "
            "Install it with `.venv/bin/python -m pip install -r "
            "requirements.mask-service.txt`."
        ) from exc
    tokenizer_data = _lmfe_tokenizer_data(model_id, local_files_only, device_map)
    parser = JsonSchemaParser(copy.deepcopy(schema))
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)


def qwen_generate(
    messages: list[dict[str, Any]],
    images: list[Any] | None = None,
    *,
    model_id: str = DEFAULT_QWEN_MODEL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    local_files_only: bool | None = None,
    device_map: str = DEFAULT_DEVICE_MAP,
    repetition_penalty: float | None = None,
    do_sample: bool | None = None,
    response_prefix: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> str:
    """Generate text from local cached Qwen-VL for Meta's SAM3 agent messages."""

    local_only = _local_only_default() if local_files_only is None else local_files_only
    if response_prefix and json_schema is not None:
        raise ValueError("response_prefix and json_schema cannot be combined")

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "qwen-vl-utils is not installed in this environment."
        ) from exc

    try:
        model, processor = _load_qwen(model_id, local_only, device_map)
    except Exception as exc:
        mode = "local Hugging Face cache" if local_only else "configured model source"
        raise RuntimeError(
            f"Could not load Qwen-VL model {model_id!r} from the {mode}: {exc}"
        ) from exc

    qwen_messages = _normalize_messages(messages, images=images)
    with contextlib.redirect_stdout(_StdoutToStderr()):
        prompt = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if response_prefix:
            prompt += response_prefix
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    penalty = (
        _repetition_penalty_default() if repetition_penalty is None else repetition_penalty
    )
    if penalty is not None:
        gen_kwargs["repetition_penalty"] = penalty
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    if json_schema is not None:
        gen_kwargs["prefix_allowed_tokens_fn"] = _json_schema_prefix_allowed_tokens_fn(
            json_schema,
            model_id=model_id,
            local_files_only=local_only,
            device_map=device_map,
        )
    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not output_text:
        raise RuntimeError("Qwen-VL returned no text.")
    return (response_prefix or "") + output_text[0]


class _StdoutToStderr:
    def write(self, text: str) -> int:
        import sys

        return sys.stderr.write(text)

    def flush(self) -> None:
        import sys

        sys.stderr.flush()
