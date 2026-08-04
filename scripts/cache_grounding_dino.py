#!/usr/bin/env python3
"""Populate and validate the offline Grounding DINO Transformers snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import grounding_dino


ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Grounding DINO safetensors and processor files into the "
            "Hugging Face cache before the robot service is put offline."
        )
    )
    parser.add_argument("--model", default=grounding_dino.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory (defaults to HF_HOME).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This is the only project command that is expected to use the network.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            allow_patterns=list(ALLOW_PATTERNS),
            local_files_only=False,
        )
    ).resolve()
    safetensors = sorted(snapshot.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(f"No safetensors were cached in {snapshot}")
    if not (snapshot / "config.json").is_file():
        raise RuntimeError(f"Cached model config is missing from {snapshot}")
    if not (snapshot / "preprocessor_config.json").is_file():
        raise RuntimeError(f"Cached processor config is missing from {snapshot}")

    # Validate configs and tokenizer/processor strictly from the completed snapshot.
    AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    print(
        json.dumps(
            {
                "model_id": args.model,
                "snapshot": str(snapshot),
                "safetensors": [path.name for path in safetensors],
                "processor_class": type(processor).__name__,
                "ready_for_offline_startup": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
