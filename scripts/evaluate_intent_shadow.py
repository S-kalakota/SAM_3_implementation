#!/usr/bin/env python3
"""Compare deterministic and Qwen structured-intent parsing on 30-50 commands."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grounding_intent
import local_qwen
import mask_service


def load_commands(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "name", "commands"
    }:
        raise ValueError("shadow corpus keys are invalid")
    if value["schema_version"] != 1:
        raise ValueError("shadow corpus schema_version must be 1")
    commands = value["commands"]
    if not isinstance(commands, list) or not 30 <= len(commands) <= 50:
        raise ValueError("shadow corpus must contain 30 to 50 commands")
    for command in commands:
        if (
            not isinstance(command, dict)
            or set(command) != {"text", "accepted"}
            or not isinstance(command["text"], str)
            or not isinstance(command["accepted"], bool)
        ):
            raise ValueError(f"invalid shadow command: {command!r}")
    return {**value, "path": str(path)}


def parse_one_deterministically(text: str) -> dict[str, Any]:
    try:
        intent = grounding_intent.parse_grounding_intent(text)
        return {
            "accepted": True,
            "intent": intent,
            "intent_hash": grounding_intent.intent_hash(intent),
            "error": None,
        }
    except grounding_intent.GroundingIntentError as exc:
        return {
            "accepted": False,
            "intent": None,
            "intent_hash": None,
            "error": {"code": exc.code, "message": str(exc)},
        }


def qwen_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        qwen_model=args.qwen_model,
        intent_max_new_tokens=args.max_new_tokens,
        allow_qwen_downloads=args.allow_downloads,
        qwen_device_map=args.device_map,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands",
        type=Path,
        default=Path("evaluation/intent_shadow_commands.json"),
    )
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--qwen-model", default=local_qwen.DEFAULT_QWEN_MODEL)
    parser.add_argument("--device-map", default=local_qwen.DEFAULT_DEVICE_MAP)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/intent_shadow_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_commands(args.commands)
    rows = []
    for command in corpus["commands"]:
        deterministic = parse_one_deterministically(command["text"])
        qwen_intent = None
        qwen_record = None
        if args.qwen:
            qwen_intent, qwen_record = mask_service.qwen_grounding_intent(
                command["text"],
                qwen_args(args),
            )
        comparison = None
        if deterministic["intent"] is not None:
            comparison = grounding_intent.compare_intents(
                deterministic["intent"],
                qwen_intent,
                shadow_error=None
                if qwen_intent is not None
                else str((qwen_record or {}).get("error")),
            )
        rows.append(
            {
                "text": command["text"],
                "expected_accepted": command["accepted"],
                "deterministic": deterministic,
                "qwen_intent": qwen_intent,
                "qwen_record": qwen_record,
                "comparison": comparison,
            }
        )
    report = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus["path"],
        "qwen_enabled": bool(args.qwen),
        "qwen_model": args.qwen_model if args.qwen else None,
        "summary": {
            "command_count": len(rows),
            "deterministic_expectation_matches": sum(
                row["deterministic"]["accepted"] == row["expected_accepted"]
                for row in rows
            ),
            "qwen_schema_valid": sum(row["qwen_intent"] is not None for row in rows),
            "active_shadow_exact_matches": sum(
                bool((row["comparison"] or {}).get("matches")) for row in rows
            ),
        },
        "commands": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
