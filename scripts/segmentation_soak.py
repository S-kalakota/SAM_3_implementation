#!/usr/bin/env python3
"""Run a no-motion 20-round live segmentation consistency/absence soak."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import grounding_intent
import grounding_v2
import mask_client


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read reference mask {path}")
    return image > 0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError(f"mask shapes differ: {first.shape} != {second.shape}")
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(first & second)) / union


def score_round(
    response: dict[str, Any],
    *,
    expected_intent: dict[str, Any] | None,
    expected_hash: str | None,
    expected: str,
    reference_mask: np.ndarray | None,
    min_iou: float,
    raw_command: str | None = None,
    expected_relationship: str | None = None,
) -> dict[str, Any]:
    relationship_types: list[str] = []
    if response.get("schema_version") == 2:
        try:
            envelope = grounding_v2.validate_command_envelope(
                response.get("command_envelope"),
                expected_raw_command=raw_command,
            )
            identity_ok = expected_hash is None or envelope["envelope_hash"] == expected_hash
            relationship_types = sorted(grounding_v2.relationship_types(envelope))
        except grounding_v2.GroundingV2Error:
            identity_ok = False
        selected = response.get("target_mask")
    else:
        identity_ok = (
            response.get("grounding_intent") == expected_intent
            and response.get("intent_hash") == expected_hash
        )
        selected = response.get("selected_mask")
    relationship_ok = (
        expected_relationship is None
        or expected_relationship in relationship_types
    )
    selected_count = int(response.get("num_kept", 0))
    iou = None
    if reference_mask is not None and isinstance(selected, dict):
        predicted = read_mask(Path(selected["mask_full"]))
        iou = mask_iou(predicted, reference_mask)
    if expected == "absent":
        passed = identity_ok and relationship_ok and selected_count == 0
    elif reference_mask is not None:
        passed = (
            identity_ok
            and relationship_ok
            and selected_count == 1
            and iou is not None
            and iou >= min_iou
        )
    else:
        passed = identity_ok and relationship_ok and selected_count == 1
    verification = response.get("verification") or {}
    malformed_attempts = sum(
        1 for attempt in verification.get("attempts", []) if attempt.get("error")
    )
    return {
        "passed": passed,
        "identity_ok": identity_ok,
        "relationship_ok": relationship_ok,
        "relationship_types": relationship_types,
        "selected_count": selected_count,
        "mask_iou": iou,
        "verification_status": verification.get("status"),
        "malformed_verifier_attempts": malformed_attempts,
        "elapsed_s": response.get("elapsed_s"),
        "result_json": response.get("result_json"),
        "overlay": (
            (response.get("overlay") or {}).get("output")
            if isinstance(response.get("overlay"), dict)
            else (response.get("audit_artifacts") or {}).get("candidate_overlay")
        ),
    }


def summarize(rounds: list[dict[str, Any]], expected: str) -> dict[str, Any]:
    return {
        "round_count": len(rounds),
        "passed_rounds": sum(item["passed"] for item in rounds),
        "failed_rounds": sum(not item["passed"] for item in rounds),
        "identity_failures": sum(not item["identity_ok"] for item in rounds),
        "relationship_failures": sum(
            not item.get("relationship_ok", True) for item in rounds
        ),
        "malformed_verifier_attempts": sum(
            item["malformed_verifier_attempts"] for item in rounds
        ),
        "wrong_acceptances_when_absent": (
            sum(item["selected_count"] > 0 for item in rounds)
            if expected == "absent" else None
        ),
        "acceptance_19_of_20": len(rounds) == 20 and sum(
            item["passed"] for item in rounds
        ) >= 19,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="+", help="Visual target phrase or command")
    parser.add_argument("--host", default="127.0.0.1:8765")
    parser.add_argument("--rounds", default=20, type=int)
    parser.add_argument("--expected", choices=("present", "absent"), required=True)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--min-iou", default=0.5, type=float)
    parser.add_argument("--interval", default=0.2, type=float)
    parser.add_argument("--timeout", default=600.0, type=float)
    parser.add_argument(
        "--relationship",
        choices=sorted(grounding_v2.VALID_RELATIONSHIPS),
        help="Relationship represented by this live release-gate soak.",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Run the retained version-1 soak instead of the default v2 path.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/segmentation_soak.json")
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not 0.0 < args.min_iou <= 1.0:
        parser.error("--min-iou must be in (0, 1]")
    if args.interval < 0.0:
        parser.error("--interval must not be negative")
    if args.expected == "absent" and args.reference_mask is not None:
        parser.error("--reference-mask is not used with --expected absent")
    if args.v1 and args.relationship is not None:
        parser.error("--relationship is available only for v2 live reports")
    return args


def main() -> None:
    args = parse_args()
    request = " ".join(args.request)
    intent = grounding_intent.parse_grounding_intent(request) if args.v1 else None
    expected_hash = grounding_intent.intent_hash(intent) if intent is not None else None
    reference = read_mask(args.reference_mask) if args.reference_mask else None
    rows = []
    for index in range(args.rounds):
        response = mask_client.segment(
            request,
            host=args.host,
            timeout=args.timeout,
            v1=args.v1,
        )
        if not args.v1 and expected_hash is None:
            envelope = response.get("command_envelope") or {}
            expected_hash = envelope.get("envelope_hash")
        row = score_round(
            response,
            expected_intent=intent,
            expected_hash=expected_hash,
            expected=args.expected,
            reference_mask=reference,
            min_iou=args.min_iou,
            raw_command=request,
            expected_relationship=args.relationship,
        )
        row["round"] = index + 1
        rows.append(row)
        print(
            f"round={index + 1} passed={row['passed']} "
            f"selected={row['selected_count']} iou={row['mask_iou']}"
        )
        if index + 1 < args.rounds and args.interval:
            time.sleep(args.interval)
    report = {
        "schema_version": 1 if args.v1 else 2,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": request,
        "relationship": args.relationship,
        "grounding_intent": intent,
        "identity_hash": expected_hash,
        "intent_hash": expected_hash if args.v1 else None,
        "envelope_hash": expected_hash if not args.v1 else None,
        "expected": args.expected,
        "reference_mask": None
        if args.reference_mask is None
        else str(args.reference_mask.expanduser().resolve()),
        "min_iou": args.min_iou,
        "summary": summarize(rows, args.expected),
        "rounds": rows,
        "note": (
            "Without --reference-mask, present-target rounds measure stable single "
            "acceptance and identity, not visual correctness."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
