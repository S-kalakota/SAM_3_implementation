#!/usr/bin/env python3
"""Create the approval artifact required for the SAM image-backend switch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

import sam3_image_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-frame masks emitted by the current multiplex adapter "
            "and the Sam3Processor image adapter."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-iou", default=0.95, type=float)
    return parser.parse_args()


def _read_mask(path: str, manifest_dir: Path):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = manifest_dir / resolved
    image = cv2.imread(str(resolved.resolve()), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read parity mask {resolved}")
    return image > 0


def evaluate_manifest(manifest: dict[str, Any], manifest_dir: Path, min_iou: float) -> dict[str, Any]:
    if set(manifest) != {"schema_version", "cases"} or manifest["schema_version"] != 1:
        raise ValueError("parity manifest requires schema_version=1 and cases")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("parity manifest must contain at least one case")
    reports = []
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "legacy_masks",
            "image_model_masks",
        }:
            raise ValueError("each parity case has invalid keys")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("parity case ids must be unique non-empty strings")
        seen_ids.add(case_id)
        legacy = [_read_mask(path, manifest_dir) for path in case["legacy_masks"]]
        image = [_read_mask(path, manifest_dir) for path in case["image_model_masks"]]
        report = sam3_image_service.frozen_frame_parity_report(
            legacy,
            image,
            min_iou=min_iou,
        )
        reports.append({"case_id": case_id, **report})
    approved = all(report["approved"] for report in reports)
    return {
        "schema_version": 1,
        "approved": approved,
        "cases": len(reports),
        "min_iou": float(min_iou),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_reports": reports,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = evaluate_manifest(manifest, manifest_path.parent, args.min_iou)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
