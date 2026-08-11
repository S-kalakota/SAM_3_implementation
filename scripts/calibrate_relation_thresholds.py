#!/usr/bin/env python3
"""Grid-search relationship thresholds with a zero-safety-acceptance constraint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import relation_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _resolve(path: str, root: Path) -> Path:
    value = Path(path).expanduser()
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def _mask(path: str, root: Path) -> np.ndarray:
    image = cv2.imread(str(_resolve(path, root)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read mask {path}")
    return image > 0


def _array(path: str | None, root: Path) -> np.ndarray | None:
    return None if path is None else np.load(_resolve(path, root), allow_pickle=False)


def load_cases(manifest: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, list[float | int]]]:
    if set(manifest) != {"schema_version", "threshold_grid", "cases"}:
        raise ValueError("calibration manifest keys are invalid")
    if manifest["schema_version"] != 1:
        raise ValueError("calibration schema_version must be 1")
    grid = manifest["threshold_grid"]
    if not isinstance(grid, dict) or not grid:
        raise ValueError("threshold_grid must be a non-empty object")
    unknown_thresholds = set(grid) - set(relation_geometry.DEFAULT_THRESHOLDS)
    if unknown_thresholds:
        raise ValueError(f"unknown calibration thresholds: {sorted(unknown_thresholds)}")
    for name, values in grid.items():
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            )
        ):
            raise ValueError(f"threshold grid {name} must contain numeric values")
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("calibration cases must be a non-empty list")
    cases = []
    seen_ids: set[str] = set()
    for raw in manifest["cases"]:
        required = {
            "case_id",
            "relationship",
            "target_mask",
            "anchor_mask",
            "depth",
            "xyz",
            "expected_pass",
            "safety_case",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("calibration case keys are invalid")
        if (
            not isinstance(raw["case_id"], str)
            or not raw["case_id"].strip()
            or raw["case_id"] in seen_ids
        ):
            raise ValueError("calibration case ids must be unique non-empty strings")
        if raw["relationship"] not in relation_geometry.SUPPORTED_RELATIONSHIPS:
            raise ValueError(f"unsupported relationship {raw['relationship']!r}")
        if not isinstance(raw["expected_pass"], bool) or not isinstance(
            raw["safety_case"], bool
        ):
            raise ValueError("expected_pass and safety_case must be booleans")
        if not isinstance(raw["target_mask"], str) or not isinstance(
            raw["anchor_mask"], str
        ):
            raise ValueError("calibration masks must be paths")
        if raw["depth"] is not None and not isinstance(raw["depth"], str):
            raise ValueError("calibration depth must be a path or null")
        if raw["xyz"] is not None and not isinstance(raw["xyz"], str):
            raise ValueError("calibration xyz must be a path or null")
        seen_ids.add(raw["case_id"])
        cases.append(
            {
                "case_id": raw["case_id"],
                "relationship": raw["relationship"],
                "target_mask": _mask(raw["target_mask"], root),
                "anchor_mask": _mask(raw["anchor_mask"], root),
                "depth": _array(raw["depth"], root),
                "xyz": _array(raw["xyz"], root),
                "expected_pass": raw["expected_pass"],
                "safety_case": raw["safety_case"],
            }
        )
    return cases, grid


def main() -> None:
    args = parse_args()
    path = args.manifest.expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases, grid = load_cases(manifest, path.parent)
    result = relation_geometry.grid_search_thresholds(cases, grid)
    relationship_case_counts = {
        relationship: sum(
            case["relationship"] == relationship for case in cases
        )
        for relationship in sorted(relation_geometry.SUPPORTED_RELATIONSHIPS)
    }
    relationship_safety_case_counts = {
        relationship: sum(
            case["relationship"] == relationship and case["safety_case"]
            for case in cases
        )
        for relationship in sorted(relation_geometry.SUPPORTED_RELATIONSHIPS)
    }
    report = {
        "schema_version": 1,
        "cases": len(cases),
        "relationship_case_counts": relationship_case_counts,
        "relationship_safety_case_counts": relationship_safety_case_counts,
        **result,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
