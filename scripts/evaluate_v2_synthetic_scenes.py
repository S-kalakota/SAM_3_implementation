#!/usr/bin/env python3
"""Exercise all relationship gates on the checked-in 60-scene safety fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import relation_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = PROJECT_ROOT / "evaluation/v2_synthetic_frozen_scenes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=DEFAULT_SCENES, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_scenes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "name",
        "synthetic",
        "limitations",
        "cases",
    }:
        raise ValueError("synthetic scene fixture keys are invalid")
    if value["schema_version"] != 1 or value["synthetic"] is not True:
        raise ValueError("synthetic scene fixture identity is invalid")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) < 60:
        raise ValueError("synthetic frozen fixture must contain at least 60 scenes")
    return cases


def _instance_mask(
    shape: tuple[int, int],
    instances: list[list[int]],
    *,
    first_only: bool,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    selected = instances[:1] if first_only else instances
    for x0, y0, x1, y1 in selected:
        mask[y0:y1, x0:x1] = True
    return mask


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    correct = 0
    false_safety_accepts = 0
    for case in cases:
        width, height = case["image_size_wh"]
        shape = (height, width)
        target = _instance_mask(
            shape,
            case["target_instances_xyxy"],
            first_only=True,
        )
        anchor = _instance_mask(
            shape,
            case["anchor_instances_xyxy"],
            first_only=True,
        )
        depth = np.full(shape, np.nan, dtype=np.float32)
        depth[anchor] = float(case["anchor_depth_m"])
        depth[target] = float(case["target_depth_m"])
        result = relation_geometry.evaluate_relationship(
            case["relationship"],
            target,
            anchor,
            depth=depth,
            thresholds={"depth_min_valid_pixels": 5},
        )
        geometry_correct = result["status"] == case["expected_geometry_status"]
        correct += int(geometry_correct)
        deterministic_final_accept = (
            result["status"] == "pass"
            and case["counterfactual"] not in {
                "swapped_target_anchor_attributes",
                "multiple_matching_entities",
            }
        )
        if case["safety_case"] and deterministic_final_accept:
            false_safety_accepts += 1
        records.append(
            {
                "case_id": case["case_id"],
                "geometry_correct": geometry_correct,
                "expected_geometry_status": case["expected_geometry_status"],
                "actual_geometry_status": result["status"],
                "expected_final_accept": case["expected_final_accept"],
                "geometry": result,
            }
        )
    return {
        "schema_version": 1,
        "cases": len(cases),
        "geometry_correct": correct,
        "geometry_accuracy": correct / len(cases),
        # Attribute swaps and duplicate entities are Qwen/ambiguity safety gates,
        # so this metric intentionally counts only what geometry could accept.
        "geometry_only_false_safety_accepts": false_safety_accepts,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    path = args.scenes.expanduser().resolve()
    scenes = json.loads(path.read_text(encoding="utf-8"))
    report = evaluate_cases(validate_scenes(scenes))
    output = args.output or PROJECT_ROOT / "outputs/v2_synthetic_scene_evaluation.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2))
    if report["geometry_accuracy"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
