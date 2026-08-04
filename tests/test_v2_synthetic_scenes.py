import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_v2_synthetic_scenes


def test_sixty_scene_fixture_covers_each_relation_and_safety_counterfactual():
    value = json.loads(
        (PROJECT_ROOT / "evaluation/v2_synthetic_frozen_scenes.json").read_text()
    )
    cases = evaluate_v2_synthetic_scenes.validate_scenes(value)
    assert len(cases) == 60
    assert len({case["relationship"] for case in cases}) == 10
    assert len({case["counterfactual"] for case in cases}) == 6
    assert sum(case["safety_case"] for case in cases) == 50
    report = evaluate_v2_synthetic_scenes.evaluate_cases(cases)
    assert report["geometry_accuracy"] == 1.0
    assert report["geometry_only_false_safety_accepts"] == 0
