import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import relation_geometry


def rectangle(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_inside_pass_and_fail_measure_fraction_and_center():
    anchor = rectangle((100, 100), 20, 20, 80, 80)
    inside = rectangle((100, 100), 35, 35, 50, 50)
    outside = rectangle((100, 100), 75, 75, 95, 95)
    depth = np.ones((100, 100), dtype=np.float32)
    passed = relation_geometry.evaluate_relationship("inside", inside, anchor, depth=depth)
    failed = relation_geometry.evaluate_relationship("inside", outside, anchor, depth=depth)
    assert passed["status"] == "pass"
    assert passed["measurements"]["target_fraction_inside_anchor_bounds"] == 1.0
    assert failed["status"] == "fail"


def test_inside_uses_compatible_depth_when_reliable():
    anchor = rectangle((40, 40), 5, 5, 35, 35)
    target = rectangle((40, 40), 15, 15, 25, 25)
    depth = np.full((40, 40), np.nan, dtype=np.float32)
    depth[anchor] = 1.0
    depth[target] = 0.5
    result = relation_geometry.evaluate_relationship("inside", target, anchor, depth=depth)
    assert result["status"] == "fail"
    assert result["measurements"]["depth_compatible"] is False


def test_on_passes_support_overlap_and_small_vertical_gap():
    anchor = rectangle((100, 100), 30, 55, 75, 80)
    target = rectangle((100, 100), 40, 35, 65, 53)
    result = relation_geometry.evaluate_relationship(
        "on", target, anchor, depth=np.ones((100, 100), dtype=np.float32)
    )
    assert result["status"] == "pass"
    assert result["measurements"]["horizontal_support_overlap_fraction"] == 1.0


def test_on_fails_without_horizontal_support():
    anchor = rectangle((100, 100), 60, 55, 90, 80)
    target = rectangle((100, 100), 5, 35, 25, 53)
    assert relation_geometry.evaluate_relationship(
        "on", target, anchor, depth=np.ones((100, 100), dtype=np.float32)
    )["status"] == "fail"


@pytest.mark.parametrize("relationship", ["inside", "on"])
def test_depth_compatible_relations_are_unavailable_without_depth(relationship):
    anchor = rectangle((100, 100), 20, 50, 80, 80)
    target = (
        rectangle((100, 100), 35, 55, 50, 65)
        if relationship == "inside"
        else rectangle((100, 100), 35, 35, 50, 49)
    )
    result = relation_geometry.evaluate_relationship(relationship, target, anchor)
    assert result["status"] == "unavailable"


@pytest.mark.parametrize(
    ("relationship", "target_box", "anchor_box"),
    [
        ("left_of", (10, 40, 20, 50), (70, 40, 80, 50)),
        ("right_of", (70, 40, 80, 50), (10, 40, 20, 50)),
        ("above", (40, 10, 50, 20), (40, 70, 50, 80)),
        ("below", (40, 70, 50, 80), (40, 10, 50, 20)),
    ],
)
def test_directional_relationships_pass_with_margin(relationship, target_box, anchor_box):
    target = rectangle((100, 100), *target_box)
    anchor = rectangle((100, 100), *anchor_box)
    assert relation_geometry.evaluate_relationship(relationship, target, anchor)["status"] == "pass"
    assert relation_geometry.evaluate_relationship(relationship, anchor, target)["status"] == "fail"


@pytest.mark.parametrize("relationship", ["near", "next_to"])
def test_proximity_uses_normalized_mask_edge_distance(relationship):
    target = rectangle((100, 100), 10, 40, 20, 50)
    near = rectangle((100, 100), 25, 40, 35, 50)
    far = rectangle((100, 100), 80, 80, 90, 90)
    assert relation_geometry.evaluate_relationship(relationship, target, near)["status"] == "pass"
    assert relation_geometry.evaluate_relationship(relationship, target, far)["status"] == "fail"


def test_proximity_checks_3d_distance_when_reliable():
    target = rectangle((30, 30), 5, 10, 10, 15)
    anchor = rectangle((30, 30), 11, 10, 16, 15)
    xyz = np.full((30, 30, 3), np.nan, dtype=np.float32)
    xyz[target] = [0.0, 0.0, 1.0]
    xyz[anchor] = [2.0, 0.0, 1.0]
    result = relation_geometry.evaluate_relationship(
        "near",
        target,
        anchor,
        xyz=xyz,
        thresholds={"depth_min_valid_pixels": 5},
    )
    assert result["status"] == "fail"
    assert result["measurements"]["median_3d_distance_m"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("relationship", "target_depth", "anchor_depth", "expected"),
    [
        ("in_front_of", 0.8, 1.0, "pass"),
        ("in_front_of", 1.2, 1.0, "fail"),
        ("behind", 1.2, 1.0, "pass"),
        ("behind", 0.8, 1.0, "fail"),
    ],
)
def test_depth_order_uses_robust_median_separation(
    relationship, target_depth, anchor_depth, expected
):
    target = rectangle((30, 30), 2, 2, 8, 8)
    anchor = rectangle((30, 30), 20, 20, 27, 27)
    depth = np.full((30, 30), np.nan, dtype=np.float32)
    depth[target] = target_depth
    depth[anchor] = anchor_depth
    result = relation_geometry.evaluate_relationship(
        relationship,
        target,
        anchor,
        depth=depth,
        thresholds={"depth_min_valid_pixels": 5},
    )
    assert result["status"] == expected


@pytest.mark.parametrize("relationship", ["in_front_of", "behind"])
def test_depth_order_is_unavailable_without_reliable_geometry(relationship):
    target = rectangle((30, 30), 2, 2, 8, 8)
    anchor = rectangle((30, 30), 20, 20, 27, 27)
    result = relation_geometry.evaluate_relationship(relationship, target, anchor)
    assert result["status"] == "unavailable"
    assert result["passed"] is None


@pytest.mark.parametrize(
    ("relationship", "expected"),
    [
        ("inside", (30, 30, 50, 50)),
        ("left_of", (0, 0, 40, 100)),
        ("right_of", (40, 0, 100, 100)),
        ("above", (0, 0, 100, 40)),
        ("below", (0, 40, 100, 100)),
        ("in_front_of", (0, 0, 100, 100)),
        ("behind", (0, 0, 100, 100)),
    ],
)
def test_relation_aware_rois(relationship, expected):
    anchor = rectangle((100, 100), 30, 30, 50, 50)
    assert relation_geometry.relation_aware_roi(anchor, relationship) == expected


def test_grid_search_maximizes_accuracy_with_zero_safety_false_acceptance():
    shape = (100, 100)
    anchor = rectangle(shape, 50, 40, 60, 50)
    valid_target = rectangle(shape, 40, 40, 50, 50)  # separation 0.10
    safety_target = rectangle(shape, 47, 40, 50, 50)  # separation 0.065
    cases = [
        {
            "relationship": "left_of",
            "target_mask": valid_target,
            "anchor_mask": anchor,
            "expected_pass": True,
            "safety_case": False,
        },
        {
            "relationship": "left_of",
            "target_mask": safety_target,
            "anchor_mask": anchor,
            "expected_pass": False,
            "safety_case": True,
        },
    ]
    report = relation_geometry.grid_search_thresholds(
        cases,
        {"direction_margin_fraction": [0.0, 0.08, 0.15]},
    )
    assert report["thresholds"]["direction_margin_fraction"] == 0.08
    assert report["valid_case_accuracy"] == 1.0
    assert report["safety_false_accepts"] == 0
