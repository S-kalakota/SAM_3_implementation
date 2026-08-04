import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import sam3_image_service


def test_parity_matching_does_not_use_a_greedy_false_rejection():
    similarities = np.asarray(
        [
            [0.96, 0.95],
            [0.95, 0.94],
        ]
    )
    matching = sam3_image_service._threshold_bipartite_matching(similarities, 0.95)
    assert matching == {0: 1, 1: 0}


def test_parity_rejects_extra_image_masks():
    empty = np.zeros((4, 4), dtype=bool)
    report = sam3_image_service.frozen_frame_parity_report(
        [empty],
        [empty, empty],
        min_iou=0.95,
    )
    assert report["approved"] is False
    assert report["unmatched_image_indices"]
