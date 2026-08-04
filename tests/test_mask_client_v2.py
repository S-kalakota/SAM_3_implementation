import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mask_client


def test_normal_client_sends_exact_interpretation_to_v2_segment():
    envelope = {"schema_version": 2, "envelope_hash": "sealed"}
    expected = {"schema_version": 2, "status": "no_match"}
    with mock.patch.object(mask_client, "interpret", return_value=envelope) as interpret:
        with mock.patch.object(mask_client, "_post_json", return_value=expected) as post:
            result = mask_client.segment(
                "find the red box",
                host="localhost:9999",
                timeout=12,
            )
    assert result == expected
    interpret.assert_called_once_with(
        "find the red box", host="localhost:9999", timeout=12
    )
    post.assert_called_once_with(
        "http://localhost:9999/v2/segment",
        envelope,
        12,
    )


def test_v1_rollback_remains_one_versioned_request():
    with mock.patch.object(mask_client, "_post_json", return_value={"schema_version": 1}) as post:
        result = mask_client.segment("pick the red box", v1=True)
    assert result["schema_version"] == 1
    assert post.call_args.args[0].endswith("/v1/segment")
    assert post.call_args.args[1]["schema_version"] == 1
