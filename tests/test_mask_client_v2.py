import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

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


def test_http_error_preserves_and_renders_fastapi_detail():
    payload = {
        "detail": {
            "code": "missing_source_evidence",
            "message": "target mention was not copied",
            "details": {"field": "target.mention", "evidence": "green box"},
        }
    }
    error = urllib.error.HTTPError(
        "http://localhost:8765/v2/interpret",
        422,
        "Unprocessable Entity",
        {},
        io.BytesIO(json.dumps(payload).encode()),
    )
    with mock.patch.object(mask_client.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(mask_client.MaskServiceHTTPError) as caught:
            mask_client._post_json(
                "http://localhost:8765/v2/interpret",
                {"raw_command": "find the green box"},
                10,
            )
    rendered = str(caught.value)
    assert "HTTP 422" in rendered
    assert "missing_source_evidence" in rendered
    assert "target mention was not copied" in rendered
    assert "green box" in rendered
    assert caught.value.payload == payload


def test_cli_turns_service_rejection_into_clean_nonzero_exit():
    error = mask_client.MaskServiceHTTPError(
        status=422,
        reason="Unprocessable Entity",
        url="http://localhost/v2/interpret",
        payload={"detail": {"code": "bad_envelope", "message": "try again"}},
    )
    parsed = mock.Mock(
        request=["find", "box"],
        interpret_only=True,
        legacy=False,
        v1=False,
        host="localhost",
        timeout=10,
    )
    with mock.patch.object(mask_client, "parse_args", return_value=parsed):
        with mock.patch.object(mask_client, "interpret", side_effect=error):
            with pytest.raises(SystemExit) as caught:
                mask_client.main()
    assert caught.value.code != 0
    assert "bad_envelope" in str(caught.value)
    assert "try again" in str(caught.value)
