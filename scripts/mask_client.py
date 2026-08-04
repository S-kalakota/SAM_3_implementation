#!/usr/bin/env python3
"""Small client for the resident SAM 3.1 mask service."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import grounding_intent


class MaskServiceHTTPError(RuntimeError):
    """Readable local-service rejection that preserves its structured payload."""

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        url: str,
        payload: Any,
    ) -> None:
        self.status = int(status)
        self.reason = reason
        self.url = url
        self.payload = payload
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        if isinstance(detail, dict):
            code = detail.get("code")
            message = detail.get("message")
            details = detail.get("details")
            lines = [f"Mask service rejected the request (HTTP {status} {reason})."]
            if code:
                lines.append(f"code: {code}")
            if message:
                lines.append(f"message: {message}")
            if details:
                lines.append("details: " + json.dumps(details, indent=2, ensure_ascii=False))
            rendered = "\n".join(lines)
        elif detail:
            rendered = f"Mask service rejected the request (HTTP {status} {reason}): {detail}"
        else:
            rendered = f"Mask service rejected the request (HTTP {status} {reason})."
        super().__init__(rendered)


def _base_url(host: str) -> str:
    base_url = host.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    return base_url


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            decoded: Any = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = body.decode("utf-8", errors="replace").strip()
        raise MaskServiceHTTPError(
            status=exc.code,
            reason=str(exc.reason),
            url=url,
            payload=decoded,
        ) from exc


def interpret(
    raw_command: str,
    *,
    host: str = "127.0.0.1:8765",
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Ask Qwen for the sealed v2 envelope without capturing a frame."""

    if not isinstance(raw_command, str) or not raw_command.strip():
        raise ValueError("raw command must be non-empty")
    command = raw_command.strip()
    return _post_json(
        f"{_base_url(host)}/v2/interpret",
        {"raw_command": command},
        timeout,
    )


def segment(
    request: str,
    *,
    host: str = "127.0.0.1:8765",
    use_agent_fallback: bool = True,
    timeout: float = 300.0,
    legacy: bool = False,
    v1: bool = False,
) -> dict[str, Any]:
    base_url = _base_url(host)
    if not legacy and not v1:
        # Version 2 intentionally performs two calls.  The first is the only
        # language interpretation; the second authenticates and consumes the
        # exact returned envelope without re-reading the sentence.
        envelope = interpret(request, host=host, timeout=timeout)
        return _post_json(f"{base_url}/v2/segment", envelope, timeout)
    if v1 and not legacy:
        intent = grounding_intent.parse_grounding_intent(request)
        payload = {
            "schema_version": 1,
            "source_phrase": grounding_intent.collapse_space(request),
            "grounding_intent": intent,
            "intent_hash": grounding_intent.intent_hash(intent),
            "use_agent_fallback": bool(use_agent_fallback),
        }
        return _post_json(f"{base_url}/v1/segment", payload, timeout)

    query = urllib.parse.urlencode(
        {
            "request": request,
            "use_agent_fallback": str(use_agent_fallback).lower(),
        }
    )
    req = urllib.request.Request(
        f"{base_url}/segment?{query}",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the local mask service.")
    parser.add_argument("request", nargs="+", help="Natural-language request.")
    parser.add_argument("--host", default="127.0.0.1:8765")
    parser.add_argument("--timeout", default=300.0, type=float)
    parser.add_argument(
        "--no-agent-fallback",
        action="store_true",
        help="Use only the direct SAM path.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the deprecated query-string /segment interface.",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Use the retained version-1 structured-intent rollback API.",
    )
    parser.add_argument(
        "--interpret-only",
        action="store_true",
        help="Return Qwen's validated v2 command envelope without segmentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = " ".join(args.request)
    try:
        if args.interpret_only:
            if args.legacy or args.v1:
                raise SystemExit("--interpret-only cannot be combined with --legacy or --v1")
            result = interpret(request, host=args.host, timeout=args.timeout)
        else:
            result = segment(
                request,
                host=args.host,
                use_agent_fallback=not args.no_agent_fallback,
                timeout=args.timeout,
                legacy=args.legacy,
                v1=args.v1,
            )
    except MaskServiceHTTPError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
