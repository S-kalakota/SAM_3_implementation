#!/usr/bin/env python3
"""Small client for the resident SAM 3.1 mask service."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any


def segment(
    request: str,
    *,
    host: str = "127.0.0.1:8765",
    use_agent_fallback: bool = False,
    timeout: float = 300.0,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "request": request,
            "use_agent_fallback": str(use_agent_fallback).lower(),
        }
    )
    req = urllib.request.Request(
        f"http://{host}/segment?{query}",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the local mask service.")
    parser.add_argument("request", nargs="+", help="Natural-language request.")
    parser.add_argument("--host", default="127.0.0.1:8765")
    parser.add_argument("--timeout", default=300.0, type=float)
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--agent-fallback",
        action="store_true",
        help="Explicitly allow the unbounded Qwen/SAM agent after bounded paths fail.",
    )
    fallback_group.add_argument(
        "--no-agent-fallback",
        action="store_false",
        dest="agent_fallback",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(agent_fallback=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = segment(
        " ".join(args.request),
        host=args.host,
        use_agent_fallback=args.agent_fallback,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
