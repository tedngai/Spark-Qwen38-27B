#!/usr/bin/env python3
"""Smoke test for the Qwen3.8-27B SGLang server (OpenAI-compatible API).

Checks /health, picks the served model from /v1/models, sends one
deterministic chat completion, and optionally writes the raw report to
evidence/.

Usage:
    python3 tests/smoke_test.py --base-url http://127.0.0.1:8000
    python3 tests/smoke_test.py --profile baseline --save-dir evidence

Exit code 0 only if the health check passes and the completion returns
non-empty output with a finish_reason. Never starts or stops the server.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.request

PROMPT = "What is 2+2? Answer in one short sentence."


def http_json(url: str, payload: dict | None = None, timeout: float = 180.0):
    """GET (payload=None) or POST a JSON document; return (status, parsed)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        return resp.status, parsed


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save(report: dict, profile: str, save_dir: str | None) -> None:
    if not save_dir:
        return
    out_dir = pathlib.Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{profile}-smoke-{utc_stamp()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report saved: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test a running Qwen3.8-27B SGLang server."
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--model",
        default=None,
        help="served model id (default: first id from /v1/models)",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0, help="request timeout (s)")
    ap.add_argument(
        "--profile",
        default="manual",
        help="profile label used in the evidence file name",
    )
    ap.add_argument(
        "--save-dir",
        default=None,
        help="write the report JSON here (e.g. evidence)",
    )
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    report: dict = {"captured_at": utc_now(), "base_url": base, "profile": args.profile}

    # 1. health
    try:
        status, health = http_json(base + "/health", timeout=15.0)
    except Exception as exc:  # report any transport failure
        print(f"FAIL: /health request error: {exc}")
        return 1
    report["health_status"] = status
    report["health_body"] = health
    print(f"/health -> HTTP {status}")
    if status != 200:
        print("FAIL: server not ready (health != 200)")
        save(report, args.profile, args.save_dir)
        return 1

    # 2. models
    model = args.model
    try:
        status, models = http_json(base + "/v1/models", timeout=15.0)
        ids = [m.get("id") for m in (models or {}).get("data", [])]
        report["models"] = ids
        print(f"/v1/models -> {ids}")
        if model is None and ids:
            model = ids[0]
    except Exception as exc:
        print(f"WARN: /v1/models failed: {exc}")
    if model is None:
        print("FAIL: no model id available (pass --model)")
        save(report, args.profile, args.save_dir)
        return 1
    report["model"] = model

    # 3. chat completion
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    started = time.monotonic()
    try:
        status, resp = http_json(
            base + "/v1/chat/completions", payload, timeout=args.timeout
        )
    except Exception as exc:
        print(f"FAIL: chat request error: {exc}")
        save(report, args.profile, args.save_dir)
        return 1
    elapsed = time.monotonic() - started
    report["request"] = payload
    report["response"] = resp
    report["client_elapsed_s"] = round(elapsed, 3)

    choice = (resp or {}).get("choices", [{}])[0]
    message = choice.get("message", {})
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    finish = choice.get("finish_reason")

    print(f"POST /v1/chat/completions -> HTTP {status} in {elapsed:.2f}s")
    print(f"  finish_reason: {finish}")
    print(f"  content: {content!r}")
    if reasoning:
        print(f"  reasoning_content[:120]: {reasoning[:120]!r}")

    # Thinking models may return empty content with reasoning only when the
    # max_tokens budget is too small to get past the reasoning phase.
    ok = status == 200 and finish is not None and (content or reasoning)
    print("SMOKE OK" if ok else "SMOKE FAILED")
    save(report, args.profile, args.save_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
