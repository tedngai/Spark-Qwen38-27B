#!/usr/bin/env python3
"""Capture a running server's configuration and health as evidence.

Collects /health, /v1/models, the full /get_server_info config dump, the
container inspect (image, command, restart policy — secrets filtered) and the
git revision, and writes one evidence JSON. Never starts or stops the server.

Usage:
    python3 tests/capture_server.py --profile baseline --save-dir evidence

Exit code 0 only if /health is 200 and /get_server_info was captured.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import urllib.request

SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def http_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
        try:
            return resp.status, json.loads(body) if body else None
        except json.JSONDecodeError:
            return resp.status, body


def docker_inspect(container: str) -> dict | None:
    """Pruned, secret-free subset of `docker inspect`; None if unavailable."""
    try:
        out = subprocess.run(
            ["docker", "inspect", container], capture_output=True, text=True, timeout=20
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        raw = json.loads(out.stdout)[0]
    except Exception:
        return None
    env = [
        e
        for e in raw.get("Config", {}).get("Env", []) or []
        if not any(m in e.split("=", 1)[0].upper() for m in SECRET_MARKERS)
    ]
    return {
        "id": raw.get("Id"),
        "created": raw.get("Created"),
        "state_status": raw.get("State", {}).get("Status"),
        "image": raw.get("Config", {}).get("Image"),
        "image_id": raw.get("Image"),
        "cmd": raw.get("Config", {}).get("Cmd"),
        "restart_policy": raw.get("HostConfig", {}).get("RestartPolicy", {}).get("Name"),
        "env": env,
    }


def git_rev() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--container", default="qwen3.8-27b-nvfp4-sglang")
    ap.add_argument("--profile", required=True, help="profile label for the file name")
    ap.add_argument("--save-dir", default=None, help="write the report JSON here")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    report: dict = {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "profile": args.profile,
        "base_url": base,
        "git": git_rev(),
    }

    ok = True
    for key, path in (
        ("health", "/health"),
        ("models", "/v1/models"),
        ("server_info", "/get_server_info"),
    ):
        try:
            status, body = http_json(base + path, timeout=30.0)
            report[key] = body
            report[f"{key}_http_status"] = status
            print(f"{path} -> HTTP {status}")
            if status != 200:
                ok = False
        except Exception as exc:
            report[key] = None
            report[f"{key}_http_status"] = None
            report[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"{path} -> ERROR: {exc}")
            ok = False

    report["container"] = docker_inspect(args.container)

    if args.save_dir:
        out_dir = pathlib.Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.profile}-server-info-{utc_stamp()}.json"
        with open(path, "x") as fh:  # never overwrite existing evidence
            json.dump(report, fh, indent=2)
            fh.write("\n")
        print(f"report saved: {path}")
    else:
        print("(no --save-dir; report not saved)")

    print("CAPTURE OK" if ok else "CAPTURE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
