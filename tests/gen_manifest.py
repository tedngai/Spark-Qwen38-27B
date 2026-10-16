#!/usr/bin/env python3
"""Regenerate docs/package-manifest.json (paths, sizes, SHA-256).

Per AGENTS.md: run after adding, removing, renaming, or materially changing
any tracked file, BEFORE committing. The manifest covers every git-tracked
file except docs/package-manifest.json itself and runtime exclusions.

Usage:
    python3 tests/gen_manifest.py
    python3 tests/gen_manifest.py --check   # exit 1 if out of date
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

MANIFEST = pathlib.Path("docs/package-manifest.json")
EXCLUDE_PREFIXES = ("logs/", ".cache/", "venv/", "__pycache__/")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    )
    files = []
    for line in out.stdout.splitlines():
        p = line.strip()
        if not p or p == ".env" or p == MANIFEST.as_posix():
            continue
        if p.startswith(EXCLUDE_PREFIXES):
            continue
        files.append(p)
    return sorted(files)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    files = []
    for p in tracked_files():
        path = pathlib.Path(p)
        files.append(
            {"path": p, "size": path.stat().st_size, "sha256": sha256(path)}
        )
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository": "Qwen3.8-27B-Deployment",
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the manifest matches; exit 1 if stale")
    args = ap.parse_args()

    manifest = build()
    if args.check:
        current = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
        if current == manifest:
            print("manifest up to date")
            return 0
        stale = (
            current is None
            or current.get("file_count") != manifest["file_count"]
            or {f["path"]: f["sha256"] for f in current.get("files", [])}
            != {f["path"]: f["sha256"] for f in manifest["files"]}
        )
        print("manifest STALE" if stale else "manifest timestamps differ (regenerate)")
        return 1 if stale else 0

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written: {MANIFEST} ({manifest['file_count']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
