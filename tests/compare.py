#!/usr/bin/env python3
"""Compare A/B benchmark evidence across profiles (stdlib only).

Reads evidence/*-bench-*.json (from tests/benchmark.py), keeps the latest run
per profile, and prints one table per scenario with the metrics that decide a
configuration: aggregate throughput, per-request decode speed, TTFT,
speculative-decode acceptance, failures and retractions.

If evidence/<profile>-vllmbench-*/result_*.json exists, a second table shows
the vllm-benchmark-suite scorecard per profile.

Usage:
    python3 tests/compare.py
    python3 tests/compare.py --scenario short-c8-512x256
    python3 tests/compare.py --evidence-dir evidence
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

METRICS = [
    ("aggregate_tok_s", "agg tok/s", "high"),
    ("decode_tok_s_p50", "dec p50", "high"),
    ("ttft_s_p50", "ttft p50", "low"),
    ("spec_accept_length_mean", "accept", "high"),
    ("requests_failed", "fails", "low"),
    ("retractions_total", "retr", "low"),
]


def load_bench_files(evidence_dir: pathlib.Path) -> dict[str, dict]:
    """Latest bench report per profile."""
    latest: dict[str, dict] = {}
    for path in sorted(evidence_dir.glob("*-bench-*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        profile = report.get("profile") or path.stem
        captured = report.get("captured_at", "")
        if profile not in latest or captured > latest[profile].get("captured_at", ""):
            latest[profile] = report
    return latest


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(latest: dict[str, dict], scenario_filter: str | None) -> None:
    scenarios: list[str] = []
    for report in latest.values():
        for scen in report.get("scenarios", []):
            name = scen.get("name")
            if name and name not in scenarios:
                scenarios.append(name)
    if scenario_filter:
        scenarios = [s for s in scenarios if s == scenario_filter]
    if not scenarios:
        print("no matching bench scenarios found in evidence/")
        return

    width = max(len(p) for p in latest) if latest else 10
    width = max(width, 12)

    for name in scenarios:
        rows = []
        for profile, report in latest.items():
            scen = next(
                (s for s in report.get("scenarios", []) if s.get("name") == name), None
            )
            if scen:
                rows.append((profile, scen))
        if not rows:
            continue
        rows.sort(key=lambda r: r[1].get("aggregate_tok_s") or 0.0, reverse=True)

        header = f"{'profile':<{width}} " + " ".join(
            f"{label:>9}" for _, label, _ in METRICS
        )
        print(f"\n== {name}")
        print(header)
        for profile, scen in rows:
            cells = []
            for key, _, _ in METRICS:
                cells.append(f"{fmt(scen.get(key)):>9}")
            print(f"{profile:<{width}} " + " ".join(cells))


def print_suite(latest_suite: dict[str, dict]) -> None:
    if not latest_suite:
        return
    print("\n== vllm-benchmark-suite scorecard (latest per profile)")
    width = max(len(p) for p in latest_suite)
    width = max(width, 12)
    print(
        f"{'profile':<{width}} {'peak tok/s':>10} {'best ttft':>10} {'score':>7} {'grade':>6}"
    )
    for profile in sorted(latest_suite):
        r = latest_suite[profile].get("results", {}) or {}
        score = latest_suite[profile].get("score", {}) or {}
        print(
            f"{profile:<{width}} {fmt(r.get('peak_throughput_tps')):>10} "
            f"{fmt(r.get('best_ttft_ms')):>10} {fmt(score.get('overall')):>7} "
            f"{str(score.get('grade', '-')):>6}"
        )


def load_suite_files(evidence_dir: pathlib.Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for path in sorted(evidence_dir.glob("*-vllmbench-*/result_*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        profile = path.parent.name.split("-vllmbench-")[0]
        mtime = path.stat().st_mtime
        if profile not in latest or mtime > latest[profile]["_mtime"]:
            data["_mtime"] = mtime
            latest[profile] = data
    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--scenario", default=None, help="only this scenario name")
    args = ap.parse_args()

    evidence_dir = pathlib.Path(args.evidence_dir)
    if not evidence_dir.is_dir():
        print(f"FAIL: no such directory: {evidence_dir}")
        return 1

    latest = load_bench_files(evidence_dir)
    if not latest:
        print(f"no *-bench-*.json files found in {evidence_dir}/")
        return 1

    print(f"profiles with A/B bench evidence: {', '.join(sorted(latest))}")
    print_table(latest, args.scenario)
    print_suite(load_suite_files(evidence_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
