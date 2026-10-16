#!/usr/bin/env python3
"""Concurrency benchmark for the Qwen3.8-27B SGLang server (stdlib only).

Measures, at several concurrency levels, what a user actually experiences:
time-to-first-token, per-request decode speed, aggregate throughput, and
(for A/B between launch profiles) speculative-decode acceptance, prefix-cache
hits and retractions.

Design notes:
  - Prompts are generated from fixed seeds, so two runs of the same scenario
    (e.g. baseline vs baseline-no-spec) send byte-identical prompts. Each
    prompt is unique, which defeats prefix sharing on purpose; `cached_tokens`
    in the report proves it.
  - Decoding is forced (`ignore_eos`, exact max_new_tokens) so every profile
    does identical work: apples to apples.
  - Requests stream from /generate and timestamps are taken client-side:
    TTFT = first chunk, decode time = last chunk - first chunk. The server's
    own `e2e_latency` is recorded alongside for cross-checking.
  - The baseline admits 8 concurrent users (`--max-running-requests 8`), so
    c12/c16 scenarios measure queueing behaviour beyond the cap.
  - Idle guard: a tiny probe request must come back fast before each scenario.
    --force overrides.

Usage:
    python3 tests/benchmark.py --profile baseline --save-dir evidence
    python3 tests/benchmark.py --only short-c1,short-c8 --profile mytest
    python3 tests/benchmark.py --skip-long --profile quick

Exit code 0 only if every request in every scenario succeeded with the exact
forced output length and no retractions. Failed runs are still saved. Never
starts or stops the server.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

DEFAULT_SCENARIOS = [
    {"name": "short-c1-512x256", "concurrency": 1, "in_tokens": 512, "out_tokens": 256},
    {"name": "short-c4-512x256", "concurrency": 4, "in_tokens": 512, "out_tokens": 256},
    {"name": "short-c8-512x256", "concurrency": 8, "in_tokens": 512, "out_tokens": 256},
    {"name": "short-c12-512x256", "concurrency": 12, "in_tokens": 512, "out_tokens": 256},
    {"name": "short-c16-512x256", "concurrency": 16, "in_tokens": 512, "out_tokens": 256},
    {"name": "long-c1-16384x256", "concurrency": 1, "in_tokens": 16384, "out_tokens": 256},
    {"name": "long-c8-16384x256", "concurrency": 8, "in_tokens": 16384, "out_tokens": 256},
    {"name": "long-c12-16384x256", "concurrency": 12, "in_tokens": 16384, "out_tokens": 256},
]

# Deterministic prompt vocabulary: ordinary English words, no tokenizer needed.
# ~1.3 tokens/word for this tokenizer family; prompt sizing uses chars/4.0.
_WORDS = (
    "the system records temperature pressure volume latency throughput cache "
    "memory kernel buffer token context window batch scheduler worker queue "
    "index column row table record entry value signal noise sample measure "
    "calibrate validate verify accept reject commit checkpoint restore buffer "
    "linear attention state vector matrix tensor graph node edge weight bias "
    "layer norm scale shift rotate project embed decode encode draft verify "
    "region cluster packet frame segment packet route gateway client server "
    "upload download stream chunk block page slot cycle epoch phase step"
).split()

CHUNK_HEADER = {"Content-Type": "application/json"}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def http_json(url: str, payload: dict | None = None, timeout: float = 180.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=CHUNK_HEADER)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        return resp.status, parsed


def build_prompt(in_tokens: int, seed: int) -> str:
    """Deterministic, unique pseudo-prose of roughly `in_tokens` tokens."""
    rng = random.Random(seed)
    parts = [
        f"Record {seed:010d}: operational notes unique to this document "
        f"serial number {seed * 7919 % 10**9}.\n"
    ]
    target_chars = int(in_tokens * 4.0)
    total = len(parts[0])
    while total < target_chars:
        n_words = rng.randint(8, 16)
        words = [rng.choice(_WORDS) for _ in range(n_words)]
        sent = " ".join(words)
        sent = sent[0].upper() + sent[1:] + ".\n"
        parts.append(sent)
        total += len(sent)
    return "".join(parts)


def run_request(base: str, prompt: str, out_tokens: int, timeout: float) -> dict:
    """One forced-decode streaming request; returns timings + final meta_info."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": out_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    rec: dict = {"ok": False, "error": None}
    t0 = time.monotonic()
    first_ts = last_ts = None
    final_meta: dict = {}
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            base + "/generate", data=data, headers=CHUNK_HEADER
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                body = line[6:].strip() if line.startswith("data: ") else line
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                if first_ts is None:
                    first_ts = now
                last_ts = now
                meta = chunk.get("meta_info") or {}
                if meta.get("finish_reason") is not None:
                    final_meta = meta
    except Exception as exc:  # record, do not abort the whole run
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = round(time.monotonic() - t0, 3)
        return rec

    wall = (last_ts or time.monotonic()) - t0
    rec.update(
        {
            "ok": True,
            "wall_s": round(wall, 3),
            "ttft_s": round(first_ts - t0, 3) if first_ts else None,
            "decode_s": round(last_ts - first_ts, 3) if first_ts and last_ts else None,
            "prompt_tokens": final_meta.get("prompt_tokens"),
            "completion_tokens": final_meta.get("completion_tokens"),
            "cached_tokens": final_meta.get("cached_tokens"),
            "finish_reason": final_meta.get("finish_reason"),
            "num_retractions": final_meta.get("num_retractions"),
            "server_e2e_latency_s": final_meta.get("e2e_latency"),
            "spec_accept_length": final_meta.get("spec_accept_length"),
            "spec_accept_rate": final_meta.get("spec_accept_rate"),
            "spec_num_proposed_drafts": final_meta.get("spec_num_proposed_drafts"),
            "spec_num_correct_drafts": final_meta.get("spec_num_correct_drafts"),
            "spec_verify_ct": final_meta.get("spec_verify_ct"),
        }
    )
    comp = rec.get("completion_tokens") or 0
    if comp and rec.get("decode_s"):
        rec["decode_tok_s"] = round(comp / rec["decode_s"], 2)
    if comp:
        rec["e2e_tok_s"] = round(comp / wall, 2)
    return rec


def pct(values: list[float], p: float):
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def summarize(name: str, scen: dict, records: list[dict], wall_s: float) -> dict:
    ok = [r for r in records if r.get("ok")]
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    decode = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    e2e = [r["e2e_tok_s"] for r in ok if r.get("e2e_tok_s")]
    accept = [r["spec_accept_length"] for r in ok if r.get("spec_accept_length")]
    comp_total = sum(r.get("completion_tokens") or 0 for r in ok)
    return {
        "name": name,
        "concurrency": scen["concurrency"],
        "input_tokens_target": scen["in_tokens"],
        "output_tokens_target": scen["out_tokens"],
        "wall_s": round(wall_s, 3),
        "requests_ok": len(ok),
        "requests_failed": len(records) - len(ok),
        "completion_tokens_total": comp_total,
        "aggregate_tok_s": round(comp_total / wall_s, 2) if wall_s > 0 else None,
        "prompt_tokens_mean": round(
            statistics.mean(r["prompt_tokens"] for r in ok), 1
        ) if ok and all(r.get("prompt_tokens") is not None for r in ok) else None,
        "cached_tokens_mean": round(
            statistics.mean(r.get("cached_tokens") or 0 for r in ok), 2
        ) if ok else None,
        "ttft_s_p50": pct(ttfts, 50),
        "ttft_s_p90": pct(ttfts, 90),
        "ttft_s_max": max(ttfts) if ttfts else None,
        "decode_tok_s_p50": pct(decode, 50),
        "decode_tok_s_p10": pct(decode, 10),
        "decode_tok_s_mean": round(statistics.mean(decode), 2) if decode else None,
        "e2e_tok_s_p50": pct(e2e, 50),
        "spec_accept_length_mean": round(statistics.mean(accept), 3) if accept else None,
        "spec_accept_length_min": min(accept) if accept else None,
        "retractions_total": sum(r.get("num_retractions") or 0 for r in records),
        "requests": records,
    }


def run_scenario(base: str, scen: dict, seed_base: int, timeout: float) -> dict:
    n = scen["concurrency"]
    barrier = threading.Barrier(n)
    results: list[dict] = [None] * n

    def worker(i: int) -> None:
        prompt = build_prompt(scen["in_tokens"], seed_base + i)
        barrier.wait()
        results[i] = run_request(base, prompt, scen["out_tokens"], timeout)

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(worker, i) for i in range(n)]
        for f in futs:
            f.result()
    wall = time.monotonic() - t0
    return summarize(scen["name"], scen, results, wall)


def probe(base: str, idle_limit_s: float) -> tuple[bool, str]:
    """Tiny forced request; idle servers answer near-instantly."""
    try:
        rec = run_request(base, "ping", 8, timeout=60.0)
    except Exception as exc:
        return False, f"probe error: {exc}"
    if not rec.get("ok"):
        return False, f"probe failed: {rec.get('error')}"
    e2e = rec.get("server_e2e_latency_s") or rec.get("wall_s") or 99.0
    if e2e > idle_limit_s:
        return False, f"server looks busy (probe e2e_latency={e2e:.2f}s > {idle_limit_s}s)"
    return True, f"probe ok (e2e_latency={e2e:.2f}s)"


def snapshot_gpu() -> dict | None:
    """Best-effort GPU memory snapshot; GB10 reports unified memory."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            used, total = (int(x) for x in out.stdout.strip().split(",")[:2])
            return {"gpu_index": 0, "memory_used_mib": used, "memory_total_mib": total}
    except Exception:
        pass
    return None


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
    ap.add_argument("--profile", required=True,
                    help="profile label for the evidence file, e.g. baseline")
    ap.add_argument("--save-dir", default=None,
                    help="write the report JSON here (e.g. evidence)")
    ap.add_argument("--only", default=None,
                    help="comma-separated scenario-name filters (exact name or "
                         "prefix ending at a '-' boundary, e.g. 'short-c1,short-c12')")
    ap.add_argument("--skip-long", action="store_true",
                    help="run only the short-prompt scenarios")
    ap.add_argument("--out-tokens", type=int, default=None,
                    help="override forced output length for every scenario")
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-request timeout (s)")
    ap.add_argument("--idle-limit", type=float, default=2.0,
                    help="probe e2e_latency above this = server busy")
    ap.add_argument("--force", action="store_true",
                    help="run even if the idle guard says the server is busy")
    ap.add_argument("--run-seed", type=int, default=0,
                    help="offset added to all prompt seeds (bump to get fresh "
                         "prompts when re-running against a live server)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    scenarios = []
    for s in DEFAULT_SCENARIOS:
        if args.skip_long and s["name"].startswith("long-"):
            continue
        if args.only:
            name = s["name"]

            def matches(filter_str: str) -> bool:
                return filter_str == name or (
                    name.startswith(filter_str)
                    and name[len(filter_str):len(filter_str) + 1] == "-"
                )

            if not any(matches(f.strip()) for f in args.only.split(",") if f.strip()):
                continue
        s = dict(s)
        if args.out_tokens:
            s["out_tokens"] = args.out_tokens
        # Deterministic, profile-independent seeds: every profile A/B with the
        # same scenario set sends byte-identical prompts. --run-seed shifts all
        # seeds when re-running against a live server (fresh radix cache).
        digest = int(hashlib.sha256(s["name"].encode()).hexdigest()[:8], 16)
        s["seed_base"] = (digest % 1_000_000) * 1000 + args.run_seed
        scenarios.append(s)
    if not scenarios:
        print("FAIL: no scenarios selected")
        return 1

    report: dict = {
        "captured_at": utc_now(),
        "profile": args.profile,
        "base_url": base,
        "git": git_rev(),
        "scenarios": [],
    }

    try:
        status, server_info = http_json(base + "/get_server_info", timeout=30.0)
        report["server_info"] = server_info
        report["server_http_status"] = status
    except Exception as exc:
        print(f"FAIL: /get_server_info error: {exc}")
        return 1

    report["gpu_before"] = snapshot_gpu()
    print(f"profile={args.profile}  scenarios={[s['name'] for s in scenarios]}")

    all_ok = True
    for scen in scenarios:
        ok_probe, msg = probe(base, args.idle_limit)
        if not ok_probe and not args.force:
            print(f"FAIL: {msg} (use --force to override)")
            all_ok = False
            break
        print(f"  [{scen['name']}] {msg}; launching c={scen['concurrency']} ...")
        summary = run_scenario(base, scen, scen["seed_base"], args.timeout)
        report["scenarios"].append(summary)
        failures = summary["requests_failed"]
        exact = all(
            r.get("completion_tokens") == scen["out_tokens"]
            for r in summary["requests"] if r.get("ok")
        )
        if failures or not exact:
            all_ok = False
        print(
            f"  [{scen['name']}] ok={summary['requests_ok']}/{scen['concurrency']} "
            f"wall={summary['wall_s']}s aggregate={summary['aggregate_tok_s']} tok/s "
            f"decode/req p50={summary['decode_tok_s_p50']} tok/s "
            f"ttft p50={summary['ttft_s_p50']}s "
            f"accept={summary['spec_accept_length_mean']} "
            f"cached_mean={summary['cached_tokens_mean']} "
            f"retractions={summary['retractions_total']}"
            + ("" if exact else "  !! output length mismatch")
        )
        time.sleep(2.0)  # let the server drain between scenarios

    report["gpu_after"] = snapshot_gpu()
    report["all_ok"] = all_ok

    if args.save_dir:
        out_dir = pathlib.Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.profile}-bench-{utc_stamp()}.json"
        with open(path, "x") as fh:  # never overwrite existing evidence
            json.dump(report, fh, indent=2)
            fh.write("\n")
        print(f"report saved: {path}")
    else:
        print("(no --save-dir; report not saved)")

    print("BENCH OK" if all_ok else "BENCH COMPLETED WITH FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
