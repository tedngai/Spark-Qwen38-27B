# Tests

Host-side validation and measurement scripts. Stdlib-only Python, run with
`python3` from the repo root. They never start or stop the server (except
`run_profile.sh`, which is the one-shot orchestrator).

| Script | What it does | Writes |
|---|---|---|
| `smoke_test.py` | `/health` + one chat completion against a running server | `evidence/<profile>-smoke-<ts>.json` with `--save-dir` |
| `capture_server.py` | `/health` + `/v1/models` + full `/get_server_info` config + container inspect (secrets filtered) | `evidence/<profile>-server-info-<ts>.json` |
| `benchmark.py` | A/B concurrency benchmark: fixed-seed prompts, forced 256-token decodes, TTFT / decode tok/s / aggregate / spec-accept / retractions at c1..c16 and 16K-token prompts | `evidence/<profile>-bench-<ts>.json` |
| `run_profile.sh` | full cycle: stop → start `<profile>` → capture → smoke → A/B bench → vllm-benchmark-suite → stop | `evidence/<profile>-*`, `logs/suite-*.log` |
| `compare.py` | tables comparing profiles from `evidence/*-bench-*.json` (+ suite scorecard) | stdout |
| `gen_manifest.py` | regenerates `docs/package-manifest.json` (`--check` to verify) | `docs/package-manifest.json` |

Examples:

```bash
# measure the current server quickly (no restart)
python3 tests/smoke_test.py --profile manual --save-dir evidence
python3 tests/benchmark.py --profile manual --save-dir evidence --skip-long

# full benchmark cycle for a profile (STOPS any running server first)
tests/run_profile.sh baseline
tests/run_profile.sh no-spec --quick --no-suite

# compare everything measured so far
python3 tests/compare.py
```

The vllm-benchmark-suite (vllm-bench) is wrapped by `run_profile.sh`; run it
manually with `vllm-bench --url http://127.0.0.1:8000 --quick -y`.

Exit codes: 0 = all checks passed; 1 = failure (reports are still saved —
retain failed runs in `evidence/`).
