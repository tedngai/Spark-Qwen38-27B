# Evidence

Timestamped, committed records of what was actually measured. Nothing in this
directory is edited after the fact; failed or invalid probes are retained and
marked, never overwritten.

Naming conventions (UTC `YYYYMMDDTHHMMSSZ`):

| Pattern | Produced by | Contents |
|---|---|---|
| `<profile>-server-info-<ts>.json` | `tests/capture_server.py` | applied server config (`/get_server_info`), health, models, container inspect |
| `<profile>-smoke-<ts>.json` | `tests/smoke_test.py` | health + one chat completion (request/response verbatim) |
| `<profile>-bench-<ts>.json` | `tests/benchmark.py` | A/B concurrency benchmark: per-scenario summaries + per-request records, GPU snapshots, git rev |
| `<profile>-vllmbench-<ts>/` | `tests/run_profile.sh` | vllm-benchmark-suite outputs (result JSON, benchmark CSV/JSON) |
| `legacy-*/` | pre-packaging | historical runs kept for lineage; see `docs/results.md` |

<profile> is the launch profile label from `configs/docker/candidates/` (or
`manual` for ad-hoc runs against a live server).

The A/B benchmark seeds are deterministic per scenario name, so two
`<profile>-bench-*.json` files can be compared directly with
`python3 tests/compare.py`.
