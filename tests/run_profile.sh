#!/usr/bin/env bash
# One-shot benchmark cycle for one launch profile:
#
#   stop current server -> start <profile> -> capture config -> smoke test
#   -> A/B concurrency benchmark -> vllm-benchmark-suite -> (stop)
#
#   tests/run_profile.sh <profile> [--quick] [--no-suite] [--standard-suite] [--keep-running]
#
#     --quick           A/B harness runs short scenarios only (vllm-bench stays --quick)
#     --no-suite        skip vllm-benchmark-suite
#     --standard-suite  use vllm-bench --standard instead of --quick
#     --keep-running    leave the server up after a successful run
#
# MANUAL USE ONLY (docs/TESTING.md): run it by hand, one profile at a time.
# It never retries and never loops; on any failure it stops and leaves the
# server running for inspection.
#
# Note: this STOPS any running server first — a profile can only be measured
# cleanly from a fresh start.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  echo "usage: tests/run_profile.sh <profile> [--quick] [--no-suite] [--standard-suite] [--keep-running]" >&2
  exit 2
fi
shift

QUICK=0
RUN_SUITE=1
SUITE_PRESET="--quick"
KEEP_RUNNING=0
for arg in "$@"; do
  case "${arg}" in
    --quick)          QUICK=1 ;;
    --no-suite)       RUN_SUITE=0 ;;
    --standard-suite) SUITE_PRESET="--standard" ;;
    --keep-running)   KEEP_RUNNING=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

# Validate the profile BEFORE stopping anything.
export PROFILE
if [[ ! -f "configs/docker/candidates/${PROFILE}.env" ]]; then
  echo "ERROR: unknown profile '${PROFILE}'" >&2
  echo "Available profiles:" >&2
  for f in configs/docker/candidates/*.env; do
    [[ -e "$f" ]] || continue
    echo "  $(basename "$f" .env)" >&2
  done
  exit 1
fi

on_error() {
  echo "" >&2
  echo "ERROR: profile run '${PROFILE}' aborted — server left running for inspection." >&2
  echo "  server log : logs/qwen38_sglang.log" >&2
  echo "  stop it    : configs/docker/stop.sh" >&2
}
trap on_error ERR

start_wall=$(date +%s)

echo "== [1/6] stopping any running server"
configs/docker/stop.sh

echo "== [2/6] starting profile '${PROFILE}'"
configs/docker/run_qwen38_sglang.sh "${PROFILE}"

echo "== [3/6] capturing server configuration"
python3 tests/capture_server.py --profile "${PROFILE}" --save-dir evidence

echo "== [4/6] smoke test"
python3 tests/smoke_test.py --profile "${PROFILE}" --save-dir evidence

echo "== [5/6] A/B concurrency benchmark"
BENCH_ARGS=(--profile "${PROFILE}" --save-dir evidence)
if [[ "${QUICK}" == "1" ]]; then BENCH_ARGS+=(--skip-long); fi
python3 tests/benchmark.py "${BENCH_ARGS[@]}"

if [[ "${RUN_SUITE}" == "1" ]]; then
  VLLM_BENCH="${VLLM_BENCH:-}"
  if [[ -z "${VLLM_BENCH}" ]]; then
    if command -v vllm-bench >/dev/null 2>&1; then
      VLLM_BENCH="$(command -v vllm-bench)"
    elif [[ -x "${HOME}/miniconda3/bin/vllm-bench" ]]; then
      VLLM_BENCH="${HOME}/miniconda3/bin/vllm-bench"
    fi
  fi
  if [[ -z "${VLLM_BENCH}" ]]; then
    echo "WARN: vllm-bench not found; skipping the suite (set VLLM_BENCH to override)" >&2
  else
    suite_ts="$(date -u +%Y%m%dT%H%M%SZ)"
    suite_dir="evidence/${PROFILE}-vllmbench-${suite_ts}"
    mkdir -p "${suite_dir}"
    echo "== [6/6] vllm-benchmark-suite (${SUITE_PRESET}) -> ${suite_dir}"
    "${VLLM_BENCH}" \
      --url "http://127.0.0.1:8000" \
      "${SUITE_PRESET}" -y --no-html --no-charts \
      --output-dir "${suite_dir}" \
      2>&1 | tee "logs/suite-${PROFILE}-${suite_ts}.log"
  fi
else
  echo "== [6/6] vllm-benchmark-suite skipped (--no-suite)"
fi

elapsed=$(( $(date +%s) - start_wall ))
echo ""
echo "== profile '${PROFILE}' benchmark cycle complete in ${elapsed}s"
echo "   evidence: evidence/${PROFILE}-*"
if [[ "${KEEP_RUNNING}" == "1" ]]; then
  echo "   server left running (--keep-running)"
else
  echo "   stopping server"
  configs/docker/stop.sh
fi
