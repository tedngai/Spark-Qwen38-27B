#!/usr/bin/env bash
# Stop and remove the Qwen3.8-27B SGLang container. Logs are preserved.
#
#   ./configs/docker/stop.sh
#
# Safe to run when nothing is running (prints "No container found").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONTAINER_NAME="qwen3.8-27b-nvfp4-sglang"
LOG_DIR="logs"
PID_FILE="${LOG_DIR}/qwen38_sglang.pid"
LOGFOLLOW_PID_FILE="${LOG_DIR}/qwen38_sglang_logfollow.pid"
LOG_FILE="${LOG_DIR}/qwen38_sglang.log"

stopped_any=false

# Stop the log follower if it is still alive (it also exits by itself once the
# container is gone).
if [[ -f "${LOGFOLLOW_PID_FILE}" ]]; then
  pid="$(cat "${LOGFOLLOW_PID_FILE}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${LOGFOLLOW_PID_FILE}"
fi

stop_container() {
  local container="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    echo "Stopping/removing container ${container}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    stopped_any=true
  fi
}

stop_container "${CONTAINER_NAME}"
# Legacy containers from earlier migration stages (TokenSpeed, vLLM, prefetch) — harmless cleanup.
stop_container "qwen3.8-27b-fp8-tokenspeed"
stop_container "qwen3.8-27b-nvfp4-vllm"
stop_container "qwen3.8-27b-nvfp4-prefetch"

if [[ "${stopped_any}" == false ]]; then
  echo "No container found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
