#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="qwen3.8-27b-nvfp4-sglang"
PREFETCH_CONTAINER_NAME="qwen3.8-27b-nvfp4-prefetch"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"

stopped_any=false

stop_container() {
  local container="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    echo "Stopping/removing container ${container}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    stopped_any=true
  fi
}

# Clean up old TokenSpeed / vLLM containers too (migration)
stop_container "qwen3.8-27b-fp8-tokenspeed"
stop_container "qwen3.8-27b-nvfp4-vllm"

stop_container "${CONTAINER_NAME}"
stop_container "${PREFETCH_CONTAINER_NAME}"

if [[ "${stopped_any}" == false ]]; then
  echo "No container found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
