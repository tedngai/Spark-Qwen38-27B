#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen3.8-27b-nvfp4-sglang"
HOST="0.0.0.0"
PORT="8000"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
ENV_FILE="${WORK_DIR}/.env"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
MAX_RUNNING_REQUESTS=8
MAMBA_CACHE_SIZE=$((MAX_RUNNING_REQUESTS * 8))
YARN=0
CONTEXT_LENGTH=262144

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi

if (( CONTEXT_LENGTH < 262144 || CONTEXT_LENGTH > 1000000 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported (use 262144..1000000)"
  exit 1
fi
case "${YARN}" in
  0|1) : ;;
  *) echo "YARN must be 0 or 1, got '${YARN}'"; exit 1 ;;
esac
NEED_YARN=0
if (( CONTEXT_LENGTH > 262144 )); then
  if [[ "${YARN}" == "1" ]] || [[ "${CONTEXT_LENGTH}" == "1000000" ]]; then
    NEED_YARN=1
  else
    echo "warning: CONTEXT_LENGTH=${CONTEXT_LENGTH} needs YaRN but YARN=0; build will stay at 262K"
  fi
fi
if (( NEED_YARN )); then
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  (( YARN_FACTOR < 1 )) && YARN_FACTOR=1
  YARN_OVERRIDE=$(printf '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": %s, "original_max_position_embeddings": 262144}}}' "${YARN_FACTOR}")
  CONTEXT_ARGS=(--json-model-override-args "${YARN_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  YARN_SUFFIX=" (YaRN factor ${YARN_FACTOR})"
else
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
  YARN_SUFFIX=""
fi

# Clean up any old TokenSpeed / vLLM containers
for old in "qwen3.8-27b-fp8-tokenspeed" "qwen3.8-27b-nvfp4-vllm"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "${old}"; then
    echo "Removing old container ${old}"
    docker rm -f "${old}" >/dev/null 2>&1 || true
  fi
done

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "Context: ${CONTEXT_LENGTH} tokens${YARN_SUFFIX:-}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"
if [[ -f "${ENV_FILE}" ]]; then
  echo "Loaded environment from ${ENV_FILE}"
fi

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --restart unless-stopped \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --served-model-name "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --mem-fraction-static 0.92 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --mamba-ssm-dtype bfloat16 \
    --mamba-full-memory-ratio 4.21 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
    "${CONTEXT_ARGS[@]}" \
    --allow-auto-truncate \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Chat: ${CHAT_URL}"
echo "SGLang is ready and responding; shell is now free."
