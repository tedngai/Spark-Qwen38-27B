#!/usr/bin/env bash
# Qwen3.8-27B (RadixArk/Qwen3.8-27B-NVFP4) — SGLang in Docker on the DGX Spark (GB10, SM121).
#
# Profile-driven launcher: every tunable comes from
# configs/docker/candidates/<profile>.env (profile wins over .env and the
# defaults below). Usage:
#
#   ./configs/docker/run_qwen38_sglang.sh [profile]      # default: baseline
#
# The container runs detached with `--restart no` (manual runs only — see
# docs/TESTING.md). This script blocks until the server answers /v1/models
# and exits 0; stop with ./configs/docker/stop.sh.
#
# Guards: refuses to start if the container is already running; if the
# container dies before becoming ready it prints the log tail and exits 1.
# No retries, no restart loops.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-baseline}"
PROFILE_FILE="configs/docker/candidates/${PROFILE}.env"

if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "ERROR: unknown profile '${PROFILE}' (no ${PROFILE_FILE})" >&2
  echo "Available profiles:" >&2
  for f in configs/docker/candidates/*.env; do
    [[ -e "$f" ]] || continue
    echo "  $(basename "$f" .env)" >&2
  done
  exit 1
fi

# ---------------------------------------------------------------- defaults --
# Validated baseline (packaged 2026-10-16). Profiles may override any of these.
MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen3.8-27b-nvfp4-sglang"
HOST="0.0.0.0"
PORT="8000"
RESTART_POLICY="no"                 # manual runs only (docs/TESTING.md)
KV_CACHE_DTYPE="fp8_e4m3"
MEM_FRACTION="0.85"
MAX_RUNNING_REQUESTS="8"
ATTENTION_BACKEND="flashinfer"
CHUNKED_PREFILL="2048"
MAMBA_SSM_DTYPE="float32"
MAMBA_FULL_MEMORY_RATIO="11.01"
MAMBA_RADIX_CACHE_STRATEGY="extra_buffer_lazy"
MAX_MAMBA_CACHE_SIZE=""             # empty = engine default
DISABLE_PREFILL_CUDA_GRAPH="0"
SPEC_ALGORITHM="DSPARK"             # DSPARK | DFLASH | EAGLE | empty (no speculative decoding)
SPEC_DRAFT_MODEL="RadixArk/Qwen3.8-27B-DSpark"
SPEC_DRAFT_ATTENTION_BACKEND="flashinfer"
SPEC_NUM_STEPS=""                   # EAGLE only
SPEC_EAGLE_TOPK=""                  # EAGLE only
SPEC_NUM_DRAFT_TOKENS=""            # optional for both
CONTEXT_LENGTH="262144"
YARN="0"

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/qwen38_sglang.log"
PID_FILE="${LOG_DIR}/qwen38_sglang.pid"
LOGFOLLOW_PID_FILE="${LOG_DIR}/qwen38_sglang_logfollow.pid"
HF_HOME="${REPO_ROOT}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# -------------------------------------------------- env / profile overrides --
if [[ -f .env ]]; then
  # shellcheck source=/dev/null
  source .env                        # HF_TOKEN (+ optional GITEA_TOKEN); may set CONTEXT_LENGTH/YARN
fi
# shellcheck source=/dev/null
source "$PROFILE_FILE"               # profile always wins

# -------------------------------------------------------------- validation --
if (( CONTEXT_LENGTH < 262144 || CONTEXT_LENGTH > 1000000 )); then
  echo "ERROR: CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported (use 262144..1000000)" >&2
  exit 1
fi
case "${YARN}" in
  0|1) : ;;
  *) echo "ERROR: YARN must be 0 or 1, got '${YARN}'" >&2; exit 1 ;;
esac
NEED_YARN=0
if (( CONTEXT_LENGTH > 262144 )); then
  if [[ "${YARN}" == "1" ]] || [[ "${CONTEXT_LENGTH}" == "1000000" ]]; then
    NEED_YARN=1
  else
    echo "ERROR: CONTEXT_LENGTH=${CONTEXT_LENGTH} needs YARN=1 (or exactly 1000000)" >&2
    exit 1
  fi
fi
if (( NEED_YARN )); then
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  if (( YARN_FACTOR < 1 )); then YARN_FACTOR=1; fi
  YARN_OVERRIDE=$(printf '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": %s, "original_max_position_embeddings": 262144}}}' "${YARN_FACTOR}")
  CONTEXT_ARGS=(--json-model-override-args "${YARN_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  YARN_SUFFIX=" (YaRN factor ${YARN_FACTOR})"
else
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
  YARN_SUFFIX=""
fi

# ------------------------------------------------------------------ guards --
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not on PATH" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: cannot talk to docker (is the daemon up?)" >&2; exit 1; }

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "ERROR: container ${CONTAINER_NAME} is already running." >&2
  echo "       Stop it first: ./configs/docker/stop.sh" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  mkdir -p "${LOG_DIR}"
  echo "Found exited container ${CONTAINER_NAME}; saving last 200 log lines to ${LOG_DIR}/qwen38_sglang_prev.log"
  echo "=== [$(date -Is)] previous ${CONTAINER_NAME} log tail ===" >> "${LOG_DIR}/qwen38_sglang_prev.log"
  docker logs --tail 200 "${CONTAINER_NAME}" >> "${LOG_DIR}/qwen38_sglang_prev.log" 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# Legacy containers from earlier migration stages (TokenSpeed, vLLM) — harmless cleanup.
for old in "qwen3.8-27b-fp8-tokenspeed" "qwen3.8-27b-nvfp4-vllm"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "${old}"; then
    echo "Removing legacy container ${old}"
    docker rm -f "${old}" >/dev/null 2>&1 || true
  fi
done

# ------------------------------------------------------------- arg assembly --
SPEC_ARGS=()
case "${SPEC_ALGORITHM}" in
  "")
    : ;;                             # speculative decoding disabled
  DSPARK|DFLASH)
    SPEC_ARGS=(--speculative-algorithm "${SPEC_ALGORITHM}"
               --speculative-draft-model-path "${SPEC_DRAFT_MODEL}")
    if [[ -n "${SPEC_DRAFT_ATTENTION_BACKEND}" ]]; then
      SPEC_ARGS+=(--speculative-draft-attention-backend "${SPEC_DRAFT_ATTENTION_BACKEND}")
    fi
    if [[ -n "${SPEC_NUM_DRAFT_TOKENS}" ]]; then
      SPEC_ARGS+=(--speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}")
    fi
    ;;
  EAGLE)
    SPEC_ARGS=(--speculative-algorithm EAGLE)
    if [[ -n "${SPEC_NUM_STEPS}" ]]; then
      SPEC_ARGS+=(--speculative-num-steps "${SPEC_NUM_STEPS}")
    fi
    if [[ -n "${SPEC_EAGLE_TOPK}" ]]; then
      SPEC_ARGS+=(--speculative-eagle-topk "${SPEC_EAGLE_TOPK}")
    fi
    if [[ -n "${SPEC_NUM_DRAFT_TOKENS}" ]]; then
      SPEC_ARGS+=(--speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}")
    fi
    ;;
  *)
    echo "ERROR: SPEC_ALGORITHM must be DSPARK, DFLASH, EAGLE, or empty; got '${SPEC_ALGORITHM}'" >&2
    exit 1
    ;;
esac

MAMBA_ARGS=(--mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
            --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}"
            --mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY}")
if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
  MAMBA_ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi

GRAPH_ARGS=()
if [[ "${DISABLE_PREFILL_CUDA_GRAPH}" == "1" ]]; then
  GRAPH_ARGS=(--disable-prefill-cuda-graph)
fi

# ------------------------------------------------------------------ launch --
mkdir -p "${LOG_DIR}" "${HF_HOME}"

echo "Starting SGLang — profile=${PROFILE}"
echo "  model=${MODEL_ID}"
if [[ "${SPEC_ALGORITHM}" == "DSPARK" || "${SPEC_ALGORITHM}" == "DFLASH" ]]; then
  echo "  spec=${SPEC_ALGORITHM} (draft: ${SPEC_DRAFT_MODEL})"
elif [[ "${SPEC_ALGORITHM}" == "EAGLE" ]]; then
  echo "  spec=EAGLE (in-checkpoint MTP head)"
else
  echo "  spec=off"
fi
echo "  context=${CONTEXT_LENGTH}${YARN_SUFFIX}  mem-fraction=${MEM_FRACTION}  max-running-requests=${MAX_RUNNING_REQUESTS}"
echo "  image=${IMAGE}  port=${PORT}  restart=${RESTART_POLICY}"
echo "  log=${LOG_FILE}"

cat >>"${LOG_FILE}" <<EOF
[$(date -Is)] launching profile=${PROFILE} (restart=${RESTART_POLICY})
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --restart "${RESTART_POLICY}" \
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
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --chunked-prefill-size "${CHUNKED_PREFILL}" \
    "${MAMBA_ARGS[@]}" \
    "${GRAPH_ARGS[@]}" \
    "${CONTEXT_ARGS[@]}" \
    --allow-auto-truncate \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    "${SPEC_ARGS[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id:0:12})"

# Follow container logs into ${LOG_FILE}; the follower exits on its own when
# the container is removed (stop.sh also kills it).
nohup docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1 &
echo $! > "${LOGFOLLOW_PID_FILE}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
echo "Waiting for ${READY_URL} + /health (warm start typically takes ~3.5 min)"
start_ts=$(date +%s)
last_report=0
while :; do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1 && curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "ERROR: SGLang container exited before becoming ready. Last 40 lines of ${LOG_FILE}:" >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  elapsed=$(( $(date +%s) - start_ts ))
  if (( elapsed - last_report >= 15 )); then
    echo "  still starting... (${elapsed}s)"
    last_report=${elapsed}
  fi
  sleep 5
done

elapsed=$(( $(date +%s) - start_ts ))
echo "SGLang is ready after ${elapsed}s (profile=${PROFILE})"
echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
echo "Next: tests/run_profile.sh ${PROFILE}   (stops the server first; a profile can only be measured cleanly from a fresh start)"
