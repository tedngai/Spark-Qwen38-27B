#!/usr/bin/env bash
# One-time prefetch for Qwen3.8-27B NVFP4: pull the SGLang image and download
# the model + DSpark draft model into the shared Hugging Face cache
# (.cache/huggingface, bind-mounted into the container).
#
#   ./configs/docker/prefetch.sh
#
# Needs HF_TOKEN in .env for gated/private repos (both models are public).
# Progress: logs/prefetch.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
DRAFT_MODEL_ID="RadixArk/Qwen3.8-27B-DSpark"
IMAGE="lmsysorg/sglang:qwen38-27b"
ENV_FILE="${REPO_ROOT}/.env"
HF_HOME="${REPO_ROOT}/.cache/huggingface"
LOG_FILE="logs/prefetch.log"
PREFETCH_CONTAINER_NAME="qwen3.8-27b-nvfp4-prefetch"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" logs

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
else
  echo "Missing ${ENV_FILE} (HF_TOKEN may be required for downloads)"
fi

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] prefetching ${MODEL_ID} + ${DRAFT_MODEL_ID}
EOF

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} already present locally"
else
  echo "Pulling image ${IMAGE} (progress in ${LOG_FILE})"
  docker pull "${IMAGE}" 2>&1 | tee -a "${LOG_FILE}"
fi

docker rm -f "${PREFETCH_CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Downloading ${MODEL_ID} and ${DRAFT_MODEL_ID} into ${HF_HOME}"
echo "This can take a while; progress in ${LOG_FILE}"
docker run --rm \
  --name "${PREFETCH_CONTAINER_NAME}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  "${IMAGE}" \
  bash -c '
    set -e
    if command -v hf >/dev/null 2>&1; then
      hf download "$1"
      hf download "$2"
    else
      python3 -c "import sys; from huggingface_hub import snapshot_download; [snapshot_download(m) for m in sys.argv[1:]]" "$1" "$2"
    fi
  ' _ "${MODEL_ID}" "${DRAFT_MODEL_ID}" 2>&1 | tee -a "${LOG_FILE}"

echo "Prefetch complete"
du -sh "${HF_HOME}"
echo "Log: ${LOG_FILE}"
