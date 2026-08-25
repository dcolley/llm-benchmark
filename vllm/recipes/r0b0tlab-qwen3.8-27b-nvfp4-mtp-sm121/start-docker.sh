#!/usr/bin/env bash
# Serve r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 on DGX Spark (preferred: Docker).
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${PROFILE:-production}"
export VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121}"
export MODEL="${MODEL:-r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-nvfp4-mtp}"
export PORT="${PORT:-8000}"

COMPOSE_FILES=(-f docker-compose.yml)
case "$PROFILE" in
  production|mtp|prod)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="production MTP K=3"
    ;;
  long|longctx|niah)
    COMPOSE_FILES+=(-f docker-compose.long.yml)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="long context 262K (no MTP)"
    ;;
  ar|nospec)
    COMPOSE_FILES+=(-f docker-compose.ar.yml)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="AR floor (no speculation)"
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE' (use: production | long | ar)" >&2
    exit 1
    ;;
esac

echo "Pulling image if needed: $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

echo "Starting Qwen3.8-27B-NVFP4-MTP ($LABEL) on :$PORT"
docker compose "${COMPOSE_FILES[@]}" up -d "$@"
echo "Logs: docker compose ${COMPOSE_FILES[*]} logs -f"
if [[ "${FOLLOW_LOGS:-1}" == "1" ]]; then
  docker compose "${COMPOSE_FILES[@]}" logs -f
fi
