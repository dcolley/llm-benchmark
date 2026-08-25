#!/usr/bin/env bash
# Serve Ornith-1.5-35B-A3B (NVFP4 default) on DGX Spark via Docker.
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${PROFILE:-production}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
export MODEL="${MODEL:-ornith-ai/Ornith-1.5-35B-A3B-NVFP4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ornith-1.5-35b-a3b-nvfp4}"
export PORT="${PORT:-8000}"

COMPOSE_FILES=(-f docker-compose.yml)
case "$PROFILE" in
  production|prod|default|mmlu)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="production/MMLU 262K eager (SM121-stable)"
    ;;
  chat)
    COMPOSE_FILES+=(-f docker-compose.chat.yml)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="chat 262K eager"
    ;;
  long|longctx)
    COMPOSE_FILES+=(-f docker-compose.long.yml)
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    LABEL="long context 262K eager"
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE' (use: production | chat | long)" >&2
    exit 1
    ;;
esac

echo "Pulling image if needed: $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

echo "Starting Ornith-1.5-35B ($LABEL) model=$MODEL on :$PORT"
docker compose "${COMPOSE_FILES[@]}" up -d "$@"
echo "Logs: docker compose ${COMPOSE_FILES[*]} logs -f"
if [[ "${FOLLOW_LOGS:-1}" == "1" ]]; then
  docker compose "${COMPOSE_FILES[@]}" logs -f
fi
