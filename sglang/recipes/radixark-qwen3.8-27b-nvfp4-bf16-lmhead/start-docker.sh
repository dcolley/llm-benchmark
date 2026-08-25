#!/usr/bin/env bash
# Serve RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead with SGLang on DGX Spark.
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${PROFILE:-production}"
export SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-qwen38-27b-dflash2}"
export MODEL="${MODEL:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-nvfp4}"
export PORT="${PORT:-8000}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

COMPOSE_FILES=(-f docker-compose.yml)
case "$PROFILE" in
  production|mtp|eagle|prod)
    export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
    LABEL="production MTP/EAGLE 32K"
    ;;
  long|longctx)
    COMPOSE_FILES+=(-f docker-compose.long.yml)
    export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
    LABEL="long context 262K + MTP"
    ;;
  dspark)
    COMPOSE_FILES+=(-f docker-compose.dspark.yml)
    export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
    export DRAFT_MODEL="${DRAFT_MODEL:-RadixArk/Qwen3.8-27B-DSpark}"
    LABEL="DSpark draft 32K"
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE' (use: production | long | dspark)" >&2
    exit 1
    ;;
esac

echo "Pulling $SGLANG_IMAGE"
docker pull "$SGLANG_IMAGE"

echo "Starting $MODEL ($LABEL) on :$PORT"
docker compose "${COMPOSE_FILES[@]}" up -d "$@"
echo "Logs: docker compose ${COMPOSE_FILES[*]} logs -f"
if [[ "${FOLLOW_LOGS:-1}" == "1" ]]; then
  docker compose "${COMPOSE_FILES[@]}" logs -f
fi
