#!/usr/bin/env bash
# Docker llama-server for unsloth/Qwen3.8-27B-GGUF on DGX Spark.
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${PROFILE:-production}"
QUANT="${QUANT:-UD-Q8_K_XL}"
REPO="${REPO:-unsloth/Qwen3.8-27B-GGUF}"
PORT="${PORT:-8000}"
ALIAS="${ALIAS:-qwen3.8-27b-gguf}"
NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
PARALLEL="${PARALLEL:-4}"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gguf/Qwen3.8-27B-GGUF}"
MODEL_FILE_NAME="Qwen3.8-27B-${QUANT}.gguf"
USE_HF="${USE_HF:-0}"
VISION="${VISION:-0}"
MMPROJ_NAME="${MMPROJ_NAME:-mmproj-F16.gguf}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
NAME="${NAME:-qwen38-27b-gguf}"

case "$PROFILE" in
  production|mtp|prod)
    CTX_SIZE="${CTX_SIZE:-32768}"
    ;;
  long|longctx|niah)
    CTX_SIZE="${CTX_SIZE:-262144}"
    PARALLEL="${PARALLEL:-2}"
    ;;
  ar|nospec)
    CTX_SIZE="${CTX_SIZE:-32768}"
    SPEC_TYPE=none
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE' (use: production | long | ar)" >&2
    exit 1
    ;;
esac

ARGS=(
  --host 0.0.0.0
  --port 8000
  --alias "$ALIAS"
  --ctx-size "$CTX_SIZE"
  --n-gpu-layers "$NGL"
  --flash-attn "$FLASH_ATTN"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --parallel "$PARALLEL"
  --jinja
)

VOLUMES=(
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)

if [[ "$USE_HF" != "1" && -f "$MODEL_DIR/$MODEL_FILE_NAME" ]]; then
  echo "Using local GGUF $MODEL_DIR/$MODEL_FILE_NAME"
  VOLUMES+=(-v "$MODEL_DIR:/models:ro")
  ARGS+=(--model "/models/$MODEL_FILE_NAME")
  if [[ "$VISION" == "1" ]]; then
    ARGS+=(--mmproj "/models/$MMPROJ_NAME")
  fi
else
  echo "Using Hugging Face $REPO:$QUANT"
  ARGS+=(--hf-repo "${REPO}:${QUANT}")
fi

if [[ "$SPEC_TYPE" != "none" && -n "$SPEC_TYPE" ]]; then
  ARGS+=(--spec-type "$SPEC_TYPE")
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Removing existing container $NAME"
  docker rm -f "$NAME" >/dev/null
fi

echo "Pulling $LLAMA_IMAGE"
docker pull "$LLAMA_IMAGE"

echo "Starting $ALIAS ($QUANT) PROFILE=$PROFILE ctx=$CTX_SIZE on :$PORT"
docker run -d --name "$NAME" --gpus all --ipc=host --restart unless-stopped \
  -p "${PORT}:8000" \
  -e HF_HOME=/root/.cache/huggingface \
  "${VOLUMES[@]}" \
  "$LLAMA_IMAGE" \
  "${ARGS[@]}"

echo "Logs: docker logs -f $NAME"
if [[ "${FOLLOW_LOGS:-1}" == "1" ]]; then
  docker logs -f "$NAME"
fi
