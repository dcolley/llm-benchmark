#!/usr/bin/env bash
# Host llama-server for unsloth/Qwen3.8-27B-GGUF on DGX Spark (GB10).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="${PIDFILE:-$ROOT/llama-server.pid}"
LOGFILE="${LOGFILE:-$ROOT/llama-server.log}"

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP/build/bin/llama-server}"
REPO="${REPO:-unsloth/Qwen3.8-27B-GGUF}"
QUANT="${QUANT:-UD-Q8_K_XL}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gguf/Qwen3.8-27B-GGUF}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-27B-${QUANT}.gguf}"
PORT="${PORT:-8000}"
ALIAS="${ALIAS:-qwen3.8-27b-gguf}"
NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
PARALLEL="${PARALLEL:-4}"
VISION="${VISION:-0}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}"
USE_HF="${USE_HF:-0}"
PROFILE="${PROFILE:-production}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"

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

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Missing llama-server at $LLAMA_SERVER" >&2
  echo "Build ~/llama.cpp with CMAKE_CUDA_ARCHITECTURES=121a-real and GGML_CUDA_COMPRESSION_MODE=speed," >&2
  echo "or use ./start-docker.sh" >&2
  exit 1
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running (pid $(cat "$PIDFILE")). Stop with ./stop.sh first." >&2
  exit 1
fi

ARGS=(
  --host 0.0.0.0
  --port "$PORT"
  --alias "$ALIAS"
  --ctx-size "$CTX_SIZE"
  --n-gpu-layers "$NGL"
  --flash-attn "$FLASH_ATTN"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --parallel "$PARALLEL"
  --jinja
)

if [[ "$USE_HF" == "1" || ! -f "$MODEL_FILE" ]]; then
  echo "Using Hugging Face repo $REPO:$QUANT (file missing locally or USE_HF=1)"
  ARGS+=(--hf-repo "${REPO}:${QUANT}")
else
  ARGS+=(--model "$MODEL_FILE")
fi

if [[ "$VISION" == "1" ]]; then
  [[ -f "$MMPROJ" ]] || { echo "Missing mmproj: $MMPROJ (VISION=1 ./download.sh)" >&2; exit 1; }
  ARGS+=(--mmproj "$MMPROJ")
fi

if [[ "$SPEC_TYPE" != "none" && -n "$SPEC_TYPE" ]]; then
  ARGS+=(--spec-type "$SPEC_TYPE")
fi

echo "Starting $ALIAS ($QUANT) PROFILE=$PROFILE ctx=$CTX_SIZE port=$PORT"
echo "Binary: $LLAMA_SERVER"
echo "Log: $LOGFILE"

nohup "$LLAMA_SERVER" "${ARGS[@]}" >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "pid $(cat "$PIDFILE")"
tail -f "$LOGFILE"
