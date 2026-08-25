#!/usr/bin/env bash
# Optional host-venv serve (prefer ./start-docker.sh on Spark).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-ornith-ai/Ornith-1.5-35B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ornith-1.5-35b-a3b-nvfp4}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-production}"

case "$PROFILE" in
  long|longctx|chat|*)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
    ;;
esac

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
