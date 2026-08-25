#!/usr/bin/env bash
# Host venv serve for Qwen/Qwen3.8-27B-FP8 on DGX Spark (GB10).
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm/.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-fp8}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --served-model-name "$SERVED_MODEL_NAME" \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"
