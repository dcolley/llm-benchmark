#!/usr/bin/env bash
# Host venv serve for r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 on DGX Spark (GB10).
# Prefer start-docker.sh with the SM121 image; host needs vLLM ≥ 0.27.x + ModelOpt NVFP4 path.
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm/.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

PROFILE="${PROFILE:-production}"
MODEL="${MODEL:-r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-nvfp4-mtp}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

ARGS=(
  "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --kv-cache-dtype fp8
  --enforce-eager
  --no-enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
)

case "$PROFILE" in
  production|mtp|prod)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    ARGS+=(
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
    )
    ;;
  long|longctx|niah)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    ARGS+=(
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    )
    ;;
  ar|nospec)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    ARGS+=(
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    )
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE' (use: production | long | ar)" >&2
    exit 1
    ;;
esac

exec vllm serve "${ARGS[@]}"
