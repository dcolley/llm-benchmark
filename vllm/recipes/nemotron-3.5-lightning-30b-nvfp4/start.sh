#!/usr/bin/env bash
# Serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on DGX Spark (GB10).
#
# Official 1× Spark recipe: DSpark speculative decoding + Marlin MoE + FlashInfer Mamba.
# See README.md and https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm/nemotron-35-lightning-env}"
if [[ ! -d "$VENV" ]]; then
  echo "Missing venv at $VENV — follow readme-venv.md first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found in $VENV — install vLLM 0.27.x (readme-venv.md)." >&2
  exit 1
fi

if ss -ltn 2>/dev/null | grep -qE ':8000\b'; then
  echo "Port 8000 is in use. Stop the other server first (Muse / Ling / ds4 / etc.)." >&2
  ss -ltnp 2>/dev/null | grep -E ':8000\b' || true
  exit 1
fi

# GB10 / sm_121
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
DSPARK="${DSPARK:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
PORT="${PORT:-8000}"
# Official card supports 1M; 262144 is a safer house default on 128 GB UMA — raise if you have headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.91}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"

ARGS=(
  serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --moe-backend marlin
  --kv-cache-dtype fp8
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --mamba-backend flashinfer
  --mamba-cache-mode align
  --reasoning-parser nemotron_v3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
)

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  ARGS+=(
    --speculative-config
    "{\"model\": \"${DSPARK}\", \"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS}}"
  )
fi

echo "Serving $MODEL on :$PORT (DSpark=$ENABLE_DSPARK, ctx=$MAX_MODEL_LEN, util=$GPU_MEMORY_UTILIZATION)."
echo "Draft: $DSPARK"
exec vllm "${ARGS[@]}"
