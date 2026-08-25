#!/usr/bin/env bash
# Serve meta-models/Muse-Glimmer-30B with vLLM on DGX Spark (GB10 / sm_121).
#
# Requires the Muse support venv (PR #51655) — see readme-venv.md.
# Optional DFlash: ENABLE_DFLASH=1 ./start.sh
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm/muse-glimmer-env}"
if [[ ! -d "$VENV" ]]; then
  echo "Missing venv at $VENV — follow readme-venv.md first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found in $VENV — install the Muse support branch (readme-venv.md)." >&2
  exit 1
fi

# GB10 / sm_121: cute-DSL and arch list for local kernels.
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

# Persist torch.compile + FlashInfer / Triton autotune across restarts.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
ENABLE_DFLASH="${ENABLE_DFLASH:-0}"
DRAFT_MODEL="${DRAFT_MODEL:-meta-models/Muse-Glimmer-30B-assistant}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-16}"

ARGS=(
  serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --attention-backend "$ATTENTION_BACKEND"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser muse_glimmer
  --reasoning-parser muse_glimmer
)

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  ARGS+=(
    --speculative-config
    "{\"method\": \"dflash\", \"model\": \"${DRAFT_MODEL}\", \"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS}}"
  )
fi

echo "Serving $MODEL on :$PORT (CUTE_DSL_ARCH=$CUTE_DSL_ARCH, util=$GPU_MEMORY_UTILIZATION)."
echo "Stop other :8000 / GPU servers first if UMA is tight."
exec vllm "${ARGS[@]}"
