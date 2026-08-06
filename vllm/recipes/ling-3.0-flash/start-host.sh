#!/bin/bash
# Serve inclusionAI/Ling-3.0-flash-fp4 via the official Ling vLLM fork (host).
# Prefer this over docker compose build on first boot (faster / matches HF card).
# For docker compose, use ./start.sh instead.
set -euo pipefail

MODEL="${MODEL:-inclusionAI/Ling-3.0-flash-fp4}"
PORT="${PORT:-8000}"
VENV="${VENV:-$HOME/my_ling_env}"
FORK_DIR="${FORK_DIR:-$HOME/src/vllm-ling-v3}"

# Persist torch.compile + FlashInfer autotune (1st start ~21 min, later ~5 min).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

if [[ ! -d "$VENV" ]]; then
  echo "Missing venv at $VENV"
  echo "Install (from https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4):"
  echo "  uv venv $VENV && source $VENV/bin/activate"
  echo "  git clone -b ling_3_0 https://github.com/inclusionAI/vllm-ling-v3.git $FORK_DIR"
  echo "  cd $FORK_DIR && VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found in $VENV — install the ling_3_0 fork first (see README.md)."
  exit 1
fi

echo "Serving $MODEL on :$PORT (stop anything else on this port first)."
exec vllm serve "$MODEL" \
  --served-model-name ling-3.0-flash \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser ling3 \
  --reasoning-parser ling3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
