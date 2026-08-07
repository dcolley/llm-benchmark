#!/bin/bash
# Serve inclusionAI/Ling-3.0-flash-int4 via SGLang (host).
# Everyday Spark defaults: thinking on, context 262144.
set -euo pipefail

MODEL="${MODEL:-inclusionAI/Ling-3.0-flash-int4}"
PORT="${PORT:-8000}"
VENV="${VENV:-$HOME/my_ling_sglang_env}"
FORK_DIR="${FORK_DIR:-$HOME/src/sglang_ling_v3}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

if [[ ! -d "$VENV" ]]; then
  echo "Missing venv at $VENV"
  echo "See README.md for install steps."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c 'import sglang' >/dev/null 2>&1; then
  echo "sglang not importable in $VENV — install a Ling-capable SGLang first (see README.md)."
  exit 1
fi

echo "Serving $MODEL on :$PORT via SGLang (everyday: thinking on, ctx=262144, NEXTN)."
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --nnodes 1 \
  --tp-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --max-running-requests 4 \
  --chunked-prefill-size 8192 \
  --context-length 262144 \
  --max-mamba-cache-size 64 \
  --enable-fp32-lm-head \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN \
  --tool-call-parser ling3 \
  --reasoning-parser ling3 \
  --served-model-name ling-3.0-flash-int4
