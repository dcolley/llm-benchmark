#! /bin/bash
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm/.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

vllm serve nvidia/Qwen3.6-27B-NVFP4 \
  --port 8000 \
  --quantization modelopt \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.75 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
