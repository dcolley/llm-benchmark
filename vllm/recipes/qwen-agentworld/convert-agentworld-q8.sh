#!/usr/bin/env bash
# Convert Qwen-AgentWorld-35B-A3B HF checkpoint to GGUF Q8_0.
#
# Needs ~40–80 GiB free RAM during conversion. Stop vLLM on this node if low on memory.
#
# Usage:
#   ./convert-agentworld-q8.sh
#   OUTDIR=~/models/gguf ./convert-agentworld-q8.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen-AgentWorld-35B-A3B/snapshots/fec3dfb109ca88f102b058a969bbcc258be866e7}"
OUTDIR="${OUTDIR:-$HOME/models/gguf}"
OUTFILE="${OUTFILE:-$OUTDIR/Qwen-AgentWorld-35B-A3B-Q8_0.gguf}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]]; then
  echo "Missing $LLAMA_CPP/convert_hf_to_gguf.py" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing model dir: $MODEL_DIR" >&2
  echo "Run: huggingface-cli download Qwen/Qwen-AgentWorld-35B-A3B" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
AVAIL_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
if [[ "$AVAIL_GB" -lt 32 ]]; then
  echo "Warning: only ${AVAIL_GB} GiB MemAvailable — stop vLLM if conversion OOMs." >&2
fi

echo "Model:  $MODEL_DIR"
echo "Output: $OUTFILE"
echo "Avail:  ${AVAIL_GB} GiB"

cd "$LLAMA_CPP"
exec "$PYTHON" convert_hf_to_gguf.py \
  --outtype q8_0 \
  --no-mtp \
  --outfile "$OUTFILE" \
  --use-temp-file \
  "$MODEL_DIR"
