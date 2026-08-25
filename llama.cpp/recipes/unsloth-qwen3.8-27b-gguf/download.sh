#!/usr/bin/env bash
# Download Unsloth Qwen3.8-27B GGUF quant into ~/models/gguf/Qwen3.8-27B-GGUF/
set -euo pipefail

REPO="${REPO:-unsloth/Qwen3.8-27B-GGUF}"
QUANT="${QUANT:-UD-Q8_K_XL}"
OUTDIR="${OUTDIR:-$HOME/models/gguf/Qwen3.8-27B-GGUF}"
VISION="${VISION:-0}"

FILE="Qwen3.8-27B-${QUANT}.gguf"
mkdir -p "$OUTDIR"

HF_BIN=(huggingface-cli)
if command -v hf >/dev/null 2>&1; then
  HF_BIN=(hf)
fi

echo "Downloading $REPO / $FILE → $OUTDIR"
"${HF_BIN[@]}" download "$REPO" "$FILE" --local-dir "$OUTDIR"

if [[ "$VISION" == "1" ]]; then
  MMPROJ="${MMPROJ:-mmproj-F16.gguf}"
  echo "Downloading vision projector $MMPROJ"
  "${HF_BIN[@]}" download "$REPO" "$MMPROJ" --local-dir "$OUTDIR"
fi

ls -lh "$OUTDIR/$FILE"
echo "Done. QUANT=$QUANT path=$OUTDIR/$FILE"
