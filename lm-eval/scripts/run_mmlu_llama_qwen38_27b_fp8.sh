#!/usr/bin/env bash
# Run stock mmlu_llama against local Qwen3.8-27B-FP8 (vLLM).
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_ID="${MODEL_ID:-qwen3.8-27b-fp8}"
OUT_DIR="${OUT_DIR:-$ROOT/results/qwen-qwen3.8-27b-fp8}"
INCLUDE_PATH="${INCLUDE_PATH:-$ROOT/custom_tasks/mmlu_llama_ds4}"
LIMIT="${LIMIT:-}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [--limit N] [--foreground]

Env:
  BASE_URL       default http://127.0.0.1:8000
  MODEL_ID       default qwen3.8-27b-fp8
  OUT_DIR        default \$ROOT/results/qwen-qwen3.8-27b-fp8
  INCLUDE_PATH   default \$ROOT/custom_tasks/mmlu_llama_ds4
  NUM_CONCURRENT default 8
  ROOT           default /home/derek/llm-benchmark/lm-eval
EOF
}

FOREGROUND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --foreground|-f) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/resume.log"
CACHE="$OUT_DIR/cache.db"

# shellcheck disable=SC1091
source "$ROOT/bin/activate"
cd "$ROOT"

if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

# Use the extraction-based mmlu_llama variant because Qwen3.8 does not honor
# stock mmlu_llama's partial-assistant continuation ("The best answer is").
# Questions, targets, five-shot examples, and strict-match metric are unchanged.
CMD=(
  lm-eval --model local-chat-completions
  --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=${NUM_CONCURRENT:-8},tokenizer_backend=none,timeout=600,max_retries=5"
  --include_path "$INCLUDE_PATH"
  --tasks mmlu_llama_ds4
  --apply_chat_template
  --fewshot_as_multiturn
  --system_instruction "You are a helpful assistant. /no_think. Give no explanation. Reply only: The best answer is X."
  --gen_kwargs '{"max_tokens":64,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"until":[],"chat_template_kwargs":{"enable_thinking":false}}'
  --use_cache "$CACHE"
  --log_samples
  --output_path "$OUT_DIR"
  "${EXTRA_ARGS[@]}"
)

echo "Running: ${CMD[*]}"
echo "Log: $LOG"

if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
else
  nohup "${CMD[@]}" >>"$LOG" 2>&1 &
  echo "lm-eval pid=$!"
fi
