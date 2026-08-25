#!/usr/bin/env bash
# Run mmlu_llama_ds4 against local DeepSeek-V4-Flash-0731 (ds4 IQ2XXS/Q2K).
#
# Uses a thinking-friendly variant of mmlu_llama:
# - no gen_prefix / continue (DeepSeek emits "." first under letter-continuation,
#   which trips the stock until=["."] stop and yields empty answers)
# - regex extract of the final "The best answer is [A-D]"
# - max_tokens=256, reasoning_effort=low
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash}"
OUT_DIR="${OUT_DIR:-$ROOT/results/deepseek-v4-flash-0731}"
INCLUDE_PATH="${INCLUDE_PATH:-$ROOT/custom_tasks/mmlu_llama_ds4}"
LIMIT="${LIMIT:-}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [--limit N] [--foreground]

Env:
  BASE_URL      default http://127.0.0.1:8000
  MODEL_ID      default deepseek-v4-flash
  OUT_DIR       default \$ROOT/results/deepseek-v4-flash-0731
  INCLUDE_PATH  default \$ROOT/custom_tasks/mmlu_llama_ds4
  ROOT          default /home/derek/llm-benchmark/lm-eval
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

CMD=(
  lm-eval --model local-chat-completions
  --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=${NUM_CONCURRENT:-4},tokenizer_backend=none,timeout=600,max_retries=5"
  --include_path "$INCLUDE_PATH"
  --tasks mmlu_llama_ds4
  --apply_chat_template
  --fewshot_as_multiturn
  --gen_kwargs 'max_tokens=512,temperature=0,reasoning_effort=low,until=[]'
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
