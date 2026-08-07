#!/usr/bin/env bash
# Run mmlu_llama against a remote OpenAI-compatible chat API (vLLM).
# Defaults target Akahsizrr/fuse-1-Lite on 192.168.10.199:8000.
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
BASE_URL="${BASE_URL:-http://192.168.10.199:8000}"
MODEL_ID="${MODEL_ID:-Akahsizrr/fuse-1-Lite}"
OUT_DIR="${OUT_DIR:-$ROOT/results/fuse-1-lite}"
LIMIT="${LIMIT:-}"   # empty = full run; set e.g. LIMIT=5 for smoke
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [--limit N] [--foreground]

Env:
  BASE_URL   default http://192.168.10.199:8000
  MODEL_ID   default Akahsizrr/fuse-1-Lite
  OUT_DIR    default \$ROOT/results/fuse-1-lite
  ROOT       default /home/derek/llm-benchmark/lm-eval
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
  --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=1,tokenizer_backend=none,timeout=180,max_retries=5"
  --tasks mmlu_llama
  --apply_chat_template
  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false'
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
