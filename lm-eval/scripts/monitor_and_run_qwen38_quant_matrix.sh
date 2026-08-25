#!/usr/bin/env bash
# Continue the active Qwen FP8 MMLU run, then benchmark r0b0tlab NVFP4 and
# Unsloth UD-Q8_K_XL GGUF in sequence. Emits monitored stage-change events.
set -Eeuo pipefail

ROOT="/home/derek/llm-benchmark/lm-eval"
RESULTS="$ROOT/results"
RUNNER="$ROOT/scripts/run_mmlu_llama_qwen38_27b_fp8.sh"
REPORTER="$ROOT/scripts/write_qwen38_quant_report.py"
STATE="$RESULTS/qwen38-quant-pipeline.state"

FP8_OLD="$RESULTS/qwen3.8-27b-fp8"
FP8_OUT="$RESULTS/qwen-qwen3.8-27b-fp8"
NVFP4_OUT="$RESULTS/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121"
GGUF_OUT="$RESULTS/unsloth-qwen3.8-27b-gguf-ud-q8-k-xl"

NVFP4_RECIPE="/home/derek/llm-benchmark/vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121"
GGUF_RECIPE="/home/derek/llm-benchmark/llama.cpp/recipes/unsloth-qwen3.8-27b-gguf"

STAGE="initializing"

emit() {
  local event="$1"
  local detail="${2:-}"
  printf '%s\t%s\t%s\n' "$(date -Is)" "$event" "$detail" >"$STATE"
  printf 'AGENT_LOOP_WAKE_QWEN38 {"prompt":"Monitor Qwen3.8 quant benchmark pipeline; stage=%s; detail=%s. Report meaningful changes and keep it running."}\n' \
    "$event" "$detail"
}

fail() {
  local rc=$?
  emit "failed" "stage=${STAGE},exit=${rc}"
  exit "$rc"
}
trap fail ERR

has_result() {
  compgen -G "$1/**/results_*.json" >/dev/null
}

cache_progress() {
  local db="$1/cache.db_rank0.db"
  [[ -f "$db" ]] || return 0
  python3 - "$db" <<'PY' 2>/dev/null || true
import sqlite3, sys
try:
    count = sqlite3.connect(sys.argv[1], timeout=2).execute(
        "select count(*) from unnamed"
    ).fetchone()[0]
    print(f"{count}/14042 ({100*count/14042:.1f}%)")
except Exception:
    pass
PY
}

wait_for_health() {
  local container="$1"
  local expected_model="$2"
  local timeout_seconds="$3"
  local started=$SECONDS
  while (( SECONDS - started < timeout_seconds )); do
    if [[ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)" == "exited" ]]; then
      return 1
    fi
    if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null |
      python3 -c "import json,sys; ids=[x['id'] for x in json.load(sys.stdin)['data']]; raise SystemExit(0 if '$expected_model' in ids else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 15
  done
  return 1
}

run_mmlu() {
  local model_id="$1"
  local out_dir="$2"
  rm -rf "$out_dir"
  MODEL_ID="$model_id" OUT_DIR="$out_dir" NUM_CONCURRENT=8 \
    "$RUNNER" --foreground
  has_result "$out_dir"
}

mkdir -p "$RESULTS"
emit "monitoring-fp8" "$(cache_progress "$FP8_OLD")"

STAGE="wait-fp8"
FP8_PID="$(pgrep -f "$ROOT/bin/lm-eval.*model=qwen3.8-27b-fp8" | sort -n | awk 'NR == 1 { print; exit }' || true)"
if [[ -n "$FP8_PID" ]]; then
  while kill -0 "$FP8_PID" 2>/dev/null; do
    printf '%s\tmonitoring-fp8\t%s\n' "$(date -Is)" "$(cache_progress "$FP8_OLD")" >"$STATE"
    sleep 300
  done
fi
sleep 5
has_result "$FP8_OLD"

STAGE="archive-fp8"
if [[ "$FP8_OLD" != "$FP8_OUT" && -d "$FP8_OLD" ]]; then
  rm -rf "$FP8_OUT"
  mv "$FP8_OLD" "$FP8_OUT"
fi
python3 "$ROOT/scripts/write_mmlu_report_qwen38_27b_fp8.py" \
  --results-dir "$FP8_OUT" \
  --out "$FP8_OUT/MMLU_LLAMA_REPORT.md"
emit "fp8-complete" "$(cache_progress "$FP8_OUT")"

STAGE="start-nvfp4"
docker rm -f qwen38-27b-fp8 >/dev/null 2>&1 || true
emit "starting-nvfp4" "r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121"
FOLLOW_LOGS=0 "$NVFP4_RECIPE/start-docker.sh"

STAGE="health-nvfp4"
if ! wait_for_health qwen38-27b-nvfp4-mtp qwen3.8-27b-nvfp4-mtp 10800; then
  docker logs qwen38-27b-nvfp4-mtp 2>&1 || true
  false
fi
emit "benchmarking-nvfp4" "MMLU 0/14042"

STAGE="benchmark-nvfp4"
run_mmlu qwen3.8-27b-nvfp4-mtp "$NVFP4_OUT"
emit "nvfp4-complete" "$(cache_progress "$NVFP4_OUT")"

STAGE="start-gguf"
docker rm -f qwen38-27b-nvfp4-mtp >/dev/null 2>&1 || true
emit "starting-gguf" "unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"
FOLLOW_LOGS=0 PARALLEL=8 "$GGUF_RECIPE/start-docker.sh"

STAGE="health-gguf"
if ! wait_for_health qwen38-27b-gguf qwen3.8-27b-gguf 10800; then
  docker logs qwen38-27b-gguf 2>&1 || true
  emit "gguf-mtp-retry" "retrying without draft-mtp"
  docker rm -f qwen38-27b-gguf >/dev/null 2>&1 || true
  FOLLOW_LOGS=0 PARALLEL=8 SPEC_TYPE=none "$GGUF_RECIPE/start-docker.sh"
  wait_for_health qwen38-27b-gguf qwen3.8-27b-gguf 10800
fi
emit "benchmarking-gguf" "MMLU 0/14042"

STAGE="benchmark-gguf"
run_mmlu qwen3.8-27b-gguf "$GGUF_OUT"
emit "gguf-complete" "$(cache_progress "$GGUF_OUT")"

STAGE="report"
python3 "$REPORTER"
emit "pipeline-complete" "$RESULTS/QWEN3.8_27B_QUANT_REPORT.md"
