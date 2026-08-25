#!/usr/bin/env bash
# Stop NVFP4, serve Unsloth UD-Q8_K_XL, run MMLU, then write the quant report.
set -euo pipefail

ROOT="/home/derek/llm-benchmark/lm-eval"
RESULTS="$ROOT/results"
STATE="$RESULTS/qwen38-quant-pipeline.state"
GGUF_RECIPE="/home/derek/llm-benchmark/llama.cpp/recipes/unsloth-qwen3.8-27b-gguf"
GGUF_OUT="$RESULTS/unsloth-qwen3.8-27b-gguf-ud-q8-k-xl"
RUNNER="$ROOT/scripts/run_mmlu_llama_qwen38_27b_fp8.sh"
DOWNLOAD="$GGUF_RECIPE/download.sh"

emit() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "${2:-}" >"$STATE"
}

wait_for_health() {
  local expected_model="$1"
  local timeout_seconds="$2"
  local started=$SECONDS
  while (( SECONDS - started < timeout_seconds )); do
    if [[ "$(docker inspect -f '{{.State.Status}}' qwen38-27b-gguf 2>/dev/null || true)" == "exited" ]]; then
      return 1
    fi
    if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null |
      python3 -c "import json,sys; ids=[x['id'] for x in json.load(sys.stdin)['data']]; raise SystemExit(0 if '$expected_model' in ids else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 15
  done
  return 1
}

emit "starting-gguf" "unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"
docker rm -f qwen38-27b-nvfp4-mtp qwen38-27b-gguf >/dev/null 2>&1 || true

if [[ ! -f "$HOME/models/gguf/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf" ]]; then
  emit "downloading-gguf" "UD-Q8_K_XL"
  "$DOWNLOAD"
fi

FOLLOW_LOGS=0 PARALLEL=8 "$GGUF_RECIPE/start-docker.sh"

if ! wait_for_health qwen3.8-27b-gguf 10800; then
  docker logs qwen38-27b-gguf 2>&1 | tail -80 || true
  emit "gguf-mtp-retry" "retrying without draft-mtp"
  docker rm -f qwen38-27b-gguf >/dev/null 2>&1 || true
  FOLLOW_LOGS=0 PARALLEL=8 SPEC_TYPE=none "$GGUF_RECIPE/start-docker.sh"
  wait_for_health qwen3.8-27b-gguf 10800
fi

emit "benchmarking-gguf" "server healthy; MMLU starting"
mkdir -p "$GGUF_OUT"
MODEL_ID=qwen3.8-27b-gguf OUT_DIR="$GGUF_OUT" NUM_CONCURRENT=8 \
  "$RUNNER" --foreground

emit "gguf-complete" "mmlu finished"
python3 "$ROOT/scripts/write_qwen38_quant_report.py"
emit "pipeline-complete" "$RESULTS/QWEN3.8_27B_QUANT_REPORT.md"
echo GGUF_BENCH_DONE
