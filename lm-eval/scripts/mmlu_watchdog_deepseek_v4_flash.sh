#!/usr/bin/env bash
# Watchdog for mmlu_llama against local DeepSeek-V4-Flash (ds4-server).
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
CACHE_DB="${CACHE_DB:-$ROOT/results/deepseek-v4-flash-0731/cache.db_rank0.db}"
LOG="${LOG:-$ROOT/results/deepseek-v4-flash-0731/resume.log}"
WATCH_LOG="${WATCH_LOG:-$ROOT/results/deepseek-v4-flash-0731/watchdog.log}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STALE_SECS="${STALE_SECS:-420}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-90}"
POLL_SECS="${POLL_SECS:-60}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
RUNNER="${RUNNER:-$ROOT/scripts/run_mmlu_llama_deepseek_v4_flash.sh}"
SERVE_CMD="${SERVE_CMD:-}"
# Short ctx is enough for mmlu_llama (~1k tokens) and frees KV for concurrency.
DS4_CTX="${DS4_CTX:-8192}"
DS4_BIN="${DS4_BIN:-/home/derek/code/ds4/ds4-server}"
GGUF="${GGUF:-$HOME/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"

mkdir -p "$(dirname "$WATCH_LOG")" "$ROOT/results/deepseek-v4-flash-0731"
log() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$WATCH_LOG"; }

cache_rows() {
  python3 - <<PY
import sqlite3, os
p=${CACHE_DB@Q}
if not os.path.exists(p):
    print(0); raise SystemExit
c=sqlite3.connect(p)
try:
    print(c.execute("SELECT COUNT(*) FROM unnamed").fetchone()[0])
except Exception:
    print(0)
PY
}

cache_age() {
  python3 - <<PY
import os, time
p=${CACHE_DB@Q}
print(int(time.time()-os.path.getmtime(p)) if os.path.exists(p) else 999999)
PY
}

lm_eval_pids() { pgrep -f "$ROOT/bin/lm-eval" || true; }
models_ok() { timeout 5 curl -sf "$BASE_URL/v1/models" >/dev/null; }

chat_ok() {
  timeout "$PROBE_TIMEOUT" curl -sf "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2? A.3 B.4\\nEnd with: The best answer is [letter].\"}],\"max_tokens\":64,\"temperature\":0,\"reasoning_effort\":\"low\"}" \
    | python3 -c "import sys,json,re; o=json.load(sys.stdin); c=(o['choices'][0]['message'].get('content') or '')+(o['choices'][0]['message'].get('reasoning_content') or ''); assert re.search(r'[ABCD]', c), 'no letter'" \
    >/dev/null 2>&1
}

wait_ready() {
  local i=0
  log "waiting for ds4 ready (timeout ${READY_TIMEOUT}s)"
  while (( i < READY_TIMEOUT )); do
    if models_ok && chat_ok; then
      log "ds4 ready after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if (( i % 60 == 0 )); then log "still waiting (${i}s)..."; fi
  done
  log "ERROR: ds4 not ready within ${READY_TIMEOUT}s"
  return 1
}

kill_lm_eval() {
  local pids
  pids=$(lm_eval_pids)
  [[ -z "$pids" ]] && { log "no lm-eval to kill"; return 0; }
  log "killing lm-eval: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 2
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
}

restart_server() {
  log "restarting ds4 (ctx=$DS4_CTX) via $DS4_BIN"
  pkill -f 'ds4-server' 2>/dev/null || true
  sleep 3
  if [[ -x "$DS4_BIN" ]]; then
    export DS4_CONT_DSPARK="${DS4_CONT_DSPARK:-1}"
    export DS4_CONT_MTP_MODE="${DS4_CONT_MTP_MODE:-2}"
    export DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
    nohup "$DS4_BIN" --cuda -m "$GGUF" -c "$DS4_CTX" --port 8000 \
      >>"$ROOT/results/deepseek-v4-flash-0731/serve.log" 2>&1 &
  elif [[ -n "${SERVE_CMD:-}" && -x "$SERVE_CMD" ]]; then
    nohup env CONTEXT="$DS4_CTX" "$SERVE_CMD" --no-install >>"$ROOT/results/deepseek-v4-flash-0731/serve.log" 2>&1 &
  else
    log "ERROR: no ds4 binary; cannot auto-restart"
    return 1
  fi
  wait_ready
}

resume_eval() {
  if [[ -n "$(lm_eval_pids)" ]]; then
    log "lm-eval already running; not launching another"
    return 0
  fi
  log "resuming lm-eval"
  nohup "$RUNNER" >>"$LOG" 2>&1 &
  log "launched runner pid=$!"
}

log "watchdog start STALE_SECS=$STALE_SECS"
while true; do
  rows=$(cache_rows)
  age=$(cache_age)
  pids=$(lm_eval_pids)
  if [[ -z "$pids" ]]; then
    if compgen -G "$ROOT/results/deepseek-v4-flash-0731/**/results_*.json" >/dev/null; then
      # ignore smoke-only folder
      newest=$(ls -1t "$ROOT"/results/deepseek-v4-flash-0731/*/results_*.json 2>/dev/null | grep -v '/smoke/' | head -1 || true)
      if [[ -n "$newest" ]]; then
        log "results present and no lm-eval — exiting ($newest)"
        exit 0
      fi
    fi
    log "lm-eval missing with no full results; resume"
    if ! models_ok || ! chat_ok; then
      restart_server || true
    fi
    resume_eval
  elif (( age > STALE_SECS )); then
    log "WEDGE suspected: cache_rows=$rows age=${age}s"
    if models_ok && chat_ok; then
      log "API still answers; waiting one more poll"
    else
      kill_lm_eval
      restart_server || true
      resume_eval
    fi
  else
    log "ok rows=$rows age=${age}s lm-eval=$(echo "$pids" | tr '\n' ' ')"
  fi
  sleep "$POLL_SECS"
done
