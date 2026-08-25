#!/usr/bin/env bash
# Watchdog for Ornith-1.5-35B-A3B-NVFP4 mmlu_llama (local Docker vLLM).
# On wedge: kill lm-eval, docker compose restart, wait, resume.
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
RECIPE_DIR="${RECIPE_DIR:-/home/derek/llm-benchmark/vllm/recipes/ornith-1.5-35b-a3b-nvfp4}"
OUT_DIR="${OUT_DIR:-$ROOT/results/ornith-1.5-35b-a3b-nvfp4}"
CACHE_DB="${CACHE_DB:-$OUT_DIR/cache.db_rank0.db}"
LOG="${LOG:-$OUT_DIR/resume.log}"
WATCH_LOG="${WATCH_LOG:-$OUT_DIR/watchdog.log}"
MODEL_ID="${MODEL_ID:-ornith-1.5-35b-a3b-nvfp4}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STALE_SECS="${STALE_SECS:-900}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-60}"
POLL_SECS="${POLL_SECS:-60}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
RUNNER="${RUNNER:-$ROOT/scripts/run_mmlu_llama_ornith.sh}"

mkdir -p "$(dirname "$WATCH_LOG")" "$OUT_DIR"
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

lm_eval_pids() {
  pgrep -f "$ROOT/bin/lm-eval" || true
}

models_ok() {
  timeout 5 curl -sf "$BASE_URL/v1/models" >/dev/null
}

chat_ok() {
  timeout "$PROBE_TIMEOUT" curl -sf "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2? A.3 B.4\"},{\"role\":\"assistant\",\"content\":\"The best answer is\"}],\"max_tokens\":8,\"temperature\":0,\"continue_final_message\":true,\"add_generation_prompt\":false}" \
    | python3 -c "import sys,json; o=json.load(sys.stdin); c=o['choices'][0]['message'].get('content') or ''; assert c.strip(), 'empty'; print(c.strip()[:40])" \
    >/dev/null 2>&1
}

wait_ready() {
  local i=0
  log "waiting for Ornith ready (timeout ${READY_TIMEOUT}s)"
  while (( i < READY_TIMEOUT )); do
    if models_ok && chat_ok; then
      log "Ornith ready after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if (( i % 60 == 0 )); then log "still waiting (${i}s)..."; fi
  done
  log "ERROR: Ornith not ready within ${READY_TIMEOUT}s"
  return 1
}

kill_lm_eval() {
  local pids
  pids=$(lm_eval_pids)
  if [[ -z "$pids" ]]; then
    log "no lm-eval to kill"
    return 0
  fi
  log "killing lm-eval: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 2
  pids=$(lm_eval_pids)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

restart_server() {
  log "restarting Ornith via docker compose in $RECIPE_DIR"
  (cd "$RECIPE_DIR" && docker compose restart ornith-15-35b)
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

has_final_results() {
  find "$OUT_DIR" -type f -name 'results_*.json' ! -path '*/smoke/*' 2>/dev/null | grep -q .
}

log "watchdog start STALE_SECS=$STALE_SECS POLL_SECS=$POLL_SECS MODEL_ID=$MODEL_ID"
prev_rows=$(cache_rows)
while true; do
  sleep "$POLL_SECS"
  rows=$(cache_rows)
  age=$(cache_age)
  pids=$(lm_eval_pids)

  if [[ -z "$pids" ]]; then
    if has_final_results; then
      log "lm-eval finished (results present outside smoke/); watchdog exiting"
      exit 0
    fi
    log "lm-eval missing with no final results; resume"
    if ! models_ok || ! chat_ok; then
      restart_server || true
    fi
    resume_eval
    prev_rows=$(cache_rows)
    continue
  fi

  if (( rows > prev_rows )); then
    log "ok progress rows=$rows (+$((rows - prev_rows))) age=${age}s"
    prev_rows=$rows
    continue
  fi

  if (( age < STALE_SECS )); then
    log "idle but fresh age=${age}s rows=$rows"
    continue
  fi

  log "WEDGE suspected: cache_rows=$rows age=${age}s"
  if models_ok && chat_ok; then
    log "API answers but cache stale — kill/resume client"
    kill_lm_eval
    resume_eval
  else
    kill_lm_eval
    restart_server || true
    resume_eval
  fi
  prev_rows=$(cache_rows)
done
