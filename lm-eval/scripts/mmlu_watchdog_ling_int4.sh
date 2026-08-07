#!/usr/bin/env bash
# Watchdog for cached mmlu_llama against local SGLang Ling-3.0-flash-int4.
# Detects "wedge": /v1/models OK but chat completions hang / cache stops growing.
# On wedge: kill lm-eval, restart SGLang container, wait for ready, resume eval.
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
COMPOSE_DIR="${COMPOSE_DIR:-/home/derek/sglang/recipes/ling-3.0-flash-int4}"
SERVICE="${SERVICE:-ling-3.0-flash-int4}"
CACHE_DB="${CACHE_DB:-$ROOT/results/ling-3.0-flash-int4/cache.db_rank0.db}"
LOG="${LOG:-$ROOT/results/ling-3.0-flash-int4/resume.log}"
WATCH_LOG="${WATCH_LOG:-$ROOT/results/ling-3.0-flash-int4/watchdog.log}"
MODEL_ID="${MODEL_ID:-ling-3.0-flash-int4}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STALE_SECS="${STALE_SECS:-300}"          # hybrid MoE can be slower
PROBE_TIMEOUT="${PROBE_TIMEOUT:-60}"
POLL_SECS="${POLL_SECS:-60}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # weight reload ~3–5m on Spark

mkdir -p "$(dirname "$WATCH_LOG")" "$ROOT/results/ling-3.0-flash-int4"
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
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2? A.3 B.4\"},{\"role\":\"assistant\",\"content\":\"The best answer is\"}],\"max_tokens\":5,\"temperature\":0,\"continue_final_message\":true,\"add_generation_prompt\":false}" \
    | python3 -c "import sys,json; o=json.load(sys.stdin); c=o['choices'][0]['message'].get('content') or ''; assert c.strip(), 'empty'; print(c.strip()[:40])" \
    >/dev/null 2>&1
}

wait_ready() {
  local i=0
  log "waiting for SGLang ready (timeout ${READY_TIMEOUT}s)"
  while (( i < READY_TIMEOUT )); do
    if models_ok && chat_ok; then
      log "SGLang ready after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if (( i % 60 == 0 )); then log "still waiting (${i}s)..."; fi
  done
  log "ERROR: SGLang not ready within ${READY_TIMEOUT}s"
  return 1
}

kill_lm_eval() {
  local pids
  pids=$(lm_eval_pids)
  if [[ -z "$pids" ]]; then
    log "no lm-eval running"
    return 0
  fi
  log "killing lm-eval: $pids"
  kill $pids 2>/dev/null || true
  sleep 2
  pids=$(lm_eval_pids)
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
}

restart_server() {
  log "restarting docker compose service $SERVICE"
  (cd "$COMPOSE_DIR" && docker compose restart "$SERVICE")
}

start_lm_eval() {
  log "starting lm-eval (resume via --use_cache)"
  # shellcheck disable=SC1091
  source "$ROOT/bin/activate"
  cd "$ROOT"
  nohup lm-eval --model local-chat-completions \
    --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=1,tokenizer_backend=none,timeout=300,max_retries=5" \
    --tasks mmlu_llama \
    --apply_chat_template \
    --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false' \
    --use_cache ./results/ling-3.0-flash-int4/cache.db \
    --log_samples \
    --output_path ./results/ling-3.0-flash-int4 \
    >>"$LOG" 2>&1 &
  log "lm-eval pid=$!"
}

recover() {
  local reason=$1
  log "WEDGE detected: $reason"
  kill_lm_eval
  restart_server
  wait_ready
  start_lm_eval
}

if [[ "${1:-}" == "--recover-now" ]]; then
  recover "manual --recover-now"
  exit 0
fi

log "watchdog start STALE_SECS=$STALE_SECS PROBE_TIMEOUT=$PROBE_TIMEOUT POLL_SECS=$POLL_SECS"
prev_rows=$(cache_rows)
while true; do
  sleep "$POLL_SECS"
  rows=$(cache_rows)
  age=$(cache_age)
  pids=$(lm_eval_pids)
  if [[ -z "$pids" ]]; then
    if compgen -G "$ROOT/results/ling-3.0-flash-int4/**/results_*.json" >/dev/null \
       || compgen -G "$ROOT/results/ling-3.0-flash-int4/results_*.json" >/dev/null; then
      log "lm-eval finished (results present); watchdog exiting"
      exit 0
    fi
    log "lm-eval not running and no results yet — restarting eval only"
    if models_ok && chat_ok; then
      start_lm_eval
    else
      recover "lm-eval dead and chat unhealthy"
    fi
    prev_rows=$rows
    continue
  fi

  grew=0
  if (( rows > prev_rows )); then grew=1; fi

  if (( grew )); then
    log "ok progress rows=$rows (+$((rows - prev_rows))) age=${age}s pids=$pids"
    prev_rows=$rows
    continue
  fi

  if (( age < STALE_SECS )); then
    log "idle but fresh age=${age}s rows=$rows (waiting)"
    continue
  fi

  if ! models_ok; then
    recover "models endpoint down; cache stale ${age}s rows=$rows"
    prev_rows=$(cache_rows)
    continue
  fi

  if ! chat_ok; then
    recover "chat probe hung/failed; cache stale ${age}s rows=$rows (models still OK)"
    prev_rows=$(cache_rows)
    continue
  fi

  recover "cache stale ${age}s while chat probe OK — lm-eval likely wedged client-side"
  prev_rows=$(cache_rows)
done
