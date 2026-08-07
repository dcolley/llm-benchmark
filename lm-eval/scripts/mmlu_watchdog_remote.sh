#!/usr/bin/env bash
# Remote-safe watchdog for cached mmlu_llama.
# Detects stall (cache not growing + chat probe fail/hang) and resumes lm-eval.
# Does NOT restart Docker or SSH into the remote host.
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
OUT_DIR="${OUT_DIR:-$ROOT/results/fuse-1-lite}"
CACHE_DB="${CACHE_DB:-$OUT_DIR/cache.db_rank0.db}"
LOG="${LOG:-$OUT_DIR/resume.log}"
WATCH_LOG="${WATCH_LOG:-$OUT_DIR/watchdog.log}"
MODEL_ID="${MODEL_ID:-Akahsizrr/fuse-1-Lite}"
BASE_URL="${BASE_URL:-http://192.168.10.199:8000}"
# Long enough to cover mmlu_llama context-build (cache mtime stays old until API starts).
STALE_SECS="${STALE_SECS:-900}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-60}"
POLL_SECS="${POLL_SECS:-60}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

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
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2? A.3 B.4\"},{\"role\":\"assistant\",\"content\":\"The best answer is\"}],\"max_tokens\":5,\"temperature\":0,\"continue_final_message\":true,\"add_generation_prompt\":false}" \
    | python3 -c "import sys,json; o=json.load(sys.stdin); c=o['choices'][0]['message'].get('content') or ''; assert c.strip(), 'empty'; print(c.strip()[:40])" \
    >/dev/null 2>&1
}

wait_ready() {
  local i=0
  log "waiting for remote API ready (timeout ${READY_TIMEOUT}s)"
  while (( i < READY_TIMEOUT )); do
    if models_ok && chat_ok; then
      log "remote API ready after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if (( i % 60 == 0 )); then log "still waiting (${i}s)..."; fi
  done
  log "ERROR: remote API not ready within ${READY_TIMEOUT}s"
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
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 2
  pids=$(lm_eval_pids)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

start_lm_eval() {
  log "starting lm-eval (resume via --use_cache)"
  # shellcheck disable=SC1091
  source "$ROOT/bin/activate"
  cd "$ROOT"
  nohup lm-eval --model local-chat-completions \
    --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=1,tokenizer_backend=none,timeout=180,max_retries=5" \
    --tasks mmlu_llama \
    --apply_chat_template \
    --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false' \
    --use_cache "$OUT_DIR/cache.db" \
    --log_samples \
    --output_path "$OUT_DIR" \
    >>"$LOG" 2>&1 &
  log "lm-eval pid=$!"
}

recover() {
  local reason=$1
  log "WEDGE detected: $reason"
  kill_lm_eval
  wait_ready
  start_lm_eval
}

if [[ "${1:-}" == "--recover-now" ]]; then
  recover "manual --recover-now"
  exit 0
fi

log "watchdog start STALE_SECS=$STALE_SECS PROBE_TIMEOUT=$PROBE_TIMEOUT POLL_SECS=$POLL_SECS BASE_URL=$BASE_URL"
prev_rows=$(cache_rows)
while true; do
  sleep "$POLL_SECS"
  rows=$(cache_rows)
  age=$(cache_age)
  pids=$(lm_eval_pids)

  if [[ -z "$pids" ]]; then
    # Only treat as finished if a full-run results JSON exists outside smoke/.
    finished=0
    while IFS= read -r -d '' f; do
      finished=1
      break
    done < <(find "$OUT_DIR" -type f -name 'results_*.json' ! -path '*/smoke/*' -print0 2>/dev/null)
    if (( finished )); then
      log "lm-eval finished (results present outside smoke/); watchdog exiting"
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

  if (( rows > prev_rows )); then
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
