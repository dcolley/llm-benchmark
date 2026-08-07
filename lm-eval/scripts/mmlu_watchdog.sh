#!/usr/bin/env bash
# Watchdog for cached mmlu_llama against local vLLM (nemotron-puzzle).
# Detects "wedge": /v1/models OK but chat completions hang / cache stops growing.
# On wedge: kill lm-eval, restart vLLM container, wait for ready, resume eval.
set -euo pipefail

ROOT="${ROOT:-/home/derek/llm-benchmark/lm-eval}"
COMPOSE_DIR="${COMPOSE_DIR:-/home/derek/vllm/recipes}"
SERVICE="${SERVICE:-nemotron-puzzle}"
CACHE_DB="${CACHE_DB:-$ROOT/results/nemotron-puzzle/cache.db_rank0.db}"
LOG="${LOG:-$ROOT/results/nemotron-puzzle/resume.log}"
WATCH_LOG="${WATCH_LOG:-$ROOT/results/nemotron-puzzle/watchdog.log}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
STALE_SECS="${STALE_SECS:-180}"          # no cache growth => suspect stall
PROBE_TIMEOUT="${PROBE_TIMEOUT:-20}"     # chat probe budget
POLL_SECS="${POLL_SECS:-60}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # vLLM restart can take ~15-20m

mkdir -p "$(dirname "$WATCH_LOG")"
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
  log "waiting for vLLM ready (timeout ${READY_TIMEOUT}s)"
  while (( i < READY_TIMEOUT )); do
    if models_ok && chat_ok; then
      log "vLLM ready after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i + 5))
    if (( i % 60 == 0 )); then log "still waiting (${i}s)..."; fi
  done
  log "ERROR: vLLM not ready within ${READY_TIMEOUT}s"
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

restart_vllm() {
  log "restarting docker compose service $SERVICE"
  (cd "$COMPOSE_DIR" && docker compose restart "$SERVICE")
}

start_lm_eval() {
  log "starting lm-eval (resume via --use_cache)"
  # shellcheck disable=SC1091
  source "$ROOT/bin/activate"
  cd "$ROOT"
  nohup lm-eval --model local-chat-completions \
    --model_args "base_url=${BASE_URL}/v1/chat/completions,model=${MODEL_ID},num_concurrent=1,tokenizer_backend=none,timeout=120,max_retries=5" \
    --tasks mmlu_llama \
    --apply_chat_template \
    --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false' \
    --use_cache ./results/nemotron-puzzle/cache.db \
    --log_samples \
    --output_path ./results/nemotron-puzzle \
    >>"$LOG" 2>&1 &
  log "lm-eval pid=$!"
}

recover() {
  local reason=$1
  log "WEDGE detected: $reason"
  kill_lm_eval
  restart_vllm
  wait_ready
  start_lm_eval
}

# --- one-shot recover if --recover-now ---
if [[ "${1:-}" == "--recover-now" ]]; then
  recover "manual --recover-now"
  exit 0
fi

# --- watch loop ---
log "watchdog start STALE_SECS=$STALE_SECS PROBE_TIMEOUT=$PROBE_TIMEOUT POLL_SECS=$POLL_SECS"
prev_rows=$(cache_rows)
while true; do
  sleep "$POLL_SECS"
  rows=$(cache_rows)
  age=$(cache_age)
  pids=$(lm_eval_pids)
  # Finished? cache full-ish and no process — exit cleanly if results exist
  if [[ -z "$pids" ]]; then
    if compgen -G "$ROOT/results/nemotron-puzzle/**/results_*.json" >/dev/null \
       || compgen -G "$ROOT/results/nemotron-puzzle/results_*.json" >/dev/null; then
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

  # no growth
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

  # chat works but lm-eval stuck (likely half-open socket / client hang)
  recover "cache stale ${age}s while chat probe OK — lm-eval likely wedged client-side"
  prev_rows=$(cache_rows)
done
