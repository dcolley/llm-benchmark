#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME="${NAME:-qwen38-27b-gguf}"
PIDFILE="${PIDFILE:-$(pwd)/llama-server.pid}"

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    echo "Stopped host llama-server pid $pid"
  fi
  rm -f "$PIDFILE"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
  echo "Stopped container $NAME"
fi

docker compose down 2>/dev/null || true
echo "Done."
