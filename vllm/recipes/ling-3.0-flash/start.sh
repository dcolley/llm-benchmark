#!/bin/bash
# Start Ling-3.0-flash-fp4 via docker compose from this directory.
# Stop nemotron-puzzle (or anything on :8000) first.
# For host-venv serve instead, use ./start-host.sh
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d --build "$@"
echo "Ling-3.0-flash starting. Follow logs with: docker compose -f $(pwd)/docker-compose.yml logs -f"
docker compose logs -f
