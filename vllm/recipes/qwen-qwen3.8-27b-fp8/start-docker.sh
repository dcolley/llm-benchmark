#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose up -d "$@"
echo "Qwen3.8-27B-FP8 starting. Logs: docker compose -f $(pwd)/docker-compose.yml logs -f"
docker compose logs -f
