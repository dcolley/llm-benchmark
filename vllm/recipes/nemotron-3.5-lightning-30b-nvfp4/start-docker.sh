#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose up -d "$@"
echo "Nemotron 3.5 Lightning starting. Logs: docker compose -f $(pwd)/docker-compose.yml logs -f"
docker compose logs -f
