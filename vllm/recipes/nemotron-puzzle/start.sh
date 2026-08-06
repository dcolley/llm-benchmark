#!/bin/bash
# Prefer docker compose from this directory.
# Equivalent: docker compose -f docker-compose.yml up -d
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d "$@"
echo "Nemotron Puzzle starting. Follow logs with: docker compose -f $(pwd)/docker-compose.yml logs -f"
docker compose logs -f
