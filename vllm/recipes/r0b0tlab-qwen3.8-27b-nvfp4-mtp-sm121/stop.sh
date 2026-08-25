#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f docker-compose.yml -f docker-compose.long.yml -f docker-compose.ar.yml down "$@" 2>/dev/null \
  || docker compose down "$@"
echo "Stopped qwen38-27b-nvfp4-mtp"
