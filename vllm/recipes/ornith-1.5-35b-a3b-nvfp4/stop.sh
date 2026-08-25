#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${PROFILE:-production}"
case "$PROFILE" in
  long|longctx) docker compose -f docker-compose.yml -f docker-compose.long.yml down "$@" ;;
  *) docker compose down "$@" ;;
esac
echo "ornith-15-35b stopped"
