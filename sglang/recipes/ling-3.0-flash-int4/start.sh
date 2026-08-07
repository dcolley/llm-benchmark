#!/bin/bash
# Start Ling-3.0-flash-int4 via SGLang Docker (preferred) or host script.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "host" ]]; then
  shift
  exec ./start-host.sh "$@"
fi

docker compose up -d "$@"
echo "Ling-3.0-flash-int4 (SGLang) starting. Logs:"
docker compose logs -f
