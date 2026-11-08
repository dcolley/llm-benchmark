#!/usr/bin/env bash
# Fetch the canonical text8 corpus (mattmahoney.net text8.zip), unzip it, and
# record sha256. The published comparison configs expect:
# 6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
load_text8_env "$ROOT"

TEXT8_URL="${TEXT8_URL:-http://mattmahoney.net/dc/text8.zip}"
TEXT8_DIR="${TEXT8_DIR:-$ROOT/data}"
TEXT8_PATH="${TEXT8_PATH:-$TEXT8_DIR/text8}"
TEXT8_ZIP="${TEXT8_ZIP:-$TEXT8_DIR/text8.zip}"
TEXT8_EXPECTED_SHA256="${TEXT8_EXPECTED_SHA256:-6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"

mkdir -p "$TEXT8_DIR" "$RESULTS_DIR"

if [[ ! -f "$TEXT8_PATH" ]]; then
  if [[ ! -f "$TEXT8_ZIP" ]]; then
    download_url "$TEXT8_URL" "$TEXT8_ZIP"
  fi
  unzip -o "$TEXT8_ZIP" -d "$TEXT8_DIR"
fi

if [[ ! -f "$TEXT8_PATH" ]]; then
  echo "Expected extracted file at $TEXT8_PATH" >&2
  exit 1
fi

digest="$(sha256_file "$TEXT8_PATH")"
zip_digest="$(sha256_file "$TEXT8_ZIP")"
size="$(wc -c < "$TEXT8_PATH" | tr -d ' ')"

cat > "$RESULTS_DIR/text8_sha256.txt" << REPORT
text8_path=$TEXT8_PATH
text8_bytes=$size
text8_sha256=$digest
text8_zip_path=$TEXT8_ZIP
text8_zip_sha256=$zip_digest
expected_text8_sha256=$TEXT8_EXPECTED_SHA256
REPORT

echo "Recorded $RESULTS_DIR/text8_sha256.txt"
echo "text8 sha256=$digest bytes=$size"

if [[ -n "$TEXT8_EXPECTED_SHA256" && "$digest" != "$TEXT8_EXPECTED_SHA256" ]]; then
  echo "SHA256 mismatch: expected $TEXT8_EXPECTED_SHA256" >&2
  exit 1
fi
if [[ "$size" != "100000000" ]]; then
  echo "Unexpected text8 size $size (expected 100000000 bytes)" >&2
  exit 1
fi
echo "text8 hash and size match the published comparison configs."
