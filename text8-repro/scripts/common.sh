# Shared helpers for text8-repro scripts. Source from other scripts; do not run directly.
# shellcheck shell=bash

text8_repro_root() {
  cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd
}

load_env_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value#\"}"
      value="${value%\"}"
      value="${value#\'}"
      value="${value%\'}"
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < "$f"
}

load_text8_env() {
  local root="$1"
  load_env_file "$root/.env"
  load_env_file "${PWD}/.env"
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

download_url() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo "Downloading $url -> $dest"
  curl -fL --retry 5 --retry-all-errors --retry-delay 2 -o "${dest}.tmp" "$url"
  mv "${dest}.tmp" "$dest"
}
