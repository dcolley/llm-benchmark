#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on one DGX Spark via Entrpi/ds4-on-spark.
#
# Official HF safetensors (~155–167 GB) do NOT fit 128 GB UMA.
# This path uses the ~81 GiB IQ2XXS/Q2K GGUF + ~7 GiB DSpark drafter.
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-131072}"
GGUF_DIR="${GGUF_DIR:-$HOME/gguf}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/ds4-on-spark}"
DS4_SERVER="${DS4_SERVER:-}"
DO_INSTALL=1
START_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--install-only] [--no-install] [--help]

Env:
  PORT      default 8000
  HOST      default 0.0.0.0
  CONTEXT   default 131072 (try 196608 after stable boot; 262144 is tight)
  GGUF_DIR  default \$HOME/gguf
  INSTALL_DIR default \$HOME/ds4-on-spark
  DS4_SERVER  explicit path to ds4-server binary (optional)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-only) START_ONLY=0; DO_INSTALL=1; INSTALL_ONLY=1; shift ;;
    --no-install) DO_INSTALL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done
INSTALL_ONLY="${INSTALL_ONLY:-0}"

if ss -ltn 2>/dev/null | grep -qE ":${PORT}\\b"; then
  echo "Port ${PORT} is already in use. Stop the other server first (Muse/Ling/etc)." >&2
  ss -ltnp 2>/dev/null | grep -E ":${PORT}\\b" || true
  exit 1
fi

avail_gi=$(free -b | awk '/Mem:/{printf "%.0f", $7/2^30}')
if (( avail_gi < 90 )); then
  echo "Warning: only ~${avail_gi} GiB available RAM. Stop other models before loading ~88 GiB weights+drafter." >&2
fi

export DS4_CONT_DSPARK="${DS4_CONT_DSPARK:-1}"
export DS4_CONT_MTP_MODE="${DS4_CONT_MTP_MODE:-2}"
export DS4_GGUF_DIR="${DS4_GGUF_DIR:-$GGUF_DIR}"
export DS4_BATCH_FIT_HEADROOM_MB="${DS4_BATCH_FIT_HEADROOM_MB:-8192}"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  echo "Installing / updating ds4-on-spark (Entrpi) into $INSTALL_DIR ..."
  mkdir -p "$INSTALL_DIR" "$GGUF_DIR"
  # One-shot installer: builds engine + pulls 0731 GGUF + DSpark drafter.
  # Prefer clone+script if already present so we do not re-pipe curl blindly every time.
  if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/entrpi/ds4-on-spark.git "$INSTALL_DIR"
  else
    git -C "$INSTALL_DIR" pull --ff-only || true
  fi
  # install.sh has no --host; bind address is passed to ds4-serve / ds4-server.
  if [[ -x "$INSTALL_DIR/install.sh" ]]; then
    bash "$INSTALL_DIR/install.sh" --port "$PORT" --ctx "$CONTEXT" --keep-old-weights --with-dspark || {
      echo "Local install.sh failed; falling back to upstream one-liner." >&2
      curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh \
        | bash -s -- --port "$PORT" --ctx "$CONTEXT" --keep-old-weights --with-dspark
    }
  else
    curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh \
      | bash -s -- --port "$PORT" --ctx "$CONTEXT" --keep-old-weights --with-dspark
  fi
fi


if [[ "$INSTALL_ONLY" -eq 1 ]]; then
  echo "Install-only complete. GGUFs under $GGUF_DIR ; re-run without --install-only to serve."
  exit 0
fi

# Locate ds4-server
if [[ -z "$DS4_SERVER" ]]; then
  for c in \
    "$HOME/.cache/huggingface/ds4-engine/ds4-server" \
    "$INSTALL_DIR/ds4-server" \
    "$INSTALL_DIR/bin/ds4-server" \
    "$(command -v ds4-server 2>/dev/null || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then
      DS4_SERVER="$c"
      break
    fi
  done
fi

if [[ -z "${DS4_SERVER:-}" || ! -x "$DS4_SERVER" ]]; then
  echo "ds4-server not found. Re-run: $0 --install-only" >&2
  exit 1
fi

# Prefer explicit 0731 GGUF names from antirez / bleysg
MODEL_GGUF="${MODEL_GGUF:-}"
if [[ -z "$MODEL_GGUF" ]]; then
  MODEL_GGUF=$(ls -1 "$GGUF_DIR"/DeepSeek-V4-Flash-IQ2XXS*0731*.gguf 2>/dev/null | head -1 || true)
fi
DRAFT_GGUF="${DRAFT_GGUF:-}"
if [[ -z "$DRAFT_GGUF" ]]; then
  DRAFT_GGUF=$(ls -1 "$GGUF_DIR"/DSpark-drafter*0731*.gguf 2>/dev/null | head -1 || true)
fi

if [[ -z "$MODEL_GGUF" ]]; then
  echo "Target GGUF not found under $GGUF_DIR (expected DeepSeek-V4-Flash-IQ2XXS*0731*.gguf)." >&2
  echo "Run: $0 --install-only   or hf download antirez/deepseek-v4-gguf" >&2
  exit 1
fi

if [[ -n "$DRAFT_GGUF" ]]; then
  export DS4_DSPARK_MODEL="$DRAFT_GGUF"
  export DS4_CONT_DSPARK=1
else
  echo "Warning: DSpark drafter GGUF not found; starting without speculation." >&2
  export DS4_CONT_DSPARK=0
fi

echo "Serving DeepSeek-V4-Flash-0731 via $DS4_SERVER"
echo "  model=$MODEL_GGUF"
echo "  draft=${DRAFT_GGUF:-none}"
echo "  context=$CONTEXT host=$HOST port=$PORT"

exec "$DS4_SERVER" --cuda \
  -m "$MODEL_GGUF" \
  -c "$CONTEXT" \
  --host "$HOST" \
  --port "$PORT"
