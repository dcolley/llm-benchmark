#!/usr/bin/env bash
# Qwen-AgentWorld-35B-A3B on 2x DGX Spark (1x GB10 each, tensor parallel).
#
# Network env vars follow NVIDIA's "Run on two Sparks" NCCL playbook:
# https://build.nvidia.com/spark/nccl/stacked-sparks
#
# Usage:
#   1. Copy qwen-agentworld-2spark.env.example -> qwen-agentworld-2spark.env
#   2. Set HEAD_ADDR, HOST_IP, NODE_RANK (0=head, 1=worker) on each Spark
#   3. Head:  ./qwen-agentworld-2spark.sh head
#   4. Worker: ./qwen-agentworld-2spark.sh worker   (after head is listening)
#
# API (head only): http://<mgmt-ip>:8000/v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/qwen-agentworld-2spark.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy qwen-agentworld-2spark.env.example and edit." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 {head|worker}" >&2
  exit 1
fi

if [[ "$ROLE" == "head" && "${NODE_RANK:-}" != "0" ]]; then
  echo "head role requires NODE_RANK=0 in $ENV_FILE" >&2
  exit 1
fi
if [[ "$ROLE" == "worker" && "${NODE_RANK:-}" != "1" ]]; then
  echo "worker role requires NODE_RANK=1 in $ENV_FILE" >&2
  exit 1
fi

: "${HEAD_ADDR:?HEAD_ADDR required}"
: "${HOST_IP:?HOST_IP required}"
: "${VLLM_VENV:?VLLM_VENV required}"
: "${NCCL_IFNAME:?NCCL_IFNAME required}"

MASTER_PORT="${MASTER_PORT:-29501}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.23.0}"

# shellcheck source=/dev/null
source "$VLLM_VENV/bin/activate"

ACTUAL_VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
if [[ "$ACTUAL_VLLM_VERSION" != "$EXPECTED_VLLM_VERSION" ]]; then
  echo "vLLM version mismatch: found $ACTUAL_VLLM_VERSION, expected $EXPECTED_VLLM_VERSION" >&2
  echo "Install the same venv on both Sparks (see README-qwen-agentworld-2spark.md)." >&2
  exit 1
fi

# NVIDIA stacked-sparks NCCL playbook exports (plus GLOO for PyTorch store).
export VLLM_HOST_IP="$HOST_IP"
export MASTER_ADDR="$HEAD_ADDR"
export MASTER_PORT="$MASTER_PORT"
export NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
export GLOO_SOCKET_IFNAME="$NCCL_IFNAME"
export UCX_NET_DEVICES="$NCCL_IFNAME"
export OMPI_MCA_btl_tcp_if_include="$NCCL_IFNAME"

# Compile parallelism (first boot only; tune per node).
# - MAX_JOBS: parallel FlashInfer ninja/nvcc jobs (default was 2 to avoid earlyoom)
# - FLASHINFER_NVCC_THREADS: threads per nvcc compile (--threads=N)
# - OMP_NUM_THREADS: must be set before vLLM starts or it forces 1 for torch ops
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

if [[ "${NCCL_DEBUG:-}" != "" ]]; then
  export NCCL_DEBUG
fi

echo "Network: HOST_IP=$HOST_IP IFACE=$NCCL_IFNAME MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "Compile: MAX_JOBS=$MAX_JOBS FLASHINFER_NVCC_THREADS=$FLASHINFER_NVCC_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS"

if [[ "$ROLE" == "worker" ]]; then
  echo "Waiting for head rendezvous at ${HEAD_ADDR}:${MASTER_PORT}..."
  for _ in $(seq 1 "${WORKER_WAIT_SECS:-120}"); do
    if (echo >/dev/tcp/"$HEAD_ADDR"/"$MASTER_PORT") 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if ! (echo >/dev/tcp/"$HEAD_ADDR"/"$MASTER_PORT") 2>/dev/null; then
    echo "Head not reachable at ${HEAD_ADDR}:${MASTER_PORT}. Start head first." >&2
    exit 1
  fi
fi

if [[ "${STOP_EARLYOOM:-0}" == "1" ]]; then
  sudo systemctl stop earlyoom 2>/dev/null || true
  trap 'sudo systemctl start earlyoom 2>/dev/null || true' EXIT
fi

COMMON_ARGS=(
  "$MODEL"
  --tensor-parallel-size 2
  --nnodes 2
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --reasoning-parser qwen3
  --trust-remote-code
  --language-model-only
  --distributed-executor-backend mp
)

# Hermes / OpenAI clients send tool_choice=auto; requires both flags.
if [[ "${ENABLE_AUTO_TOOL_CHOICE:-1}" == "1" ]]; then
  COMMON_ARGS+=(--enable-auto-tool-choice)
  COMMON_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER:-qwen3_xml}")
fi

if [[ "$ROLE" == "head" ]]; then
  echo "Starting head (API on port ${PORT:-8000})..."
  exec vllm serve "${COMMON_ARGS[@]}" --port "${PORT:-8000}"
else
  echo "Starting worker (headless, master ${HEAD_ADDR}:${MASTER_PORT})..."
  exec vllm serve "${COMMON_ARGS[@]}" --headless
fi
