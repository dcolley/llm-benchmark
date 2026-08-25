# Venv — Qwen3.8-27B-NVFP4-MTP-sm121

Prefer **Docker** with `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121` (`./start-docker.sh`). Host venv is a fallback when you already have a Spark-capable vLLM with ModelOpt NVFP4.

```bash
mkdir -p ~/vllm
cd ~/vllm
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install 'vllm==0.27.1'
```

```bash
export VLLM_VENV=$HOME/vllm/.venv   # optional
PROFILE=production ./start.sh
```

Confirm:

```bash
source "${VLLM_VENV:-$HOME/vllm/.venv}/bin/activate"
command -v vllm
vllm --version
```

Always keep **`--kv-cache-dtype fp8`** for this checkpoint.
