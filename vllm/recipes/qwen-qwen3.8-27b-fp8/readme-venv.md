# Venv — Qwen3.8-27B-FP8

Stock vLLM ≥ **0.17** (Spark: prefer **0.27.x** for GDN / MTP / GB10). Same layout as other host recipes:

```bash
mkdir -p ~/vllm
cd ~/vllm
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install 'vllm==0.27.1'
```

The start script activates `$VLLM_VENV` or `$HOME/vllm/.venv`:

```bash
export VLLM_VENV=$HOME/vllm/.venv   # optional override
./start.sh
```

Confirm:

```bash
source "${VLLM_VENV:-$HOME/vllm/.venv}/bin/activate"
command -v vllm
vllm --version
```

**Docker** (preferred on Spark) does not need this venv — use `start-docker.sh` / `docker compose`.
