# Venv — Qwen3.6-27B-NVFP4

Stock vLLM (Python 3.11). Same layout as other Qwen host recipes:

```bash
mkdir -p ~/vllm
cd ~/vllm
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install 'vllm==0.23.0'
```

The start script activates `$VLLM_VENV` or `$HOME/vllm/.venv`:

```bash
export VLLM_VENV=$HOME/vllm/.venv   # optional override
./start-nvidia-Qwen3.6-27B-NVFP4.sh
```

Confirm:

```bash
source "${VLLM_VENV:-$HOME/vllm/.venv}/bin/activate"
command -v vllm
```
