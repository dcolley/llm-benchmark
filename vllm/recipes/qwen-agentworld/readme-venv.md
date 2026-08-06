# Venv — Qwen-AgentWorld (2× Spark)

Both Sparks need the **same** stock vLLM install (Python 3.11, vLLM **0.23.0**).

```bash
mkdir -p ~/vllm
cd ~/vllm
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install 'vllm==0.23.0'
```

Point `VLLM_VENV` in `qwen-agentworld-2spark.env` at that venv:

```bash
VLLM_VENV=$HOME/vllm/.venv
```

Confirm on each node before launch:

```bash
source "$VLLM_VENV/bin/activate"
python -c 'import vllm; print(vllm.__version__)'  # expect 0.23.0
```

The launch script refuses to start if `EXPECTED_VLLM_VERSION` does not match the installed package.
