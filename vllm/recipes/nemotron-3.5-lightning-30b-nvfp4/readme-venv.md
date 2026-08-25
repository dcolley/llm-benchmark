# Venv — Nemotron 3.5 Lightning 30B-A3B NVFP4

Uses **vLLM 0.27.x** (NVIDIA card: `vllm/vllm-openai:v0.27.1`). Dedicated venv so it
does not clobber stock `~/vllm/.venv` (0.23.0) used by older Qwen recipes.

## Install (host, aarch64 / GB10)

```bash
mkdir -p ~/vllm
cd ~/vllm
uv venv nemotron-35-lightning-env --python 3.12
source nemotron-35-lightning-env/bin/activate

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"

# Pin to the release NVIDIA tested; bump when a newer aarch64 wheel is published.
uv pip install 'vllm==0.27.1'
```

If `0.27.1` is not yet on PyPI for your platform, use the matching nightly container
(`docker-compose.yml` in this folder) or install from the vLLM nightly aarch64 index:

```bash
# Example — pick a current hash from https://wheels.vllm.ai/nightly/cu130/vllm/
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

Confirm Nemotron parsers:

```bash
source "${VLLM_VENV:-$HOME/vllm/nemotron-35-lightning-env}/bin/activate"
command -v vllm
python -c "import vllm; print(vllm.__version__)"
vllm serve --help | grep -E 'nemotron_v3|qwen3_coder' || true
```

`start.sh` activates `$VLLM_VENV` or `$HOME/vllm/nemotron-35-lightning-env`:

```bash
export VLLM_VENV=$HOME/vllm/nemotron-35-lightning-env   # optional override
./start.sh
```
