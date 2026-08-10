# Venv — Muse Glimmer-30B (DGX Spark)

Muse needs the support branch until it merges into a release
([vLLM PR #51655](https://github.com/vllm-project/vllm/pull/51655)). Use a **dedicated**
venv — do not overwrite the stock `~/vllm/.venv` (0.23.0) used by other Qwen recipes.

## Install (host, aarch64 / GB10)

```bash
mkdir -p ~/vllm ~/src
cd ~/vllm
uv venv muse-glimmer-env --python 3.12
source muse-glimmer-env/bin/activate

git clone --branch tiezhen/new-model-support --single-branch \
  https://github.com/xianbaoqian/vllm.git ~/src/vllm-muse-glimmer
cd ~/src/vllm-muse-glimmer

# Prefer precompiled CUDA kernels. The Muse PR tip often has no matching
# nightly yet — pin a known aarch64/cu130 wheel commit (update as needed):
#   https://wheels.vllm.ai/nightly/cu130/vllm/
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-3a79957b62ade336010cc052e322d0005eb091a2}"
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

If that commit 404s, pick a newer hash from the nightly index above (must list
`manylinux_2_28_aarch64`), or install `vllm==0.26.0` then retry editable with
`VLLM_PRECOMPILED_WHEEL_LOCATION` pointing at that aarch64 wheel URL.

Confirm Muse parsers are present:

```bash
source "${VLLM_VENV:-$HOME/vllm/muse-glimmer-env}/bin/activate"
command -v vllm
python -c "import vllm; print(vllm.__version__)"
python -c "from vllm.model_executor.models.registry import ModelRegistry; \
print('MuseGlimmerForConditionalGeneration' in ModelRegistry.models)"
```

`start.sh` activates `$VLLM_VENV` or `$HOME/vllm/muse-glimmer-env`:

```bash
export VLLM_VENV=$HOME/vllm/muse-glimmer-env   # optional override
./start.sh
```

## After Muse merges upstream

When a PyPI/nightly build includes `muse_glimmer`, switch this venv to that release and
drop the fork clone. Keep `CUTE_DSL_ARCH=sm_121a` on Spark.
