# Venv — Ling-3.0-flash (host)

Required only for **`./start-host.sh`**. Docker compose (`./start.sh`) builds the fork in the image and does not need this venv.

Matches the [model card](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4):

```bash
uv venv ~/my_ling_env
source ~/my_ling_env/bin/activate
git clone -b ling_3_0 https://github.com/inclusionAI/vllm-ling-v3.git ~/src/vllm-ling-v3
cd ~/src/vllm-ling-v3
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

Defaults used by `start-host.sh`:

| Env | Default |
|-----|---------|
| `VENV` | `$HOME/my_ling_env` |
| `FORK_DIR` | `$HOME/src/vllm-ling-v3` |
| `VLLM_CACHE_ROOT` | `$HOME/.cache/vllm` |

Stock `vllm` / `vllm/vllm-openai` does **not** include `ling3` parsers or `BailingMoeV3` support — use this fork (or the recipe Dockerfile).
