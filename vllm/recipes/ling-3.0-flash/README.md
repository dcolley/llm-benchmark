# Ling-3.0-flash (official FP4) on DGX Spark

Serve [`inclusionAI/Ling-3.0-flash-fp4`](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4) (~70 GB) with the Ling vLLM fork. Port **8000** — do not run alongside Nemotron Puzzle.

## Why this checkpoint

| Checkpoint | ~Disk | Single Spark (128 GB) |
|---|---|---|
| BF16 | ~255 GB | No |
| FP8 | ~128 GB | Barely weights |
| **FP4 (this recipe)** | **~70 GB** | **Best — room for KV** |
| INT4 | ~77 GB | Yes (slightly better quality, less KV) |
| Community MXFP4 | ~78 GB | Yes |
| AtomicChat NVFP4 | TBD | Not ready (building) |
| GGUF | varies | llama.cpp only |

## Download weights

Needs ~70 GB+ free under the Hugging Face cache:

```bash
pip install -U "huggingface_hub[cli]"
hf download inclusionAI/Ling-3.0-flash-fp4
```

Or let vLLM pull on first serve (same cache mount / `HF_HOME`).

## Venv (host path)

For `./start-host.sh`, install the Ling vLLM fork in a host venv — see **[readme-venv.md](./readme-venv.md)**. Docker compose does not need that venv.

## Run

Each model has its own folder and `start.sh`:

| Folder | Start |
|---|---|
| `recipes/nemotron-puzzle/` | `./start.sh` → docker compose |
| `recipes/ling-3.0-flash/` | `./start.sh` → docker compose |

**Docker compose** (builds the Ling fork image — can be slow on Spark):

```bash
# Stop Nemotron first if it owns :8000
cd vllm/recipes/nemotron-puzzle && docker compose down

cd vllm/recipes/ling-3.0-flash
./start.sh
```

**Host venv** (faster first boot; matches HF card — [readme-venv.md](./readme-venv.md)):

```bash
cd vllm/recipes/ling-3.0-flash
./start-host.sh
```

## Startup / vLLM cache (avoid crash-loop pain)

Compose mounts host cache and sets `VLLM_CACHE_ROOT` so torch.compile + FlashInfer autotune survive container recreate:

| | |
|---|---|
| Mount | `~/.cache/vllm` → `/root/.cache/vllm` |
| Env | `VLLM_CACHE_ROOT=/root/.cache/vllm` |

| Start | Typical wall time on Spark |
|---|---|
| **1st** (cold autotune) | **~21 min** |
| **Later** (warm cache) | **~5 min** |

Without the mount, every fresh container redoes autotune (~21 min). That looks like a hang/crash-loop under `restart: unless-stopped` if you recreate often. Host path uses the same `~/.cache/vllm` via `VLLM_CACHE_ROOT` in `start-host.sh`.

## Memory tuning (`--gpu-memory-utilization`)

This flag caps **total** vLLM memory (weights + KV), not “free for context.” After ~70 GB weights load, leftover budget becomes the KV / Mamba-state pool.

| util | Approx budget on 128 GB | After FP4 weights |
|---|---|---|
| 0.70 | ~90 GB | ~15–20 GB KV |
| **0.70 (default)** | ~90 GB | **~15–20 GB KV** |
| 0.80 | ~102 GB | ~25–30 GB KV |
| 0.85 | ~109 GB | ~35 GB KV |

Defaults in this recipe: `util=0.70`, `max-model-len=131072`, `max-num-seqs=4`. Raise util toward `0.80` only if the box has headroom after a stable boot. Drop MTP (`--speculative-config`) to free more memory and speed cold start.

### Spark / sm_121 notes

If MoE kernels fail (illegal instruction / PTX errors), try Marlin / FlashInfer fallbacks, e.g. add to compose `environment` or the host command:

```bash
--moe-backend marlin
# and/or
export VLLM_USE_FLASHINFER_MOE_FP4=0
```

If the container crashes with `Python.h: No such file or directory` while inspecting `BailingMoeV3ForCausalLM`, rebuild the image so Triton can JIT (`python3-dev` is required in the Dockerfile):

```bash
docker compose build --no-cache && ./start.sh
```

## Client sampling

From the model card: `temperature=0.6`, `top_p=0.95`, `top_k=20`, thinking on by default.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ling-3.0-flash",
    "messages": [{"role": "user", "content": "hello!"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

## Alternates

- Quality lean: swap model id to `inclusionAI/Ling-3.0-flash-int4` (~77 GB).
- Community MXFP4: `olka-fi/Ling-3.0-flash-MXFP4` (apply their `vllm_patch/` if needed).
- When published: `AtomicChat/Ling-3.0-flash-NVFP4` (test Marlin on GB10).
