# Qwen3.8-27B-FP8 on DGX Spark

Serve [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) with vLLM on GB10. Port **8000**.

Dense **27B** hybrid GDN + attention VLM (text / image / video). Fine-grained FP8 (block size 128). Native context **262,144** (YaRN up to **1M**). Thinking on by default; MTP speculative decode supported.

Official sibling guidance: [vLLM Qwen3.5/3.6 27B recipes](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) (same architecture family).

## RAM / memory

| Item | Size |
|------|------|
| Checkpoint on disk | **~31 GB** (`usedStorage` ≈ 28.8 GiB) |
| Weights in VRAM/UMA (FP8 + some BF16) | **~28–32 GB** |
| Official FP8 floor | **1× 40 GB GPU** (H100 / L40S class) |
| Comfortable on Spark | **≥ 64 GB** free UMA for 262K + MTP + vision |
| Full Spark budget | Fits **one** GB10 (~121–128 GB UMA) with room for KV / GDN state |

Rough runtime split on Spark at `--max-model-len 262144` and `--gpu-memory-utilization 0.85`:

- Weights ≈ 30 GB  
- CUDA graphs / activations ≈ 5–10 GB  
- Remainder ≈ KV (fp8) + Mamba/GDN state (fp16) for context and concurrency  

For **text-only** throughput, add `--language-model-only` to skip the vision encoder and free more KV. For **1M** context, use YaRN (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` + `rope_parameters` overrides from the [model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)) and expect much higher KV pressure.

**Stop other `:8000` servers** before starting (Nemotron, Muse, Ling, etc.).

## Prerequisites

- DGX Spark (GB10) or other GPU with ≥ ~40 GB for weights + modest KV
- Hugging Face login if needed: `huggingface-cli login`
- Docker image `vllm/vllm-openai:v0.27.1` (or host venv — see [readme-venv.md](./readme-venv.md))

## Download weights

```bash
huggingface-cli download Qwen/Qwen3.8-27B-FP8
```

Or let vLLM pull on first serve.

## Run (Docker — preferred)

```bash
cd ~/llm-benchmark/vllm/recipes/qwen-qwen3.8-27b-fp8
./start-docker.sh
# or: docker compose up -d && docker compose logs -f
```

| Flag | Default | Notes |
|------|---------|--------|
| `--max-model-len` | `262144` | Card native; YaRN for 1M |
| `--gpu-memory-utilization` | `0.85` | Leave headroom for GDN + vision |
| `--kv-cache-dtype` | `fp8` | |
| `--mamba-backend` | `flashinfer` | Spark / Blackwell |
| `--mamba-cache-mode` | `align` | Required for hybrid GDN prefix cache |
| MTP | `3` tokens | Speculative decode |
| `--reasoning-parser` | `qwen3` | Thinking mode |
| `--tool-call-parser` | `qwen3_coder` | Auto tool choice on |
| Served name | `qwen3.8-27b-fp8` | Use this in API `model` |

## Run (host venv)

```bash
cd ~/llm-benchmark/vllm/recipes/qwen-qwen3.8-27b-fp8
./start.sh
```

```bash
MAX_MODEL_LEN=131072 GPU_MEMORY_UTILIZATION=0.80 NUM_SPEC_TOKENS=1 ./start.sh
```

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp8",
    "messages": [{"role": "user", "content": "Say hi in one sentence."}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq .
```

Thinking on (default): omit `enable_thinking: false`, use `temperature=1.0`, `top_p=0.95`. Non-thinking: `temperature=0.7`, `top_p=0.8`, `presence_penalty=1.5`.

## Multimodal

Image / video via OpenAI-style `image_url` / `video_url` content parts — see the [model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).

## License

Apache-2.0.
