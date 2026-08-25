# Nemotron 3.5 Lightning 30B-A3B NVFP4 on DGX Spark

Serve [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) with vLLM **0.27.x** on GB10. Port **8000**.

30B total / **3B active** MoE (Mamba-2 + MoE + selective Attention). NVFP4 checkpoint with optional **DSpark** speculative decoding (recommended on Spark).

| Variant | Hugging Face |
|---------|----------------|
| Main (NVFP4) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| DSpark drafter | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` |

License: [OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).

## Prerequisites

- DGX Spark (GB10) or other Blackwell with enough UMA for weights + KV
- Hugging Face token if the repo is gated: `huggingface-cli login`
- **Stop other `:8000` servers** (Muse, Ling, Nemotron Puzzle, ds4, Qwen, etc.)

## Venv (recommended)

See **[readme-venv.md](./readme-venv.md)** — dedicated `~/vllm/nemotron-35-lightning-env`, vLLM 0.27.1.

## Download weights

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
huggingface-cli download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

Or let vLLM pull on first serve.

## Run (host)

```bash
cd ~/llm-benchmark/vllm/recipes/nemotron-3.5-lightning-30b-nvfp4
./start.sh
```

Defaults (NVIDIA 1× Spark recipe, with conservative ctx):

| Flag | Default | Notes |
|------|---------|--------|
| DSpark draft | on | `ENABLE_DSPARK=0` to disable |
| `--max-model-len` | `262144` | Card supports **1M** — try `MAX_MODEL_LEN=1048576` if UMA allows |
| `--gpu-memory-utilization` | `0.91` | Official Spark value |
| `--moe-backend` | `marlin` | NVFP4 on Blackwell |
| `--kv-cache-dtype` | `fp8` | |
| Speculative tokens | `3` | DSpark |

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[0].id'
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4","messages":[{"role":"user","content":"Say hi in one sentence."}],"max_tokens":64}'
```

## Run (Docker)

Uses `vllm/vllm-openai:v0.27.1` — same flags as `start.sh`.

```bash
cd ~/llm-benchmark/vllm/recipes/nemotron-3.5-lightning-30b-nvfp4
docker compose up -d
docker compose logs -f
```

## Tunables (env)

```bash
MAX_MODEL_LEN=1048576 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_NUM_SEQS=4 \
./start.sh
```

## Official NVIDIA benchmarks (NVFP4)

| Task | Score |
|------|------:|
| MMLU Pro | 81.62 |
| GPQA Diamond | 75.57 |
| SWE-bench Verified | 52.80 |
| IFBench (loose) | 72.88 |

Full table on the [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4#benchmarks).

## Related recipes

| Recipe | Model |
|--------|--------|
| [`nemotron-puzzle/`](../nemotron-puzzle/) | Nemotron Labs 3 Puzzle 75B-A9B NVFP4 |
| [`qwen3.6-27b-nvfp4/`](../qwen3.6-27b-nvfp4/) | Qwen3.6-27B-NVFP4 |
| [`meta-muse-glimmer-30b/`](../meta-muse-glimmer-30b/) | Muse Glimmer-30B BF16 |

Only one server on **:8000**.
