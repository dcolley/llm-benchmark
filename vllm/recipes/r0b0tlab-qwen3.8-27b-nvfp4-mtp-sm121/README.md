# Qwen3.8-27B-NVFP4-MTP (SM121) on DGX Spark

Serve [`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121) with the author’s SM121 vLLM image on GB10. Port **8000**.

Self-quantized **NVFP4** (ModelOpt `w4a16_nvfp4-fp8_attn-kv_fp8_cast`) + trained **MTP** head in BF16. Built for **DGX Spark / SM121**. Verified on vLLM `v0.27.2rc0-sm121`.

Companion evidence / configs: [r0b0tlab/qwen38-27b-nvfp4-sm121-vllm](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm).

## Best options on this hardware (chosen defaults)

| Choice | Value | Why |
|--------|-------|-----|
| Runtime image | `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121` | Checkpoint + SM121 kernels verified together |
| KV cache | **`--kv-cache-dtype fp8` (required)** | Checkpoint ships `kv_cache_quant_algo: "FP8"`; omitting / mismatching breaks arithmetic on 0.27.x |
| Speculation | **MTP K=3** | Best throughput on Spark at ≥c8 (~83 tok/s); short decode wins vs AR (~2.5×) |
| Context | **32 768** | Production latency/throughput balance with MTP + util 0.70 |
| GPU util | **0.70** | Measured production profile; leaves UMA headroom for MTP |
| Eager | **`--enforce-eager`** | Stable path on this SM121 build |
| Prefix cache | **off** (`--no-enable-prefix-caching`) | Matches verified serve configs |
| Tools | **`qwen3_xml`** + auto tool choice | Card sanity suite / agentic gates |

**Not default (optional):** DSpark K=7 for c1–c4 latency / reasoning — needs external draft [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co) + config normalization from the companion repo. MTP is better for throughput and short decodes on Spark.

**Long context:** `PROFILE=long` → **262 144**, util **0.85**, **no MTP** (NIAH 8/8 verified).

## RAM / memory

| Item | Size |
|------|------|
| Quant map | 193× W4A16_NVFP4 + 208× FP8 attn + BF16 remainder + 15× BF16 MTP |
| Single GB10 | Fits (~121–128 GB UMA) for production or 262K |
| Production (32K + MTP) | util **0.70** |
| Long (262K, no MTP) | util **0.85**; KV ≈ 2.5M tokens fp8 capacity per card notes |

**Stop other `:8000` servers** before starting (FP8 Qwen3.8, Nemotron, Muse, etc.).

## Prerequisites

- DGX Spark (GB10 / SM121)
- Hugging Face login if the repo is gated / rate-limited: `huggingface-cli login`
- Docker (preferred). Host venv needs a recent vLLM with ModelOpt NVFP4 — see [readme-venv.md](./readme-venv.md)

## Download weights

```bash
huggingface-cli download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121
```

Or let the container pull on first serve (HF cache mounted at `~/.cache/huggingface`).

## Run (Docker — preferred)

```bash
cd ~/llm-benchmark/vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121
./start-docker.sh
# or: PROFILE=long ./start-docker.sh
# or: PROFILE=ar ./start-docker.sh
./stop.sh
```

| Profile | Flags (high level) | Use when |
|---------|--------------------|----------|
| `production` (default) | MTP K=3, 32K, util 0.70, eager, no prefix cache | Daily serving / throughput |
| `long` | 262K, util 0.85, no MTP | Needle-in-haystack / long docs |
| `ar` | 32K, no speculation | Baseline tok/s |

| Flag | Production | Long |
|------|------------|------|
| `--max-model-len` | `32768` | `262144` |
| `--gpu-memory-utilization` | `0.70` | `0.85` |
| `--kv-cache-dtype` | `fp8` | `fp8` |
| MTP | K=3 | off |
| Served name | `qwen3.8-27b-nvfp4-mtp` | same |

Override image / model:

```bash
VLLM_IMAGE=vllm/vllm-openai:v0.27.1 MODEL=/path/to/local/ckpt ./start-docker.sh
```

Stock `vllm/vllm-openai:v0.27.1` may work, but the SM121 image above is what this checkpoint was measured against.

### DSpark (optional latency)

Mount the draft and replace speculative config (see companion repo for normalization):

```text
--speculative-config '{"method":"dspark","model":"/draft","num_speculative_tokens":7}'
```

Measured on this checkpoint: DSpark wins c1–c4 (~28–44 tok/s); MTP wins ≥c8 and short decodes.

## Run (host venv)

```bash
cd ~/llm-benchmark/vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121
PROFILE=production ./start.sh
```

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-nvfp4-mtp",
    "messages": [{"role": "user", "content": "What is 19 times 23? Reply with the number only."}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

If you get nonsense arithmetic (e.g. `417` for 19×23), the FP8 KV path is wrong — ensure `--kv-cache-dtype fp8` and that the checkpoint still has `kv_cache_quant_algo: "FP8"`.

## Measured reference (single Spark, author card)

| Gate | Result |
|------|--------|
| MTP K=3 dedicated c1 | ~27.8–28.1 tok/s (~2.45× AR) |
| MTP K=3 c8 | ~82.9–84.3 tok/s |
| AR c1 floor | ~11.35 tok/s |
| NIAH @ 262 144 | 8/8 PASS |
| GSM8K flex (this ckpt, FP8 KV) | 81.25% exact / 91.25% numeric-norm |

## Provenance

- Source BF16: Qwen3.8-27B; PTQ via NVIDIA ModelOpt 0.46.0rc1 family recipe
- License: Apache-2.0 (weights follow upstream Qwen terms)
- Runtime digest (author): `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121`
