# Qwen3.8-27B-GGUF (Unsloth) on DGX Spark

Serve [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) with **llama.cpp** on GB10. Port **8000** (same as the vLLM recipes so lm-eval can swap backends).

Unsloth Dynamic V3 GGUF of Qwen3.8-27B (qwen35 hybrid GDN + attention). Native context **262 144**. Text-only by default; optional `mmproj` for vision.

This is the GGUF leg of the Qwen3.8-27B quant comparison (alongside FP8 + NVFP4-MTP).

## Best options on this hardware (chosen defaults)

| Choice | Value | Why |
|--------|-------|-----|
| Quant | **`UD-Q8_K_XL`** (~31.5 GB) | Best Unsloth dynamic 8-bit; quality peer to official FP8 (~30 GB) for the three-way report |
| Runtime | Local `~/llama.cpp` **or** `ghcr.io/ggml-org/llama.cpp:server-cuda13` | GB10 aarch64 + CUDA 13; local build should use `GGML_CUDA_COMPRESSION_MODE=speed` |
| Offload | **`-ngl 99`** | Full GPU / UMA offload |
| Flash Attn | **`-fa on`** | Playbook win on SM121 |
| KV cache | **`-ctk q8_0 -ctv q8_0`** | Fits long context on 121 GB UMA without crushing quality |
| Context | **32 768** (production) | Matches FP8/NVFP4 production profiles for fair API benches |
| Speculation | **`--spec-type draft-mtp`** | Card notes MTP; enable when GGUF ships MTP tensors |
| API alias | `qwen3.8-27b-gguf` | Stable `model` id for lm-eval |

**Override for Unsloth’s general default:** `QUANT=UD-Q4_K_XL` (~17.9 GB) — faster download / more KV headroom, slightly more quant loss.

| Quant | Size | Use |
|-------|------|-----|
| `UD-Q4_K_XL` | 17.9 GB | Unsloth recommended general default |
| `UD-Q5_K_XL` | 20.2 GB | Mid quality / speed |
| `UD-Q6_K_XL` | 25.9 GB | High quality |
| **`UD-Q8_K_XL`** | **31.5 GB** | **Default here — report peer to FP8** |
| `Q8_0` | 29.0 GB | Classic 8-bit (non-UD) |
| `BF16` | 54.7 GB | Quality ceiling (less KV room) |

**Stop other `:8000` servers** (vLLM FP8 / NVFP4) before starting.

## Prerequisites

- DGX Spark (GB10 / SM121), ~121 GB UMA
- Hugging Face CLI: `huggingface-cli login` if needed
- Either:
  - Local build: `~/llama.cpp/build/bin/llama-server` with `CMAKE_CUDA_ARCHITECTURES=121a-real` and **`GGML_CUDA_COMPRESSION_MODE=speed`** (see [playbook](../../../Qwen36-GB10-Performance-Playbook.md)), or
  - Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda13`

## Download weights

```bash
cd ~/llm-benchmark/llama.cpp/recipes/unsloth-qwen3.8-27b-gguf
./download.sh                 # UD-Q8_K_XL
QUANT=UD-Q4_K_XL ./download.sh
```

Files land under `~/models/gguf/Qwen3.8-27B-GGUF/`. Or let `llama-server -hf …` pull into the HF cache.

## Run (host — preferred when local speed build exists)

```bash
cd ~/llm-benchmark/llama.cpp/recipes/unsloth-qwen3.8-27b-gguf
./start.sh
# QUANT=UD-Q4_K_XL ./start.sh
# PROFILE=long ./start.sh
# PROFILE=ar SPEC_TYPE=none ./start.sh
./stop.sh
```

## Run (Docker)

```bash
./start-docker.sh
# PROFILE=long ./start-docker.sh
./stop.sh
```

| Profile | Context | Notes |
|---------|---------|--------|
| `production` (default) | 32 768 | MTP on, q8 KV, `-ngl 99`, `-fa on` |
| `long` | 262 144 | Same KV types; drop concurrency pressure |
| `ar` | 32 768 | `SPEC_TYPE=none` — no draft-mtp |

| Flag / env | Production default |
|------------|-------------------|
| `QUANT` | `UD-Q8_K_XL` |
| `-c` / `CTX_SIZE` | `32768` |
| `-ngl` | `99` |
| `-fa` | `on` |
| `-ctk` / `-ctv` | `q8_0` |
| `--spec-type` | `draft-mtp` |
| `--alias` | `qwen3.8-27b-gguf` |
| Port | `8000` |

Vision (optional):

```bash
VISION=1 ./start.sh   # also loads mmproj-F16.gguf (download.sh pulls it when VISION=1)
```

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-gguf",
    "messages": [{"role": "user", "content": "What is 19 times 23? Reply with the number only."}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

## llama-bench (optional)

```bash
~/llama.cpp/build/bin/llama-bench \
  -m ~/models/gguf/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  -p 2048,4096,8192 -n 32,128 -ngl 99 -fa on -r 3 -o md
```

## Report lineup (planned)

| Recipe | Runtime | Quant |
|--------|---------|-------|
| `vllm/recipes/qwen-qwen3.8-27b-fp8` | vLLM | Official FP8 |
| `vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121` | vLLM SM121 | NVFP4 + MTP |
| **`llama.cpp/recipes/unsloth-qwen3.8-27b-gguf`** | llama.cpp | **UD-Q8_K_XL** |

## Notes

- Host `llama-server` older than `llama-bench` may lack newer Qwen3.8 / MTP bits — prefer a rebuilt binary or Docker if load fails.
- Unsloth card default `UD-Q4_K_XL` is fine for interactive use; keep **Q8** for the quality report unless you explicitly want a Q4 data point too.
