# Ornith-1.5-35B-A3B on DGX Spark

Serve [Ornith-1.5](https://huggingface.co/collections/ornith-ai/ornith-15) mid-size MoE ([`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) family) with stock vLLM on **one** GB10 (~121–128 GB UMA). Port **8000**.

Official card: reasoning model (`<think>` → `reasoning_content`), Qwen3 tool/reasoning parsers, vLLM ≥ 0.19.1 (this recipe uses **`vllm/vllm-openai:v0.27.1`**).

## Which checkpoint for this hardware?

| Variant | Best for Spark? | Why |
|---------|-----------------|-----|
| **[`…-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)** | **Yes — default** | Smallest weight footprint on Blackwell; same pattern as your Qwen/Nemotron NVFP4 recipes; most KV / context headroom on 128 GB UMA |
| [`…-FP8`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8) | Fallback | Larger than NVFP4; use if NVFP4 kernels misbehave on SM121 |
| [`…-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (BF16) | Only if needed | Card ≈ **70 GB** weights; fits UMA but squeezes long-context KV. Official serve assumes **2×80 GB**; on one Spark keep `max-model-len` modest |
| [`…-MLX`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX) | **No** | Apple Silicon / MLX only — not for CUDA / DGX Spark / this Docker recipe |

**Recommendation:** ship **`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`**. Override with `MODEL=ornith-ai/Ornith-1.5-35B-A3B-FP8` (or BF16) only if you need to A/B quality.

Single-GPU (`--tensor-parallel-size 1`). Stop other `:8000` servers first.

## Defaults (chosen for Spark)

| Flag | Value | Why |
|------|-------|-----|
| Model | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | Best UMA fit |
| Image | `vllm/vllm-openai:v0.27.1` | Matches other Spark recipes; ≥ 0.19.1 required |
| Context | **262144** (prod / chat / long) | Card window; agents packing ~128K need headroom above 131072 |
| GPU util | **0.85** | Leaves UMA headroom; raise only after a stable boot |
| Eager | **`--enforce-eager`** | Disables CUDA graphs — required on SM121 (illegal instruction / illegal address with FULL_AND_PIECEWISE) |
| Prefix cache | **off** (`--no-enable-prefix-caching`) | Hybrid Mamba prefix-cache is experimental and correlated with engine deaths |
| Max seqs | **2** | Reduces concurrent decode pressure on UMA |
| Reasoning | `--reasoning-parser qwen3` | Per [model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Tools | `qwen3_xml` + auto tool choice | Per card |
| Thinking off (optional) | `enable_thinking: false` via chat-template kwargs | Safer for letter-style `mmlu_llama` / clients that break on empty `content` |
| Cache mounts | HF + `VLLM_CACHE_ROOT` | Avoid re-autotune after recreate |

Sampling (from card): general `temperature=0.6`, `top_p=0.95`, `top_k=20`; their published benches use `temperature=1.0`.

## Download

```bash
huggingface-cli download ornith-ai/Ornith-1.5-35B-A3B-NVFP4
# optional: ornith-ai/Ornith-1.5-35B-A3B-FP8
# optional: ornith-ai/Ornith-1.5-35B-A3B
```

Or let the container pull on first start (cache at `~/.cache/huggingface`).

## Run (Docker)

```bash
cd ~/llm-benchmark/vllm/recipes/ornith-1.5-35b-a3b-nvfp4
./start-docker.sh
# long context:
PROFILE=long ./start-docker.sh
# FP8 / BF16 override:
MODEL=ornith-ai/Ornith-1.5-35B-A3B-FP8 SERVED_MODEL_NAME=ornith-1.5-35b-a3b-fp8 ./start-docker.sh
./stop.sh
```

| Profile | `max-model-len` | util | Use |
|---------|-----------------|------|-----|
| `production` / `mmlu` (default) | 262144 | 0.85 | Eval + agents (full card window) |
| `chat` | 262144 | 0.85 | Same as production |
| `long` | 262144 | 0.85 | Alias for max context |

## Smoke

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-1.5-35b-a3b-nvfp4",
    "messages": [{"role":"user","content":"Say hi in one word."}],
    "max_tokens": 64,
    "temperature": 0.6
  }' | python3 -m json.tool
```

Expect `reasoning_content` and/or `content` depending on thinking settings.

## Host venv (optional)

Needs a recent vLLM with NVFP4 / Qwen3 MoE support on aarch64 — prefer Docker on Spark. If you already have a working host venv:

```bash
./start.sh
```

## Notes

- Collection: [ornith-ai/ornith-15](https://huggingface.co/collections/ornith-ai/ornith-15)
- Blog / method: [Ornith-1.5](https://ornith.ai/ornith_1_5.html)
- YaRN to ~1M tokens is documented on the BF16 card; only enable when you truly need it (hurts normal-length quality if left on statically).
