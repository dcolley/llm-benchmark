# LM Evaluation Harness

**venv:** `/home/derek/llm-benchmark/lm-eval` (Python 3.11, created with uv)

**Activate:** `source ~/llm-benchmark/lm-eval/bin/activate`

## Quick start

```bash
source ~/llm-benchmark/lm-eval/bin/activate

# Evaluate an HF model on MMLU
lm-eval --model hf \
  --model_args pretrained=EleutherAI/pythia-70m \
  --tasks mmlu \
  --num_fewshot 5 \
  --limit 100
```

## Ling-3.0-flash-int4 (local SGLang)

Serve from [`../sglang/recipes/ling-3.0-flash-int4/`](../sglang/recipes/ling-3.0-flash-int4/) (thinking off for letter answers), then:

```bash
# smoke
./scripts/run_mmlu_llama_ling_int4.sh --limit 5 --foreground

# full (background) + watchdog
./scripts/run_mmlu_llama_ling_int4.sh
nohup ./scripts/mmlu_watchdog_ling_int4.sh >/dev/null 2>&1 &
```

Defaults: `BASE_URL=http://127.0.0.1:8000`, `MODEL_ID=ling-3.0-flash-int4`, output under `results/ling-3.0-flash-int4/`.

Report: [`results/ling-3.0-flash-int4/MMLU_LLAMA_REPORT.md`](results/ling-3.0-flash-int4/MMLU_LLAMA_REPORT.md) — **83.61%** overall (2026-08-07).

## Qwen3.8-27B quant matrix (local vLLM / llama.cpp)

Same `mmlu_llama_ds4` extraction task for three Spark recipes:

```bash
# FP8 (vLLM) — MODEL_ID=qwen3.8-27b-fp8
./scripts/run_mmlu_llama_qwen38_27b_fp8.sh --limit 5 --foreground

# NVFP4 / GGUF: start the matching recipe, then reuse the runner with MODEL_ID / OUT_DIR
MODEL_ID=qwen3.8-27b-nvfp4-mtp OUT_DIR=./results/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121 \
  ./scripts/run_mmlu_llama_qwen38_27b_fp8.sh --foreground
```

| Variant | Recipe | MMLU |
|--------|--------|-----:|
| FP8 | [`../vllm/recipes/qwen-qwen3.8-27b-fp8/`](../vllm/recipes/qwen-qwen3.8-27b-fp8/) | **84.48%** |
| NVFP4 | [`../vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/`](../vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/) | **84.11%** |
| UD-Q8_K_XL | [`../llama.cpp/recipes/unsloth-qwen3.8-27b-gguf/`](../llama.cpp/recipes/unsloth-qwen3.8-27b-gguf/) | **83.50%** |

Comparison: [`results/QWEN3.8_27B_QUANT_REPORT.md`](results/QWEN3.8_27B_QUANT_REPORT.md).

## Ornith-1.5-35B-A3B-NVFP4 (local vLLM)

Serve from [`../vllm/recipes/ornith-1.5-35b-a3b-nvfp4/`](../vllm/recipes/ornith-1.5-35b-a3b-nvfp4/), then:

```bash
# smoke
./scripts/run_mmlu_llama_ornith.sh --limit 5 --foreground

# full (background) + watchdog
./scripts/run_mmlu_llama_ornith.sh
nohup ./scripts/mmlu_watchdog_ornith.sh >/dev/null 2>&1 &
```

Defaults: `BASE_URL=http://127.0.0.1:8000`, `MODEL_ID=ornith-1.5-35b-a3b-nvfp4`, output under `results/ornith-1.5-35b-a3b-nvfp4/`.

Report: [`results/ornith-1.5-35b-a3b-nvfp4/MMLU_LLAMA_REPORT.md`](results/ornith-1.5-35b-a3b-nvfp4/MMLU_LLAMA_REPORT.md) — **82.12%** overall (2026-08-21).

## DeepSeek-V4-Flash-0731 (local ds4)

Serve from [`../ds4/recipes/deepseek-v4-flash-0731/`](../ds4/recipes/deepseek-v4-flash-0731/), then:

```bash
# smoke
./scripts/run_mmlu_llama_deepseek_v4_flash.sh --limit 5 --foreground

# full (background) + watchdog
./scripts/run_mmlu_llama_deepseek_v4_flash.sh
nohup ./scripts/mmlu_watchdog_deepseek_v4_flash.sh >/dev/null 2>&1 &
```

Defaults: `BASE_URL=http://127.0.0.1:8000`, `MODEL_ID=deepseek-v4-flash`, output under `results/deepseek-v4-flash-0731/`.

Report: [`results/deepseek-v4-flash-0731/MMLU_LLAMA_REPORT.md`](results/deepseek-v4-flash-0731/MMLU_LLAMA_REPORT.md) — **82.55%** overall (2026-08-12).

## vLLM integration (API server pattern)

vLLM cannot be pip-installed on aarch64. Use the NGC container instead:

```bash
# 1. Start vLLM server
docker run --gpus all -d --rm \
  --name vllm-server \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:26.02-py3 \
  --model meta-llama/Llama-3.2-3B-Instruct

# 2. Wait for server
until curl -s http://localhost:8000/v1/models > /dev/null; do sleep 2; done

# 3. Run evaluation
source ~/llm-benchmark/lm-eval/bin/activate
lm-eval --model local-chat \
  --model_args base_url=http://localhost:8000/v1 \
  --tasks mmlu

# 4. Stop server
docker stop vllm-server
```

A helper script is available: `~/vllm-eval.sh`

## Remote mmlu_llama (fuse-1-Lite)

```bash
# smoke
./scripts/run_mmlu_llama_remote.sh --limit 5 --foreground

# full (background) + watchdog
./scripts/run_mmlu_llama_remote.sh
nohup ./scripts/mmlu_watchdog_remote.sh >/dev/null 2>&1 &
```

Defaults: `BASE_URL=http://192.168.10.199:8000`, `MODEL_ID=Akahsizrr/fuse-1-Lite`, output under `results/fuse-1-lite/`.

Report: [`results/fuse-1-lite/MMLU_LLAMA_REPORT.md`](results/fuse-1-lite/MMLU_LLAMA_REPORT.md)

## Installed packages

- lm-eval 0.4.12
- torch 2.12.0+cu130
- transformers 5.12.0
- accelerate 1.14.0
- datasets 5.0.0
