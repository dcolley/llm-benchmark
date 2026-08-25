# llm-benchmark

LLM, agents, and AI experiments.

## SGLang recipes

Host and Docker launch recipes for DGX Spark live under [`sglang/recipes/`](sglang/recipes/):

| Recipe | Notes |
|--------|--------|
| [`ling-3.0-flash-int4/`](sglang/recipes/ling-3.0-flash-int4/) | Ling-3.0-flash INT4 (SGLang Docker; preferred on Spark; thinking + 262K + NEXTN) |
| [`radixark-qwen3.8-27b-nvfp4-bf16-lmhead/`](sglang/recipes/radixark-qwen3.8-27b-nvfp4-bf16-lmhead/) | Qwen3.8-27B NVFP4, BF16 `lm_head` (SGLang; MTP; Spark 0.80 mem) |

## ds4 recipes (DeepSeek-V4 on one Spark)

Native C/CUDA via [`Entrpi/ds4-on-spark`](https://github.com/entrpi/ds4-on-spark) — **not** stock vLLM/SGLang (official HF weights are too large for 128 GB UMA).

| Recipe | Notes |
|--------|--------|
| [`deepseek-v4-flash-0731/`](ds4/recipes/deepseek-v4-flash-0731/) | DeepSeek-V4-Flash-0731 (~81 GiB IQ2XXS GGUF + DSpark); fits **one** Spark |

## vLLM recipes

Host and Docker launch recipes for DGX Spark live under [`vllm/recipes/`](vllm/recipes/) (also `~/vllm/recipes/`):

| Recipe | Notes |
|--------|--------|
| [`ling-3.0-flash/`](vllm/recipes/ling-3.0-flash/) | Ling-3.0-flash FP4 (Docker or host Ling fork; OOM’d on single Spark) |
| [`nemotron-puzzle/`](vllm/recipes/nemotron-puzzle/) | Nemotron Puzzle (Docker) |
| [`nemotron-3.5-lightning-30b-nvfp4/`](vllm/recipes/nemotron-3.5-lightning-30b-nvfp4/) | Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark (host venv or Docker v0.27.1) |
| [`ornith-1.5-35b-a3b-nvfp4/`](vllm/recipes/ornith-1.5-35b-a3b-nvfp4/) | Ornith-1.5-35B-A3B **NVFP4** (Docker preferred; FP8/BF16 via `MODEL=`) |
| [`qwen-agentworld/`](vllm/recipes/qwen-agentworld/) | Qwen-AgentWorld on 2× Spark (host venv) |
| [`qwen3.6-27b-nvfp4/`](vllm/recipes/qwen3.6-27b-nvfp4/) | Qwen3.6-27B-NVFP4 (host venv) |
| [`qwen-qwen3.8-27b-fp8/`](vllm/recipes/qwen-qwen3.8-27b-fp8/) | Qwen3.8-27B-FP8 VLM + MTP (Docker preferred) |
| [`r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/`](vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/) | Qwen3.8-27B NVFP4 + MTP (SM121 Docker image) |

## llama.cpp recipes

| Recipe | Notes |
|--------|--------|
| [`unsloth-qwen3.8-27b-gguf/`](llama.cpp/recipes/unsloth-qwen3.8-27b-gguf/) | Unsloth Qwen3.8-27B UD-Q8_K_XL (llama.cpp Docker) |

Recipes that need a Python venv include a `readme-venv.md` in that folder.

Copy `*.env.example` to `*.env` locally (env files are gitignored). Do not commit private IPs, tokens, or venvs.

## lm-eval

Harness and benchmark runs live under [`lm-eval/`](lm-eval/).

```bash
source ~/llm-benchmark/lm-eval/bin/activate
```

| Run | Notes |
|-----|--------|
| [`lm-eval/results/ling-3.0-flash-int4/`](lm-eval/results/ling-3.0-flash-int4/) | Ling INT4 on local SGLang — [MMLU report](lm-eval/results/ling-3.0-flash-int4/MMLU_LLAMA_REPORT.md) (**83.61%**) |
| [`lm-eval/results/fuse-1-lite/`](lm-eval/results/fuse-1-lite/) | `Akahsizrr/fuse-1-Lite` on `192.168.10.199:8000` — [MMLU report](lm-eval/results/fuse-1-lite/MMLU_LLAMA_REPORT.md) |
| [`lm-eval/results/nemotron-puzzle/`](lm-eval/results/nemotron-puzzle/) | Nemotron Puzzle vs Qwen comparison |
| [`lm-eval/results/QWEN3.8_27B_QUANT_REPORT.md`](lm-eval/results/QWEN3.8_27B_QUANT_REPORT.md) | Qwen3.8-27B FP8 vs NVFP4 vs GGUF Q8 — FP8 **84.48%** |
| [`lm-eval/results/ornith-1.5-35b-a3b-nvfp4/`](lm-eval/results/ornith-1.5-35b-a3b-nvfp4/) | Ornith-1.5-35B-A3B NVFP4 on local vLLM — [MMLU report](lm-eval/results/ornith-1.5-35b-a3b-nvfp4/MMLU_LLAMA_REPORT.md) (**82.12%**) |
| [`lm-eval/results/deepseek-v4-flash-0731/`](lm-eval/results/deepseek-v4-flash-0731/) | DeepSeek-V4-Flash-0731 on local ds4 — [MMLU report](lm-eval/results/deepseek-v4-flash-0731/MMLU_LLAMA_REPORT.md) (**82.55%**) |

Runners: [`lm-eval/scripts/run_mmlu_llama_ling_int4.sh`](lm-eval/scripts/run_mmlu_llama_ling_int4.sh), [`lm-eval/scripts/run_mmlu_llama_remote.sh`](lm-eval/scripts/run_mmlu_llama_remote.sh), [`lm-eval/scripts/run_mmlu_llama_qwen38_27b_fp8.sh`](lm-eval/scripts/run_mmlu_llama_qwen38_27b_fp8.sh), [`lm-eval/scripts/run_mmlu_llama_ornith.sh`](lm-eval/scripts/run_mmlu_llama_ornith.sh), [`lm-eval/scripts/run_mmlu_llama_deepseek_v4_flash.sh`](lm-eval/scripts/run_mmlu_llama_deepseek_v4_flash.sh). Watchdogs: `mmlu_watchdog_ling_int4.sh`, `mmlu_watchdog_remote.sh`, `mmlu_watchdog_ornith.sh`, `mmlu_watchdog_deepseek_v4_flash.sh`.
