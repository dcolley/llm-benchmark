# llm-benchmark

LLM, agents, and AI experiments.

## vLLM recipes

Host and Docker launch recipes for DGX Spark live under [`vllm/recipes/`](vllm/recipes/):

| Recipe | Notes |
|--------|--------|
| [`ling-3.0-flash/`](vllm/recipes/ling-3.0-flash/) | Ling-3.0-flash FP4 (Docker or host Ling fork) |
| [`nemotron-puzzle/`](vllm/recipes/nemotron-puzzle/) | Nemotron Puzzle (Docker) |
| [`qwen-agentworld/`](vllm/recipes/qwen-agentworld/) | Qwen-AgentWorld on 2× Spark (host venv) |
| [`qwen3.6-27b-nvfp4/`](vllm/recipes/qwen3.6-27b-nvfp4/) | Qwen3.6-27B-NVFP4 (host venv) |

Recipes that need a Python venv include a `readme-venv.md` in that folder.

Copy `*.env.example` to `*.env` locally (env files are gitignored). Do not commit private IPs, tokens, or venvs.

## lm-eval

Harness and benchmark runs live under [`lm-eval/`](lm-eval/).

```bash
source ~/llm-benchmark/lm-eval/bin/activate
```

| Run | Notes |
|-----|--------|
| [`lm-eval/results/fuse-1-lite/`](lm-eval/results/fuse-1-lite/) | `Akahsizrr/fuse-1-Lite` on `192.168.10.199:8000` — [MMLU report](lm-eval/results/fuse-1-lite/MMLU_LLAMA_REPORT.md) |
| [`lm-eval/results/nemotron-puzzle/`](lm-eval/results/nemotron-puzzle/) | Nemotron Puzzle vs Qwen comparison |

Remote runner + watchdog: [`lm-eval/scripts/run_mmlu_llama_remote.sh`](lm-eval/scripts/run_mmlu_llama_remote.sh), [`lm-eval/scripts/mmlu_watchdog_remote.sh`](lm-eval/scripts/mmlu_watchdog_remote.sh).