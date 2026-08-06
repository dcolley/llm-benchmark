# delbot

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
