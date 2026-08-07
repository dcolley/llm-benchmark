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
