# Qwen3.6-27B-NVFP4 on DGX Spark

Serve [`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) with stock vLLM on the host. Port **8000**.

## Venv

See **[readme-venv.md](./readme-venv.md)**.

## Download weights

```bash
huggingface-cli download nvidia/Qwen3.6-27B-NVFP4
```

Or let vLLM pull on first serve.

## Run

```bash
cd vllm/recipes/qwen3.6-27b-nvfp4
./start-nvidia-Qwen3.6-27B-NVFP4.sh
```

Defaults in the script: `max-model-len=262144`, `max-num-seqs=4`, `gpu-memory-utilization=0.75`, `quantization=modelopt`, Qwen3 reasoning + XML tool parsers.
