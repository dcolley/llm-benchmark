# Qwen3.8-27B quant comparison on DGX Spark

Generative MMLU, five-shot, 14,042 samples, identical extraction task and non-thinking sampling. Higher is better.

| Model | Quant | Runtime | MMLU | Valid extractions |
|-------|-------|---------|-----:|------------------:|
| `Qwen/Qwen3.8-27B-FP8` | FP8 | vLLM 0.27.1 | **84.48% ± 0.29%** | 14,031/14,042 |
| `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` | NVFP4 + FP8 KV | vLLM 0.27.2rc0-sm121, MTP K=3 | **84.11% ± 0.30%** | 42,121/42,126 |
| `unsloth/Qwen3.8-27B-GGUF` | UD-Q8_K_XL GGUF | llama.cpp CUDA 13, MTP | **83.50% ± 0.30%** | 14,033/14,042 |

**Best MMLU result:** `Qwen/Qwen3.8-27B-FP8` at **84.48%**.

## Domain comparison

| Domain | Qwen FP8 | r0b0tlab NVFP4 | Unsloth GGUF Q8 |
|--------|----------:|-----------------:|------------------:|
| Social sciences | 90.93% | 90.22% | 90.71% |
| Other | 86.03% | 85.42% | 84.74% |
| STEM | 82.59% | 82.02% | 81.13% |
| Humanities | 80.49% | 80.66% | 79.55% |
| **Overall** | **84.48%** | **84.11%** | **83.50%** |

## Method

- Task: `mmlu_llama_ds4`; stock questions, targets, and five-shot examples.
- Prompt: concise `/no_think` answer-only instruction.
- Sampling: temperature 0.7, top-p 0.8, top-k 20, max 64 tokens.
- API concurrency: 8.
- The serving stacks differ, so this primarily compares end-to-end accuracy. Runtime throughput should be reported separately.

## Result artifacts

- `Qwen/Qwen3.8-27B-FP8`: `results/qwen-qwen3.8-27b-fp8/qwen3.8-27b-fp8/results_2026-08-16T21-47-46.448403.json`
- `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`: `results/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/qwen3.8-27b-nvfp4-mtp/results_2026-08-17T15-03-04.080277.json`
- `unsloth/Qwen3.8-27B-GGUF`: `results/unsloth-qwen3.8-27b-gguf-ud-q8-k-xl/qwen3.8-27b-gguf/results_2026-08-17T20-00-08.332345.json`
