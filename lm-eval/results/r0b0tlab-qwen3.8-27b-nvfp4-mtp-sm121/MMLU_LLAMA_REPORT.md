# mmlu_llama: Qwen3.8-27B-NVFP4-MTP-sm121

| NVFP4 | Qwen FP8 | Δ vs FP8 | Qwen3.6 MTP | Δ vs 3.6 |
|------:|---------:|---------:|------------:|---------:|
| **84.11%** | **84.48%** | **-0.37 pp** | **84.76%** | **-0.65 pp** |

Generative MMLU, five-shot, `strict_match` / `exact_match`, 14,042 samples.

| | **NVFP4** | **Qwen FP8** | **Qwen3.6 MTP** |
|--|--:|--:|--:|
| **Overall** | **84.11% ± 0.30%** | **84.48%** | **84.76% ± 0.29%** |
| Samples | 14,042 | 14,042 | 14,042 |
| Quant | NVFP4 weights, FP8 KV | block FP8, FP8 KV | MoE 35B-A3B |
| Serving | vLLM 0.27.2rc0-sm121, MTP K=3 | vLLM 0.27.1, MTP-3 | LM Studio |
| Date | 2026-08-17 | 2026-08-16 | ~2026-06-16 |

**Headline:** NVFP4 trails Qwen FP8 by **0.37 pp** (84.11% vs 84.48%) and Qwen3.6 by **0.65 pp**.

---

## Domain scores

| Domain | NVFP4 | Qwen FP8 | Δ vs FP8 | Qwen3.6 | Δ vs 3.6 |
|--------|------:|---------:|---------:|--------:|---------:|
| Social sciences | 90.22% ± 0.53% | 90.93% | -0.71 pp | 91.55% | -1.33 pp |
| Other | 85.42% ± 0.60% | 86.03% | -0.61 pp | 87.22% | -1.80 pp |
| STEM | 82.02% ± 0.66% | 82.59% | -0.57 pp | 83.44% | -1.42 pp |
| Humanities | 80.66% ± 0.56% | 80.49% | +0.17 pp | 79.57% | +1.09 pp |
| **Overall** | **84.11% ± 0.30%** | **84.48%** | **-0.37 pp** | **84.76%** | **-0.65 pp** |

---

## Setup

- Image: `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121`
- `--kv-cache-dtype fp8`, MTP K=3, ctx 32K, `enforce-eager`, no prefix cache.
- Same `mmlu_llama_ds4` extraction task and non-thinking sampling as FP8.
- Extraction validity: 42,121/42,126 valid (5 invalid).
- Results: `results/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/qwen3.8-27b-nvfp4-mtp/results_2026-08-17T15-03-04.080277.json`.

---

## Weakest and strongest subjects

Lowest:

| Subject | Score |
|---------|------:|
| global facts | 53.00% |
| virology | 57.23% |
| high school mathematics | 61.48% |
| college chemistry | 67.00% |
| college physics | 67.65% |
| professional law | 69.56% |
| college mathematics | 70.00% |
| professional accounting | 70.92% |

Highest:

| Subject | Score |
|---------|------:|
| high school government and politics | 97.41% |
| high school computer science | 96.00% |
| college biology | 95.83% |
| medical genetics | 95.00% |
| high school geography | 94.95% |
| high school biology | 94.52% |
| marketing | 94.44% |
| international law | 94.21% |

*Generated from `results_2026-08-17T15-03-04.080277.json`.*
