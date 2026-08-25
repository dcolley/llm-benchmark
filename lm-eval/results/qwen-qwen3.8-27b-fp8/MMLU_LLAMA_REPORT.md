# mmlu_llama comparison: Qwen3.8-27B-FP8 vs Qwen3.6-35B-A3B-MTP

| Qwen3.8-27B-FP8 | Qwen3.6-35B-A3B-MTP | Δ |
|------------------:|--------------------:|--:|
| **84.48%** | **84.76%** | **-0.28 pp** |

Generative MMLU, five-shot, `strict_match` / `exact_match`, 14,042 samples.

| | **Qwen3.8-27B-FP8** | **Qwen3.6-35B-A3B-MTP** |
|--|--:|--:|
| **Overall** | **84.48% ± 0.29%** | **84.76% ± 0.29%** |
| Δ vs Qwen3.6 | **-0.28 pp** | baseline |
| Samples | 14,042 | 14,042 |
| Architecture | Dense 27B, block FP8, MTP-3 | MoE 35B-A3B, MTP |
| Serving | vLLM 0.27.1 on DGX Spark | LM Studio |
| Date | 2026-08-16 | ~2026-06-16 |

**Headline:** Qwen3.8-27B-FP8 trails Qwen3.6-35B-A3B-MTP by **0.28 points** overall (84.48% vs 84.76%).

> The questions, five-shot examples, targets, and strict-match metric are shared, but answer extraction and serving stacks differ. Treat small deltas as directional.

---

## Domain scores

| Domain | Qwen3.8-27B-FP8 | Qwen3.6 MTP | Δ (3.8 − 3.6) |
|--------|-----------------:|------------:|--------------:|
| Social sciences | 90.93% ± 0.51% | 91.55% ± 0.49% | -0.62 pp |
| Other | 86.03% ± 0.59% | 87.22% ± 0.57% | -1.19 pp |
| STEM | 82.59% ± 0.65% | 83.44% ± 0.64% | -0.85 pp |
| Humanities | 80.49% ± 0.56% | 79.57% ± 0.57% | +0.92 pp |
| **Overall** | **84.48% ± 0.29%** | **84.76% ± 0.29%** | **-0.28 pp** |

---

## Setup differences

### Qwen3.8-27B-FP8 (this run)

```bash
./scripts/run_mmlu_llama_qwen38_27b_fp8.sh --foreground
```

- Local vLLM 0.27.1, `qwen3.8-27b-fp8`, FP8 weights and FP8 KV cache.
- MTP speculative decoding: 3 tokens; eight concurrent requests.
- `mmlu_llama_ds4`: stock MMLU questions, targets, and five-shot examples, but full assistant turns with regex extraction of `The best answer is [A-D]`.
- Thinking disabled using `chat_template_kwargs.enable_thinking=false`; concise answer-only system instruction; Qwen-recommended non-thinking sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`).
- Extraction validity: 14,031/14,042 valid (11 invalid).
- Results: `results/qwen-qwen3.8-27b-fp8/qwen3.8-27b-fp8/results_2026-08-16T21-47-46.448403.json`.

Stock `mmlu_llama` was rejected during smoke testing because Qwen3.8 ignored its partial assistant continuation (`The best answer is`) and produced a fresh prose answer, falsely scoring 0%. The extraction task avoids that protocol artifact.

### Qwen3.6-35B-A3B-MTP (prior baseline)

- Stock `mmlu_llama` partial-assistant continuation under LM Studio.
- Thinking disabled with `/no_think`; `max_tokens=512`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`.

---

## Weakest and strongest subjects (Qwen3.8)

Lowest:

| Subject | Score |
|---------|------:|
| global facts | 53.00% |
| virology | 57.23% |
| college physics | 63.73% |
| high school mathematics | 64.81% |
| college chemistry | 66.00% |
| professional law | 69.43% |
| college mathematics | 71.00% |
| abstract algebra | 73.00% |

Highest:

| Subject | Score |
|---------|------:|
| high school government and politics | 97.41% |
| medical genetics | 97.00% |
| high school microeconomics | 95.80% |
| college biology | 95.14% |
| high school us history | 95.10% |
| high school world history | 94.94% |
| high school biology | 94.84% |
| high school psychology | 94.50% |

---

## Takeaways

1. Overall difference versus Qwen3.6 is **-0.28 pp**.
2. Domain-level differences are more informative than tiny subject-level changes.
3. For a strict apples-to-apples comparison, rerun Qwen3.6 with the same vLLM server and extraction task.

*Generated from `results_2026-08-16T21-47-46.448403.json` and the published Qwen3.6-35B-A3B-MTP table.*
