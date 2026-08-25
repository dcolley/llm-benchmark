# mmlu_llama comparison: Ornith-1.5-35B-A3B-NVFP4 vs Qwen3.6-35B-A3B-MTP

| Ornith-1.5-35B-A3B-NVFP4 | Qwen3.6-35B-A3B-MTP |
|------------------------:|--------------------:|
| **82.12%** | **84.76%** |

Side-by-side `mmlu_llama` (5-shot, `strict_match` / `exact_match`) on local OpenAI-compatible serving.

| | **ornith-ai/Ornith-1.5-35B-A3B-NVFP4** | **unsloth/qwen3.6-35b-a3b-mtp** |
|--|--|--|
| **Overall** | **82.12% ± 0.31%** | **84.76% ± 0.29%** |
| Δ vs Qwen | **-2.64 pp** | baseline |
| Samples | 14,042 | 14,042 |
| Serving | vLLM Docker (`:8000`) | LM Studio (`:1234`) |
| Quant | NVFP4 | Unsloth MTP |

**Headline:** Qwen3.6-35B-A3B-MTP is **2.6 pp** ahead overall (84.76% vs 82.12%).

> Methodologies are not identical (decode settings / stack). Treat small subtask gaps as noise.

## Domain scores

| Domain | Ornith | Qwen MTP | Δ (Ornith − Qwen) |
|--------|-------:|---------:|------------------:|
| Social sciences | 89.34% ± 0.55% | 91.55% ± 0.49% | -2.21 pp |
| Other | 85.03% ± 0.61% | 87.22% ± 0.57% | -2.19 pp |
| Stem | 79.99% ± 0.69% | 83.44% ± 0.64% | -3.45 pp |
| Humanities | 76.90% ± 0.59% | 79.57% ± 0.57% | -2.67 pp |
| **Overall** | **82.12% ± 0.31%** | **84.76% ± 0.29%** | **-2.64 pp** |

## Setup (Ornith)

```bash
lm-eval --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:8000/v1/chat/completions,model=ornith-1.5-35b-a3b-nvfp4,..." \
  --tasks mmlu_llama --apply_chat_template \
  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false'
```

Results JSON: `/home/derek/llm-benchmark/lm-eval/results/ornith-1.5-35b-a3b-nvfp4/ornith-1.5-35b-a3b-nvfp4/results_2026-08-21T17-19-28.964073.json`

Thinking disabled server-side (`enable_thinking: false`). Recipe: `vllm/recipes/ornith-1.5-35b-a3b-nvfp4`.

## Largest gaps (|Δ| ≥ 4 pp)

Ornith has no subtasks ≥4 pp ahead of Qwen.

### Qwen ahead (biggest)

| Subtask | Ornith | Qwen | Δ |
|---------|-------:|-----:|--:|
| professional accounting | 66.3% | 75.5% | **-9.2** |
| college physics | 61.8% | 69.6% | **-7.8** |
| high school physics | 77.5% | 84.1% | **-6.6** |
| us foreign policy | 89.0% | 95.0% | **-6.0** |
| high school mathematics | 59.3% | 65.2% | **-5.9** |
| high school statistics | 79.2% | 84.7% | **-5.6** |
| machine learning | 75.0% | 80.4% | **-5.4** |
| econometrics | 78.1% | 83.3% | **-5.3** |
| anatomy | 83.0% | 88.2% | **-5.2** |
| conceptual physics | 90.6% | 95.7% | **-5.1** |
| philosophy | 85.5% | 89.7% | **-4.2** |
| professional law | 66.8% | 70.9% | **-4.2** |

## Weak spots (Ornith < 75%)

| Subtask | Ornith | Qwen |
|---------|-------:|-----:|
| global facts | 52.0% | 56.0% |
| virology | 59.0% | 59.0% |
| high school mathematics | 59.3% | 65.2% |
| college physics | 61.8% | 69.6% |
| professional accounting | 66.3% | 75.5% |
| professional law | 66.8% | 70.9% |
| abstract algebra | 67.0% | 70.0% |
| college mathematics | 67.0% | 70.0% |
| moral scenarios | 67.3% | 69.7% |
| college chemistry | 69.0% | 67.0% |
| formal logic | 70.6% | 73.8% |

## Full subtask table

| Tasks | Ornith | ± | Qwen | Δ pp |
|-------|-------:|--:|-----:|-----:|
| **mmlu_llama** | **0.8212** | 0.0031 | **0.8476** | **-2.64** |
| abstract algebra | 0.6700 | 0.0473 | 0.7000 | -3.00 |
| anatomy | 0.8296 | 0.0325 | 0.8815 | -5.19 |
| astronomy | 0.9079 | 0.0235 | 0.9276 | -1.97 |
| business ethics | 0.8100 | 0.0394 | 0.8400 | -3.00 |
| clinical knowledge | 0.8604 | 0.0213 | 0.9019 | -4.15 |
| college biology | 0.9167 | 0.0231 | 0.9306 | -1.39 |
| college chemistry | 0.6900 | 0.0465 | 0.6700 | +2.00 |
| college computer science | 0.7500 | 0.0435 | 0.7800 | -3.00 |
| college mathematics | 0.6700 | 0.0473 | 0.7000 | -3.00 |
| college medicine | 0.8497 | 0.0272 | 0.8439 | +0.58 |
| college physics | 0.6176 | 0.0484 | 0.6961 | -7.85 |
| computer security | 0.8500 | 0.0359 | 0.8900 | -4.00 |
| conceptual physics | 0.9064 | 0.0190 | 0.9574 | -5.10 |
| econometrics | 0.7807 | 0.0389 | 0.8333 | -5.26 |
| electrical engineering | 0.8207 | 0.0320 | 0.8621 | -4.14 |
| elementary mathematics | 0.7989 | 0.0206 | 0.8095 | -1.06 |
| formal logic | 0.7063 | 0.0407 | 0.7381 | -3.18 |
| global facts | 0.5200 | 0.0502 | 0.5600 | -4.00 |
| high school biology | 0.9355 | 0.0140 | 0.9548 | -1.93 |
| high school chemistry | 0.8227 | 0.0269 | 0.8374 | -1.47 |
| high school computer science | 0.9100 | 0.0288 | 0.9400 | -3.00 |
| high school european history | 0.8727 | 0.0260 | 0.8848 | -1.21 |
| high school geography | 0.9040 | 0.0210 | 0.9444 | -4.04 |
| high school government and politics | 0.9741 | 0.0115 | 0.9845 | -1.04 |
| high school macroeconomics | 0.8897 | 0.0159 | 0.8949 | -0.52 |
| high school mathematics | 0.5926 | 0.0300 | 0.6519 | -5.93 |
| high school microeconomics | 0.9286 | 0.0167 | 0.9580 | -2.94 |
| high school physics | 0.7748 | 0.0341 | 0.8411 | -6.63 |
| high school psychology | 0.9303 | 0.0109 | 0.9596 | -2.93 |
| high school statistics | 0.7917 | 0.0277 | 0.8472 | -5.55 |
| high school us history | 0.9363 | 0.0171 | 0.9363 | -0.00 |
| high school world history | 0.9156 | 0.0181 | 0.9325 | -1.69 |
| human aging | 0.8475 | 0.0241 | 0.8610 | -1.35 |
| human sexuality | 0.9237 | 0.0233 | 0.9466 | -2.29 |
| international law | 0.9091 | 0.0262 | 0.9256 | -1.65 |
| jurisprudence | 0.8796 | 0.0315 | 0.8981 | -1.85 |
| logical fallacies | 0.8834 | 0.0252 | 0.8957 | -1.23 |
| machine learning | 0.7500 | 0.0411 | 0.8036 | -5.36 |
| management | 0.8932 | 0.0306 | 0.9029 | -0.97 |
| marketing | 0.9359 | 0.0160 | 0.9402 | -0.43 |
| medical genetics | 0.9300 | 0.0256 | 0.9500 | -2.00 |
| miscellaneous | 0.9464 | 0.0081 | 0.9502 | -0.38 |
| moral disputes | 0.8382 | 0.0198 | 0.8584 | -2.02 |
| moral scenarios | 0.6726 | 0.0157 | 0.6972 | -2.46 |
| nutrition | 0.8758 | 0.0189 | 0.9020 | -2.62 |
| philosophy | 0.8553 | 0.0200 | 0.8971 | -4.18 |
| prehistory | 0.8981 | 0.0168 | 0.9167 | -1.86 |
| professional accounting | 0.6631 | 0.0282 | 0.7553 | -9.22 |
| professional law | 0.6675 | 0.0120 | 0.7093 | -4.18 |
| professional medicine | 0.9081 | 0.0175 | 0.9338 | -2.57 |
| professional psychology | 0.8824 | 0.0130 | 0.9003 | -1.79 |
| public relations | 0.8000 | 0.0383 | 0.7818 | +1.82 |
| security studies | 0.7918 | 0.0260 | 0.8286 | -3.68 |
| sociology | 0.9254 | 0.0186 | 0.9254 | -0.00 |
| us foreign policy | 0.8900 | 0.0314 | 0.9500 | -6.00 |
| virology | 0.5904 | 0.0383 | 0.5904 | -0.00 |
| world religions | 0.9064 | 0.0223 | 0.8947 | +1.17 |

## Interpretation

1. Overall gap vs Qwen MTP: **-2.64 pp** on the same `mmlu_llama` harness.
2. Ornith is a coding/agentic-focused MoE; MMLU letter accuracy is a different axis than Terminal-Bench / SWE-bench.
3. This run used greedy short continuations with thinking off — closer to Puzzle/Fuse methodology than Qwen’s sampling `/no_think` setup.

*Generated from `results_2026-08-21T17-19-28.964073.json` vs Derek’s Qwen3.6-35B-A3B-MTP table.*
