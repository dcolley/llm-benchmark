# mmlu_llama comparison: DeepSeek-V4-Flash-0731 vs Qwen3.6-35B-A3B-MTP

| DeepSeek-V4-Flash-0731 | Qwen3.6-35B-A3B-MTP | Ling INT4 (context) | Muse Glimmer (context) |
|-----------------------:|--------------------:|--------------------:|-----------------------:|
| **82.55%** | **84.76%** | 83.61% | 82.13% |

Side-by-side generative MMLU (`mmlu_llama` family, 5-shot, `strict_match` / `exact_match`).

| | **DeepSeek-V4-Flash-0731 (ds4 IQ2XXS/Q2K)** | **unsloth/qwen3.6-35b-a3b-mtp** |
|--|--|--|
| **Overall** | **82.55% ± 0.30%** | **84.76% ± 0.29%** |
| Δ vs Qwen | **-2.21 pp** | baseline |
| Samples | 14042 | 14,042 |
| Model | DeepSeek-V4-Flash-0731 (GGUF IQ2XXS/Q2K + DSpark) | 35B-A3B (Unsloth MTP) |
| Serving | local ds4-server (`:8000`, ctx 8192 for eval) | LM Studio (`:1234`) |
| Date | results_2026-08-12T08-12-26.727181.json | ~2026-06-16 |

**Headline:** DeepSeek-V4-Flash-0731 trails Qwen3.6-35B-A3B-MTP by about **2.2 points** overall (82.55% vs 84.76%) under this harness.

> Methodologies are **not identical** (see [Setup differences](#setup-differences)). Treat domain gaps as directional.

---

## Domain scores

| Domain | DeepSeek V4 Flash | Qwen3.6 MTP | Δ (DS − Qwen) |
|--------|------------------:|------------:|--------------:|
| Social sciences | 89.18% ± 0.55% | 91.55% ± 0.49% | -2.37 pp |
| Other | 85.19% ± 0.61% | 87.22% ± 0.57% | -2.03 pp |
| STEM | 87.35% ± 0.58% | 83.44% ± 0.64% | +3.91 pp |
| Humanities | 73.26% ± 0.61% | 79.57% ± 0.57% | -6.31 pp |
| **Overall** | **82.55% ± 0.30%** | **84.76% ± 0.29%** | **-2.21 pp** |

---

## Setup differences

### DeepSeek-V4-Flash-0731 (this run)

```bash
# Serve (short ctx is enough for MMLU and frees KV for concurrency)
ds4-server --cuda -m $HOME/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-....gguf -c 8192 --port 8000

# Eval
./scripts/run_mmlu_llama_deepseek_v4_flash.sh
nohup ./scripts/mmlu_watchdog_deepseek_v4_flash.sh >/dev/null 2>&1 &
```

Harness notes:

- Task pack `mmlu_llama_ds4`: same prompts/targets as `mmlu_llama`, but **no `gen_prefix` continuation**.
- Stock `mmlu_llama` uses `until: ["."]` after prefacing `The best answer is`; this quantised DeepSeek stack often emits `.` first under continuation, which yields empty answers.
- Instead: full assistant turns, `max_tokens=512`, `temperature=0`, `reasoning_effort=low`, regex extract of `The best answer is [A-D]` or a bare letter reply.
- `num_concurrent=4` against local ds4-server.
- Results JSON: `/home/derek/llm-benchmark/lm-eval/results/deepseek-v4-flash-0731/deepseek-v4-flash/results_2026-08-12T08-12-26.727181.json`

### Qwen3.6-35B-A3B-MTP (Derek’s prior run)

- Thinking disabled via `/no_think` system instruction; LM Studio sampling (`max_tokens=512`).
- Shared: generative MMLU letter answers, chat template, `strict_match` exact match.

---

## Notable weak / strong subjects (DeepSeek)

Lowest:

| Subtask | Score |
|---------|------:|
| professional law | 53.19% |
| virology | 56.02% |
| global facts | 62.00% |
| college chemistry | 64.00% |
| public relations | 71.82% |
| security studies | 73.06% |
| college computer science | 75.00% |
| high school european history | 76.36% |

Highest:

| Subtask | Score |
|---------|------:|
| high school government and politics | 97.41% |
| elementary mathematics | 97.09% |
| high school microeconomics | 95.80% |
| medical genetics | 95.00% |
| high school psychology | 94.50% |
| conceptual physics | 94.47% |
| college biology | 93.75% |
| miscellaneous | 93.74% |

---

## Takeaways

1. **Overall Δ vs Qwen: -2.21 pp** on this generative letter-answer harness.
2. Domain rank order should be compared carefully given the DeepSeek-specific extraction path.
3. Cross-stack caveats: DeepSeek used greedy decode on ds4 GGUF; Qwen used LM Studio sampling with `/no_think`.

*Generated from `results_2026-08-12T08-12-26.727181.json` and Derek Colley’s published Qwen3.6-35B-A3B-MTP mmlu_llama table (2026-06-16).*
