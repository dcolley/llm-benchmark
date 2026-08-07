# mmlu_llama: Ling-3.0-flash-int4 (SGLang on DGX Spark)

| Ling-3.0-flash-int4 | Nemotron Puzzle | Qwen3.6-35B-A3B-MTP |
|--------------------:|----------------:|--------------------:|
| **83.61%** | 82.42% | 84.76% |

`mmlu_llama` (5-shot, `strict_match` / `exact_match`) against local OpenAI-compatible SGLang.

| | **ling-3.0-flash-int4** |
|--|--|
| **Overall** | **83.61% ± 0.30%** |
| Δ vs Qwen3.6 MTP | **-1.15 pp** |
| Δ vs Nemotron Puzzle | **+1.19 pp** |
| Samples | 14,042 |
| Model | `inclusionAI/Ling-3.0-flash-int4` (served as `ling-3.0-flash-int4`) |
| Serving | SGLang `lmsysorg/sglang:dev-cu13-Ling-3.0-flash` on DGX Spark (`:8000`) |
| Date | 2026-08-07 |

**Headline:** Ling INT4 lands between Nemotron Puzzle and Qwen3.6-35B-A3B-MTP on this harness (~83.6%).

> Cross-run methodologies are not identical (see [Setup](#setup)). Treat small gaps (~1 pp) as directional.

---

## Domain scores

| Domain | Ling INT4 | Nemotron* | Qwen MTP* |
|--------|----------:|----------:|----------:|
| Social sciences | 90.67% ± 0.52% | — | 91.55% |
| Other | 86.55% ± 0.58% | — | 87.22% |
| STEM | 85.44% ± 0.62% | — | 83.44% |
| Humanities | 75.83% ± 0.60% | — | 79.57% |
| **Overall** | **83.61% ± 0.30%** | **82.42%** | **84.76%** |

\*Prior house runs; domain cells for Nemotron omitted here (see their reports). Ling rank order: social sciences ≫ other ≳ STEM ≫ humanities (same pattern as Qwen).

---

## Setup

```bash
# Server (thinking off for letter answers)
cd ~/sglang/recipes/ling-3.0-flash-int4 && ./start.sh
# includes --default-chat-template-kwargs '{"enable_thinking":false}'

# Eval
./scripts/run_mmlu_llama_ling_int4.sh          # full, background
./scripts/run_mmlu_llama_ling_int4.sh --limit 5 --foreground  # smoke
nohup ./scripts/mmlu_watchdog_ling_int4.sh >/dev/null 2>&1 &
```

Harness:

```bash
lm-eval --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:8000/v1/chat/completions,model=ling-3.0-flash-int4,num_concurrent=1,tokenizer_backend=none,timeout=300,max_retries=5" \
  --tasks mmlu_llama --apply_chat_template \
  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false' \
  --use_cache ./results/ling-3.0-flash-int4/cache.db \
  --log_samples --output_path ./results/ling-3.0-flash-int4
```

- Thinking disabled server-side: `--default-chat-template-kwargs '{"enable_thinking":false}'`
- `mmlu_llama` continuation: assistant prefaced with `The best answer is`, then short letter completion
- Deterministic decode: `temperature=0`, `max_tokens=10`
- Serve recipe: tp=1, `mem-fraction-static=0.70`, `context-length=8192`, `max-running-requests=1`, `max-mamba-cache-size=64`, no NEXTN
- Stack: SGLang OpenAI API on GB10 (unified memory)

Results JSON: `results/ling-3.0-flash-int4/ling-3.0-flash-int4/results_2026-08-07T12-42-25.506701.json`

Smoke (`--limit 5` per subject, 285 requests): 87.72% ± 1.87% — not comparable to the full run.

---

## Notable weak / strong subjects (Ling INT4)

Lowest (full run):

| Subtask | Score |
|---------|------:|
| Virology | 56.0% |
| Moral scenarios | 60.4% |
| Global facts | 65.0% |
| College chemistry | 66.0% |
| Professional law | 66.0% |

Highest:

| Subtask | Score |
|---------|------:|
| Medical genetics | 96.0% |
| Marketing | 95.7% |
| Astronomy | 94.7% |
| Sociology | 94.0% |
| Professional medicine | 93.4% |
| Miscellaneous | 93.4% |
