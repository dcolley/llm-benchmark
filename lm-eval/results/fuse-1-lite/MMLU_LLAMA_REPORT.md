# mmlu_llama comparison: Fuse-1-Lite vs Qwen3.6-35B-A3B-MTP

| Fuse-1-Lite | Qwen3.6-35B-A3B-MTP | Nemotron Puzzle (context) |
|------------:|--------------------:|--------------------------:|
| **60.84%** | **84.76%** | 82.42% |

Side-by-side `mmlu_llama` (5-shot, `strict_match` / `exact_match`) on local / remote OpenAI-compatible serving.

| | **Akahsizrr/fuse-1-Lite** | **unsloth/qwen3.6-35b-a3b-mtp** |
|--|--|--|
| **Overall** | **60.84% ± 0.40%** | **84.76% ± 0.29%** |
| Δ vs Qwen | **-23.92 pp** | baseline |
| Samples | 14,042 | 14,042 |
| Model | Fuse-1-Lite | 35B-A3B (Unsloth MTP) |
| Serving | remote vLLM (`192.168.10.199:8000`) | LM Studio (`:1234`) |
| Date | 2026-08-07 | ~2026-06-16 |

**Headline:** Qwen3.6-35B-A3B-MTP leads by about **23.9 points** overall (84.76% vs 60.84%). Fuse-1-Lite trails across every domain bucket under this harness.

> Methodologies are **not identical** (see [Setup differences](#setup-differences)). Treat domain gaps as directional; small subtask deltas under ~3–4 pp with large stderr are often noise.

---

## Domain scores

| Domain | Fuse-1-Lite | Qwen3.6 MTP | Δ (Fuse − Qwen) |
|--------|------------:|------------:|----------------:|
| Social sciences | 70.13% ± 0.81% | 91.55% ± 0.49% | -21.42 pp |
| Other | 63.50% ± 0.84% | 87.22% ± 0.57% | -23.72 pp |
| STEM | 58.26% ± 0.86% | 83.44% ± 0.64% | -25.18 pp |
| Humanities | 54.73% ± 0.70% | 79.57% ± 0.57% | -24.84 pp |
| **Overall** | **60.84% ± 0.40%** | **84.76% ± 0.29%** | **-23.92 pp** |

Same rank order on both models: social sciences ≫ other ≳ STEM ≫ humanities — but Fuse sits ~20–24 pp lower in each bucket.

---

## Setup differences

### Fuse-1-Lite (this run)

```bash
lm-eval --model local-chat-completions \
  --model_args "base_url=http://192.168.10.199:8000/v1/chat/completions,model=Akahsizrr/fuse-1-Lite,..." \
  --tasks mmlu_llama --apply_chat_template \
  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false'
```

- Served on remote vLLM at `192.168.10.199:8000`, `max_model_len` 4096
- `mmlu_llama` continuation: assistant prefaced with `The best answer is`, then short letter completion
- Deterministic decode: `temperature=0`, `max_tokens=10`
- Date: 2026-08-07

Results JSON: `results/fuse-1-lite/Akahsizrr__fuse-1-Lite/results_2026-08-07T03-02-31.412326.json`

### Qwen3.6-35B-A3B-MTP (Derek’s prior run)

```bash
lm-eval --model local-chat-completions \
  --model_args "base_url=http://localhost:1234/v1/chat/completions,model=qwen3.6-35b-a3b-mtp" \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --system_instruction "You are a helpful assistant. /no_think" \
  --gen_kwargs 'max_tokens=512,temperature=0.7,top_p=0.8,top_k=20,min_p=0.0' \
  --log_samples \
  --output_path ./results/debug-llama-nothink
```

- Thinking disabled via `/no_think` system instruction
- Sampling decode with larger `max_tokens`
- Stack: LM Studio API

Shared: `mmlu_llama`, chat template, `strict_match` exact match.

---

## Where each model wins (largest gaps)

Subtasks where **|Δ| ≥ 4 pp** (Fuse − Qwen). Positive = Fuse ahead.

### Fuse-1-Lite ahead

No subtasks with Fuse ahead by ≥ 4 pp (Fuse does not lead any subtask by that margin).

### Qwen ahead (biggest of 57 with |Δ| ≥ 4 pp)

| Subtask | Fuse | Qwen | Δ |
|---------|-----:|-----:|--:|
| Professional medicine | 54.8% | 93.4% | **-38.6** |
| High school physics | 45.7% | 84.1% | **-38.4** |
| Econometrics | 50.0% | 83.3% | **-33.3** |
| Professional accounting | 42.9% | 75.5% | **-32.6** |
| Formal logic | 41.3% | 73.8% | **-32.5** |
| Anatomy | 57.8% | 88.1% | **-30.4** |
| Professional law | 41.5% | 70.9% | **-29.5** |
| Conceptual physics | 66.4% | 95.7% | **-29.4** |
| Professional psychology | 61.6% | 90.0% | **-28.4** |
| High school statistics | 56.5% | 84.7% | **-28.2** |
| Abstract algebra | 42.0% | 70.0% | **-28.0** |
| Elementary mathematics | 53.2% | 81.0% | **-27.8** |
| Machine learning | 52.7% | 80.4% | **-27.7** |
| Medical genetics | 68.0% | 95.0% | **-27.0** |
| Human sexuality | 67.9% | 94.7% | **-26.7** |

Pattern: Qwen’s lead is near-universal — **57 / 57** subtasks trail by ≥ 4 pp. Fuse’s worst holes include professional law, moral scenarios, anatomy, and several high-school STEM / social subjects. Closest subtasks still favor Qwen (see full table).

---

## Shared weak spots (< 75%)

Both models score under 75% on:

| Subtask | Fuse | Qwen | Notes |
|---------|-----:|-----:|-------|
| Global facts | 39.0% ± 4.9% | 56.0% | World trivia; both weak |
| Formal logic | 41.3% ± 4.4% | 73.8% | Fuse far below |
| Professional law | 41.5% ± 1.3% | 70.9% | Large-n; Qwen better but both soft |
| Abstract algebra | 42.0% ± 5.0% | 70.0% |  |
| High school mathematics | 42.6% ± 3.0% | 65.2% | Formal math without CoT |
| College physics | 44.1% ± 4.9% | 69.6% | Qwen also soft; Fuse much weaker |
| Moral scenarios | 44.2% ± 1.7% | 69.7% | Ethics without CoT |
| College chemistry | 46.0% ± 5.0% | 67.0% |  |
| Virology | 46.4% ± 3.9% | 59.0% | Niche medical |
| College mathematics | 47.0% ± 5.0% | 70.0% |  |

**Fuse-only extras under 75% (Qwen ≥ 75%):** 40 subtasks — e.g. Professional accounting (42.9%), High school physics (45.7%), Econometrics (50.0%), Machine learning (52.7%), Elementary mathematics (53.2%), College computer science (54.0%), Professional medicine (54.8%), High school statistics (56.5%), Anatomy (57.8%), Business ethics (59.0%), High school chemistry (59.6%), College medicine (61.3%) (+28 more).

---

## Full subtask table

| Tasks | Fuse | ± | Qwen | ± | Δ pp |
|-------|-----:|--:|-----:|--:|-----:|
| **mmlu_llama** | **0.6084** | 0.0040 | **0.8476** | 0.0029 | **-23.92** |
| **humanities** | **0.5473** | 0.0070 | **0.7957** | 0.0057 | **-24.84** |
| formal logic | 0.4127 | 0.0440 | 0.7381 | 0.0393 | -32.54 |
| high school european history | 0.7455 | 0.0340 | 0.8848 | 0.0249 | -13.93 |
| high school us history | 0.7451 | 0.0306 | 0.9363 | 0.0171 | -19.12 |
| high school world history | 0.7468 | 0.0283 | 0.9325 | 0.0163 | -18.57 |
| international law | 0.7190 | 0.0410 | 0.9256 | 0.0240 | -20.66 |
| jurisprudence | 0.7037 | 0.0441 | 0.8981 | 0.0292 | -19.44 |
| logical fallacies | 0.6687 | 0.0370 | 0.8957 | 0.0240 | -22.70 |
| moral disputes | 0.6185 | 0.0262 | 0.8584 | 0.0188 | -23.99 |
| moral scenarios | 0.4425 | 0.0166 | 0.6972 | 0.0154 | -25.47 |
| philosophy | 0.6688 | 0.0267 | 0.8971 | 0.0173 | -22.83 |
| prehistory | 0.6512 | 0.0265 | 0.9167 | 0.0154 | -26.55 |
| professional law | 0.4146 | 0.0126 | 0.7093 | 0.0116 | -29.47 |
| world religions | 0.7836 | 0.0316 | 0.8947 | 0.0235 | -11.11 |
| **other** | **0.6350** | 0.0084 | **0.8722** | 0.0057 | **-23.72** |
| business ethics | 0.5900 | 0.0494 | 0.8400 | 0.0368 | -25.00 |
| clinical knowledge | 0.6453 | 0.0294 | 0.9019 | 0.0183 | -25.66 |
| college medicine | 0.6127 | 0.0371 | 0.8439 | 0.0277 | -23.12 |
| global facts | 0.3900 | 0.0490 | 0.5600 | 0.0499 | -17.00 |
| human aging | 0.6592 | 0.0318 | 0.8610 | 0.0232 | -20.18 |
| management | 0.7961 | 0.0399 | 0.9029 | 0.0293 | -10.68 |
| marketing | 0.7735 | 0.0274 | 0.9402 | 0.0155 | -16.67 |
| medical genetics | 0.6800 | 0.0469 | 0.9500 | 0.0219 | -27.00 |
| miscellaneous | 0.7203 | 0.0161 | 0.9502 | 0.0078 | -22.99 |
| nutrition | 0.6830 | 0.0266 | 0.9020 | 0.0170 | -21.90 |
| professional accounting | 0.4291 | 0.0295 | 0.7553 | 0.0256 | -32.62 |
| professional medicine | 0.5478 | 0.0302 | 0.9338 | 0.0151 | -38.60 |
| virology | 0.4639 | 0.0388 | 0.5904 | 0.0383 | -12.65 |
| **social sciences** | **0.7013** | 0.0081 | **0.9155** | 0.0049 | **-21.42** |
| econometrics | 0.5000 | 0.0470 | 0.8333 | 0.0351 | -33.33 |
| high school geography | 0.7929 | 0.0289 | 0.9444 | 0.0163 | -15.15 |
| high school government and politics | 0.8135 | 0.0281 | 0.9845 | 0.0089 | -17.10 |
| high school macroeconomics | 0.6462 | 0.0242 | 0.8949 | 0.0156 | -24.87 |
| high school microeconomics | 0.7059 | 0.0296 | 0.9580 | 0.0130 | -25.21 |
| high school psychology | 0.7780 | 0.0178 | 0.9596 | 0.0084 | -18.16 |
| human sexuality | 0.6794 | 0.0409 | 0.9466 | 0.0197 | -26.72 |
| professional psychology | 0.6160 | 0.0197 | 0.9003 | 0.0121 | -28.43 |
| public relations | 0.6182 | 0.0465 | 0.7818 | 0.0396 | -16.36 |
| security studies | 0.7347 | 0.0283 | 0.8286 | 0.0241 | -9.39 |
| sociology | 0.7463 | 0.0308 | 0.9254 | 0.0186 | -17.91 |
| us foreign policy | 0.7900 | 0.0409 | 0.9500 | 0.0219 | -16.00 |
| **stem** | **0.5826** | 0.0086 | **0.8344** | 0.0064 | **-25.18** |
| abstract algebra | 0.4200 | 0.0496 | 0.7000 | 0.0461 | -28.00 |
| anatomy | 0.5778 | 0.0427 | 0.8815 | 0.0279 | -30.37 |
| astronomy | 0.7434 | 0.0355 | 0.9276 | 0.0211 | -18.42 |
| college biology | 0.7361 | 0.0369 | 0.9306 | 0.0213 | -19.45 |
| college chemistry | 0.4600 | 0.0501 | 0.6700 | 0.0473 | -21.00 |
| college computer science | 0.5400 | 0.0501 | 0.7800 | 0.0416 | -24.00 |
| college mathematics | 0.4700 | 0.0502 | 0.7000 | 0.0461 | -23.00 |
| college physics | 0.4412 | 0.0494 | 0.6961 | 0.0458 | -25.49 |
| computer security | 0.7100 | 0.0456 | 0.8900 | 0.0314 | -18.00 |
| conceptual physics | 0.6638 | 0.0309 | 0.9574 | 0.0132 | -29.36 |
| electrical engineering | 0.6345 | 0.0401 | 0.8621 | 0.0287 | -22.76 |
| elementary mathematics | 0.5317 | 0.0257 | 0.8095 | 0.0202 | -27.78 |
| high school biology | 0.7323 | 0.0252 | 0.9548 | 0.0118 | -22.25 |
| high school chemistry | 0.5961 | 0.0345 | 0.8374 | 0.0260 | -24.13 |
| high school computer science | 0.7300 | 0.0446 | 0.9400 | 0.0239 | -21.00 |
| high school mathematics | 0.4259 | 0.0301 | 0.6519 | 0.0290 | -22.60 |
| high school physics | 0.4570 | 0.0407 | 0.8411 | 0.0299 | -38.41 |
| high school statistics | 0.5648 | 0.0338 | 0.8472 | 0.0245 | -28.24 |
| machine learning | 0.5268 | 0.0474 | 0.8036 | 0.0377 | -27.68 |

---

## Interpretation

1. **Qwen wins the head-to-head by a wide margin** on this harness: ~84.76% vs ~60.84% overall (**-23.92 pp**), with ~20–24 pp deficits for Fuse in every domain.
2. **No domain is competitive.** Social sciences is Fuse’s best bucket (~70%) and still trails Qwen (~92%) by more than 20 points.
3. **Shared weak spots still look like “no-think / short-answer” pain:** global facts, virology, moral scenarios, professional law, and several math/chem items — but Fuse is weaker still on almost all of them.
4. **Largest individual holes** vs Qwen include professional law, moral scenarios, anatomy, and multiple STEM/high-school subjects with |Δ| well above 4 pp (see gaps section).
5. **Context:** Nemotron Puzzle (~82.4%) sits between Fuse and Qwen on the same task family; Fuse (~60.8%) is a distinctly lower tier under this `mmlu_llama` continuation setup.
6. **Caveat on decode settings:** Fuse used greedy `max_tokens=10` + message continuation on remote vLLM; Qwen used `temperature=0.7` / `max_tokens=512` on LM Studio. Harness differences may inflate the gap, but the magnitude (~24 pp) is unlikely to reverse with modest gen tweaks alone.

---

## Suggested follow-ups

- Spot-check Fuse sample logs for empty / malformed letter completions (continuation / chat-template issues)
- Re-run Fuse with closer gen settings to Qwen (`temperature` / larger `max_tokens`) if letter extraction looks broken
- Compare against Nemotron Puzzle (82.42%) under the same Fuse-style greedy continuation recipe
- `--tasks gsm8k`, `arc_challenge`, `mmlu_pro` for a fuller Fuse profile
- Confirm remote vLLM chat template / `continue_final_message` behavior matches the local Puzzle run

---

*Generated from Fuse results `results_2026-08-07T03-02-31.412326.json` and Derek Colley’s published Qwen3.6-35B-A3B-MTP mmlu_llama table (2026-06-16).*
