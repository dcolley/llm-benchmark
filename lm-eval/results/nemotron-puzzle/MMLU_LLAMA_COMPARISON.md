# mmlu_llama comparison: Nemotron Puzzle vs Qwen3.6-35B-A3B-MTP

Side-by-side `mmlu_llama` (5-shot, `strict_match` / `exact_match`) on DGX Spark-class local serving.

| | **nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4** | **unsloth/qwen3.6-35b-a3b-mtp** |
|--|--|--|
| **Overall** | **82.42% ± 0.31%** | **84.76% ± 0.29%** |
| Δ vs Qwen | **−2.34 pp** | baseline |
| Samples | 14,042 | 14,042 |
| Params (MoE) | 75B-A9B (NVFP4) | 35B-A3B (Unsloth MTP) |
| Serving | vLLM (`:8000`) | LM Studio (`:1234`) |
| Date | 2026-08-06 | ~2026-06-16 |

**Headline:** Qwen3.6-35B-A3B-MTP leads by about **2.3 points** overall under broadly similar `mmlu_llama` setups. Nemotron Puzzle remains strong on social sciences and several applied subtasks, but trails on humanities (especially professional law) and several STEM / “other” subjects.

> Methodologies are **not identical** (see [Setup differences](#setup-differences)). Treat domain gaps as directional; small subtask deltas under ~3–4 pp with large stderr are often noise.

---

## Domain scores

| Domain | Nemotron Puzzle | Qwen3.6 MTP | Δ (Puzzle − Qwen) |
|--------|----------------:|------------:|------------------:|
| Social sciences | 89.93% ± 0.53% | 91.55% ± 0.49% | −1.62 pp |
| Other | 84.45% ± 0.62% | 87.22% ± 0.57% | −2.77 pp |
| STEM | 81.22% ± 0.68% | 83.44% ± 0.64% | −2.22 pp |
| Humanities | 76.96% ± 0.59% | 79.57% ± 0.57% | −2.61 pp |
| **Overall** | **82.42% ± 0.31%** | **84.76% ± 0.29%** | **−2.34 pp** |

Both models keep the same rank order: social sciences ≫ other ≳ STEM ≫ humanities.

---

## Setup differences

### Nemotron Puzzle (this run)

```bash
lm-eval --model local-chat-completions \
  --model_args "base_url=http://localhost:8000/v1/chat/completions,model=nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4,num_concurrent=1,tokenizer_backend=none,timeout=120,max_retries=5" \
  --tasks mmlu_llama \
  --apply_chat_template \
  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false' \
  --use_cache ./results/nemotron-puzzle/cache.db \
  --log_samples \
  --output_path ./results/nemotron-puzzle
```

- Thinking disabled server-side: `--default-chat-template-kwargs '{"enable_thinking": false}'`
- `mmlu_llama` continuation: assistant prefaced with `The best answer is`, then short letter completion
- Deterministic decode: `temperature=0`, `max_tokens=10`
- Stack: vLLM OpenAI-compatible API, FlashInfer MoE, MTP speculative decode on the serve recipe

Results JSON: `results/nemotron-puzzle/nvidia__NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4/results_2026-08-06T14-23-45.720107.json`

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

Shared: `mmlu_llama`, chat template, multiturn few-shot (defaulted true for Nemotron), `strict_match` exact match.

---

## Where each model wins (largest gaps)

Subtasks where **|Δ| ≥ 4 pp** (Puzzle − Qwen). Positive = Puzzle ahead.

### Nemotron Puzzle ahead

| Subtask | Puzzle | Qwen | Δ |
|---------|-------:|-----:|--:|
| Formal logic | 79.4% | 73.8% | **+5.6** |
| College physics | 73.5% | 69.6% | **+3.9** |
| Elementary mathematics | 83.9% | 81.0% | +2.9 |
| Logical fallacies | 92.0% | 89.6% | +2.5 |
| Marketing | 96.2% | 94.0% | +2.1 |
| College computer science | 80.0% | 78.0% | +2.0 |
| Management | 92.2% | 90.3% | +1.9 |
| High school macroeconomics | 91.3% | 89.5% | +1.8 |
| College biology | 94.4% | 93.1% | +1.4 |

### Qwen ahead (biggest)

| Subtask | Puzzle | Qwen | Δ |
|---------|-------:|-----:|--:|
| Econometrics | 70.2% | 83.3% | **−13.2** |
| Anatomy | 76.3% | 88.2% | **−11.9** |
| High school physics | 74.8% | 84.1% | **−9.3** |
| Conceptual physics | 88.5% | 95.7% | **−7.2** |
| Clinical knowledge | 83.8% | 90.2% | **−6.4** |
| High school computer science | 88.0% | 94.0% | **−6.0** |
| Human sexuality | 89.3% | 94.7% | **−5.3** |
| Professional law | 65.6% | 70.9% | **−5.3** |
| Philosophy | 85.5% | 89.7% | **−4.2** |
| High school european history | 84.2% | 88.5% | **−4.2** |
| Electrical engineering | 82.1% | 86.2% | **−4.1** |
| World religions | 85.4% | 89.5% | **−4.1** |
| College medicine | 80.3% | 84.4% | **−4.0** |

Pattern: Qwen’s lead is broad across domains and especially large on anatomy, econometrics, and several physics / clinical subjects. Puzzle’s wins cluster in formal logic and a few STEM / applied niches, but not enough to offset the aggregate gap.

---

## Shared weak spots (< 75%)

Both models struggle on the same “no-think hurts formal reasoning” set:

| Subtask | Puzzle | Qwen | Notes |
|---------|-------:|-----:|-------|
| Global facts | 54.0% ± 5.0% | 56.0% ± 5.0% | World trivia; both weak |
| Virology | 57.2% ± 3.9% | 59.0% ± 3.8% | Niche medical |
| High school mathematics | 64.4% ± 2.9% | 65.2% ± 2.9% | Tight stderr; real gap |
| Professional law | 65.6% ± 1.2% | 70.9% ± 1.2% | Large-n; Qwen better but both soft |
| College chemistry | 64.0% ± 4.8% | 67.0% ± 4.7% | |
| Abstract algebra | 67.0% ± 4.7% | 70.0% ± 4.6% | |
| College mathematics | 69.0% ± 4.6% | 70.0% ± 4.6% | |
| Moral scenarios | 69.6% ± 1.5% | 69.7% ± 1.5% | Ethics without CoT |
| College physics | 73.5% ± 4.4% | 69.6% ± 4.6% | Puzzle slightly better |

**Puzzle-only extras under 75%:** high school physics (74.8%), econometrics (70.2%), professional accounting (72.0%).

---

## Full subtask table

| Tasks | Puzzle | ± | Qwen | ± | Δ pp |
|-------|-------:|--:|-----:|--:|-----:|
| **mmlu_llama** | **0.8242** | 0.0031 | **0.8476** | 0.0029 | **−2.34** |
| **humanities** | **0.7696** | 0.0059 | **0.7957** | 0.0057 | **−2.61** |
| formal logic | 0.7937 | 0.0362 | 0.7381 | 0.0393 | +5.55 |
| high school european history | 0.8424 | 0.0285 | 0.8848 | 0.0249 | −4.24 |
| high school us history | 0.9265 | 0.0183 | 0.9363 | 0.0171 | −0.98 |
| high school world history | 0.9072 | 0.0189 | 0.9325 | 0.0163 | −2.53 |
| international law | 0.9008 | 0.0273 | 0.9256 | 0.0240 | −2.48 |
| jurisprudence | 0.8889 | 0.0304 | 0.8981 | 0.0292 | −0.92 |
| logical fallacies | 0.9202 | 0.0213 | 0.8957 | 0.0240 | +2.46 |
| moral disputes | 0.8295 | 0.0202 | 0.8584 | 0.0188 | −2.89 |
| moral scenarios | 0.6961 | 0.0154 | 0.6972 | 0.0154 | −0.11 |
| philosophy | 0.8553 | 0.0200 | 0.8971 | 0.0173 | −4.18 |
| prehistory | 0.9074 | 0.0161 | 0.9167 | 0.0154 | −0.93 |
| professional law | 0.6565 | 0.0121 | 0.7093 | 0.0116 | −5.29 |
| world religions | 0.8538 | 0.0271 | 0.8947 | 0.0235 | −4.09 |
| **other** | **0.8445** | 0.0062 | **0.8722** | 0.0057 | **−2.77** |
| business ethics | 0.8300 | 0.0378 | 0.8400 | 0.0368 | −1.00 |
| clinical knowledge | 0.8377 | 0.0227 | 0.9019 | 0.0183 | −6.41 |
| college medicine | 0.8035 | 0.0303 | 0.8439 | 0.0277 | −4.05 |
| global facts | 0.5400 | 0.0501 | 0.5600 | 0.0499 | −2.00 |
| human aging | 0.8386 | 0.0247 | 0.8610 | 0.0232 | −2.24 |
| management | 0.9223 | 0.0265 | 0.9029 | 0.0293 | +1.94 |
| marketing | 0.9615 | 0.0126 | 0.9402 | 0.0155 | +2.14 |
| medical genetics | 0.9500 | 0.0219 | 0.9500 | 0.0219 | 0.00 |
| miscellaneous | 0.9119 | 0.0101 | 0.9502 | 0.0078 | −3.83 |
| nutrition | 0.8725 | 0.0191 | 0.9020 | 0.0170 | −2.94 |
| professional accounting | 0.7199 | 0.0268 | 0.7553 | 0.0256 | −3.55 |
| professional medicine | 0.9007 | 0.0182 | 0.9338 | 0.0151 | −3.31 |
| virology | 0.5723 | 0.0385 | 0.5904 | 0.0383 | −1.81 |
| **social sciences** | **0.8993** | 0.0053 | **0.9155** | 0.0049 | **−1.62** |
| econometrics | 0.7018 | 0.0430 | 0.8333 | 0.0351 | −13.16 |
| high school geography | 0.9343 | 0.0176 | 0.9444 | 0.0163 | −1.01 |
| high school government and politics | 0.9845 | 0.0089 | 0.9845 | 0.0089 | 0.00 |
| high school macroeconomics | 0.9128 | 0.0143 | 0.8949 | 0.0156 | +1.79 |
| high school microeconomics | 0.9454 | 0.0148 | 0.9580 | 0.0130 | −1.26 |
| high school psychology | 0.9523 | 0.0091 | 0.9596 | 0.0084 | −0.73 |
| human sexuality | 0.8931 | 0.0271 | 0.9466 | 0.0197 | −5.34 |
| professional psychology | 0.8693 | 0.0136 | 0.9003 | 0.0121 | −3.10 |
| public relations | 0.7545 | 0.0412 | 0.7818 | 0.0396 | −2.73 |
| security studies | 0.8204 | 0.0246 | 0.8286 | 0.0241 | −0.82 |
| sociology | 0.9303 | 0.0180 | 0.9254 | 0.0186 | +0.50 |
| us foreign policy | 0.9200 | 0.0273 | 0.9500 | 0.0219 | −3.00 |
| **stem** | **0.8122** | 0.0068 | **0.8344** | 0.0064 | **−2.22** |
| abstract algebra | 0.6700 | 0.0473 | 0.7000 | 0.0461 | −3.00 |
| anatomy | 0.7630 | 0.0367 | 0.8815 | 0.0279 | −11.85 |
| astronomy | 0.9342 | 0.0202 | 0.9276 | 0.0211 | +0.66 |
| college biology | 0.9444 | 0.0192 | 0.9306 | 0.0213 | +1.39 |
| college chemistry | 0.6400 | 0.0482 | 0.6700 | 0.0473 | −3.00 |
| college computer science | 0.8000 | 0.0402 | 0.7800 | 0.0416 | +2.00 |
| college mathematics | 0.6900 | 0.0465 | 0.7000 | 0.0461 | −1.00 |
| college physics | 0.7353 | 0.0439 | 0.6961 | 0.0458 | +3.92 |
| computer security | 0.8600 | 0.0349 | 0.8900 | 0.0314 | −3.00 |
| conceptual physics | 0.8851 | 0.0208 | 0.9574 | 0.0132 | −7.23 |
| electrical engineering | 0.8207 | 0.0320 | 0.8621 | 0.0287 | −4.14 |
| elementary mathematics | 0.8386 | 0.0189 | 0.8095 | 0.0202 | +2.91 |
| high school biology | 0.9484 | 0.0126 | 0.9548 | 0.0118 | −0.64 |
| high school chemistry | 0.8030 | 0.0280 | 0.8374 | 0.0260 | −3.45 |
| high school computer science | 0.8800 | 0.0327 | 0.9400 | 0.0239 | −6.00 |
| high school mathematics | 0.6444 | 0.0292 | 0.6519 | 0.0290 | −0.74 |
| high school physics | 0.7483 | 0.0354 | 0.8411 | 0.0299 | −9.27 |
| high school statistics | 0.8102 | 0.0267 | 0.8472 | 0.0245 | −3.70 |
| machine learning | 0.7857 | 0.0389 | 0.8036 | 0.0377 | −1.79 |

---

## Interpretation

1. **Qwen wins the head-to-head** on this harness: ~85% vs ~82% overall, with a consistent ~1.5–2.8 pp edge in every domain bucket.
2. **Size ≠ automatic win on MMLU-letter.** Puzzle is a larger MoE (75B-A9B NVFP4) with thinking off and very short greedy continuations; Qwen’s Unsloth MTP quant under LM Studio with sampling still scores higher here.
3. **Shared failure modes match the earlier Qwen write-up:** global facts, virology, moral scenarios, professional law, and math/chem/physics college items — i.e. multi-step / niche knowledge without chain-of-thought.
4. **Puzzle’s distinctive holes** vs Qwen: econometrics (−13 pp), anatomy (−12 pp), high-school / conceptual physics (−7 to −9 pp). Those alone explain a large share of the STEM / social / other gaps.
5. **Caveat on decode settings:** Puzzle used greedy `max_tokens=10` + message continuation; Qwen used `temperature=0.7` / `max_tokens=512`. A re-run of Puzzle with closer sampling (or thinking-on for hard STEM/law) would be needed before calling the gap structural rather than harness/settings.

---

## Suggested follow-ups

- Re-run Puzzle with closer gen settings to Qwen (`temperature` / larger `max_tokens`) for a fairer API-layer comparison
- Enable thinking on Puzzle for STEM + professional law only (expect uplift on the shared weak set)
- `--tasks gsm8k`, `arc_challenge`, `mmlu_pro` for a fuller profile
- Optional: compare both against prior `nvidia-nemotron-3-super` results under the same `mmlu_llama` recipe

---

*Generated from Puzzle results `results_2026-08-06T14-23-45.720107.json` and Derek Colley’s published Qwen3.6-35B-A3B-MTP mmlu_llama table (2026-06-16).*
