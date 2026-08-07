# Ling-3.0-flash INT4 on DGX Spark (SGLang)

Serve [`inclusionAI/Ling-3.0-flash-int4`](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4) (~72 GB) with the **official SGLang fork**.

## vLLM or SGLang?

| | SGLang | vLLM |
|---|---|---|
| Official INT4 card | **Documented** (`sglang_ling_v3` / `ling_v3_support`) | Not on the INT4 card |
| Proven long-context INT4 | 4×3090 TP4, 262K (community) | — |
| Our Spark FP4 attempts | — | OOM’d at load (~82→121 GiB) |

**Use SGLang for INT4.** Keep the Ling vLLM fork for FP4/FP8 experiments only.

## Install

The INT4 card’s `git clone …/sglang_ling_v3` URL currently **404s** publicly. Prefer the published Ling runtime image (CUDA 13 for Spark):

```bash
docker pull lmsysorg/sglang:dev-cu13-Ling-3.0-flash
```

Or try upstream/org main if you have access to a private fork:

```bash
git clone https://github.com/inclusionAI/sglang.git ~/src/sglang
# (ling_v3_support branch is not published on this repo as of Aug 2026)
```

Host venv (only if the image lacks something you need):

```bash
uv venv ~/my_ling_sglang_env
source ~/my_ling_sglang_env/bin/activate
# install from a Ling-capable SGLang tree when available
```

## Download

```bash
hf download inclusionAI/Ling-3.0-flash-int4   # ~72 GiB
```

## Run (everyday Spark)

**Docker (recommended):**

```bash
cd ~/sglang/recipes/ling-3.0-flash-int4
./start.sh
```

**Host** (once a Ling SGLang install exists):

```bash
./start-host.sh
# or: ./start.sh host
```

Defaults:

| Flag | Value | Why |
|---|---|---|
| `--tp-size` | `1` | Single GB10 |
| `--mem-fraction-static` | `0.80` | Needed for NEXTN (0.70 left no KV room after draft) |
| `--context-length` | `262144` | Model native max |
| `--max-running-requests` | `4` | Concurrency capped for NEXTN on Spark |
| `--max-mamba-cache-size` | `64` | Caps hybrid state so 262K can fit |
| `--chunked-prefill-size` | `8192` | Stable long prefill |
| Thinking | **on** (model default) | Everyday chat / reasoning |
| NEXTN / MTP | **on** (`--speculative-algorithm NEXTN`) | Built-in MTP draft (~3.5 GiB); lower latency |
| Prefix cache | radix on + `--enable-cache-report` | `usage.prompt_tokens_details.cached_tokens` |
| `restart` | `unless-stopped` | Stay up across reboots/SSH drops |

Port **8000**. Stop other servers first.

Pass `"chat_template_kwargs": {"enable_thinking": false}` per request when you need non-thinking answers (e.g. MMLU).

### Prefix cache

Radix (prefix) cache is **on by default** — do not pass `--disable-radix-cache`. `--enable-cache-report` exposes hits in OpenAI-compatible usage:

```json
"usage": {
  "prompt_tokens": 96046,
  "completion_tokens": 50,
  "prompt_tokens_details": { "cached_tokens": 95040 }
}
```

Keep system prompts / tool schemas byte-stable; put dynamic text after the stable prefix. Warm once, then measure the second identical-prefix call (first is a miss/write).

### Bring-up status (this Spark)

**Works without NEXTN** (`lmsysorg/sglang:dev-cu13-Ling-3.0-flash`, INT4, tp=1). Weight load ~70 GiB.

**NEXTN + `mem-fraction-static=0.80` works** (target ~70 GiB + draft ~3.5 GiB; `max_model_len=262144`). `0.70` failed KV init.

`mmlu_llama` (2026-08-07, thinking off, no NEXTN): **83.61%** — see [`~/llm-benchmark/lm-eval/results/ling-3.0-flash-int4/MMLU_LLAMA_REPORT.md`](/home/derek/llm-benchmark/lm-eval/results/ling-3.0-flash-int4/MMLU_LLAMA_REPORT.md).

### If NEXTN OOMs

Lower `--max-running-requests` or `--context-length`, or raise `mem-fraction-static` further (card recipes use ~0.80–0.85).

### Optional host sysctl (helps load on Spark)

```bash
sudo sysctl -w vm.min_free_kbytes=5242880
```

## Client

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ling-3.0-flash-int4",
    "messages": [{"role": "user", "content": "hello!"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```
