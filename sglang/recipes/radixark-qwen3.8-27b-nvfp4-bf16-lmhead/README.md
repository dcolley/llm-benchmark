# Qwen3.8-27B NVFP4 (BF16 lm_head) on DGX Spark

Serve [`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) with **SGLang** on GB10. Port **8000**.

This is the [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) W4A4 checkpoint with **`lm_head` left in BF16** (not NVFP4). MLP is NVFP4, attention weights FP8, MTP + vision stay BF16. Official runtime is SGLang ([cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)); the card’s GB300 sample is TP4 + NEXTN.

On Spark we run **TP=1**, `--mem-fraction-static 0.80` (cookbook: 0.85 trips DGX OS earlyoom), FlashInfer attention, and in-checkpoint **MTP / EAGLE 3/1/4**.

## Defaults

| Knob | Value | Why |
|------|--------|-----|
| Image | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | Cookbook Qwen3.8 image |
| Quant | NVFP4 W4A4, BF16 `lm_head` | Same as the HF repo you asked for |
| KV | `auto` (FP8 from checkpoint) | Card sets `kv_cache_quant_algo: FP8` |
| Speculation | MTP/EAGLE 3/1/4 | In-checkpoint MTP; no extra draft download |
| Context | **32 768** | Daily serve; `PROFILE=long` for 262K |
| Mem fraction | **0.80** | Spark UMA + earlyoom |
| Prefill chunk | **2048** | Cookbook hybrid-GDN decode latency |
| Attention | `flashinfer` | SM121 (not `trtllm_mha`) |

**Optional:** `PROFILE=dspark` adds [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark). Field reports on Spark: ~34–38 tok/s with DSpark vs ~22 tok/s without a tuned draft ([forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)).

**Stop other `:8000` servers first** (Ornith, FP8 Qwen3.8, r0b0tlab NVFP4, Ling, …).

## Download

```bash
hf download RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
# optional:
# hf download RadixArk/Qwen3.8-27B-DSpark
```

Or let the container pull into `~/.cache/huggingface` on first serve.

## Run (Docker)

```bash
cd ~/llm-benchmark/sglang/recipes/radixark-qwen3.8-27b-nvfp4-bf16-lmhead
./start-docker.sh
# PROFILE=long ./start-docker.sh
# PROFILE=dspark ./start-docker.sh
./stop.sh
```

| Profile | Serve |
|---------|--------|
| `production` (default) | 32K, MTP/EAGLE 3/1/4 |
| `long` | 262K, MTP |
| `dspark` | 32K, DSpark draft |

Served name: `qwen3.8-27b-nvfp4`.

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-nvfp4",
    "messages": [{"role": "user", "content": "What is 19 times 23? Reply with the number only."}],
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq -r '.choices[0].message.content'
```

Thinking is on by default; pass `enable_thinking: false` for short MMLU-style answers.

## Notes

- Weights ~16–18 GB NVFP4 vs ~31 GB FP8. Fits one Spark with KV headroom.
- Card eval (GB300 TP4, different stack): GSM8K **96.13%**, Terminal-Bench 2.1 **69.4%**. Not a Spark number.
- Compared with [`r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121`](../../vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121/): that one is a Spark-tuned vLLM SM121 image + their MTP quant. This recipe is RadixArk’s ModelOpt NVFP4 + unquantized `lm_head` on SGLang.
