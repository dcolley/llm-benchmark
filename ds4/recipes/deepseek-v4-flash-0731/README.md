# DeepSeek-V4-Flash-0731 on DGX Spark (single node)

Serve [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) on **one** GB10 Spark.

## Can I run this on one server?

**Yes — but not the official Hugging Face safetensors as-is.**

| Artifact | Approx size | Fits one Spark (~128 GB UMA)? |
|---|---|---|
| Official HF checkpoint (MXFP4 experts + DSpark head) | ~155–167 GB weights | **No** |
| Unsloth UD-Q4_K_XL GGUF | ~155 GB | **No** (needs ~2 Sparks / ≥160 GB) |
| Unsloth UD-IQ3_XXS | ~104 GB | Tight / risky after OS + KV |
| **antirez IQ2XXS/Q2K hybrid GGUF (recommended)** | **~81 GiB** + **~7 GiB** DSpark drafter | **Yes** with headroom |
| UD-IQ1_S floor | ~82.5 GB | Yes, lower quality |

This box has **~121 GiB** total unified memory (~119 GiB usable after OS). MoE experts must stay resident; you cannot “layer-offload” like a dense model.

**Official card recipes** ([vLLM](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-with-vllm) / [SGLang](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-with-sglang)) target **4×GB300**-class nodes with DSpark (`method: dspark` / `--speculative-algorithm DSPARK`). They are the right path for full-fidelity FP8 weights — **not** this single Spark.

## RAM budget (this Spark, single-stream)

| Piece | GiB (approx) |
|---|---:|
| IQ2XXS/Q2K target GGUF | ~81 |
| DSpark Q2K drafter | ~7 |
| KV + activations @ 131K ctx | ~10–20 |
| OS / desktop / headroom | ~15–25 |
| **Practical ceiling** | **≤ ~119 usable** |

Community notes: at **262K** context the memory floor can reject batches / 503; **131K–196K** is the safer everyday window on one Spark ([NVIDIA forum](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855)).

Stop other GPU servers first (Muse / Ling / Nemotron) — they will OOM the box.

## Recommended runtime: ds4-on-spark

Use [`Entrpi/ds4-on-spark`](https://github.com/entrpi/ds4-on-spark) (Spark-tuned fork of [`antirez/ds4`](https://github.com/antirez/ds4)): native C/CUDA, continuous batching, prefix cache, DSpark, OpenAI-compatible server.

### One-shot install + start

```bash
# Stop whatever owns :8000 / UMA first, e.g.:
#   pkill -f 'vllm serve' || true
#   cd ~/sglang/recipes/ling-3.0-flash-int4 && docker compose stop

curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh \
  | bash -s -- --start --port 8000 --ctx 131072 --keep-old-weights --with-dspark

# Optional: re-bind for LAN access (installer defaults to 127.0.0.1)
# ds4-serve --host 0.0.0.0 --port 8000 -c 131072
```

That downloads:

- Target: [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) (~81 GiB IQ2XXS/Q2K 0731)
- Drafter: [`bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF) (~7 GiB)

### Or use this recipe’s wrapper

```bash
cd ~/llm-benchmark/ds4/recipes/deepseek-v4-flash-0731
./start.sh                 # installs if needed, then serves :8000
./start.sh --install-only  # download/build only
./start.sh --no-install    # assume ds4 already installed
```

Defaults (overridable via env):

| Env / flag | Default | Notes |
|---|---|---|
| `PORT` | `8000` | OpenAI-compatible |
| `CONTEXT` | `131072` | Raise toward `196608` only after a clean boot |
| `HOST` | `0.0.0.0` | Installer often defaults to localhost |
| `DS4_CONT_DSPARK` | `1` | Enable DSpark speculation |
| `GGUF_DIR` | `$HOME/gguf` | Weight + drafter location |

### Client

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"Say hi in one sentence."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 256
  }'
```

Card sampling guidance: `temperature=1.0`; `top_p=0.95` for agentic, `top_p=1.0` otherwise. Reasoning effort: `low` / `high` / `max` (via encoding / server API — see HF [chat template](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#chat-template) notes).

## Official multi-GPU (reference only)

If you later have ≥4 large Blackwell GPUs or a second Spark for Q4:

```bash
# vLLM (card example — not for 1× Spark)
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

```bash
# SGLang (card example — not for 1× Spark)
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --mem-fraction-static 0.90
```

## Links

- Model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Spark installer: https://github.com/entrpi/ds4-on-spark
- Single-Spark throughput thread: https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855
- Hardware sizing: https://kingy.ai/ai/ai-guides/deepseek-v4-flash-0731-local-hardware-guide/
