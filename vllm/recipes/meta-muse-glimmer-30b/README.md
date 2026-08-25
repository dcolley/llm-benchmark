# Muse Glimmer-30B on DGX Spark

Serve [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) (BF16 multimodal, ~60 GB) with vLLM on the host. Port **8000**.

## Hardware fit (this box)

| | |
|---|---|
| GPU | NVIDIA GB10 (compute capability **12.1** / `sm_121`) |
| Memory | ~128 GB unified (CPU+GPU) |
| Weights | BF16 shards ≈ **59.5 GB** + ViT-G/14 perception encoder in-checkpoint |
| Context | 131,072 (model card); most layers are sliding-window 2048 |

BF16 fits a single Spark with headroom for KV if nothing else is pinning UMA. Stop LM Studio / other GPU servers first (`nvidia-smi` should show near-idle compute apps).

## Why not stock `~/vllm/.venv` (0.23.0)?

`MuseGlimmerForConditionalGeneration` is **not** in released vLLM yet. Native support (model + `muse_glimmer` reasoning/tool parsers + DFlash draft head) lands in [vllm-project/vllm#51655](https://github.com/vllm-project/vllm/pull/51655). This recipe uses a **dedicated** venv built from that branch until a release includes it.

## Venv

See **[readme-venv.md](./readme-venv.md)**.

## Download weights

Needs ~60 GB free under the Hugging Face cache:

```bash
hf download meta-models/Muse-Glimmer-30B
# optional DFlash drafter (~5 GB):
hf download meta-models/Muse-Glimmer-30B-assistant
```

Or let vLLM pull on first serve.

## Run

```bash
cd ~/llm-benchmark/vllm/recipes/meta-muse-glimmer-30b
./start.sh
```

Defaults in the script:

| Flag / env | Default | Notes |
|---|---|---|
| `CUTE_DSL_ARCH` | `sm_121a` | GB10 cute-DSL kernels |
| `--attention-backend` | `TRITON_ATTN` | Reliable on Spark (same as DiffusionGemma recipe) |
| `--gpu-memory-utilization` | `0.70` | UMA-safe budget after ~60 GB weights |
| `--max-model-len` | `131072` | Full card context |
| `--max-num-seqs` | `2` | Keep KV modest on UMA |
| `--reasoning-parser` / `--tool-call-parser` | `muse_glimmer` | Must be used **together** (channel-scoped framing) |

Optional overrides: `MODEL`, `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `ATTENTION_BACKEND`, `VLLM_VENV`, `ENABLE_DFLASH=1`.

### Spark / sm_121 notes

- Set `CUTE_DSL_ARCH=sm_121a` (done in `start.sh`). Do not point this recipe at x86-only NVFP4 assumptions from Hopper/RTX guides.
- If attention backends misbehave, keep `TRITON_ATTN`. FlashInfer can be tried via `ATTENTION_BACKEND=FLASHINFER`.
- If the machine OOMs during load (desktop + browser + weights), drop util toward `0.60`–`0.65` or lower `MAX_MODEL_LEN` / `MAX_NUM_SEQS`.
- Cold start caches compile/autotune under `VLLM_CACHE_ROOT` (`~/.cache/vllm`).

### Optional DFlash speculative decoding

```bash
ENABLE_DFLASH=1 ./start.sh
```

Uses [`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) (`MuseGlimmerAssistantModel`, ~5 GB). Extra UMA cost — prefer after a clean BF16 smoke test.

**Local patch:** PR #51655 registers `MuseGlimmerAssistantModel`, but vLLM’s `EAGLEConfig(method=dflash)` rewrites that to `DFlashMuseGlimmerAssistantModel`. The editable install under `~/src/vllm-muse-glimmer` adds the prefixed registry alias (needed for `ENABLE_DFLASH=1`).

## Client sampling

From the model card: `temperature=1.0`, `top_p=0.95`, `top_k=64`. Reasoning strength via system prompt: `Reasoning strength: low|medium|high|xhigh`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Glimmer-30B",
    "messages": [
      {"role": "system", "content": "Reasoning strength: high"},
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in one sentence."},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            }
          }
        ]
      }
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64
  }'
```

## Alternates (not this recipe’s default)

| Checkpoint | Notes |
|---|---|
| Official BF16 (this recipe) | Best quality; fits Spark UMA |
| [`Inferact/Muse-Glimmer-30B-NVFP4-W4A4`](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) | ModelOpt NVFP4; try after Muse lands + verify W4A16 fallback on sm_121 |
| [`RadixArk/Muse-Glimmer-NVFP4`](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4) | Tuned for **SGLang** on GB10 (text-only quant); not the vLLM path here |
| GGUF / llama.cpp | Outside vLLM |

## Port conflicts

Only one server on **:8000**. Stop sibling recipes (Qwen NVFP4, Ling, Nemotron, etc.) first.
