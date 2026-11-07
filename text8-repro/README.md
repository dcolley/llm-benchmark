# PhaseMesh text8 independent re-check

Scripts to re-score and optionally re-train the published PhaseMesh text8
comparison, without taking the published numbers on faith.

This directory is container-agnostic application code plus a draft runtime
(`Dockerfile`, `docker-compose.yml`, `.env.example`). Operational acceptance
of the runtime is owned by Ops. The same image is intended to run on two
machines; host paths and the single GPU id live only in a gitignored `.env`.

## Pins

| Item | Value |
| --- | --- |
| Hugging Face snapshot | [MarxistLeninist/phasemesh-text8-comparison](https://huggingface.co/MarxistLeninist/phasemesh-text8-comparison) revision `2eb1bef95b39e9745035044679f9d18b8ac3513a` |
| Upstream source | [ashhart/PhaseMesh](https://github.com/ashhart/PhaseMesh) commit `3ae43ffac6c0699857cada0865f156405fb95956` |
| text8 | `http://mattmahoney.net/dc/text8.zip`, extracted file sha256 `6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7` (100,000,000 bytes) |
| Runtime base image | `nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` (NGC PyTorch 26.07, torch 2.13, arm64 / GB10 sm_121) |

The published snapshot includes `MANIFEST.json`. `scripts/fetch_hf.sh` verifies
every listed file against it. If a future pin omitted that file, the script
would download the tree via the public Hugging Face API, write
`results/generated_manifest.json`, and the operator should record that fact
here.

## Claims being checked

Published as sampled validation bits-per-byte (50 batches of 64, sequence 1024,
contiguous text8 90/5/5 split, seed 1, effective batch 64, 5000 updates, BF16
train / FP32 sampled val):

| Checkpoint | Step | Claimed sampled val bpc |
| --- | ---: | ---: |
| `transformer/best.pt` | 4000 | 1.2401772355061287 |
| `transformer/final.pt` | 5000 | 1.257564388922874 |
| `ssm/best.pt` | 4750 | 1.2638741508689288 |
| `ssm/final.pt` | 5000 | 1.268647525903973 |

These checkpoints are model-only (`model_state`, `cfg`, `model_type`, `step`,
`val_bpc`). They load with `torch.load(..., weights_only=True)` and do not
include optimizer or RNG state. An independent 50-batch re-sample with seed 1
uses the same *procedure* as the runners, but it is not a replay of the
in-training RNG. Full non-overlapping val/test bpc is the more trustworthy
comparison.

## Layout

```
text8-repro/
  README.md
  RESULTS.md                 # fill in after a run
  Dockerfile                 # draft runtime; Ops accepts
  docker-compose.yml         # score is default; train is a profile
  .env.example               # placeholders only
  requirements.txt           # extra pip pins (none beyond the NGC image today)
  score.py
  device_check.py
  train_transformer.py
  train_ssm.py
  train_common.py
  scripts/fetch_hf.sh
  scripts/fetch_text8.sh
  scripts/provenance_diff.sh
  scripts/check_score_gate.py
```

Generated and ignored: `.env`, downloaded HF snapshot and text8, `runs/`,
`results/scores.json`, `results/provenance.md`, cache directories, `*.pt`.

## Environment variables

Copy `.env.example` to `.env` and set absolute host paths. Never commit `.env`.
Do not put hostnames or IP addresses in either file.

| Variable | Purpose |
| --- | --- |
| `HF_DIR` | Host directory for the pinned HF snapshot (compose mounts it read-only into score/train) |
| `TEXT8_DIR` | Host directory that contains the extracted `text8` file (read-only in score/train) |
| `TEXT8_PATH` | Extracted text8 file; host scripts default to `$TEXT8_DIR/text8` or `data/text8` |
| `RESULTS_DIR` | Writable scores and reports |
| `RUNS_DIR` | Writable training outputs (`progress.json`, checkpoints, `wrapper_timing.json`) |
| `CACHE_DIR` | Writable cache (`TORCH_HOME`, `XDG_CACHE_HOME`, Triton/HF caches) |
| `NVIDIA_VISIBLE_DEVICES` | Single GPU index visible inside each service. Do not list more than one. |
| `SEED` | Training and sampled-eval seed (default 1) |
| `DEVICE` | `cuda` or `cpu` |
| `MICROBATCH` | Microbatch for eval/train accumulation (published configs use 4) |
| `WALL_SECONDS` | Runner deadline forwarded to the published scripts (default 259200). Their own default of 10800 is too short for 5000 steps. |
| `IMAGE_NAME` | Tag for the built image, shared by both machines |
| `HF_REPO`, `HF_REVISION`, `HF_BASE_URL` | Fetch pin |
| `TEXT8_URL`, `TEXT8_EXPECTED_SHA256` | Corpus fetch and check |
| `UPSTREAM_REPO`, `UPSTREAM_COMMIT` | Provenance diff pin |

## Scoring gate before any training

Training is **not** started by `docker compose up`. The `train-transformer` and
`train-ssm` services are behind the `train` profile.

The gate is: for every scored checkpoint, the independently reproduced sampled
validation bpc is within **0.01** of the claimed value.

```
python scripts/check_score_gate.py results/scores.json
```

Only start a training profile after that command exits 0, or after an operator
records a documented exception. A miss can be caused by the fresh-seed sampled
protocol rather than a bad checkpoint; inspect `checkpoint_val_bpc`, the
CPU/GPU logit parity fields, and full-split bpc before deciding.

The two training jobs are independent and may run on two machines, one
architecture each, using the same image and different `.env` files (paths and
`NVIDIA_VISIBLE_DEVICES`). There is no tensor parallelism.

## Commands

Host-side, from this directory, after filling `.env` or exporting the same
variables:

```
./scripts/fetch_hf.sh
./scripts/fetch_text8.sh
./scripts/provenance_diff.sh
python score.py --ckpt-dir "$HF_DIR" --data "$TEXT8_PATH" --out-dir "$RESULTS_DIR" --device cuda --seed 1
python scripts/check_score_gate.py "$RESULTS_DIR/scores.json"
python train_transformer.py --out-dir "$RUNS_DIR/transformer" --seed 1 --data "$TEXT8_PATH" --hf-dir "$HF_DIR"
python train_ssm.py --out-dir "$RUNS_DIR/ssm" --seed 1 --data "$TEXT8_PATH" --hf-dir "$HF_DIR"
```

Compose (draft runtime):

```
cp .env.example .env   # then edit host paths
docker compose --profile fetch run --rm fetch
docker compose up --build score
python scripts/check_score_gate.py /absolute/path/to/results/scores.json
# only after the gate passes:
docker compose --profile train run --rm train-transformer
docker compose --profile train run --rm train-ssm
```

`score`, `train-transformer` and `train-ssm` use `network_mode: none`. Fetch
keeps default networking so it can download the snapshot and text8. No service
uses privileged mode, host networking, or the Docker socket. Each run/score
service requests exactly one GPU.

`score.py` writes `results/scores.json` with load checks, device capability
logging, CPU vs GPU logit parity, sampled val bpc, and full val/test bpc.
Training wrappers invoke the published `run_matched.py` / `run_ssm.py` with
arguments taken from that snapshot's `config.json`, and write `wrapper_timing.json`
containing wall time plus the runner's `trained_seconds` (optimizer-only).

## What the scorer does

1. `torch.load(..., map_location="cpu", weights_only=True)`.
2. Rebuild the model from the checkpoint `cfg` using the matching vendored
   package (`transformer/vendor` or `ssm/vendor`).
3. `load_state_dict(..., strict=True)` and reject non-finite tensors.
4. Evaluate in FP32.
5. On a 2 x 64 batch of fixed-seed random byte ids, compare CPU vs CUDA logits
   and record max absolute and max relative difference. Warn if max abs >
   `1e-4` (`--parity-warn-threshold`).
6. Sampled val: `torch.manual_seed(seed)` then the published loop (50 iterations,
   batch 64, sequence 1024, microbatched).
7. Full val and test: contiguous 90/5/5 `ByteData` split, non-overlapping
   windows of 1024, plus a padded remainder window so leftover tail tokens are
   included. Mean CE in nats / ln(2), token-weighted.

`--max-windows N` limits the full-split pass for smoke tests.
`--allow-missing-data` loads checkpoints without text8.

Startup (score and train wrappers) logs `torch.cuda.get_device_capability()`
and `torch.cuda.get_arch_list()`. GB10 is `sm_121` and may be absent from that
list even when the image is intended for it.

## Kernel and arch notes (from inspecting the published vendor trees)

These need confirmation on the target GPU; they are source findings, not
hardware results.

- **Used SSM path:** published `ssm/config.json` sets `ssm_backend` to `fft`.
  The forward is PyTorch FFT causal convolution (`torch.fft.rfft` / `irfft`),
  not a custom CUDA extension.
- **Unused Triton path:** `model.py` implements `fixed_triton` through
  `_FixedKernelSSM`, which imports `.triton_scan.ssm_recurrent_triton` when the
  activation is on CUDA. **`triton_scan.py` is not in either vendor tree.**
  Backend `auto` would pick `fixed_triton` only above `ssm_auto_threshold`
  (32768); published sequence length is 1024.
- **Short conv:** `nn.Conv1d` depthwise with right padding trimmed. Not the
  `causal-conv1d` or Mamba package.
- **`torch.compile`:** present in vendored `train_core.py`, but the published
  runners only import `lr_at` and run their own loop, so compile is not used
  for this re-check's training path.
- **Transformer:** `F.scaled_dot_product_attention`. Which SDPA backend runs on
  sm_121 (flash / mem-efficient / math) should be checked at runtime.
- No `mamba_ssm` import appears in the vendor copies.
- `scripts/provenance_diff.sh` records these keyword hits plus a file-level
  diff against upstream `phase_ssm/` in `results/provenance.md`.

## Surprises in the pinned Hugging Face files

- `MANIFEST.json` **is** present at `2eb1bef` and lists sha256 + byte sizes,
  including the four checkpoints (~522–527 MB each).
- Shared vendor files (`__init__.py`, `data.py`, `model.py`, `recurrent.py`,
  `train_core.py`) are byte-identical between `transformer/` and `ssm/`.
- The SSM vendor tree omits `train.py` and `transformer.py` (only the
  transformer side vendors those). Upstream `phase_ssm/` has additional modules
  such as `triton_scan.py` that were not vendored.
- `transformer/progress.json` uses keys `best` and `last_eval`. The published
  `run_matched.py` now writes `best_val_bpc`, `target_steps`,
  `training_byte_presentations`, and `complete` — the same schema as
  `ssm/progress.json`. A wrapper re-run will produce the newer schema.
- Runner CLI defaults include `--microbatch 8` and `--wall-seconds 10800`. The
  published configs used `--microbatch 4`. The wrappers pass the config.json
  values and a much larger wall limit. The SSM snapshot README example uses
  `--wall-seconds 86400`.
- `run_ssm.py` accepts `--heads` and records it in `config.json` even though
  `PhaseSSMConfig` does not use head count. The wrapper still passes `--heads
  17` so a re-run can match the published spec.
- The SSM snapshot README mentions `MODEL_MANIFEST.json`; the published
  package instead has the top-level `MANIFEST.json`.

## Dependencies

The NGC image supplies PyTorch and numpy. `requirements.txt` installs any extra
pinned Python packages (none today). Fetch needs `curl`, `git`, and `unzip`,
which the Dockerfile installs. Host-side scripts need those plus `python3` and
`sha256sum` or `shasum`.
