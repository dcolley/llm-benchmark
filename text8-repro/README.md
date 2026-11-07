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
  scripts/export_artifacts.py
  tests/test_check_score_gate.py
  apt-packages.txt
  artifacts/                 # reviewed copies only; see artifacts/README.md
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
| `UPSTREAM_DIR` | Persistent writable checkout of upstream PhaseMesh used by `provenance_diff.sh` |
| `UID`, `GID` | Host user/group that should own container-written outputs (`user: ${UID}:${GID}`) |

## Scoring gate before any training

Training is **not** started by `docker compose up`. The `train-transformer` and
`train-ssm` services are behind the `train` profile.

The gate is fail-closed (`scripts/check_score_gate.py`). Exit codes:

- `0` PASS
- `1` FAIL
- `2` missing or invalid `scores.json`
- `3` `PARITY HOLD - needs PM review`

It passes only if all of the following hold:

- All four published checkpoints are present exactly once (`transformer/best.pt`,
  `transformer/final.pt`, `ssm/best.pt`, `ssm/final.pt`), with no duplicates
  and no extras.
- Each checkpoint loaded with `weights_only=True`, strict `state_dict` loading,
  and finite-tensor checks.
- Full val and full test both have `complete: true`, `truncated` present and
  false, and `n_tokens` equal to the expected coverage: text8 is 100,000,000
  bytes with a contiguous 90/5/5 split, so val and test are each 5,000,000
  bytes. Non-overlapping windows of 1024 plus a padded remainder score every
  byte except index 0 of the split, which is **4,999,999** tokens.
- Scoring ran on GPU. Skipped or CPU-only parity fails. The gate recomputes
  the parity band from `cpu_gpu_parity.max_abs_diff` and does not trust payload
  warn flags. UKLLM's policy: `max_abs <= 1e-4` PASS; `1e-4 < max_abs <= 1e-3`
  HOLD (exit 3); `max_abs > 1e-3` or missing/non-finite `max_abs` FAIL.
  Different CPU vs GPU reduction order (MKL/OpenBLAS vs cuBLAS) and FFT
  libraries (pocketfft vs cuFFT for the SSM FFT conv) over 11–12 layers at
  width 1088, with logits around 10 and FP32 eps about 1.2e-7, produce honest
  diffs from about 1e-5 to a few 1e-4. Real bugs (wrong kernel, dtype, or a
  TF32 leak) show up around 1e-3 to 1e-1. TF32 is gated separately. A HOLD
  means Ops brings the numbers to UKLLM, who decides.
- Both `sampled_val_bpc` and `full_val_bpc` are within the fixed **0.01**
  tolerance of the hard-coded published claims (not a claim copied from the
  payload). `full_test_bpc` is printed for information and must be present and
  complete, but is not compared to the claim. A `--threshold` other than 0.01
  fails.
- Recorded TF32 metadata must show the overrides off. `tf32.env`
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` and `NVIDIA_TF32_OVERRIDE` are always
  written by `score.py` (null if unset); a missing `tf32` block, a missing key,
  or any value other than `"0"` fails. `cuda_matmul_allow_tf32` and
  `cudnn_allow_tf32` must be false, and `float32_matmul_precision` must be
  `"highest"`. This is a gate failure, not a warning.
- The run did not set a smoke/skip/limit flag (`--skip-full`, `--skip-sampled`,
  `--skip-parity`, `--allow-missing-data`, or `--max-windows`).

Non-finite scores or deltas fail the gate (a NaN comparison must never pass).

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
# Persistent writable upstream checkout; set UPSTREAM_DIR in .env so later
# runs fetch/checkout the pin instead of cloning from scratch.
./scripts/provenance_diff.sh
python score.py --ckpt-dir "$HF_DIR" --data "$TEXT8_PATH" --out-dir "$RESULTS_DIR" --device cuda --seed 1
python scripts/check_score_gate.py "$RESULTS_DIR/scores.json"
python scripts/export_artifacts.py --src "$RESULTS_DIR" --dest artifacts
# only after the gate passes:
python train_transformer.py --out-dir "$RUNS_DIR/transformer" --seed 1 --data "$TEXT8_PATH" --hf-dir "$HF_DIR"
python train_ssm.py --out-dir "$RUNS_DIR/ssm" --seed 1 --data "$TEXT8_PATH" --hf-dir "$HF_DIR"
```

Compose (draft runtime):

```
cp .env.example .env   # then edit host paths, UID/GID, and UPSTREAM_DIR
docker compose --profile fetch run --rm fetch
./scripts/provenance_diff.sh   # not a compose service; needs network and UPSTREAM_DIR
docker compose up --build score
python scripts/check_score_gate.py /absolute/path/to/results/scores.json
python scripts/export_artifacts.py --src /absolute/path/to/results --dest artifacts
# only after the gate passes:
docker compose --profile train run --rm train-transformer
docker compose --profile train run --rm train-ssm
```

`provenance_diff.sh` diffs both vendored `phase_ssm` trees against upstream
`phase_ssm/` at the pinned commit and writes `results/provenance.md`. Point
`UPSTREAM_DIR` at a persistent, writable checkout (created on first run). Do
not use a throwaway path if you want to reuse the clone.

Compose services run as `user: ${UID}:${GID}` from `.env`, so files written
into `results/`, `runs/`, and `cache/` are owned by that host user rather
than root. Fetch has no GPU reservation. Score and train each request one GPU.

The score service sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` and
`NVIDIA_TF32_OVERRIDE=0`. The scorer also sets
`torch.backends.cuda.matmul.allow_tf32=False`,
`torch.backends.cudnn.allow_tf32=False`, and
`torch.set_float32_matmul_precision("highest")` before any CUDA work, and
records the effective env vars, flags, and matmul precision in
`device_report` / `scores.json`. If an override env is still set to something
other than `0`, the scorer prints a loud warning and `check_score_gate.py`
fails closed. Training services do not set those env vars and do not change
TF32 flags; they only record whatever the container inherited, matching the
published runners (which also leave TF32 untouched).

`score`, `train-transformer` and `train-ssm` use `network_mode: none`. Fetch
keeps default networking so it can download the snapshot and text8, and it
does not reserve a GPU. No service uses privileged mode, host networking, or
the Docker socket. Each score/train service requests exactly one GPU.

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
   and record `max_abs_diff`, `relative_error` (`max_abs / max|cpu_logit|`),
   and the parity-batch bpc on CPU and GPU plus their delta. The gate applies
   UKLLM's three-band rule to `max_abs_diff` (see above); it does not trust
   the informational `warn` flag.
6. Sampled val: `torch.manual_seed(seed)` then the published loop (50 iterations,
   batch 64, sequence 1024, microbatched).
7. Full val and test: contiguous 90/5/5 `ByteData` split, non-overlapping
   windows of 1024, plus a padded remainder window so leftover tail tokens are
   included. Mean CE in nats / ln(2), token-weighted. Because the windows do
   not overlap, the first token of each window has no prior context from
   earlier bytes in the split. Full-split bpc is therefore not directly
   comparable to a sliding-window (stride-1) number. Each full split writes
   `complete` and `n_tokens` (expected 4,999,999 for val and test).

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
pinned Python packages (none today). Fetch needs `curl`, `git`, and `unzip`.
Those apt packages are pinned in `apt-packages.txt` (Ubuntu 24.04 / noble
versions for the NGC 26.07 userspace) and recorded again inside the image at
`/opt/text8-repro-apt-versions.txt` after install. Host-side scripts need those
plus `python3` and `sha256sum` or `shasum`.

The Compose build context is this directory. `.dockerignore` excludes `.env*`,
weights (`*.pt`, `*.safetensors`, `*.bin`), `data/`, caches, `runs/`,
`results/`, `.git`, and secret-like files. The repo root is not the build
context, so there is no extra root `.dockerignore`.
