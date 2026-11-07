# PhaseMesh text8 re-check results

Fill this in after scoring (and only then, if requested, after a from-scratch
re-train). Copy numbers from `results/scores.json` and training
`wrapper_timing.json` / `progress.json`. Do not treat a blank cell as a pass.

Hardware, image digest, `torch.cuda.get_device_capability()`, and
`torch.cuda.get_arch_list()` should be pasted from the scorer's `device_report`.

## Claims vs reproduced

| Checkpoint | Claimed step | Claimed sampled val bpc | Checkpoint `val_bpc` | Reproduced sampled val bpc | Full val bpc | Full test bpc | \|sampled − claimed\| | \|full val − claimed\| | Gate (≤ 0.01 on sampled and full val) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| transformer/best.pt | 4000 | 1.2401772355061287 |  |  |  |  |  |  |  |
| transformer/final.pt | 5000 | 1.257564388922874 |  |  |  |  |  |  |  |
| ssm/best.pt | 4750 | 1.2638741508689288 |  |  |  |  |  |  |  |
| ssm/final.pt | 5000 | 1.268647525903973 |  |  |  |  |  |  |  |

Sampling note: claimed scores were recorded during training. Independent
sampled validation uses the published 50 x 64 protocol with a fresh
`torch.manual_seed(seed)` and cannot replay the original RNG.

Full-split note: windows are non-overlapping, so the first token of each
window has no prior context from earlier bytes in the split. Full bpc is not
directly comparable to a sliding-window number. Full test is informational
but must be present and complete for the gate.

## CPU vs GPU logit parity

Fixed batch: 2 sequences of 64 byte ids, seed 1, FP32, TF32 off. The gate
uses UKLLM's three-band policy on `max_abs_diff`: `<= 1e-4` PASS;
`1e-4 < max_abs <= 1e-3` HOLD (exit 3, needs PM review); `> 1e-3` FAIL.
Missing or non-finite `max_abs` is FAIL. Also record `relative_error`
(`max_abs / max|cpu_logit|`) and the parity-batch bpc delta (CPU vs GPU)
as information only. Measured values cannot be filled until Ops runs
scoring on hardware.

| Checkpoint | max abs diff | max rel diff | Warning? |
| --- | ---: | ---: | --- |
| transformer/best.pt | *(not yet measured; fill after hardware run)* | *(not yet measured)* |  |
| transformer/final.pt | *(not yet measured; fill after hardware run)* | *(not yet measured)* |  |
| ssm/best.pt | *(not yet measured; fill after hardware run)* | *(not yet measured)* |  |
| ssm/final.pt | *(not yet measured; fill after hardware run)* | *(not yet measured)* |  |

## Device report

- Image digest:
- `torch` version:
- Device name:
- `get_device_capability()`:
- `get_arch_list()`:
- Capability listed?:
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` / `NVIDIA_TF32_OVERRIDE`:
- `allow_tf32` (cuda matmul / cudnn):
- `float32_matmul_precision`:
- Notes (SDPA backend, FFT, Triton unused, etc.):

## From-scratch training (optional; only after the scoring gate)

| Run | Seed | Wall seconds | Optimizer-only seconds (`trained_seconds`) | Best sampled val bpc | Final sampled val bpc |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformer | 1 |  |  |  |  |
| ssm | 1 |  |  |  |  |

Published optimizer-only times (for comparison, not a throughput claim):
transformer 8665.858 s, SSM 30490.677 s.

## Provenance

Summarise `results/provenance.md`: whether the two vendor copies match, whether
overlapping files match upstream `3ae43ff`, and any kernel hits.
