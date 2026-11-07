# PhaseMesh text8 re-check results

Fill this in after scoring (and only then, if requested, after a from-scratch
re-train). Copy numbers from `results/scores.json` and training
`wrapper_timing.json` / `progress.json`. Do not treat a blank cell as a pass.

Hardware, image digest, `torch.cuda.get_device_capability()`, and
`torch.cuda.get_arch_list()` should be pasted from the scorer's `device_report`.

## Claims vs reproduced

| Checkpoint | Claimed step | Claimed sampled val bpc | Checkpoint `val_bpc` | Reproduced sampled val bpc | Full val bpc | Full test bpc | \|sampled − claimed\| | Gate (≤ 0.01) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| transformer/best.pt | 4000 | 1.2401772355061287 |  |  |  |  |  |  |
| transformer/final.pt | 5000 | 1.257564388922874 |  |  |  |  |  |  |
| ssm/best.pt | 4750 | 1.2638741508689288 |  |  |  |  |  |  |
| ssm/final.pt | 5000 | 1.268647525903973 |  |  |  |  |  |  |

Sampling note: claimed scores were recorded during training. Independent
sampled validation uses the published 50 x 64 protocol with a fresh
`torch.manual_seed(seed)` and cannot replay the original RNG. Full-split
numbers are the primary held-out check.

## CPU vs GPU logit parity

Fixed batch: 2 sequences of 64 byte ids, seed 1, FP32. Warn if max abs diff >
1e-4.

| Checkpoint | max abs diff | max rel diff | Warning? |
| --- | ---: | ---: | --- |
| transformer/best.pt |  |  |  |
| transformer/final.pt |  |  |  |
| ssm/best.pt |  |  |  |
| ssm/final.pt |  |  |  |

## Device report

- Image digest:
- `torch` version:
- Device name:
- `get_device_capability()`:
- `get_arch_list()`:
- Capability listed?:
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
