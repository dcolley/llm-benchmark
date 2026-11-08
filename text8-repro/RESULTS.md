# Independent check of the UK LLM text8 results (PhaseMesh vs Transformer)

## Summary

All four published checkpoints reproduce the reported sampled validation scores on the exact evaluation batches. Independent seed-1 Transformer training reproduces within 0.01 bpc on all eight post-training metrics. SSM retraining is in progress, with status below as of 8 November 2026, 10:09 GMT. These findings support this specific reproduction, not a general architectural ranking.

The reference run checked here, the published checkpoints at MarxistLeninist/phasemesh-text8-comparison revision 2eb1bef, is Marx's reproduction of Ash's PhaseSSM/PhaseMesh work; the upstream code is Ash's [ashhart/PhaseMesh](https://github.com/ashhart/PhaseMesh) at commit 3ae43ff.

## Conclusions so far

**Checkpoint results independently reproduced**

All four published checkpoints reproduce Marx’s reported sampled validation bpc within 3e-8 when scored on the reference run’s exact batches. The initial sampled-score failures came from different evaluation batches, not faulty checkpoints. The Transformer’s training also reproduces end to end on independent hardware: all eight post-training metric differences are below 0.0026 bpc, within the 0.01 tolerance. SSM training is still in progress.

| Best checkpoint | Full val bpc | Test bpc |
| --- | --- | --- |
| Transformer | 1.24538 | 1.31518 |
| PhaseMesh SSM | 1.26433 | 1.34613 |
| SSM minus Transformer | 0.01895 | 0.03095 |

Lower bits per character (bpc) is better. The Transformer leads on both complete splits in this comparison: about 0.019 bpc on validation and 0.031 on test, not a uniform 0.02 gap. In Marx’s reproduction, the SSM optimiser time was 3.52 times the Transformer’s. These findings do not establish a general architectural winner.

| Metric (bpc) | Transformer best | PhaseMesh SSM best |
| --- | --- | --- |
| Full validation | 1.2453799533401848 | 1.2643274427202766 |
| Test | 1.3151808592846739 | 1.3461314026968687 |

## Scope and method

> Would you fancy independently checking the UK LLM text8 results?
>
> Ash, via X

The check covers the pinned published artefacts, code provenance, CPU/GPU numerical parity, checkpoint scoring and an independent seed-1 retraining attempt.

| Item | Configuration |
| --- | --- |
| Transformer | 130,575,232 parameters; 11 layers; width 1088; 17 heads; RoPE |
| PhaseMesh SSM | 131,688,256 parameters; 12 layers; FFT backend |
| Data | text8; contiguous 90/5/5 train/validation/test split |
| Training | Seed 1; context 1024; effective batch 64; 5,000 steps; BF16 |
| Evaluation | Every 250 steps; 50 validation batches of 64; no autocast in the reference run’s evaluate(), so FP32 |

## Provenance and environment

| Check | Finding |
| --- | --- |
| Published snapshot | Pinned to HF revision 2eb1bef95b39e9745035044679f9d18b8ac3513a |
| Source provenance | Vendored code matches overlapping upstream files at 3ae43ff; no content differences |
| Artefact integrity | All 26 HF manifest files and the text8 SHA verified |
| Dataset SHA-256 | 6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7 |
| Checkpoint loads | Strict, weights-only loads; finite tensors in all four |
| Hardware | Two NVIDIA DGX Spark (GB10) systems: Transformer on one, SSM on another |
| Runtime | NGC PyTorch 26.07 arm64; torch 2.13.0a0+9186a08b2c.nv26.07; CUDA 13.3; numpy 2.1.0 |
| Scoring precision | FP32; TF32 disabled; float32 matmul precision highest |

The expected sm_121 arch-list warning is recorded: GB10 reports capability 12.1, which is not explicitly listed by this PyTorch build. It is a runtime caveat, not evidence of a scoring failure; all four CPU/GPU parity checks pass. Published SSM scoring uses PyTorch FFT, not a custom Triton kernel.

## Baseline scoring

| Checkpoint | Sampled bpc | Full val bpc | Test bpc |
| --- | --- | --- | --- |
| Transformer best | 1.23740 | 1.24538 | 1.31518 |
| Transformer final | 1.24461 | 1.25325 | 1.32361 |
| SSM best | 1.25672 | 1.26433 | 1.34613 |
| SSM final | 1.25714 | 1.26479 | 1.34676 |

| Checkpoint | Max logit diff | Sample delta | Gate |
| --- | --- | --- | --- |
| Transformer best | 2.86e-5 | 0.00277619 | PASS |
| Transformer final | 6.29e-5 | 0.0129532 | FAIL |
| SSM best | 2.29e-5 | 0.00714931 | PASS |
| SSM final | 2.19e-5 | 0.0115049 | FAIL |

Parity bands: PASS at max absolute logit difference ≤1e-4; HOLD above 1e-4 through 1e-3; FAIL above 1e-3. Every checkpoint passes parity. The score gate requires sampled and full-validation differences from the published sampled claim ≤0.01 bpc. The largest full-validation difference is 0.00520272 bpc; only the two final-checkpoint samples breach the gate.

Full-split scoring covers 4,883 non-overlapping context windows and 4,999,999 target tokens per split, including the padded remainder. Each window starts independently without preceding context. These numbers are not directly comparable with stride-1 sliding-window scores. Test scoring is complete but informational in the gate.

## Why the gate failed - and exact replay

Our baseline scorer resets the CPU RNG to seed 1 on every call: it therefore draws the reference run’s step-0 validation sample. Marx’s reproduction instead continues a shared CPU RNG stream during training. The seed is reset after model construction; each training step draws one batch, and each evaluation draws 50 validation then 10 training batches. Dropout is zero, so it does not consume RNG. At the scheduled evaluation steps, the validation sample follows 1.24 × step earlier batch draws.

Exact offset replay was added. The four offset SHA checks match the CPU references; the two step-5000 checkpoints share the same offset hash. Strict FP32 replay with TF32 off gives:

| Checkpoint / step | Replay bpc | Absolute delta |
| --- | --- | --- |
| Transformer best / 4000 | 1.2401772450 | 9.46e-9 |
| Transformer final / 5000 | 1.2575644103 | 2.14e-8 |
| SSM best / 4750 | 1.2638741546 | 3.76e-9 |
| SSM final / 5000 | 1.2686475277 | 1.83e-9 |

The checkpoints are consistent with the published artefacts and reported scores. The formal baseline gate files still say FAIL (exit 1): they only assess the fresh-seed sample and have not been rewritten to accept replay evidence. The discrepancy was resolved by exact replay evidence before both retraining runs began. This is an evidence-based resolution, not a claim that the original gate passed.

The matched-precision runs returned identical numbers to strict FP32. The TF32 option may not have taken effect; this remains a minor implementation uncertainty and is immaterial to the strict-FP32 reproduction.

## Retraining: Transformer complete, SSM in progress

| Transformer seed 1 | Result |
| --- | --- |
| Placement / exit | NVIDIA DGX Spark (GB10) / exit 0 |
| Start | 7 November 2026, 23:19:29 GMT |
| Finish | 8 November 2026, 04:16:19 GMT |
| Training wall time | 17,807.811 seconds |
| Optimiser time | 15,543.533 seconds versus Marx’s 8,665.858 seconds |
| Optimiser-time ratio | 1.79× Marx; about 1.8× slower per step |

The training reproduces numerically despite slower execution on different hardware. This is not a controlled hardware-speed comparison. Wall time includes evaluation and saving; optimiser time excludes evaluation, saving and pauses. The timestamps are the recorded run times; wrapper_timing.json independently confirms duration and exit status, but does not contain absolute timestamps.

Every same-step in-training validation point from 250 through 5000 remains within 0.0023320896 bpc of Marx’s curve. The maximum absolute difference occurs at step 4500 (signed delta -0.0023320896). The chart excludes the untrained step-0 point so the training trajectory remains readable.

| Step | Retrained Transformer (val bpc) | Marx (val bpc) | Delta (retrained minus Marx) |
| --- | --- | --- | --- |
| 250 | 1.9446893812444455 | 1.9426932876098857 | +0.0019960936345598 |
| 500 | 1.712584031666759 | 1.711670936432297 | +0.000913095234462 |
| 750 | 1.5276179488243888 | 1.5269934478948264 | +0.0006245009295624 |
| 1000 | 1.4469354865119621 | 1.4460930360535493 | +0.0008424504584128 |
| 1250 | 1.399921254066031 | 1.3978991878265163 | +0.0020220662395147 |
| 1500 | 1.375895092859147 | 1.375442339286683 | +0.000452753572464 |
| 1750 | 1.3443524117306362 | 1.3431843025562797 | +0.0011681091743565 |
| 2000 | 1.3280537402733223 | 1.3270683195623556 | +0.0009854207109667 |
| 2250 | 1.3067856955802242 | 1.3063774800323866 | +0.0004082155478376 |
| 2500 | 1.2915964221823826 | 1.2929188392430837 | -0.0013224170607011 |
| 2750 | 1.279575182399516 | 1.2797478125650594 | -0.0001726301655434 |
| 3000 | 1.2636244091646855 | 1.2646277712132794 | -0.0010033620485939 |
| 3250 | 1.2572056931996127 | 1.2565381182477393 | +0.0006675749518734 |
| 3500 | 1.2554522126959071 | 1.2563155760296867 | -0.0008633633337796 |
| 3750 | 1.249396438114555 | 1.250121944239991 | -0.000725506125436 |
| 4000 | 1.2408376278175672 | 1.2401772355061287 | +0.0006603923114385 |
| 4250 | 1.251728308150664 | 1.249788509382233 | +0.001939798768431 |
| 4500 | 1.248885714833666 | 1.2512178044163778 | -0.0023320895827118 |
| 4750 | 1.2472362764169809 | 1.2482063418719436 | -0.0009700654549627 |
| 5000 | 1.2587719577966443 | 1.257564388922874 | +0.0012075688737703 |

Post-training scoring below is strict FP32 with TF32 off, distinct from in-training history. Marx columns use the independently scored published checkpoints, not a different evaluation sample. Deltas are retrained minus Marx; positive is worse.

### Transformer best - step 4000

| Metric (bpc) | Retrained | Marx | Delta |
| --- | --- | --- | --- |
| Full validation | 1.24561448 | 1.24537995 | +0.00023453 |
| Full test | 1.31609167 | 1.31518086 | +0.00091081 |
| Exact replay | 1.24083958 | 1.24017724 | +0.00066234 |
| Default sampled | 1.23719504 | 1.23740104 | -0.00020600 |

### Transformer final - step 5000

| Metric (bpc) | Retrained | Marx | Delta |
| --- | --- | --- | --- |
| Full validation | 1.25383224 | 1.25324812 | +0.00058412 |
| Full test | 1.32611238 | 1.32360605 | +0.00250633 |
| Exact replay | 1.25877857 | 1.25756441 | +0.00121416 |
| Default sampled | 1.24513482 | 1.24461115 | +0.00052367 |

All eight absolute differences are ≤0.00250634 bpc, comfortably inside 0.01. Both evaluations exit 0; strict finite weights-only loading and CPU/GPU parity pass. Full validation and test each cover 4,999,999 tokens. Training used BF16 with TF32 settings left unmodified and recorded as enabled; post-training scoring explicitly disabled TF32.

The trainer’s latest.pt was not directly scoreable. A new model-only export script at PR commit f8ef3c6 converted the final checkpoint without altering the original training checkpoints or Marx’s published data. All 45 tests pass (35 gate, 6 replay, 4 export); export loads with weights_only=True, without fallback. Training/scorer source remains 6a70a5b and scorer dependencies are unchanged.

### SSM - in progress, results to follow

| Status as of 8 November 2026, 10:09 GMT | Observation |
| --- | --- |
| Start / hardware | 7 November 2026, 23:42:15 GMT / NVIDIA DGX Spark (GB10), separate from the Transformer system |
| Progress | Step 1892/5000 |
| Step time | Steady ~18.57 seconds versus Marx’s 6.10; about 3x slower |
| Estimated finish | Monday 9 November 2026, about 03:10 GMT |
| Checkpoint | Resumable |

| Step | Retrained SSM val bpc | Marx val bpc | Delta (bpc, reported) |
| --- | --- | --- | --- |
| 250 | 1.83445 | 1.83581 | -0.0014 |
| 500 | 1.66197 | 1.66014 | +0.0018 |
| 750 | 1.56457 | 1.56413 | +0.0004 |
| 1000 | 1.50283 | 1.50169 | +0.0011 |
| 1250 | 1.46184 | 1.45992 | +0.0019 |
| 1500 | 1.43632 | 1.43412 | +0.0022 |
| 1750 | 1.40894 | 1.40625 | +0.0027 |

These SSM timings and interim deltas are progress observations, not independently verified by the Transformer files. Best/final validation, test, exact replay and completed wall/optimiser comparisons remain outstanding. Final SSM results will be added when training finishes.

Reported GPU power is about 38 W for the SSM versus 56 W for the Transformer, at about 96% utilisation. This suggests a possible memory or FFT bottleneck on this hardware, but power and utilisation alone do not prove the cause. Treat it as an unprofiled observation requiring profiling, not an architectural efficiency conclusion.

The provisional SSM:Transformer optimiser-time ratio on GB10 is roughly 6x, based on SSM 18.57 s/step versus Transformer 3.11 s/step, compared with 3.52x in Marx’s reproduction. This is not a controlled hardware-speed comparison.

## Process and changes

| Change | Purpose / outcome |
| --- | --- |
| 0375545 → fad34b5 | Scoring moved from TF32-on and an older gate to the reviewed base |
| 319d1c0 | Superseded curl package pin updated to 8.5.0-2ubuntu10.16 (curl and libcurl4t64) |
| 7d78584 | UID/GID renamed HOST_UID/HOST_GID to avoid the shell’s readonly UID variable |
| 6a70a5b | Added scripts/replay_eval_batches.py and score.py options --val-offsets and --match-training-precision; 41 tests pass |
| f8ef3c6 - model-only export | Final latest.pt converted for scoring; originals untouched; 45 tests pass |
| Multi-seed diagnostic study | Cancelled once exact replay identified the batch-sampling cause |

PR #3, branch text8-repro/phasemesh, is merged into main at commit 2fb258e9801fba074e29dccba40ba63219502b53. The cancelled diagnostic study is distinct from a possible future multi-seed architecture comparison.

## Recommendations

1. Transformer seed-1 training now reproduces end to end. Review completed SSM metrics before extending that conclusion to both models.
2. Use complete validation and test bpc as headline metrics, with the windowing protocol stated. Keep sampled scores for exact historical replay rather than treating different samples as interchangeable.
3. Retain the published replay script and make evaluation offsets available so others can reproduce the same batches.
4. Extend the formal gate to accept replay offsets (low priority), while retaining the original failed gate as an auditable record.
5. Fix or remove the matched-precision flag; verify the effective TF32 settings rather than relying on an option label.
6. Consider multiple seeds and uncertainty estimates before claiming either architecture wins. The observed best-checkpoint gap is 0.01895 bpc on validation and 0.03095 on test, under these configurations only.

## Evidence and references

- [Published checkpoints, code and configs - pinned snapshot](https://huggingface.co/MarxistLeninist/phasemesh-text8-comparison/tree/2eb1bef95b39e9745035044679f9d18b8ac3513a)
- [Independent-check workspace](https://github.com/dcolley/llm-benchmark)
- [Merged reproduction PR #3](https://github.com/dcolley/llm-benchmark/pull/3)

Reviewed baseline evidence includes `scores.json`, `gate.json`, `gate.txt`, `provenance.md` and `device_report.json`. Reviewed replay evidence includes `summary.csv`. Score gaps were calculated from the unrounded JSON values. Artefact-manifest verification and offset-reference matches are recorded project findings; this report directly rechecked the listed baseline and replay files. These are evidence filenames, not claims that the evidence files are committed to the repository. Reproduction tooling is linked below.

Transformer retraining evidence directly reviewed includes `comparison.md`, `scores.json`, `curve-comparison.json` and `wrapper_timing.json`. The Transformer’s training reproduces end to end on independent hardware within the stated tolerance, not bit for bit. SSM training remains unfinished; final results will be added when complete.

## How to reproduce

Start with [text8-repro/README.md](README.md), which documents the environment and workflow. These files exist in main at merge commit `2fb258e9801fba074e29dccba40ba63219502b53`. Links are relative to this report in `text8-repro/`:

- Fetch pinned artefacts: [scripts/fetch_hf.sh](scripts/fetch_hf.sh).
- Fetch and verify text8: [scripts/fetch_text8.sh](scripts/fetch_text8.sh).
- Check provenance: [scripts/provenance_diff.sh](scripts/provenance_diff.sh).
- Check the device: [device_check.py](device_check.py).
- Score checkpoints: [score.py](score.py), including `--val-offsets` and `--match-training-precision` (see the precision caveat above).
- Recreate evaluation batches: [scripts/replay_eval_batches.py](scripts/replay_eval_batches.py).
- Inspect the original gate: [scripts/check_score_gate.py](scripts/check_score_gate.py).
- Train: [train_transformer.py](train_transformer.py) and [train_ssm.py](train_ssm.py).
- Export a resumable checkpoint for scoring: [scripts/export_model_checkpoint.py](scripts/export_model_checkpoint.py), using `weights_only=True` without fallback.

Keep pinned published artefacts unchanged and write new checkpoints and exports to separate destinations.
