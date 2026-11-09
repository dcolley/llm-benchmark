# Independent check of the UK LLM text8 results (PhaseMesh vs Transformer)

## Summary

All four published checkpoints reproduce the reported sampled validation scores on the exact evaluation batches. Independent seed-1 retraining of both the Transformer and the PhaseMesh SSM reproduces within 0.01 bpc on the decisive post-training metrics (full validation, full test and exact replay). The only fresh-seed sampled breach is the known evaluation-batch artefact already documented for published final checkpoints. These findings support this specific reproduction, not a general architectural ranking.

The reference run checked here, the published checkpoints at [MarxistLeninist/phasemesh-text8-comparison](https://huggingface.co/MarxistLeninist/phasemesh-text8-comparison) revision 2eb1bef, is Marx's reproduction of Ash's PhaseSSM/PhaseMesh work; the upstream code is Ash's [ashhart/PhaseMesh](https://github.com/ashhart/PhaseMesh) at commit 3ae43ff.

## Conclusions so far

**Checkpoint results independently reproduced**

All four published checkpoints reproduce Marx’s reported sampled validation bpc within 3e-8 when scored on the reference run’s exact batches. The initial sampled-score failures came from different evaluation batches, not faulty checkpoints. Seed-1 retraining of both architectures also reproduces end to end on independent hardware: every Transformer post-training delta is below 0.0026 bpc, and the SSM matches within 0.01 bpc on 7 of 8 post-training metrics, with the remaining breach a fresh-seed sampled-batch artefact rather than a training failure.

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

## Retraining: Transformer and SSM complete

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

### SSM - complete

| SSM seed 1 | Result |
| --- | --- |
| Placement / exit | NVIDIA DGX Spark (GB10), separate from the Transformer system / exit 0 |
| Start | 7 November 2026, 23:42:15 GMT |
| Finish | 9 November 2026, about 03:18 GMT |
| Training wall time | 99,369.823 seconds (27 hours 36 minutes), including evaluation and checkpointing |
| Optimiser time | 92,850.181 seconds versus Marx’s 30,490.677 seconds |
| Optimiser-time ratio | 3.05× Marx; 18.570 seconds/step versus 6.098 seconds/step |
| Steps | 5,000/5,000 |
| Training image | `text8-repro:6a70a5b-spark3` |
| Scoring code | main at `8acd4d9` (wording-only changes since `f8ef3c6`) |

The training reproduces numerically despite slower execution on different hardware. This is not a controlled hardware-speed comparison. Wall time includes evaluation and saving; optimiser time excludes evaluation, saving and pauses. Absolute finish time is inferred from the recorded start plus wrapper wall time; wrapper_timing.json confirms duration and exit 0, but does not contain absolute timestamps.

Every same-step in-training validation point from 250 through 5000 remains within 0.0026924435411948 bpc of Marx’s curve. The maximum absolute difference occurs at step 1750 (signed delta +0.0026924435411948). Best in-training validation is 1.2638209158966265 at step 4750 versus Marx’s 1.2638741508689288 (delta -0.0000532349723023). Final in-training validation at step 5000 is 1.268237859360008 versus Marx’s 1.268647525903973 (delta -0.000409666543965).

| Step | Retrained SSM (val bpc) | Marx (val bpc) | Delta (retrained minus Marx) |
| --- | --- | --- | --- |
| 250 | 1.8344510121281743 | 1.835812197625522 | -0.00136118549734765 |
| 500 | 1.6619740352941113 | 1.6601418494024889 | +0.0018321858916223732 |
| 750 | 1.5645714161176603 | 1.5641321236272796 | +0.00043929249038066054 |
| 1000 | 1.5028255840550144 | 1.50169415835013 | +0.0011314257048844478 |
| 1250 | 1.4618411524568442 | 1.4599168781323497 | +0.0019242743244944904 |
| 1500 | 1.436322913304183 | 1.4341153437827294 | +0.002207569521453623 |
| 1750 | 1.408942829744269 | 1.4062503862030742 | +0.0026924435411948 |
| 2000 | 1.387485878770017 | 1.3854908531477192 | +0.001995025622297808 |
| 2250 | 1.364508517535574 | 1.3642272775312927 | +0.00028124000428131346 |
| 2500 | 1.3448452355409246 | 1.343896172284818 | +0.0009490632561066636 |
| 2750 | 1.32974406380313 | 1.3288723761865813 | +0.0008716876165486553 |
| 3000 | 1.3080423433580495 | 1.307802269228568 | +0.00024007412948146012 |
| 3250 | 1.2958448145862318 | 1.2963670377557936 | -0.0005222231695618529 |
| 3500 | 1.2894649307988757 | 1.2895530304166285 | -8.80996177528015e-05 |
| 3750 | 1.2794491201913252 | 1.2806939624350606 | -0.0012448422437354445 |
| 4000 | 1.2657696451866711 | 1.2659000191074083 | -0.00013037392073722565 |
| 4250 | 1.275290425124055 | 1.275563765105269 | -0.000273339981213816 |
| 4500 | 1.2710846438942076 | 1.2706390528617064 | +0.00044559103250119136 |
| 4750 | 1.2638209158966265 | 1.2638741508689288 | -5.323497230236285e-05 |
| 5000 | 1.268237859360008 | 1.268647525903973 | -0.00040966654396479285 |

Post-training scoring below is strict FP32 with TF32 off, distinct from in-training history. Gate comparisons for sampled and full validation use Marx’s claimed sampled validation values (threshold 0.01 bpc). Full validation and full test Marx columns use the independently scored published checkpoints. Deltas against those published scored values are taken from summary.csv. Positive is worse.

### SSM best - step 4750

| Metric (bpc) | Retrained | Marx | Delta |
| --- | --- | --- | --- |
| Default sampled | 1.25639896 | 1.26387415 | -0.00747519 |
| Full validation (vs published scored) | 1.26397875 | 1.26432744 | -0.00034869 |
| Full test (vs published scored) | 1.34517032 | 1.34613140 | -0.00096108 |
| Exact replay (vs claim) | 1.26381987 | 1.26387415 | -0.00005428 |
| Exact replay (vs own progress) | 1.26381987 | 1.26382092 | -0.00000104 |

Sampled gate: PASS. Full-validation gate versus claim: PASS (+0.00010460). Parity max absolute logit difference: 2.94e-5. Full validation and test each cover 4,999,999 tokens (complete). The full-split deltas above match summary.csv (`delta_full_val_vs_published_scored`, `delta_full_test_vs_published_scored`).

### SSM final - step 5000

| Metric (bpc) | Retrained | Marx | Delta |
| --- | --- | --- | --- |
| Default sampled | 1.25745285 | 1.26864753 | -0.01119467 |
| Full validation (vs published scored) | 1.26488207 | 1.26478601 | +0.00009605 |
| Full test (vs published scored) | 1.34592964 | 1.34675570 | -0.00082606 |
| Exact replay (vs claim) | 1.26823856 | 1.26864753 | -0.00040897 |
| Exact replay (vs own progress) | 1.26823856 | 1.26823786 | +0.00000070 |

Sampled gate: FAIL (about 0.0012 bpc outside the 0.01 threshold). Full-validation gate versus claim: PASS (-0.00376546). Parity max absolute logit difference: 3.29e-5. Full validation and test each cover 4,999,999 tokens (complete). summary.csv reports full-split deltas of +0.00009207 and -0.00083036 against slightly rounded published figures; the table uses the unrounded independently scored published checkpoints.

The only gate breach is the final checkpoint’s fresh-seed sampled score. That is the same sampling artefact already documented for the published final checkpoints: the default scorer reseeds and draws the step-0 sample, while exact replay and the full splits agree closely with the reference. Treat exact replay and full-split scores as decisive. Across the eight post-training metrics analogous to the Transformer comparison, seven lie within 0.01 bpc; the eighth is this sampled-batch artefact.

Reported GPU power was about 38 W for the SSM versus 56 W for the Transformer, at about 96% utilisation. This suggests a possible memory or FFT bottleneck on this hardware, but power and utilisation alone do not prove the cause. Treat it as an unprofiled observation requiring profiling, not an architectural efficiency conclusion.

The SSM:Transformer optimiser-time ratio on GB10 is about 6.0× (18.570 versus about 3.11 seconds/step), compared with 3.52× in Marx’s reproduction. This is not a controlled hardware-speed comparison.

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

1. Both Transformer and SSM seed-1 training now reproduce end to end on the decisive metrics. Keep the sampled-batch caveat for final checkpoints when citing gates.
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

Transformer retraining evidence directly reviewed includes `comparison.md`, `scores.json`, `curve-comparison.json` and `wrapper_timing.json`. SSM retraining evidence directly reviewed includes `summary.csv`, `progress.json`, `wrapper_timing.json`, `provenance.json`, `best-scores.json` and `final-scores.json`. Both architectures’ training reproduce end to end on independent hardware within the stated tolerance, not bit for bit.

## How to reproduce

Start with [text8-repro/README.md](README.md), which documents the environment and workflow. These files exist in main at commit `8acd4d97b7b58be7c0e2cc8292e4937ef0e9ca11`. Links are relative to this report in `text8-repro/`:

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


</cursor_untrusted_data_1337
