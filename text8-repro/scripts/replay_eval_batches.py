#!/usr/bin/env python3
"""Replay the Marx reference run's in-training sampled-val start offsets (no model).

Mirrors transformer/run_matched.py and ssm/run_ssm.py: after the
post-construction torch.manual_seed(seed), every corpus window is drawn by
ByteData.get_batch(..., 'cpu') which does torch.randint(0, high, (B,)) on
the default CPU generator.

Draw order (shared by both published runners):
  * at each eval step (step % eval_interval == 0), draw eval_iters val
    batches of eval_batch, then max(10, eval_iters // 5) train batches of
    eval_batch (run_matched.py:102, run_ssm.py:109);
  * then, unless step == steps, draw one training batch of effective_batch
    and increment step (run_matched.py:112-124, run_ssm.py:119-131).
  * The eval at step == steps therefore happens inside the loop, and no
    training batch is drawn at that step.

Offsets are not a fixed eval set: each eval step consumes further CPU RNG.
score.py's default sampled_bpc reseeds to --seed and therefore matches the
step-0 val draw, not later evals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULTS = {
    "steps": 5000,
    "eval_interval": 250,
    "eval_iters": 50,
    "eval_batch": 64,
    "seq": 1024,
    "effective_batch": 64,
    "seed": 1,
    "val_bytes": 5_000_000,
    "train_bytes": 90_000_000,
}
CONFIG_KEYS = (
    "steps", "eval_interval", "eval_iters", "eval_batch", "seq",
    "effective_batch", "seed",
)


def train_eval_iters(eval_iters: int) -> int:
    return max(10, eval_iters // 5)


def split_lengths(n: int, val_frac: float = 0.05, test_frac: float = 0.05) -> tuple[int, int]:
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return n_train, n_val


def high_for(n_src: int, seq: int) -> int:
    return n_src - seq - 1


def draw_offsets(high: int, batch: int):
    """Identical call shape/dtype to ByteData.get_batch (data.py:50)."""
    import torch

    return torch.randint(0, high, (batch,))


def offsets_sha256(batches: list[list[int]]) -> str:
    payload = json.dumps(batches, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_published_files(path: Path | None) -> tuple[dict, dict | None]:
    """Load a config.json and/or progress.json. Progress is detected by history."""
    if path is None:
        return {}, None
    data = load_json(path)
    if "history" in data and "eval_interval" not in data and "eval_iters" not in data:
        return {}, data
    progress = None
    if "history" in data:
        progress = {k: data[k] for k in data if k in {"history", "step", "last_eval", "complete", "best", "best_val_bpc"}}
    cfg = {k: data[k] for k in CONFIG_KEYS if k in data}
    return cfg, progress


def ambiguity_reasons(cfg: dict, progress: dict | None, step: int) -> list[str]:
    reasons = []
    if cfg.get("resume") is True:
        reasons.append("config sets resume=true")
    if not progress:
        return reasons
    if progress.get("complete") is False:
        reasons.append("progress.complete is false")
    history = progress.get("history") or []
    interval = int(cfg["eval_interval"])
    if history:
        hist_steps = [int(row["step"]) for row in history]
        expected = list(range(0, hist_steps[-1] + 1, interval)) if hist_steps else []
        if hist_steps != expected:
            reasons.append(f"progress history steps are not contiguous evals: {hist_steps}")
        if step not in hist_steps:
            reasons.append(f"requested step {step} is not in progress history")
    last_eval = progress.get("last_eval")
    if last_eval is not None and int(last_eval) < step:
        reasons.append(f"progress.last_eval {last_eval} < requested step {step}")
    return reasons


def history_val_bpc(progress: dict | None, step: int) -> float | None:
    if not progress:
        return None
    for row in progress.get("history") or []:
        if int(row.get("step", -1)) == step and "val_bpc" in row:
            return float(row["val_bpc"])
    return None


def replay_val_offsets(
    *,
    seed: int,
    step: int,
    eval_interval: int,
    eval_iters: int,
    eval_batch: int,
    seq: int,
    effective_batch: int,
    steps: int,
    val_bytes: int,
    train_bytes: int,
) -> list[list[int]]:
    if eval_interval <= 0 or step % eval_interval != 0:
        raise SystemExit(f"step {step} is not an eval step for eval_interval={eval_interval}")
    if step > steps:
        raise SystemExit(f"step {step} is beyond configured steps={steps}")
    val_high = high_for(val_bytes, seq)
    train_high = high_for(train_bytes, seq)
    if val_high <= 0 or train_high <= 0:
        raise SystemExit("val/train high must be positive; check split lengths and seq")
    import torch

    torch.manual_seed(seed)
    n_train_eval = train_eval_iters(eval_iters)
    n_prior_evals = step // eval_interval
    for _ in range(n_prior_evals):
        for _ in range(eval_iters):
            draw_offsets(val_high, eval_batch)
        for _ in range(n_train_eval):
            draw_offsets(train_high, eval_batch)
    for _ in range(step):
        draw_offsets(train_high, effective_batch)
    batches = [draw_offsets(val_high, eval_batch).tolist() for _ in range(eval_iters)]
    return batches


def build_output(
    *,
    model: str,
    step: int,
    cfg: dict,
    batches: list[list[int]],
    history_bpc: float | None = None,
) -> dict:
    import torch

    val_high = high_for(int(cfg["val_bytes"]), int(cfg["seq"]))
    train_high = high_for(int(cfg["train_bytes"]), int(cfg["seq"]))
    payload = {
        "step": step,
        "seed": int(cfg["seed"]),
        "model": model,
        "draw_order": "shared_run_matched_and_run_ssm",
        "config": {
            "steps": int(cfg["steps"]),
            "eval_interval": int(cfg["eval_interval"]),
            "eval_iters": int(cfg["eval_iters"]),
            "eval_batch": int(cfg["eval_batch"]),
            "seq": int(cfg["seq"]),
            "effective_batch": int(cfg["effective_batch"]),
            "seed": int(cfg["seed"]),
            "val_bytes": int(cfg["val_bytes"]),
            "train_bytes": int(cfg["train_bytes"]),
            "train_eval_iters": train_eval_iters(int(cfg["eval_iters"])),
        },
        "torch_version": torch.__version__,
        "val_high": val_high,
        "train_high": train_high,
        "n_batches": len(batches),
        "batch_size": len(batches[0]) if batches else 0,
        "offsets": batches,
        "offsets_sha256": offsets_sha256(batches),
    }
    if history_bpc is not None:
        payload["progress_val_bpc"] = history_bpc
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("transformer", "ssm"),
                   help="Recorded in the output; both published runners share one draw order")
    p.add_argument("--step", type=int, required=True, help="Eval step to capture (e.g. 4000, 4750, 5000)")
    p.add_argument("--config", type=Path, help="Published config.json (or a progress.json)")
    p.add_argument("--progress", type=Path, help="Published progress.json for history/resume checks")
    p.add_argument("--dest", type=Path, help="Write offsets JSON here")
    p.add_argument("--seed", type=int, help="Override seed (default 1 or from --config)")
    p.add_argument("--steps", type=int)
    p.add_argument("--eval-interval", type=int, dest="eval_interval")
    p.add_argument("--eval-iters", type=int, dest="eval_iters")
    p.add_argument("--eval-batch", type=int, dest="eval_batch")
    p.add_argument("--seq", type=int)
    p.add_argument("--effective-batch", type=int, dest="effective_batch")
    p.add_argument("--val-bytes", type=int, dest="val_bytes")
    p.add_argument("--train-bytes", type=int, dest="train_bytes")
    p.add_argument("--data", type=Path, help="Optional corpus file; split lengths from its byte length (90/5/5)")
    p.add_argument("--force-assume-clean", action="store_true",
                   help="Ignore resume/abort ambiguity detected in --config/--progress")
    p.add_argument("--compare-progress", type=Path,
                   help="Print history val_bpc at --step from this progress.json")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> tuple[dict, dict | None]:
    file_cfg, file_progress = merge_published_files(args.config)
    extra_progress = load_json(args.progress) if args.progress else None
    progress = extra_progress or file_progress
    cfg = dict(DEFAULTS)
    cfg.update(file_cfg)
    if args.data is not None:
        n = args.data.stat().st_size
        train_n, val_n = split_lengths(n)
        cfg["train_bytes"] = train_n
        cfg["val_bytes"] = val_n
    for key in (*CONFIG_KEYS, "val_bytes", "train_bytes"):
        override = getattr(args, key, None)
        if override is not None:
            cfg[key] = override
    return cfg, progress


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, progress = resolve_config(args)
    reasons = ambiguity_reasons(cfg, progress, args.step)
    if reasons and not args.force_assume_clean:
        raise SystemExit(
            "refusing to replay because published files look ambiguous "
            f"({'; '.join(reasons)}); pass --force-assume-clean to override"
        )
    batches = replay_val_offsets(
        seed=int(cfg["seed"]),
        step=args.step,
        eval_interval=int(cfg["eval_interval"]),
        eval_iters=int(cfg["eval_iters"]),
        eval_batch=int(cfg["eval_batch"]),
        seq=int(cfg["seq"]),
        effective_batch=int(cfg["effective_batch"]),
        steps=int(cfg["steps"]),
        val_bytes=int(cfg["val_bytes"]),
        train_bytes=int(cfg["train_bytes"]),
    )
    history_bpc = history_val_bpc(progress, args.step)
    payload = build_output(model=args.model, step=args.step, cfg=cfg, batches=batches, history_bpc=history_bpc)
    text = json.dumps(payload, indent=2) + "\n"
    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(text, encoding="utf-8")
        print(f"wrote {args.dest} sha256={payload['offsets_sha256']} batches={payload['n_batches']}", flush=True)
    else:
        sys.stdout.write(text)
    compare_path = args.compare_progress
    if compare_path:
        compare = load_json(compare_path)
        bpc = history_val_bpc(compare, args.step)
        print(f"progress {compare_path} step {args.step} val_bpc={bpc}", flush=True)
    elif history_bpc is not None:
        print(f"progress step {args.step} val_bpc={history_bpc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
