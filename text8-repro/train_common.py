"""Shared launcher for the published PhaseMesh training runners."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def add_common_args(parser: argparse.ArgumentParser, *, default_out: str) -> None:
    root = Path(__file__).resolve().parent
    load_env_file(root / ".env")
    load_env_file(Path.cwd() / ".env")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory passed through as the runner --out path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data", type=Path,
                        default=Path(os.environ.get("TEXT8_PATH", root / "data/text8")))
    parser.add_argument("--hf-dir", type=Path,
                        default=Path(os.environ.get("HF_DIR", root / "data/hf/phasemesh-text8-comparison")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=float(os.environ.get("WALL_SECONDS", "259200")),
        help="Runner deadline. The published default of 10800s is too short for a 5000-step run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-steps", type=int, default=None)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER,
                        help="Optional extra args forwarded to the published runner after --")


def config_value(cfg: dict, *keys, default=None):
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def runner_command(*, runner: Path, cfg: dict, args, extra_flags: list[str]) -> list[str]:
    cmd = [
        sys.executable, str(runner),
        "--data", str(args.data),
        "--out", str(args.out_dir),
        "--device", args.device,
        "--dtype", str(config_value(cfg, "dtype", default="bfloat16")),
        "--microbatch", str(config_value(cfg, "microbatch", default=4)),
        "--effective-batch", str(config_value(cfg, "effective_batch", default=64)),
        "--steps", str(config_value(cfg, "steps", default=5000)),
        "--seq", str(config_value(cfg, "seq", default=1024)),
        "--d-model", str(config_value(cfg, "d_model", default=1088)),
        "--layers", str(config_value(cfg, "layers", default=12)),
        "--heads", str(config_value(cfg, "heads", default=17)),
        "--eval-interval", str(config_value(cfg, "eval_interval", default=250)),
        "--eval-iters", str(config_value(cfg, "eval_iters", default=50)),
        "--eval-batch", str(config_value(cfg, "eval_batch", default=64)),
        "--seed", str(args.seed),
        "--lr", str(config_value(cfg, "lr", default=0.0003)),
        "--min-lr", str(config_value(cfg, "min_lr", default=0.0001)),
        "--warmup", str(config_value(cfg, "warmup", default=300)),
        "--checkpoint-every", str(config_value(cfg, "checkpoint_every", default=25)),
        "--threads", str(config_value(cfg, "threads", default=2)),
        "--wall-seconds", str(args.wall_seconds),
        "--max-new-steps", str(args.max_new_steps or config_value(cfg, "steps", default=5000)),
    ]
    cmd.extend(extra_flags)
    if args.resume:
        cmd.append("--resume")
    extras = list(args.runner_args)
    if extras and extras[0] == "--":
        extras = extras[1:]
    cmd.extend(extras)
    return cmd


def launch(kind: str, rel_runner: str, rel_config: str, extra_flags_from_cfg) -> int:
    parser = argparse.ArgumentParser(
        description=f"Thin wrapper around the published {kind} runner, using that snapshot's config.json."
    )
    add_common_args(parser, default_out=f"runs/{kind}")
    args = parser.parse_args()

    hf_dir = args.hf_dir.resolve()
    runner = hf_dir / rel_runner
    config_path = hf_dir / rel_config
    if not runner.is_file():
        raise SystemExit(f"Published runner not found: {runner}. Run scripts/fetch_hf.sh first.")
    if not config_path.is_file():
        raise SystemExit(f"Published config not found: {config_path}. Run scripts/fetch_hf.sh first.")
    if not args.data.is_file():
        raise SystemExit(f"text8 not found: {args.data}. Run scripts/fetch_text8.sh first.")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    extra_flags = extra_flags_from_cfg(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = runner_command(runner=runner, cfg=cfg, args=args, extra_flags=extra_flags)
    from device_check import log_device_report
    device_report = log_device_report(args.device)
    print(json.dumps({"event": "launch", "kind": kind, "command": cmd, "seed": args.seed,
                      "out_dir": str(args.out_dir), "device_report": device_report}, indent=2), flush=True)

    start = time.monotonic()
    completed = subprocess.run(cmd, check=False)
    wall_seconds = time.monotonic() - start

    progress_path = args.out_dir / "progress.json"
    progress = None
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    optimizer_only = None
    if isinstance(progress, dict):
        optimizer_only = progress.get("trained_seconds")

    timing = {
        "kind": kind,
        "seed": args.seed,
        "out_dir": str(args.out_dir.resolve()),
        "data": str(args.data.resolve()),
        "hf_dir": str(hf_dir),
        "runner": str(runner),
        "config": str(config_path),
        "command": cmd,
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
        "optimizer_only_seconds": optimizer_only,
        "progress_present": progress is not None,
        "note": (
            "wall_seconds is measured by this wrapper around the published runner. "
            "optimizer_only_seconds is trained_seconds from the runner progress.json, "
            "which excludes evaluation, saving, and pauses."
        ),
    }
    timing_path = args.out_dir / "wrapper_timing.json"
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "finished", "returncode": completed.returncode,
                      "wall_seconds": wall_seconds,
                      "optimizer_only_seconds": optimizer_only,
                      "timing": str(timing_path)}, indent=2), flush=True)
    return completed.returncode
