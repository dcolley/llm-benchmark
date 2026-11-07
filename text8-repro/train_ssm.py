#!/usr/bin/env python3
"""Launch the published ssm/run_ssm.py with that snapshot's config.json."""
from train_common import launch


def extra_flags(cfg: dict) -> list[str]:
    flags = ["--state-dim", str(cfg.get("state_dim", 64))]
    if cfg.get("activation_checkpoint"):
        flags.append("--activation-checkpoint")
    return flags


if __name__ == "__main__":
    raise SystemExit(launch(
        "ssm",
        "ssm/run_ssm.py",
        "ssm/config.json",
        extra_flags,
    ))
