#!/usr/bin/env python3
"""Launch the published transformer/run_matched.py with that snapshot's config.json."""
from train_common import launch


def extra_flags(_cfg: dict) -> list[str]:
    return []


if __name__ == "__main__":
    raise SystemExit(launch(
        "transformer",
        "transformer/run_matched.py",
        "transformer/config.json",
        extra_flags,
    ))
