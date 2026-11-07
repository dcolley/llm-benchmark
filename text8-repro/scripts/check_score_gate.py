#!/usr/bin/env python3
"""Exit 0 if independently sampled validation bpc is within 0.01 of each claim.

Training profiles should be used only after this gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THRESHOLD = 0.01


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="?", default=Path("results/scores.json"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if not args.scores.is_file():
        print(f"scores file not found: {args.scores}", file=sys.stderr)
        return 2
    payload = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints") or []
    if not rows:
        print("scores.json contains no checkpoints", file=sys.stderr)
        return 2
    failed = False
    for row in rows:
        name = row.get("name")
        claimed = row.get("claimed_sampled_val_bpc")
        got = row.get("sampled_val_bpc")
        embedded = row.get("checkpoint_val_bpc")
        if claimed is None or got is None:
            print(f"FAIL {name}: missing claimed or sampled_val_bpc")
            failed = True
            continue
        delta = abs(float(got) - float(claimed))
        status = "PASS" if delta <= args.threshold else "FAIL"
        extra = ""
        if embedded is not None and math.isfinite(float(embedded)):
            extra = f" checkpoint_val_bpc={embedded} (vs claim {abs(float(embedded)-float(claimed)):.6g})"
        print(f"{status} {name}: sampled={got} claimed={claimed} abs_delta={delta:.6g}{extra}")
        if delta > args.threshold:
            failed = True
    print(f"threshold={args.threshold}")
    if failed:
        print(
            "Gate failed. Do not start training until reproduced sampled bpc "
            "is within the threshold, or an operator has reviewed a documented exception "
            "(fresh-seed sampled eval is not a replay of in-training RNG).",
            file=sys.stderr,
        )
        return 1
    print("Gate passed. Training profiles may be started if Ops wants a from-scratch rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
