#!/usr/bin/env python3
"""Fail-closed scoring gate.

Passes only when every published checkpoint is present once, loads cleanly,
finishes full val and test, passes CPU/GPU parity, and both sampled val bpc
and full val bpc are within the threshold of the claim. Full test bpc is
printed for information and must be present and complete, but is not compared
to the claim.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

THRESHOLD = 0.01
EXPECTED_CHECKPOINTS = (
    "transformer/best.pt",
    "transformer/final.pt",
    "ssm/best.pt",
    "ssm/final.pt",
)
SKIP_FLAGS = (
    "skip_full",
    "skip_sampled",
    "skip_parity",
    "allow_missing_data",
)


def is_finite_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _threshold_error(threshold) -> str | None:
    if not is_finite_number(threshold):
        return f"invalid threshold {threshold!r}: must be a finite number"
    if float(threshold) <= 0:
        return f"invalid threshold {threshold}: must be > 0"
    return None


def evaluate_gate(payload: dict, threshold: float = THRESHOLD) -> dict:
    """Return a structured gate result. Never treats NaN as a pass."""
    errors: list[str] = []
    lines: list[str] = []
    threshold_problem = _threshold_error(threshold)
    if threshold_problem:
        return {
            "ok": False,
            "errors": [threshold_problem],
            "lines": [f"FAIL {threshold_problem}"],
            "threshold": threshold,
        }
    threshold = float(threshold)

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "errors": ["scores payload is missing or not an object"],
            "lines": ["FAIL scores payload is missing or not an object"],
            "threshold": threshold,
        }

    eval_block = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    for flag in SKIP_FLAGS:
        if eval_block.get(flag):
            errors.append(f"eval flag {flag} is set; the gate rejects smoke/skip runs")
    max_windows = eval_block.get("max_windows")
    if max_windows not in (None, 0, False):
        errors.append(f"eval.max_windows={max_windows!r} looks like a smoke/limit run")

    rows = payload.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        errors.append("scores.json contains no checkpoints")
        rows = []

    names = [row.get("name") for row in rows if isinstance(row, dict)]
    counts = Counter(names)
    for expected in EXPECTED_CHECKPOINTS:
        count = counts.get(expected, 0)
        if count == 0:
            errors.append(f"missing checkpoint {expected}")
        elif count > 1:
            errors.append(f"duplicate checkpoint {expected} ({count} times)")
    extras = [name for name in names if name not in EXPECTED_CHECKPOINTS]
    if extras:
        errors.append(f"unexpected checkpoints: {extras}")

    device_report = payload.get("device_report") if isinstance(payload.get("device_report"), dict) else {}
    cuda_available = bool(device_report.get("cuda_available"))
    requested = str(device_report.get("requested_device") or "cuda")
    cuda_expected = cuda_available or requested.startswith("cuda")

    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    for expected in EXPECTED_CHECKPOINTS:
        if counts.get(expected, 0) != 1:
            continue
        row = by_name.get(expected) or {}
        name = expected

        load = row.get("load") if isinstance(row.get("load"), dict) else {}
        for key in ("weights_only", "strict", "finite_tensors"):
            if load.get(key) is not True:
                errors.append(f"{name}: load.{key} is not true")

        claimed = row.get("claimed_sampled_val_bpc")
        sampled = row.get("sampled_val_bpc")
        full_val = row.get("full_val_bpc")
        full_test = row.get("full_test_bpc")
        missing = []
        for label, value in (
            ("claimed_sampled_val_bpc", claimed),
            ("sampled_val_bpc", sampled),
            ("full_val_bpc", full_val),
            ("full_test_bpc", full_test),
        ):
            if not is_finite_number(value):
                missing.append(label)
                errors.append(f"{name}: {label} is missing or non-finite ({value!r})")

        status = "PASS"
        extras_txt = ""
        if not missing:
            delta_sampled = abs(float(sampled) - float(claimed))
            delta_full_val = abs(float(full_val) - float(claimed))
            if not math.isfinite(delta_sampled) or not math.isfinite(delta_full_val):
                errors.append(f"{name}: non-finite delta")
                status = "FAIL"
            else:
                if delta_sampled > threshold:
                    errors.append(
                        f"{name}: sampled_val_bpc {sampled} differs from claim {claimed} "
                        f"by {delta_sampled:.6g} (threshold {threshold})"
                    )
                    status = "FAIL"
                if delta_full_val > threshold:
                    errors.append(
                        f"{name}: full_val_bpc {full_val} differs from claim {claimed} "
                        f"by {delta_full_val:.6g} (threshold {threshold})"
                    )
                    status = "FAIL"
                extras_txt = f" sampled_delta={delta_sampled:.6g} full_val_delta={delta_full_val:.6g}"
        else:
            status = "FAIL"

        for split in ("val", "test"):
            info = row.get(f"full_{split}")
            if not isinstance(info, dict):
                errors.append(f"{name}: full_{split} details are missing")
                status = "FAIL"
                continue
            if info.get("truncated"):
                errors.append(f"{name}: full_{split} is truncated (smoke/limit)")
                status = "FAIL"
            tokens = info.get("n_tokens")
            if not is_finite_number(tokens) or int(tokens) <= 0:
                errors.append(f"{name}: full_{split} n_tokens is missing, non-finite, or zero")
                status = "FAIL"
            bpc = row.get(f"full_{split}_bpc")
            if not is_finite_number(bpc):
                errors.append(f"{name}: full_{split}_bpc is missing or non-finite")
                status = "FAIL"

        parity = row.get("cpu_gpu_parity") if isinstance(row.get("cpu_gpu_parity"), dict) else {}
        if parity.get("warn") is True:
            errors.append(f"{name}: cpu_gpu_parity.warn is true")
            status = "FAIL"
        skipped = bool(parity.get("skipped"))
        if skipped and cuda_expected:
            errors.append(f"{name}: cpu_gpu_parity was skipped while CUDA was available or requested")
            status = "FAIL"
        if not skipped:
            if parity.get("warn") is not False:
                errors.append(f"{name}: cpu_gpu_parity.warn must be false when parity ran")
                status = "FAIL"
            if not is_finite_number(parity.get("max_abs_diff")) or not is_finite_number(parity.get("max_rel_diff")):
                errors.append(f"{name}: cpu_gpu_parity max_abs_diff/max_rel_diff missing or non-finite")
                status = "FAIL"
        if not parity:
            errors.append(f"{name}: cpu_gpu_parity output is missing")
            status = "FAIL"

        lines.append(
            f"{status} {name}: sampled={sampled} full_val={full_val} "
            f"full_test={full_test} claimed={claimed}{extras_txt}"
        )

    ok = not errors
    lines.append(f"threshold={threshold}")
    if ok:
        lines.append(
            "Gate passed. Training profiles may be started if Ops wants a from-scratch rerun."
        )
    else:
        lines.append("Gate failed. Do not start training.")
        lines.extend(f"  - {item}" for item in errors)
    return {"ok": ok, "errors": errors, "lines": lines, "threshold": threshold}


def write_gate_outputs(result: dict, json_path: Path | None, text_path: Path | None) -> None:
    text = "\n".join(result["lines"]) + "\n"
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": result["ok"],
            "threshold": result["threshold"],
            "errors": result["errors"],
            "lines": result["lines"],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if text_path is not None:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="?", default=Path("results/scores.json"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write structured gate output (default: alongside scores as gate.json)",
    )
    parser.add_argument(
        "--write-txt",
        type=Path,
        default=None,
        help="Write plaintext gate output (default: alongside scores as gate.txt)",
    )
    args = parser.parse_args(argv)
    write_json = args.write_json if args.write_json is not None else args.scores.parent / "gate.json"
    write_txt = args.write_txt if args.write_txt is not None else args.scores.parent / "gate.txt"

    if not args.scores.is_file():
        result = {
            "ok": False,
            "errors": [f"scores file not found: {args.scores}"],
            "lines": [f"FAIL scores file not found: {args.scores}"],
            "threshold": args.threshold,
        }
        write_gate_outputs(result, write_json, write_txt)
        return 2

    try:
        payload = json.loads(args.scores.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result = {
            "ok": False,
            "errors": [f"scores.json is not valid JSON: {exc}"],
            "lines": [f"FAIL scores.json is not valid JSON: {exc}"],
            "threshold": args.threshold,
        }
        write_gate_outputs(result, write_json, write_txt)
        return 2

    result = evaluate_gate(payload, args.threshold)
    write_gate_outputs(result, write_json, write_txt)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
