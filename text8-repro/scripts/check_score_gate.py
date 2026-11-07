#!/usr/bin/env python3
"""Fail-closed scoring gate.

Exit codes:
  0  PASS
  1  FAIL
  2  usage / missing or invalid scores file
  3  PARITY HOLD - needs PM review

The gate never treats NaN as a pass. BPC tolerance is the fixed constant
0.01. Parity uses UKLLM's three-band policy on independently recomputed
max_abs (payload warn flags are ignored):
  max_abs <= 1e-4                 PASS
  1e-4 < max_abs <= 1e-3          HOLD (exit 3)
  max_abs > 1e-3                  FAIL
  missing or non-finite max_abs   FAIL
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_PARITY_HOLD = 3

BPC_TOLERANCE = 0.01
PARITY_PASS_MAX = 1e-4
PARITY_HOLD_MAX = 1e-3

TEXT8_BYTES = 100_000_000
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEQ_LEN = 1024

PUBLISHED_CLAIMS = {
    "transformer/best.pt": 1.2401772355061287,
    "transformer/final.pt": 1.257564388922874,
    "ssm/best.pt": 1.2638741508689288,
    "ssm/final.pt": 1.268647525903973,
}
EXPECTED_CHECKPOINTS = tuple(PUBLISHED_CLAIMS)
SKIP_FLAGS = (
    "skip_full",
    "skip_sampled",
    "skip_parity",
    "allow_missing_data",
)
TF32_ENV_KEYS = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


def expected_split_bytes(split: str) -> int:
    n_test = int(TEXT8_BYTES * TEST_FRAC)
    n_val = int(TEXT8_BYTES * VAL_FRAC)
    if split == "val":
        return n_val
    if split == "test":
        return n_test
    raise ValueError(split)


def expected_full_tokens(split_bytes: int, seq_len: int = SEQ_LEN) -> int:
    """Token count for score.py's non-overlapping windows plus remainder.

    ByteData uses a contiguous 90/5/5 split of the 100,000,000-byte text8
    file, so val and test are each 5,000,000 bytes. Windows start at
    0, seq, 2*seq, ... while start < split_bytes - seq. Each full window
    contributes ``seq`` next-token targets. A padded remainder window
    then covers ``split_bytes - last_start - seq - 1`` further targets.
    That equals ``split_bytes - 1``: every byte except index 0 of the
    split is scored. For val and test this is 4,999,999.
    """
    starts = list(range(0, max(0, split_bytes - seq_len), seq_len))
    tokens = len(starts) * seq_len
    next_start = starts[-1] + seq_len if starts else 0
    remaining = split_bytes - next_start - 1
    if remaining > 0:
        tokens += remaining
    return tokens


EXPECTED_FULL_TOKENS = {
    "val": expected_full_tokens(expected_split_bytes("val")),
    "test": expected_full_tokens(expected_split_bytes("test")),
}


def is_finite_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _tf32_block(payload: dict):
    """Return the tf32 object written by score.py, or None if absent.

    score.py writes the same object at the top-level ``tf32`` key and under
    ``device_report.tf32``. Prefer the top-level copy.
    """
    block = payload.get("tf32")
    if isinstance(block, dict):
        return block
    report = payload.get("device_report")
    if isinstance(report, dict) and isinstance(report.get("tf32"), dict):
        return report["tf32"]
    return None


def _tf32_env_is_zero(value) -> bool:
    """Only the exact recorded string ``0`` (after strip) is acceptable."""
    if value is None:
        return False
    return str(value).strip() == "0"


def check_tf32(payload: dict) -> list[str]:
    """Fail closed if TF32 overrides or flags are not fully disabled.

    score.py always records ``tf32.env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` and
    ``tf32.env.NVIDIA_TF32_OVERRIDE`` (null if the process env lacked them).
    A missing ``tf32`` block, a missing env key, or any value other than ``0``
    therefore fails, including null/unset. ``cuda_matmul_allow_tf32`` and
    ``cudnn_allow_tf32`` must be exactly false, and
    ``float32_matmul_precision`` must be exactly ``highest``.
    """
    errors: list[str] = []
    block = _tf32_block(payload)
    if not isinstance(block, dict):
        return ["tf32 block is missing from scores.json"]
    env = block.get("env")
    if not isinstance(env, dict):
        errors.append("tf32.env is missing")
        env = {}
    for key in TF32_ENV_KEYS:
        if key not in env:
            errors.append(f"tf32.env.{key} is missing")
            continue
        value = env[key]
        if not _tf32_env_is_zero(value):
            errors.append(
                f"tf32.env.{key}={value!r} is not \"0\"; scoring requires the override off"
            )
    if block.get("cuda_matmul_allow_tf32") is not False:
        errors.append(
            f"tf32.cuda_matmul_allow_tf32 is {block.get('cuda_matmul_allow_tf32')!r}, expected false"
        )
    if block.get("cudnn_allow_tf32") is not False:
        errors.append(
            f"tf32.cudnn_allow_tf32 is {block.get('cudnn_allow_tf32')!r}, expected false"
        )
    precision = block.get("float32_matmul_precision")
    if precision != "highest":
        errors.append(
            f"tf32.float32_matmul_precision is {precision!r}, expected \"highest\""
        )
    return errors


def parity_band(max_abs) -> str:
    """Return pass, hold, or fail from max_abs. Does not trust payload flags."""
    if not is_finite_number(max_abs):
        return "fail"
    value = float(max_abs)
    if value <= PARITY_PASS_MAX:
        return "pass"
    if value <= PARITY_HOLD_MAX:
        return "hold"
    return "fail"


def check_device_is_gpu(payload: dict) -> list[str]:
    errors = []
    eval_block = payload.get("eval") if isinstance(payload.get("eval"), dict) else {}
    report = payload.get("device_report") if isinstance(payload.get("device_report"), dict) else {}
    requested = str(eval_block.get("device") or report.get("requested_device") or "")
    if not requested.startswith("cuda"):
        errors.append(f"GPU scoring is required; device is {requested!r}")
    if report.get("cuda_available") is not True:
        errors.append("device_report.cuda_available is not true; GPU parity is required")
    return errors


def evaluate_gate(payload: dict, threshold: float = BPC_TOLERANCE) -> dict:
    """Return a structured gate result."""
    errors: list[str] = []
    holds: list[str] = []
    lines: list[str] = []

    if threshold != BPC_TOLERANCE:
        return {
            "ok": False,
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "errors": [f"threshold must be the fixed {BPC_TOLERANCE}, got {threshold!r}"],
            "holds": [],
            "lines": [f"FAIL threshold must be the fixed {BPC_TOLERANCE}, got {threshold!r}"],
            "threshold": threshold,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "fail",
            "exit_code": EXIT_FAIL,
            "errors": ["scores payload is missing or not an object"],
            "holds": [],
            "lines": ["FAIL scores payload is missing or not an object"],
            "threshold": threshold,
        }

    errors.extend(check_tf32(payload))
    errors.extend(check_device_is_gpu(payload))

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

    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    for expected in EXPECTED_CHECKPOINTS:
        if counts.get(expected, 0) != 1:
            continue
        row = by_name.get(expected) or {}
        name = expected
        published = PUBLISHED_CLAIMS[expected]

        load = row.get("load") if isinstance(row.get("load"), dict) else {}
        for key in ("weights_only", "strict", "finite_tensors"):
            if load.get(key) is not True:
                errors.append(f"{name}: load.{key} is not true")

        payload_claim = row.get("claimed_sampled_val_bpc")
        if payload_claim is not None and (
            not is_finite_number(payload_claim) or float(payload_claim) != published
        ):
            errors.append(
                f"{name}: claimed_sampled_val_bpc {payload_claim!r} does not match "
                f"the hard-coded published claim {published}"
            )

        sampled = row.get("sampled_val_bpc")
        full_val = row.get("full_val_bpc")
        full_test = row.get("full_test_bpc")
        missing = []
        for label, value in (
            ("sampled_val_bpc", sampled),
            ("full_val_bpc", full_val),
            ("full_test_bpc", full_test),
        ):
            if not is_finite_number(value):
                missing.append(label)
                errors.append(f"{name}: {label} is missing or non-finite ({value!r})")

        extras_txt = ""
        if not missing:
            delta_sampled = abs(float(sampled) - published)
            delta_full_val = abs(float(full_val) - published)
            if not math.isfinite(delta_sampled) or not math.isfinite(delta_full_val):
                errors.append(f"{name}: non-finite delta")
            else:
                if delta_sampled > BPC_TOLERANCE:
                    errors.append(
                        f"{name}: sampled_val_bpc {sampled} differs from published claim "
                        f"{published} by {delta_sampled:.6g} (tolerance {BPC_TOLERANCE})"
                    )
                if delta_full_val > BPC_TOLERANCE:
                    errors.append(
                        f"{name}: full_val_bpc {full_val} differs from published claim "
                        f"{published} by {delta_full_val:.6g} (tolerance {BPC_TOLERANCE})"
                    )
                extras_txt = f" sampled_delta={delta_sampled:.6g} full_val_delta={delta_full_val:.6g}"

        for split in ("val", "test"):
            info = row.get(f"full_{split}")
            if not isinstance(info, dict):
                errors.append(f"{name}: full_{split} details are missing")
                continue
            if "truncated" not in info:
                errors.append(f"{name}: full_{split}.truncated is missing")
            elif info.get("truncated") is not False:
                errors.append(f"{name}: full_{split}.truncated must be false")
            if "complete" not in info:
                errors.append(f"{name}: full_{split}.complete is missing")
            elif info.get("complete") is not True:
                errors.append(f"{name}: full_{split}.complete must be true")
            tokens = info.get("n_tokens")
            expected_tokens = EXPECTED_FULL_TOKENS[split]
            if not is_finite_number(tokens) or int(tokens) != expected_tokens:
                errors.append(
                    f"{name}: full_{split} n_tokens is {tokens!r}, expected {expected_tokens}"
                )
            bpc = row.get(f"full_{split}_bpc")
            if not is_finite_number(bpc):
                errors.append(f"{name}: full_{split}_bpc is missing or non-finite")

        parity = row.get("cpu_gpu_parity") if isinstance(row.get("cpu_gpu_parity"), dict) else {}
        row_status = "PASS"
        if not parity:
            errors.append(f"{name}: cpu_gpu_parity output is missing")
            row_status = "FAIL"
        elif parity.get("skipped"):
            errors.append(f"{name}: cpu_gpu_parity was skipped; GPU parity is required")
            row_status = "FAIL"
        else:
            max_abs = parity.get("max_abs_diff")
            band = parity_band(max_abs)
            relative_error = parity.get("relative_error")
            bpc_delta = parity.get("parity_batch_bpc_delta")
            lines.append(
                f"PARITY {name}: max_abs={max_abs} band={band} "
                f"relative_error={relative_error} parity_batch_bpc_delta={bpc_delta}"
            )
            if band == "fail":
                if not is_finite_number(max_abs):
                    errors.append(f"{name}: cpu_gpu_parity.max_abs_diff is missing or non-finite")
                else:
                    errors.append(
                        f"{name}: cpu_gpu_parity.max_abs_diff {max_abs} > {PARITY_HOLD_MAX} (FAIL band)"
                    )
                row_status = "FAIL"
            elif band == "hold":
                holds.append(
                    f"{name}: cpu_gpu_parity.max_abs_diff {max_abs} is in the HOLD band "
                    f"({PARITY_PASS_MAX} < max_abs <= {PARITY_HOLD_MAX})"
                )
                row_status = "HOLD"

        if extras_txt or missing or row_status == "FAIL":
            if missing or any(item.startswith(f"{name}:") and "differs" in item for item in errors):
                if row_status != "HOLD":
                    row_status = "FAIL" if (missing or any(
                        item.startswith(f"{name}:") for item in errors
                    )) and row_status != "HOLD" else row_status
        # Row-level bpc/load errors already recorded; print summary line.
        if any(item.startswith(f"{name}:") for item in errors) and row_status != "HOLD":
            row_status = "FAIL"
        lines.append(
            f"{row_status} {name}: sampled={sampled} full_val={full_val} "
            f"full_test={full_test} published_claim={published}{extras_txt}"
        )

    if errors:
        status = "fail"
        exit_code = EXIT_FAIL
        ok = False
    elif holds:
        status = "hold"
        exit_code = EXIT_PARITY_HOLD
        ok = False
    else:
        status = "pass"
        exit_code = EXIT_PASS
        ok = True

    lines.append(f"bpc_tolerance={BPC_TOLERANCE}")
    lines.append(
        f"parity_bands: PASS if max_abs<={PARITY_PASS_MAX}; "
        f"HOLD if {PARITY_PASS_MAX}<max_abs<={PARITY_HOLD_MAX}; "
        f"FAIL if max_abs>{PARITY_HOLD_MAX}"
    )
    if status == "pass":
        lines.append(
            "Gate passed. Training profiles may be started if Ops wants a from-scratch rerun."
        )
    elif status == "hold":
        lines.append("PARITY HOLD - needs PM review")
        lines.extend(f"  - {item}" for item in holds)
    else:
        lines.append("Gate failed. Do not start training.")
        lines.extend(f"  - {item}" for item in errors)
        if holds:
            lines.append("Also recorded parity HOLD items:")
            lines.extend(f"  - {item}" for item in holds)
    return {
        "ok": ok,
        "status": status,
        "exit_code": exit_code,
        "errors": errors,
        "holds": holds,
        "lines": lines,
        "threshold": BPC_TOLERANCE,
    }


def write_gate_outputs(result: dict, json_path: Path | None, text_path: Path | None) -> None:
    text = "\n".join(result["lines"]) + "\n"
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": result.get("ok"),
            "status": result.get("status"),
            "exit_code": result.get("exit_code"),
            "threshold": result.get("threshold"),
            "errors": result.get("errors"),
            "holds": result.get("holds"),
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=BPC_TOLERANCE,
        help=f"Must be the fixed {BPC_TOLERANCE}; any other value fails closed",
    )
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

    if args.threshold != BPC_TOLERANCE:
        result = evaluate_gate({}, threshold=args.threshold)
        write_gate_outputs(result, write_json, write_txt)
        return EXIT_FAIL

    if not args.scores.is_file():
        result = {
            "ok": False,
            "status": "fail",
            "exit_code": EXIT_USAGE,
            "errors": [f"scores file not found: {args.scores}"],
            "holds": [],
            "lines": [f"FAIL scores file not found: {args.scores}"],
            "threshold": BPC_TOLERANCE,
        }
        write_gate_outputs(result, write_json, write_txt)
        return EXIT_USAGE

    try:
        payload = json.loads(args.scores.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result = {
            "ok": False,
            "status": "fail",
            "exit_code": EXIT_USAGE,
            "errors": [f"scores.json is not valid JSON: {exc}"],
            "holds": [],
            "lines": [f"FAIL scores.json is not valid JSON: {exc}"],
            "threshold": BPC_TOLERANCE,
        }
        write_gate_outputs(result, write_json, write_txt)
        return EXIT_USAGE

    result = evaluate_gate(payload, BPC_TOLERANCE)
    write_gate_outputs(result, write_json, write_txt)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
