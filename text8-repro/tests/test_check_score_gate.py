#!/usr/bin/env python3
"""Tiny fail-closed tests for check_score_gate.evaluate_gate."""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_score_gate import EXPECTED_CHECKPOINTS, evaluate_gate  # noqa: E402


CLAIMS = {
    "transformer/best.pt": 1.2401772355061287,
    "transformer/final.pt": 1.257564388922874,
    "ssm/best.pt": 1.2638741508689288,
    "ssm/final.pt": 1.268647525903973,
}


def _ckpt(name: str, **overrides) -> dict:
    claimed = CLAIMS[name]
    row = {
        "name": name,
        "claimed_sampled_val_bpc": claimed,
        "sampled_val_bpc": claimed,
        "full_val_bpc": claimed,
        "full_test_bpc": claimed + 0.002,
        "load": {"weights_only": True, "strict": True, "finite_tensors": True},
        "full_val": {"n_tokens": 4999999, "truncated": False},
        "full_test": {"n_tokens": 4999999, "truncated": False},
        "cpu_gpu_parity": {
            "skipped": False,
            "warn": False,
            "max_abs_diff": 1e-6,
            "max_rel_diff": 1e-6,
        },
    }
    row.update(overrides)
    return row


def valid_payload() -> dict:
    return {
        "device_report": {"cuda_available": True, "requested_device": "cuda"},
        "eval": {
            "skip_full": False,
            "skip_sampled": False,
            "skip_parity": False,
            "allow_missing_data": False,
            "max_windows": None,
        },
        "checkpoints": [_ckpt(name) for name in EXPECTED_CHECKPOINTS],
    }


def assert_fail(payload, *needles, threshold=0.01):
    result = evaluate_gate(payload, threshold)
    assert result["ok"] is False, f"expected FAIL, got PASS: {result}"
    blob = " ".join(result["errors"]).lower()
    for needle in needles:
        assert needle.lower() in blob, f"missing {needle!r} in {result['errors']}"


def test_clean_pass():
    result = evaluate_gate(valid_payload())
    assert result["ok"] is True, result
    assert result["errors"] == []


def test_nan_rejects():
    payload = valid_payload()
    payload["checkpoints"][0]["sampled_val_bpc"] = math.nan
    assert_fail(payload, "non-finite", "sampled_val_bpc")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val_bpc"] = float("nan")
    assert_fail(payload, "non-finite", "full_val_bpc")
    payload = valid_payload()
    payload["checkpoints"][0]["claimed_sampled_val_bpc"] = math.nan
    payload["checkpoints"][0]["sampled_val_bpc"] = 1.24
    payload["checkpoints"][0]["full_val_bpc"] = 1.24
    assert_fail(payload, "non-finite")


def test_missing_ckpt():
    payload = valid_payload()
    payload["checkpoints"] = payload["checkpoints"][:-1]
    assert_fail(payload, "missing checkpoint", "ssm/final.pt")


def test_duplicate_ckpt():
    payload = valid_payload()
    payload["checkpoints"].append(deepcopy(payload["checkpoints"][0]))
    assert_fail(payload, "duplicate checkpoint", "transformer/best.pt")


def test_smoke_flag():
    payload = valid_payload()
    payload["eval"]["max_windows"] = 4
    assert_fail(payload, "max_windows")
    payload = valid_payload()
    payload["eval"]["skip_full"] = True
    assert_fail(payload, "skip_full")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val"]["truncated"] = True
    assert_fail(payload, "truncated")


def test_parity_warn():
    payload = valid_payload()
    payload["checkpoints"][1]["cpu_gpu_parity"]["warn"] = True
    assert_fail(payload, "cpu_gpu_parity.warn")
    payload = valid_payload()
    payload["checkpoints"][1]["cpu_gpu_parity"] = {"skipped": True, "reason": "--skip-parity"}
    assert_fail(payload, "skipped")


def main() -> int:
    tests = [
        test_clean_pass,
        test_nan_rejects,
        test_missing_ckpt,
        test_duplicate_ckpt,
        test_smoke_flag,
        test_parity_warn,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failed:
        print(f"{failed} test(s) failed")
        return 1
    print(f"All {len(tests)} gate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
