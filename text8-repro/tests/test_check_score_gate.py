#!/usr/bin/env python3
"""Fail-closed tests for check_score_gate and export_artifacts."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_score_gate import (  # noqa: E402
    BPC_TOLERANCE,
    EXIT_FAIL,
    EXIT_PARITY_HOLD,
    EXIT_PASS,
    EXPECTED_CHECKPOINTS,
    EXPECTED_FULL_TOKENS,
    PUBLISHED_CLAIMS,
    evaluate_gate,
)
from export_artifacts import export, resolve_inside  # noqa: E402


def _ckpt(name: str, **overrides) -> dict:
    claimed = PUBLISHED_CLAIMS[name]
    row = {
        "name": name,
        "claimed_sampled_val_bpc": claimed,
        "sampled_val_bpc": claimed,
        "full_val_bpc": claimed,
        "full_test_bpc": claimed + 0.002,
        "load": {"weights_only": True, "strict": True, "finite_tensors": True},
        "full_val": {
            "n_tokens": EXPECTED_FULL_TOKENS["val"],
            "truncated": False,
            "complete": True,
        },
        "full_test": {
            "n_tokens": EXPECTED_FULL_TOKENS["test"],
            "truncated": False,
            "complete": True,
        },
        "cpu_gpu_parity": {
            "skipped": False,
            "max_abs_diff": 5e-5,
            "relative_error": 1e-6,
            "parity_batch_bpc_delta": 1e-7,
            "max_rel_diff": 1e-6,
        },
    }
    # Allow nested updates for cpu_gpu_parity / full_* via top-level replace.
    row.update(overrides)
    return row


def _tf32() -> dict:
    return {
        "env": {
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def valid_payload() -> dict:
    return {
        "device_report": {"cuda_available": True, "requested_device": "cuda"},
        "tf32": _tf32(),
        "eval": {
            "device": "cuda",
            "skip_full": False,
            "skip_sampled": False,
            "skip_parity": False,
            "allow_missing_data": False,
            "max_windows": None,
        },
        "checkpoints": [_ckpt(name) for name in EXPECTED_CHECKPOINTS],
    }


def assert_status(payload, status, *needles, threshold=BPC_TOLERANCE):
    result = evaluate_gate(payload, threshold)
    assert result["status"] == status, f"expected {status}, got {result}"
    blob = " ".join(result["errors"] + result["holds"] + result["lines"]).lower()
    for needle in needles:
        assert needle.lower() in blob, f"missing {needle!r} in {result}"
    return result


def assert_fail(payload, *needles, threshold=BPC_TOLERANCE):
    result = assert_status(payload, "fail", *needles, threshold=threshold)
    assert result["ok"] is False
    assert result["exit_code"] == EXIT_FAIL
    return result


def test_clean_pass():
    result = evaluate_gate(valid_payload())
    assert result["status"] == "pass", result
    assert result["ok"] is True
    assert result["exit_code"] == EXIT_PASS
    assert result["errors"] == []
    assert result["holds"] == []


def test_nan_rejects():
    payload = valid_payload()
    payload["checkpoints"][0]["sampled_val_bpc"] = math.nan
    assert_fail(payload, "non-finite", "sampled_val_bpc")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val_bpc"] = float("nan")
    assert_fail(payload, "non-finite", "full_val_bpc")
    payload = valid_payload()
    payload["checkpoints"][0]["claimed_sampled_val_bpc"] = math.nan
    payload["checkpoints"][0]["sampled_val_bpc"] = PUBLISHED_CLAIMS["transformer/best.pt"]
    payload["checkpoints"][0]["full_val_bpc"] = PUBLISHED_CLAIMS["transformer/best.pt"]
    assert_fail(payload, "hard-coded published claim")


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


def test_parity_skipped():
    payload = valid_payload()
    payload["checkpoints"][1]["cpu_gpu_parity"] = {"skipped": True, "reason": "--skip-parity"}
    assert_fail(payload, "skipped", "gpu parity")


def test_cpu_device():
    payload = valid_payload()
    payload["eval"]["device"] = "cpu"
    payload["device_report"]["requested_device"] = "cpu"
    payload["device_report"]["cuda_available"] = False
    assert_fail(payload, "gpu scoring is required")


def test_wrong_threshold():
    assert_fail(valid_payload(), "threshold must be the fixed", threshold=0.02)


def test_full_metrics_complete_and_tokens():
    payload = valid_payload()
    del payload["checkpoints"][0]["full_val"]["complete"]
    assert_fail(payload, "complete is missing")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val"]["complete"] = False
    assert_fail(payload, "complete must be true")
    payload = valid_payload()
    del payload["checkpoints"][0]["full_test"]["truncated"]
    assert_fail(payload, "truncated is missing")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val"]["n_tokens"] = 12
    assert_fail(payload, "n_tokens", "4999999")


def test_hardcoded_claim():
    payload = valid_payload()
    payload["checkpoints"][0]["claimed_sampled_val_bpc"] = 9.99
    assert_fail(payload, "hard-coded published claim")


def test_tf32_cublas_override():
    payload = valid_payload()
    payload["tf32"]["env"]["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    assert_fail(payload, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")


def test_tf32_nvidia_override():
    payload = valid_payload()
    payload["tf32"]["env"]["NVIDIA_TF32_OVERRIDE"] = "1"
    assert_fail(payload, "NVIDIA_TF32_OVERRIDE")


def test_tf32_cuda_matmul_true():
    payload = valid_payload()
    payload["tf32"]["cuda_matmul_allow_tf32"] = True
    assert_fail(payload, "cuda_matmul_allow_tf32")


def test_tf32_cudnn_true():
    payload = valid_payload()
    payload["tf32"]["cudnn_allow_tf32"] = True
    assert_fail(payload, "cudnn_allow_tf32")


def test_tf32_precision_high():
    payload = valid_payload()
    payload["tf32"]["float32_matmul_precision"] = "high"
    assert_fail(payload, "float32_matmul_precision", "highest")


def test_tf32_missing_block():
    payload = valid_payload()
    del payload["tf32"]
    assert_fail(payload, "tf32 block is missing")


def _set_all_parity(payload, max_abs):
    for row in payload["checkpoints"]:
        row["cpu_gpu_parity"]["max_abs_diff"] = max_abs


def test_parity_pass_band():
    payload = valid_payload()
    _set_all_parity(payload, 5e-5)
    result = evaluate_gate(payload)
    assert result["status"] == "pass", result
    assert result["exit_code"] == EXIT_PASS


def test_parity_hold_band():
    payload = valid_payload()
    _set_all_parity(payload, 5e-4)
    result = assert_status(payload, "hold", "PARITY HOLD")
    assert result["exit_code"] == EXIT_PARITY_HOLD
    assert result["ok"] is False


def test_parity_fail_band():
    payload = valid_payload()
    _set_all_parity(payload, 5e-3)
    assert_fail(payload, "FAIL band")


def test_parity_boundary_pass():
    payload = valid_payload()
    _set_all_parity(payload, 1e-4)
    result = evaluate_gate(payload)
    assert result["status"] == "pass", result
    assert result["exit_code"] == EXIT_PASS


def test_parity_boundary_hold():
    payload = valid_payload()
    _set_all_parity(payload, 1e-3)
    result = assert_status(payload, "hold", "HOLD")
    assert result["exit_code"] == EXIT_PARITY_HOLD


def test_parity_nan():
    payload = valid_payload()
    payload["checkpoints"][0]["cpu_gpu_parity"]["max_abs_diff"] = math.nan
    assert_fail(payload, "non-finite")


def test_parity_ignores_warn_flag():
    payload = valid_payload()
    _set_all_parity(payload, 5e-5)
    for row in payload["checkpoints"]:
        row["cpu_gpu_parity"]["warn"] = True
    result = evaluate_gate(payload)
    assert result["status"] == "pass", result


def test_export_allowlist_and_model_paths(tmp_path: Path | None = None):
    base = tmp_path or Path("/tmp/gate-tf32-export-test")
    src = base / "results"
    dest = base / "artifacts"
    if src.exists():
        import shutil
        shutil.rmtree(base)
    src.mkdir(parents=True)
    (src / "transformer").mkdir()
    (src / "ssm").mkdir()
    for name in ("scores.json", "device_report.json", "provenance.md", "gate.json", "gate.txt"):
        (src / name).write_text("{}\n" if name.endswith(".json") else "ok\n", encoding="utf-8")
    (src / "transformer" / "progress.json").write_text("{}\n", encoding="utf-8")
    (src / "transformer" / "wrapper_timing.json").write_text("{}\n", encoding="utf-8")
    (src / "ssm" / "progress.json").write_text("{}\n", encoding="utf-8")
    (src / "ssm" / "wrapper_timing.json").write_text("{}\n", encoding="utf-8")
    copied = export(src, dest)
    assert "scores.json" in copied
    assert "transformer/progress.json" in copied
    assert "ssm/wrapper_timing.json" in copied


def test_export_refuses_symlink_and_escape(tmp_path: Path | None = None):
    import shutil
    base = tmp_path or Path("/tmp/gate-tf32-export-refuse")
    if base.exists():
        shutil.rmtree(base)
    src = base / "results"
    src.mkdir(parents=True)
    outside = base / "secret.json"
    outside.write_text("nope\n", encoding="utf-8")
    (src / "scores.json").symlink_to(outside)
    try:
        resolve_inside(src, "scores.json")
        raise AssertionError("symlink should have been refused")
    except SystemExit as exc:
        assert "symlink" in str(exc).lower()
    try:
        resolve_inside(src, "../secret.json")
        raise AssertionError("escape should have been refused")
    except SystemExit as exc:
        assert "outside" in str(exc).lower() or "unsafe" in str(exc).lower()


def main() -> int:
    tests = [
        test_clean_pass,
        test_nan_rejects,
        test_missing_ckpt,
        test_duplicate_ckpt,
        test_smoke_flag,
        test_parity_skipped,
        test_cpu_device,
        test_wrong_threshold,
        test_full_metrics_complete_and_tokens,
        test_hardcoded_claim,
        test_tf32_cublas_override,
        test_tf32_nvidia_override,
        test_tf32_cuda_matmul_true,
        test_tf32_cudnn_true,
        test_tf32_precision_high,
        test_tf32_missing_block,
        test_parity_pass_band,
        test_parity_hold_band,
        test_parity_fail_band,
        test_parity_boundary_pass,
        test_parity_boundary_hold,
        test_parity_nan,
        test_parity_ignores_warn_flag,
        test_export_allowlist_and_model_paths,
        test_export_refuses_symlink_and_escape,
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
