#!/usr/bin/env python3
"""Fail-closed tests for check_score_gate and export_artifacts."""
from __future__ import annotations

import json
import os
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
from export_artifacts import (  # noqa: E402
    artifacts_root,
    export,
    main as export_main,
    prepare_fresh_dest,
    resolve_inside,
)


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
    payload = valid_payload()
    payload["checkpoints"][0]["full_val"]["n_tokens"] = True
    assert_fail(payload, "n_tokens")
    payload = valid_payload()
    payload["checkpoints"][0]["full_val"]["n_tokens"] = float(EXPECTED_FULL_TOKENS["val"])
    assert_fail(payload, "n_tokens", "integer")


def test_n_tokens_rejects_bool_and_float():
    payload = valid_payload()
    payload["checkpoints"][2]["full_test"]["n_tokens"] = False
    assert_fail(payload, "n_tokens")


def test_negative_max_abs():
    payload = valid_payload()
    payload["checkpoints"][0]["cpu_gpu_parity"]["max_abs_diff"] = -1e-6
    assert_fail(payload, "negative")


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


def _write_export_fixtures(base: Path):
    import shutil
    if base.exists():
        shutil.rmtree(base)
    src = base / "results"
    runs = base / "runs"
    src.mkdir(parents=True)
    (runs / "transformer").mkdir(parents=True)
    (runs / "ssm").mkdir(parents=True)
    for name in ("scores.json", "device_report.json", "provenance.md", "gate.json", "gate.txt"):
        (src / name).write_text("{0}\n".format("{}" if name.endswith(".json") else "ok"), encoding="utf-8")
    for model in ("transformer", "ssm"):
        (runs / model / "progress.json").write_text("{}\n", encoding="utf-8")
        (runs / model / "wrapper_timing.json").write_text("{}\n", encoding="utf-8")
    return src, runs


def _test_dest(name: str) -> Path:
    import shutil
    dest = artifacts_root() / name
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    return dest


def test_export_happy_path():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-happy"))
    dest = _test_dest("test-reviewed-complete")
    copied = export(src, dest, runs, stage="complete")
    assert "scores.json" in copied
    assert "runs/transformer/progress.json" in copied
    assert "runs/ssm/wrapper_timing.json" in copied
    assert (dest / "runs" / "ssm" / "wrapper_timing.json").is_file()
    _test_dest("test-reviewed-complete")


def test_export_missing_progress():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-missing"))
    (runs / "transformer" / "progress.json").unlink()
    dest = _test_dest("test-reviewed-complete-missing")
    try:
        export(src, dest, runs, stage="complete")
        raise AssertionError("missing progress should have failed")
    except SystemExit as exc:
        assert "missing required" in str(exc).lower()
        assert "transformer/progress.json" in str(exc)
    finally:
        _test_dest("test-reviewed-complete-missing")


def test_export_path_escape():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-escape"))
    try:
        resolve_inside(runs, "../results/scores.json", required=True, label="runs")
        raise AssertionError("escape should have been refused")
    except SystemExit as exc:
        assert "outside" in str(exc).lower() or "unsafe" in str(exc).lower()


def test_export_symlink_dest():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-symdest"))
    target = Path("/tmp/gate-export-symdest/target")
    dest = _test_dest("test-reviewed-symdest")
    target.mkdir()
    dest.symlink_to(target)
    try:
        export(src, dest, runs, stage="complete")
        raise AssertionError("symlink dest should have been refused")
    except SystemExit as exc:
        assert "symlink" in str(exc).lower()
    finally:
        _test_dest("test-reviewed-symdest")


def test_export_stale_dest():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-stale"))
    dest = _test_dest("test-reviewed-stale")
    dest.mkdir()
    (dest / "old.txt").write_text("stale\n", encoding="utf-8")
    try:
        export(src, dest, runs, stage="complete")
        raise AssertionError("stale dest should have been refused")
    except SystemExit as exc:
        assert "non-empty" in str(exc).lower()
    finally:
        _test_dest("test-reviewed-stale")


def test_export_refuses_symlink_and_escape():
    src, _runs = _write_export_fixtures(Path("/tmp/gate-export-refuse"))
    outside = Path("/tmp/gate-export-refuse/secret.json")
    outside.write_text("nope\n", encoding="utf-8")
    scores = src / "scores.json"
    scores.unlink()
    scores.symlink_to(outside)
    try:
        resolve_inside(src, "scores.json")
        raise AssertionError("symlink should have been refused")
    except SystemExit as exc:
        assert "symlink" in str(exc).lower()


def test_export_scoring_without_runs_dir():
    src, _runs = _write_export_fixtures(Path("/tmp/gate-export-scoring"))
    dest = _test_dest("test-reviewed-scoring")
    saved = os.environ.pop("RUNS_DIR", None)
    try:
        copied = export(src, dest, None, stage="scoring")
        assert "scores.json" in copied
        assert "gate.json" in copied
        assert not any(name.startswith("runs/") for name in copied)
        assert not (dest / "runs").exists()
        rc = export_main(["--stage", "scoring", "--src", str(src), "--dest", str(_test_dest("test-reviewed-scoring-cli"))])
        assert rc == 0
    finally:
        if saved is not None:
            os.environ["RUNS_DIR"] = saved
        _test_dest("test-reviewed-scoring")
        _test_dest("test-reviewed-scoring-cli")


def test_export_scoring_refuses_runs_src():
    src, runs = _write_export_fixtures(Path("/tmp/gate-export-scoring-runs"))
    dest = _test_dest("test-reviewed-scoring-runs")
    try:
        export_main([
            "--stage", "scoring",
            "--src", str(src),
            "--dest", str(dest),
            "--runs-src", str(runs),
        ])
        raise AssertionError("scoring should have refused --runs-src")
    except SystemExit as exc:
        assert "--runs-src" in str(exc)
        assert "not allowed" in str(exc).lower()
    finally:
        _test_dest("test-reviewed-scoring-runs")


def test_export_symlink_ancestor():
    src, _runs = _write_export_fixtures(Path("/tmp/gate-export-anc"))
    link = _test_dest("test-link-ancestor")
    target = Path("/tmp/gate-export-anc/target")
    target.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    dest = link / "reviewed"
    try:
        export(src, dest, None, stage="scoring")
        raise AssertionError("symlink ancestor should have been refused")
    except SystemExit as exc:
        message = str(exc).lower()
        assert "symlink" in message
        assert "ancestor" in message
    finally:
        if dest.exists() and not dest.is_symlink():
            import shutil
            shutil.rmtree(dest)
        _test_dest("test-link-ancestor")


def test_export_empty_dest_ok():
    src, _runs = _write_export_fixtures(Path("/tmp/gate-export-empty"))
    dest = _test_dest("test-reviewed-empty")
    dest.mkdir()
    try:
        copied = export(src, dest, None, stage="scoring")
        assert "scores.json" in copied
        assert (dest / "scores.json").is_file()
    finally:
        _test_dest("test-reviewed-empty")


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
        test_n_tokens_rejects_bool_and_float,
        test_negative_max_abs,
        test_export_happy_path,
        test_export_missing_progress,
        test_export_path_escape,
        test_export_symlink_dest,
        test_export_stale_dest,
        test_export_refuses_symlink_and_escape,
        test_export_scoring_without_runs_dir,
        test_export_scoring_refuses_runs_src,
        test_export_symlink_ancestor,
        test_export_empty_dest_ok,
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
