"""Log CUDA device capability, arch list, and TF32 settings (sm_121 is GB10)."""
from __future__ import annotations

import json
import os
import sys


TF32_ENV_VARS = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _truthy_override_env() -> list[str]:
    return [name for name in TF32_ENV_VARS if _env_enabled(name)]


def snapshot_tf32(torch_module=None) -> dict:
    """Record TF32-related env vars and torch flags without changing them."""
    snap = {
        "env": {name: os.environ.get(name) for name in TF32_ENV_VARS},
        "override_env_active": bool(_truthy_override_env()),
        "override_env_names": _truthy_override_env(),
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
        "float32_matmul_precision": None,
    }
    torch_mod = torch_module
    if torch_mod is None:
        try:
            import torch as torch_mod
        except ImportError:
            snap["torch_available"] = False
            return snap
    snap["torch_available"] = True
    cuda = getattr(getattr(torch_mod, "backends", None), "cuda", None)
    matmul = getattr(cuda, "matmul", None) if cuda is not None else None
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        snap["cuda_matmul_allow_tf32"] = bool(matmul.allow_tf32)
    cudnn = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        snap["cudnn_allow_tf32"] = bool(cudnn.allow_tf32)
    getter = getattr(torch_mod, "get_float32_matmul_precision", None)
    if callable(getter):
        snap["float32_matmul_precision"] = getter()
    return snap


def disable_tf32_for_scoring(torch_module=None) -> dict:
    """Force TF32 off for independent scoring. Call before any CUDA work.

    NGC images may set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which overrides
    the torch flags. The score compose service must set that env to 0; if it
    is still 1 here we warn loudly because the flags may not take effect.
    """
    torch_mod = torch_module
    if torch_mod is None:
        import torch as torch_mod

    warnings = []
    active = _truthy_override_env()
    if active:
        msg = (
            "WARNING: TF32 override environment variable(s) still enabled: "
            + ", ".join(f"{name}={os.environ.get(name)!r}" for name in active)
            + ". NGC sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 by default and that "
            "overrides torch.backends.*.allow_tf32. Scoring requires both "
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 and NVIDIA_TF32_OVERRIDE=0."
        )
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)
        warnings.append(msg)

    cuda = getattr(getattr(torch_mod, "backends", None), "cuda", None)
    matmul = getattr(cuda, "matmul", None) if cuda is not None else None
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = False
    cudnn = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = False
    setter = getattr(torch_mod, "set_float32_matmul_precision", None)
    if callable(setter):
        setter("highest")

    snap = snapshot_tf32(torch_mod)
    snap["policy"] = "disabled_for_scoring"
    snap["warnings"] = warnings
    return snap


def collect_device_report(requested_device: str = "cuda", *, tf32_policy: str = "record_only") -> dict:
    """Collect device metadata.

    tf32_policy:
      - ``disabled_for_scoring``: set TF32-off flags before any CUDA query.
      - ``record_only``: do not change flags (used by training wrappers).
    """
    report = {"requested_device": requested_device, "tf32_policy": tf32_policy}
    try:
        import torch
    except ImportError:
        report.update(torch_available=False, warning="torch is not installed")
        report["tf32"] = snapshot_tf32()
        report["tf32"]["policy"] = tf32_policy
        return report

    if tf32_policy == "disabled_for_scoring":
        tf32 = disable_tf32_for_scoring(torch)
    else:
        tf32 = snapshot_tf32(torch)
        tf32["policy"] = "unmodified"
        tf32["note"] = (
            "Training wrappers do not set or clear TF32. Published "
            "run_matched.py, run_ssm.py, and train_core.py also leave TF32 "
            "untouched. Record these values so a container image can be "
            "compared with a stock pip install (for example on an RTX 3090)."
        )
    report["tf32"] = tf32

    report.update(
        torch_available=True,
        torch_version=torch.__version__,
        cuda_built=getattr(torch.version, "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
    )
    arch_list = []
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception as exc:  # pragma: no cover - defensive
            report["arch_list_error"] = repr(exc)
    report["arch_list"] = arch_list

    if torch.cuda.is_available():
        index = 0
        major, minor = torch.cuda.get_device_capability(index)
        sm = f"sm_{major}{minor}"
        report.update(
            device_index=index,
            device_name=torch.cuda.get_device_name(index),
            device_capability={"major": major, "minor": minor, "sm": sm},
            capability_in_arch_list=sm in arch_list,
        )
        if sm not in arch_list:
            report["warning"] = (
                f"Device {sm} is not listed in torch.cuda.get_arch_list()={arch_list}. "
                "GB10 reports sm_121; kernels compiled without that arch may fail "
                "or silently fall back."
            )
    elif str(requested_device).startswith("cuda"):
        report["warning"] = "CUDA was requested but torch.cuda.is_available() is False"
    return report


def log_device_report(requested_device: str = "cuda", *, tf32_policy: str = "record_only") -> dict:
    report = collect_device_report(requested_device, tf32_policy=tf32_policy)
    print(json.dumps({"event": "device_report", **report}, indent=2), flush=True)
    if report.get("warning"):
        print(f"WARNING: {report['warning']}", file=sys.stderr, flush=True)
    return report


if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    policy = sys.argv[2] if len(sys.argv) > 2 else "record_only"
    log_device_report(requested, tf32_policy=policy)
