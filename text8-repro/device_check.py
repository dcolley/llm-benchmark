"""Log CUDA device capability and the PyTorch arch list (sm_121 is GB10)."""
from __future__ import annotations

import json
import sys


def collect_device_report(requested_device: str = "cuda") -> dict:
    report = {"requested_device": requested_device}
    try:
        import torch
    except ImportError:
        report.update(torch_available=False, warning="torch is not installed")
        return report

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


def log_device_report(requested_device: str = "cuda") -> dict:
    report = collect_device_report(requested_device)
    print(json.dumps({"event": "device_report", **report}, indent=2), flush=True)
    if report.get("warning"):
        print(f"WARNING: {report['warning']}", file=sys.stderr, flush=True)
    return report


if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    log_device_report(requested)
