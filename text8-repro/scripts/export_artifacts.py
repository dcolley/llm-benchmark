#!/usr/bin/env python3
"""Copy only reviewed scoring artifacts into a committable folder.

Allowlist (top-level): scores.json, device_report.json, provenance.md,
gate.json, gate.txt.

Per-model run records are copied only from these explicit relative paths
under the results directory (no globs):
  transformer/progress.json
  transformer/wrapper_timing.json
  ssm/progress.json
  ssm/wrapper_timing.json

Refuses data, weights, secrets, symlinks, and any path that resolves
outside the results directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TOP_LEVEL = (
    "scores.json",
    "device_report.json",
    "provenance.md",
    "gate.json",
    "gate.txt",
)
MODEL_PATHS = (
    "transformer/progress.json",
    "transformer/wrapper_timing.json",
    "ssm/progress.json",
    "ssm/wrapper_timing.json",
)
REQUIRED = ("scores.json", "device_report.json", "provenance.md", "gate.json")
REFUSE_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt",
    ".zip", ".tar", ".gz",
}
REFUSE_NAMES = {".env", ".env.example", ".netrc", "id_rsa", "secrets"}


def is_refused_name(name: str) -> str | None:
    if name in REFUSE_NAMES or name.startswith(".env"):
        return f"{name} looks like an env or secret file"
    if name.startswith("id_") or "credential" in name.lower() or "secret" in name.lower():
        return f"{name} looks like a secret"
    suffix = Path(name).suffix.lower()
    if suffix in REFUSE_SUFFIXES:
        return f"{name} looks like a weight or archive"
    return None


def resolve_inside(src: Path, rel: str) -> Path | None:
    """Resolve rel under src without following symlinks or leaving src."""
    rel_path = Path(rel)
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        raise SystemExit(f"refusing unsafe path outside results dir: {rel}")
    if src.is_symlink():
        raise SystemExit(f"refusing symlink {src}")
    current = src
    for part in rel_path.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"refusing to copy symlink {current}")
    if not current.exists():
        return None
    src_resolved = src.resolve()
    try:
        current.resolve().relative_to(src_resolved)
    except ValueError:
        raise SystemExit(f"refusing path outside results dir: {rel}")
    if not current.is_file():
        raise SystemExit(f"refusing to copy non-file {current}")
    reason = is_refused_name(current.name)
    if reason:
        raise SystemExit(f"refusing to copy {current}: {reason}")
    return current


def export(src: Path, dest: Path) -> list[str]:
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in (*TOP_LEVEL, *MODEL_PATHS):
        candidate = resolve_inside(src, rel)
        if candidate is None:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, target, follow_symlinks=False)
        copied.append(rel)
        print(f"copied {candidate} -> {target}")

    missing_required = [name for name in REQUIRED if not (dest / name).is_file()]
    if missing_required:
        raise SystemExit(f"export incomplete, missing required artifacts: {missing_required}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("results"),
                        help="Ignored results directory produced by scoring/provenance/gate")
    parser.add_argument("--dest", type=Path, default=Path("artifacts"),
                        help="Committed folder that may be included in a follow-up PR")
    args = parser.parse_args()
    copied = export(args.src, args.dest)
    print(f"exported {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
