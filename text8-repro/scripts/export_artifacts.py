#!/usr/bin/env python3
"""Copy only reviewed scoring artifacts into a committable folder.

Allowlist: scores.json, device_report.json, provenance.md, gate.json, gate.txt.
Refuses data, weights, secrets, or any other filename.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ALLOWLIST = {
    "scores.json",
    "device_report.json",
    "provenance.md",
    "gate.json",
    "gate.txt",
}
REQUIRED = ("scores.json", "device_report.json", "provenance.md", "gate.json")
REFUSE_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt",
    ".zip", ".tar", ".gz",
}
REFUSE_NAMES = {".env", ".env.example", ".netrc", "id_rsa", "secrets"}


def is_refused(path: Path) -> str | None:
    name = path.name
    if name in REFUSE_NAMES or name.startswith(".env"):
        return f"{name} looks like an env or secret file"
    if name.startswith("id_") or "credential" in name.lower() or "secret" in name.lower():
        return f"{name} looks like a secret"
    if path.suffix.lower() in REFUSE_SUFFIXES:
        return f"{name} looks like a weight or archive"
    if name not in ALLOWLIST:
        return f"{name} is not on the artifact allowlist"
    return None


def export(src: Path, dest: Path) -> list[str]:
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    # Never walk into data/weights; only consider top-level allowlisted names.
    for name in sorted(ALLOWLIST):
        candidate = src / name
        if not candidate.exists():
            continue
        reason = is_refused(candidate)
        if reason:
            raise SystemExit(f"refusing to copy {candidate}: {reason}")
        if candidate.is_symlink():
            raise SystemExit(f"refusing to copy symlink {candidate}")
        if not candidate.is_file():
            raise SystemExit(f"refusing to copy non-file {candidate}")
        target = dest / name
        shutil.copy2(candidate, target)
        copied.append(name)
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
    extra = [path.name for path in args.src.glob("*") if path.name not in ALLOWLIST and path.name not in {".", ".."}]
    refused_extras = [name for name in extra if is_refused(args.src / name)]
    # Extra files stay behind on purpose; report them but do not copy.
    if extra:
        print(f"leaving non-allowlisted files in place (not copied): {sorted(extra)}")
    copied = export(args.src, args.dest)
    print(f"exported {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
