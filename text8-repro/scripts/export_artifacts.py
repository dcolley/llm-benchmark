#!/usr/bin/env python3
"""Copy only reviewed scoring artifacts into a committable folder.

Scoring outputs are read from --src (the ignored results directory):
  scores.json, device_report.json, provenance.md, gate.json, gate.txt

Per-model training records are read from --runs-src (RUNS_DIR from env/.env,
matching train_common.py and compose --out-dir "$RUNS_DIR/{transformer,ssm}"):
  transformer/progress.json
  transformer/wrapper_timing.json
  ssm/progress.json
  ssm/wrapper_timing.json

Those four run files are required. A missing file fails instead of being
skipped. Paths are explicit (no globs), symlinks are refused, and resolved
paths must stay inside --src or --runs-src respectively.

--dest must be a fresh trusted directory: it is created without following
symlinks, and is refused if it already exists and is non-empty or is/contains
a symlink. Files are written with O_EXCL and O_NOFOLLOW.
"""
from __future__ import annotations

import argparse
import os
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
REQUIRED_SRC = TOP_LEVEL[:4]  # gate.txt is optional
REFUSE_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt",
    ".zip", ".tar", ".gz",
}
REFUSE_NAMES = {".env", ".env.example", ".netrc", "id_rsa", "secrets"}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def load_export_env() -> None:
    root = Path(__file__).resolve().parent.parent
    load_env_file(root / ".env")
    load_env_file(Path.cwd() / ".env")


def is_refused_name(name: str) -> str | None:
    if name in REFUSE_NAMES or name.startswith(".env"):
        return f"{name} looks like an env or secret file"
    if name.startswith("id_") or "credential" in name.lower() or "secret" in name.lower():
        return f"{name} looks like a secret"
    suffix = Path(name).suffix.lower()
    if suffix in REFUSE_SUFFIXES:
        return f"{name} looks like a weight or archive"
    return None


def path_or_child_is_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    if not path.is_dir():
        return False
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        if root_path.is_symlink():
            return True
        for name in (*dirnames, *filenames):
            if (root_path / name).is_symlink():
                return True
    return False


def resolve_inside(src: Path, rel: str, *, required: bool = False, label: str = "source") -> Path | None:
    """Resolve rel under src without following symlinks or leaving src."""
    rel_path = Path(rel)
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        raise SystemExit(f"refusing unsafe path outside {label} dir: {rel}")
    if src.is_symlink():
        raise SystemExit(f"refusing symlink {src}")
    current = src
    for part in rel_path.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"refusing to copy symlink {current}")
    if not current.exists():
        if required:
            raise SystemExit(f"missing required {label} artifact: {rel} under {src}")
        return None
    try:
        current.resolve().relative_to(src.resolve())
    except ValueError:
        raise SystemExit(f"refusing path outside {label} dir: {rel}")
    if not current.is_file():
        raise SystemExit(f"refusing to copy non-file {current}")
    reason = is_refused_name(current.name)
    if reason:
        raise SystemExit(f"refusing to copy {current}: {reason}")
    return current


def mkdir_nofollow(path: Path) -> None:
    if path.exists():
        if path.is_symlink():
            raise SystemExit(f"refusing symlink {path}")
        if not path.is_dir():
            raise SystemExit(f"refusing non-directory {path}")
        return
    parent = path.parent
    if parent != path:
        mkdir_nofollow(parent)
        if parent.is_symlink():
            raise SystemExit(f"refusing symlink parent {parent}")
    os.mkdir(path)


def prepare_fresh_dest(dest: Path) -> None:
    if dest.is_symlink():
        raise SystemExit(f"refusing symlink destination {dest}")
    if dest.exists():
        if not dest.is_dir():
            raise SystemExit(f"refusing destination that is not a directory: {dest}")
        if path_or_child_is_symlink(dest):
            raise SystemExit(f"refusing destination that contains a symlink: {dest}")
        if any(dest.iterdir()):
            raise SystemExit(
                f"refusing non-empty destination {dest}; pass a fresh empty directory"
            )
        return
    mkdir_nofollow(dest)
    if dest.is_symlink():
        raise SystemExit(f"refusing symlink destination {dest}")


def write_exclusive(src: Path, dest_file: Path) -> None:
    mkdir_nofollow(dest_file.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if getattr(os, "O_NOFOLLOW", 0):
        flags |= os.O_NOFOLLOW
    fd = os.open(dest_file, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as out:
            fd = None
            out.write(src.read_bytes())
    finally:
        if fd is not None:
            os.close(fd)


def export(src: Path, dest: Path, runs_src: Path) -> list[str]:
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    if not runs_src.is_dir():
        raise SystemExit(f"runs directory not found: {runs_src}")
    planned = []
    for rel in TOP_LEVEL:
        required = rel in REQUIRED_SRC
        candidate = resolve_inside(src, rel, required=required, label="results")
        if candidate is None:
            continue
        planned.append((rel, candidate, dest / rel))
    for rel in MODEL_PATHS:
        candidate = resolve_inside(runs_src, rel, required=True, label="runs")
        planned.append((f"runs/{rel}", candidate, dest / "runs" / rel))
    prepare_fresh_dest(dest)
    copied = []
    for rel, candidate, target in planned:
        write_exclusive(candidate, target)
        copied.append(rel)
        print(f"copied {candidate} -> {target}")
    missing_required = [name for name in REQUIRED_SRC if not (dest / name).is_file()]
    if missing_required:
        raise SystemExit(f"export incomplete, missing required artifacts: {missing_required}")
    return copied


def default_runs_src() -> Path | None:
    raw = os.environ.get("RUNS_DIR")
    if raw:
        return Path(raw)
    return None


def main() -> int:
    load_export_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(os.environ.get("RESULTS_DIR", "results")),
                        help="Ignored results directory produced by scoring/provenance/gate")
    parser.add_argument(
        "--runs-src",
        type=Path,
        default=default_runs_src(),
        help="Training output directory (default: RUNS_DIR from env/.env)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Fresh empty destination directory (created if missing; refused if non-empty or a symlink)",
    )
    args = parser.parse_args()
    if args.runs_src is None:
        raise SystemExit("RUNS_DIR is not set; pass --runs-src or set RUNS_DIR in .env")
    copied = export(args.src, args.dest, args.runs_src)
    print(f"exported {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
