#!/usr/bin/env python3
"""Copy only reviewed artifacts into a committable folder under artifacts/.

Required --stage:

  scoring   copies scores.json, device_report.json, provenance.md, gate.json,
            and gate.txt from --src. Does not read RUNS_DIR and rejects
            --runs-src.

  complete  copies the scoring files plus the four training records from
            --runs-src (default RUNS_DIR from env/.env, matching train_common
            and compose): transformer|ssm progress.json and wrapper_timing.json.
            Those four files are required; a missing file fails the export.

Each stage needs its own fresh destination, for example:

  python scripts/export_artifacts.py --stage scoring --src "$RESULTS_DIR" \\
      --dest artifacts/reviewed-scoring
  python scripts/export_artifacts.py --stage complete --src "$RESULTS_DIR" \\
      --runs-src "$RUNS_DIR" --dest artifacts/reviewed-complete

Paths are explicit (no globs). Symlinks are refused, and resolved source
paths must stay inside --src or --runs-src respectively.

Destination hardening (both rules, the stricter combination):
  1. --dest must live under this tree's artifacts/ directory, validated
     component by component without following symlinks. Dest must be a
     subdirectory of artifacts/, not artifacts/ itself.
  2. Every ancestor of the absolute dest path is lstat'd from the filesystem
     root; any symlink ancestor is refused.

--dest may be created if missing. An existing dest is accepted only when it
is an empty real directory, is not a symlink, contains no symlinks, and all
ancestors pass the checks above. mkdir_nofollow creates a path and refuses
if that path already exists. Files are written with O_EXCL and O_NOFOLLOW.
"""
from __future__ import annotations

import argparse
import os
import stat
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
REQUIRED_SRC = TOP_LEVEL[:4]  # gate.txt is copied when the gate wrote it
REFUSE_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".onnx", ".ckpt",
    ".zip", ".tar", ".gz",
}
REFUSE_NAMES = {".env", ".env.example", ".netrc", "id_rsa", "secrets"}
STAGES = ("scoring", "complete")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_root() -> Path:
    return repo_root() / "artifacts"


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
    load_env_file(repo_root() / ".env")
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


def is_lstat_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def check_no_symlink_ancestors(path: Path) -> None:
    """lstat every ancestor of path from the filesystem root; refuse symlinks."""
    abs_path = Path(os.path.abspath(path))
    parts = abs_path.parts
    if not parts:
        raise SystemExit("refusing empty destination path")
    current = Path(parts[0])
    if is_lstat_symlink(current):
        raise SystemExit(f"refusing symlink ancestor {current}")
    for part in parts[1:-1]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # Remaining components cannot exist yet; they will be created.
            return
        if stat.S_ISLNK(mode):
            raise SystemExit(f"refusing symlink ancestor {current}")


def require_dest_under_artifacts(dest: Path) -> None:
    dest_abs = Path(os.path.abspath(dest))
    root = Path(os.path.abspath(artifacts_root()))
    if dest_abs == root:
        raise SystemExit(f"destination must be a subdirectory of {root}, not {root} itself")
    try:
        rel = dest_abs.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"destination must live under {root}: {dest_abs}") from exc
    if any(part in ("..", "") for part in rel.parts):
        raise SystemExit(f"destination must live under {root}: {dest_abs}")
    current = root
    for part in rel.parts:
        current = current / part
        if current == dest_abs:
            break
        if is_lstat_symlink(current):
            raise SystemExit(f"refusing symlink ancestor {current}")


def mkdir_nofollow(path: Path) -> None:
    """Create path. Refuses if path already exists. Does not follow symlinks."""
    check_no_symlink_ancestors(path)
    if path.is_symlink() or path.exists():
        raise SystemExit(f"refusing mkdir on existing path {path}")
    parent = path.parent
    if parent != path:
        if is_lstat_symlink(parent):
            raise SystemExit(f"refusing symlink parent {parent}")
        if not parent.exists():
            mkdir_nofollow(parent)
        elif not parent.is_dir():
            raise SystemExit(f"refusing non-directory parent {parent}")
    os.mkdir(path)


def prepare_fresh_dest(dest: Path) -> None:
    require_dest_under_artifacts(dest)
    check_no_symlink_ancestors(dest)
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
    parent = dest_file.parent
    if not parent.exists():
        mkdir_nofollow(parent)
    else:
        check_no_symlink_ancestors(parent)
        if parent.is_symlink():
            raise SystemExit(f"refusing symlink parent {parent}")
        if not parent.is_dir():
            raise SystemExit(f"refusing non-directory parent {parent}")
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


def export(src: Path, dest: Path, runs_src: Path | None = None, *, stage: str) -> list[str]:
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; expected scoring or complete")
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    planned = []
    for rel in TOP_LEVEL:
        required = rel in REQUIRED_SRC
        candidate = resolve_inside(src, rel, required=required, label="results")
        if candidate is None:
            continue
        planned.append((rel, candidate, dest / rel))
    if stage == "complete":
        if runs_src is None:
            raise SystemExit("complete stage requires --runs-src or RUNS_DIR")
        if not runs_src.is_dir():
            raise SystemExit(f"runs directory not found: {runs_src}")
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
    if stage == "complete":
        missing_runs = [f"runs/{rel}" for rel in MODEL_PATHS if not (dest / "runs" / rel).is_file()]
        if missing_runs:
            raise SystemExit(f"export incomplete, missing required artifacts: {missing_runs}")
    return copied


def default_runs_src() -> Path | None:
    raw = os.environ.get("RUNS_DIR")
    if raw:
        return Path(raw)
    return None


def flag_present(name: str, argv: list[str]) -> bool:
    return any(a == name or a.startswith(f"{name}=") for a in argv)


def resolve_runs_src(stage: str, runs_src: Path | None, argv: list[str] | None = None) -> Path | None:
    argv = sys.argv[1:] if argv is None else argv
    if stage == "scoring":
        if flag_present("--runs-src", argv):
            raise SystemExit("--runs-src is not allowed with --stage scoring")
        return None
    chosen = runs_src if runs_src is not None else default_runs_src()
    if chosen is None:
        raise SystemExit("RUNS_DIR is not set; pass --runs-src or set RUNS_DIR in .env")
    return chosen


def main(argv: list[str] | None = None) -> int:
    load_export_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES,
        help="scoring: results only. complete: results plus training records",
    )
    parser.add_argument("--src", type=Path, default=Path(os.environ.get("RESULTS_DIR", "results")),
                        help="Ignored results directory produced by scoring/provenance/gate")
    parser.add_argument(
        "--runs-src",
        type=Path,
        default=None,
        help="Training output directory (complete stage; default: RUNS_DIR from env/.env)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Fresh destination under artifacts/ (e.g. artifacts/reviewed-scoring)",
    )
    args = parser.parse_args(argv)
    cli = sys.argv[1:] if argv is None else argv
    runs_src = resolve_runs_src(args.stage, args.runs_src, cli)
    copied = export(args.src, args.dest, runs_src, stage=args.stage)
    print(f"exported {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
