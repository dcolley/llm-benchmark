#!/usr/bin/env python3
"""Convert a training latest.pt into the model-only format score.py accepts.

The published runners in Marx's reproduction (HF MarxistLeninist/phasemesh-text8-comparison @
2eb1bef / ashhart/PhaseMesh @ 3ae43ff) write two checkpoint shapes:

* latest.pt (run_matched.py:75-78, run_ssm.py:82-85):
    spec, model, optimizer, step, best, history, last_eval,
    trained_seconds, rng_cpu, rng_cuda
  ``spec['model']`` is ``dataclasses.asdict(cfg)``. The live config
  dataclass is not pickled.
* best.pt (run_matched.py:110, run_ssm.py:117):
    model_state, cfg, model_type ('transformer' | 'phasessm'), step, val_bpc

score.py:132-140 loads with torch.load(..., weights_only=True) and requires
``model_state`` plus a model-only ``cfg`` dict. That already works for
best.pt. Training latest.pt cannot be scored as-is because the weights live
under ``model`` rather than ``model_state``.

This script reads a latest.pt, drops optimizer and RNG state, and writes a
best.pt-shaped file plus export metadata. It does not change the training
scripts or scoring numerics.

Loading
=======
The runner payload is plain Python containers plus tensors (dicts, lists,
numbers, strings, bools, and CPU uint8 RNG tensors). A synthetic latest.pt
in that exact shape loads with ``torch.load(..., weights_only=True)`` and
needs no ``torch.serialization.add_safe_globals`` / ``safe_globals``
allowlist. This script therefore loads with ``weights_only=True`` and does
not register any safe globals.

If a file truly cannot be loaded that way (for example a future runner that
pickles a config dataclass or RNG object torch does not allowlist), pass
``--expect-sha256 <hex>`` matching the file. Only after that digest matches
will the script fall back to ``weights_only=False``. There is no unrestricted
unsafe load path.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


class ExportError(Exception):
    """User-facing conversion failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ExportError(f"expected a 64-character hex SHA256, got {value!r}")
    return text


def load_training_checkpoint(
    path: Path,
    expect_sha256: str | None = None,
) -> tuple[dict, str, str]:
    """Load a runner latest.pt. Prefer weights_only=True; no safe globals."""
    import torch

    digest = sha256_file(path)
    if expect_sha256 is not None:
        wanted = normalize_sha256(expect_sha256)
        if digest != wanted:
            raise ExportError(
                f"source sha256 mismatch: got {digest}, expected {wanted}"
            )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        how = "weights_only=True"
    except Exception as exc:
        if expect_sha256 is None:
            raise ExportError(
                "torch.load(..., weights_only=True) failed for this file "
                f"({type(exc).__name__}: {exc}). "
                "The published runner latest.pt shape (dicts/lists/tensors "
                "only; config stored via asdict under spec['model']) is "
                "expected to load without allowlisted safe globals. "
                "To permit a restricted weights_only=False fallback, pass "
                f"--expect-sha256 matching this file ({digest})."
            ) from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
        how = "weights_only=False (SHA256-matched fallback)"

    if not isinstance(payload, dict):
        raise ExportError(f"{path}: expected a dict checkpoint, got {type(payload)}")
    return payload, digest, how


def infer_model_type(spec: dict, cli_model: str | None) -> str:
    if cli_model == "transformer":
        return "transformer"
    if cli_model == "ssm":
        return "phasessm"
    if not isinstance(spec, dict):
        raise ExportError("latest.pt spec must be a dict; pass --model transformer|ssm")
    cfg = spec.get("model")
    if isinstance(cfg, dict):
        if "state_dim" in cfg or "ssm_backend" in cfg:
            return "phasessm"
        if "n_heads" in cfg or "pos_enc" in cfg:
            return "transformer"
    raise ExportError("cannot infer model type from spec['model']; pass --model transformer|ssm")


def extract_cfg(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ExportError("latest.pt is missing a spec dict")
    cfg = spec.get("model")
    if not isinstance(cfg, dict):
        raise ExportError("latest.pt spec['model'] must be a dict (asdict of the config)")
    return cfg


def extract_val_bpc(payload: dict):
    step = payload.get("step")
    history = payload.get("history") or []
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict) and row.get("step") == step and "val_bpc" in row:
                return row["val_bpc"]
    return None


def to_score_payload(
    latest: dict,
    *,
    source_path: Path,
    source_sha256: str,
    model_type: str,
    torch_version: str,
) -> dict:
    if "model" not in latest:
        raise ExportError(
            f"{source_path}: missing 'model' state_dict "
            f"(keys={sorted(latest)})"
        )
    if "spec" not in latest:
        raise ExportError(f"{source_path}: missing 'spec' (keys={sorted(latest)})")
    if "step" not in latest:
        raise ExportError(f"{source_path}: missing 'step' (keys={sorted(latest)})")
    return {
        "model_state": latest["model"],
        "cfg": extract_cfg(latest["spec"]),
        "model_type": model_type,
        "step": latest["step"],
        "val_bpc": extract_val_bpc(latest),
        "source_basename": source_path.name,
        "source_sha256": source_sha256,
        "torch_version": torch_version,
    }


def write_atomic(payload: dict, dest: Path) -> None:
    import torch

    if dest.exists() or dest.is_symlink():
        raise ExportError(f"refusing to overwrite existing destination {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists() and not tmp.is_symlink():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def export_checkpoint(
    src: Path,
    dest: Path,
    *,
    model: str | None = None,
    expect_sha256: str | None = None,
) -> dict:
    import torch

    src = src.resolve()
    dest = dest.resolve()
    if not src.is_file():
        raise ExportError(f"source not found: {src}")
    latest, digest, how = load_training_checkpoint(src, expect_sha256=expect_sha256)
    model_type = infer_model_type(latest.get("spec"), model)
    out = to_score_payload(
        latest,
        source_path=src,
        source_sha256=digest,
        model_type=model_type,
        torch_version=str(torch.__version__),
    )
    write_atomic(out, dest)
    return {
        "dest": str(dest),
        "loader": how,
        "safe_globals": [],
        "source_basename": out["source_basename"],
        "source_sha256": out["source_sha256"],
        "step": out["step"],
        "model_type": out["model_type"],
        "torch_version": out["torch_version"],
        "val_bpc": out["val_bpc"],
        "keys": sorted(out),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--src", type=Path, required=True, help="training latest.pt")
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="model-only checkpoint path (must not already exist)",
    )
    parser.add_argument(
        "--model",
        choices=("transformer", "ssm"),
        help="required only if spec['model'] does not uniquely identify the architecture",
    )
    parser.add_argument(
        "--expect-sha256",
        dest="expect_sha256",
        default=None,
        help="optional SHA256 of --src; always verified if given, and required "
        "to enable a weights_only=False fallback",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        info = export_checkpoint(
            args.src,
            args.dest,
            model=args.model,
            expect_sha256=args.expect_sha256,
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        "wrote {dest}\n"
        "  loader: {loader}\n"
        "  safe_globals: {safe_globals}\n"
        "  model_type: {model_type}\n"
        "  step: {step}\n"
        "  val_bpc: {val_bpc}\n"
        "  source_basename: {source_basename}\n"
        "  source_sha256: {source_sha256}\n"
        "  torch_version: {torch_version}\n"
        "  keys: {keys}".format(**info)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
