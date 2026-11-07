#!/usr/bin/env python3
"""Score published PhaseMesh text8 checkpoints independently.

Loads each model-only checkpoint with the matching vendored code,
torch.load(..., weights_only=True), strict state_dict loading, and
finite-tensor checks. Evaluation is FP32.

Two evaluations are reported:
  * sampled_val: the published protocol (50 batches of 64, seq 1024) using
    ByteData.get_batch and seed --seed. Model-only checkpoints do not include
    RNG state, so this is a fresh seeded sample, not a replay of the
    in-training eval that produced the claimed scores.
  * full val/test: contiguous 90/5/5 split, non-overlapping windows of 1024
    with a padded remainder window so leftover tail tokens are included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


CLAIMS = {
    "transformer/best.pt": {
        "claimed_sampled_val_bpc": 1.2401772355061287,
        "claimed_step": 4000,
    },
    "transformer/final.pt": {
        "claimed_sampled_val_bpc": 1.257564388922874,
        "claimed_step": 5000,
    },
    "ssm/best.pt": {
        "claimed_sampled_val_bpc": 1.2638741508689288,
        "claimed_step": 4750,
    },
    "ssm/final.pt": {
        "claimed_sampled_val_bpc": 1.268647525903973,
        "claimed_step": 5000,
    },
}

EXPECTED_TEXT8_SHA256 = "6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7"
HF_REVISION_DEFAULT = "2eb1bef95b39e9745035044679f9d18b8ac3513a"
SOURCE_COMMIT_DEFAULT = "3ae43ffac6c0699857cada0865f156405fb95956"


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_vendor(vendor_dir: Path):
    vendor_dir = vendor_dir.resolve()
    if not vendor_dir.is_dir():
        raise FileNotFoundError(f"vendor directory not found: {vendor_dir}")
    stale = [name for name in sys.modules if name == "phase_ssm" or name.startswith("phase_ssm.")]
    for name in stale:
        del sys.modules[name]
    # Drop any previously inserted vendor path so the two copies cannot mix.
    sys.path[:] = [item for item in sys.path if Path(item).resolve() != vendor_dir]
    other_vendors = []
    for item in list(sys.path):
        resolved = Path(item).resolve()
        if resolved.name == "vendor" and resolved != vendor_dir:
            other_vendors.append(item)
    for item in other_vendors:
        sys.path.remove(item)
    sys.path.insert(0, str(vendor_dir))
    import phase_ssm.data as data_mod  # type: ignore
    import phase_ssm.model as model_mod  # type: ignore

    transformer_mod = None
    try:
        import phase_ssm.transformer as transformer_mod  # type: ignore
    except ImportError:
        transformer_mod = None
    return data_mod, model_mod, transformer_mod


def check_finite_state_dict(state, label: str) -> None:
    bad = []
    for key, tensor in state.items():
        if hasattr(tensor, "isfinite") and tensor.is_floating_point():
            if not bool(tensor.isfinite().all().item()):
                bad.append(key)
    if bad:
        raise RuntimeError(f"{label}: non-finite tensors: {bad[:20]}")


def build_model(payload: dict, model_mod, transformer_mod, expected_kind: str):
    cfg_dict = payload.get("cfg")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("checkpoint is missing a cfg dict")
    model_type = payload.get("model_type", expected_kind)
    if expected_kind == "transformer" or model_type == "transformer":
        if transformer_mod is None:
            raise RuntimeError("vendored transformer.py is required to load this checkpoint")
        cfg = transformer_mod.TransformerConfig(**cfg_dict)
        model = transformer_mod.TransformerLM(cfg)
        kind = "transformer"
    else:
        cfg = model_mod.PhaseSSMConfig(**cfg_dict)
        model = model_mod.PhaseSSMLM(cfg)
        kind = "phasessm"
    return model, cfg, kind


def load_checkpoint(path: Path, vendor_dir: Path, expected_kind: str, device: str):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a dict checkpoint, got {type(payload)}")
    if "model_state" not in payload:
        raise RuntimeError(f"{path}: missing model_state (keys={sorted(payload)})")
    data_mod, model_mod, transformer_mod = import_vendor(vendor_dir)
    model, cfg, kind = build_model(payload, model_mod, transformer_mod, expected_kind)
    check_finite_state_dict(payload["model_state"], f"{path} model_state")
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"{path}: strict load failed missing={missing} unexpected={unexpected}")
    check_finite_state_dict(model.state_dict(), f"{path} loaded model")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return {
        "model": model,
        "cfg": cfg,
        "kind": kind,
        "data_mod": data_mod,
        "step": payload.get("step"),
        "checkpoint_val_bpc": payload.get("val_bpc"),
        "model_type": payload.get("model_type", kind),
        "params": model.num_params(),
    }


def window_tensors(src, seq_len: int):
    import torch

    n = int(src.shape[0])
    starts = list(range(0, max(0, n - seq_len), seq_len))
    xs = []
    ys = []
    token_counts = []
    for start in starts:
        xs.append(src[start : start + seq_len])
        ys.append(src[start + 1 : start + seq_len + 1])
        token_counts.append(seq_len)
    next_start = starts[-1] + seq_len if starts else 0
    remaining = n - next_start - 1
    if remaining > 0:
        x = torch.zeros(seq_len, dtype=src.dtype)
        y = torch.full((seq_len,), -100, dtype=src.dtype)
        x[:remaining] = src[next_start : next_start + remaining]
        y[:remaining] = src[next_start + 1 : next_start + 1 + remaining]
        xs.append(x)
        ys.append(y)
        token_counts.append(remaining)
    if not xs:
        raise RuntimeError(f"split is too short for seq={seq_len}")
    return torch.stack(xs), torch.stack(ys), token_counts


def full_bpc(model, src, seq_len: int, microbatch: int, device: str, max_windows: int | None):
    import torch
    import torch.nn.functional as F

    xs, ys, token_counts = window_tensors(src, seq_len)
    available_windows = int(xs.shape[0])
    if max_windows is not None:
        xs = xs[:max_windows]
        ys = ys[:max_windows]
        token_counts = token_counts[:max_windows]
    total_nats = 0.0
    total_tokens = 0
    n_windows = int(xs.shape[0])
    model.eval()
    with torch.no_grad():
        for begin in range(0, n_windows, microbatch):
            end = min(begin + microbatch, n_windows)
            x = xs[begin:end].to(device)
            y = ys[begin:end].to(device)
            logits, _ = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            if not torch.isfinite(loss_sum):
                raise RuntimeError("non-finite full-split loss")
            tokens = int(sum(token_counts[begin:end]))
            total_nats += float(loss_sum.item())
            total_tokens += tokens
    if total_tokens == 0:
        raise RuntimeError("no tokens scored on this split")
    return {
        "bpc": total_nats / total_tokens / math.log(2),
        "n_windows": n_windows,
        "n_tokens": total_tokens,
        "truncated": max_windows is not None and n_windows < available_windows,
        "complete": not (max_windows is not None and n_windows < available_windows),
    }


def sampled_bpc(model, data, split: str, eval_iters: int, eval_batch: int, seq_len: int,
                microbatch: int, device: str, seed: int):
    import torch

    torch.manual_seed(seed)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data.get_batch(split, eval_batch, seq_len, "cpu")
            total = 0.0
            for start in range(0, eval_batch, microbatch):
                xx = x[start : start + microbatch].to(device)
                yy = y[start : start + microbatch].to(device)
                _, loss = model(xx, yy)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite sampled loss")
                total += float(loss.item()) * int(xx.shape[0]) / eval_batch
            losses.append(total)
    return sum(losses) / len(losses) / math.log(2)


def logit_parity(model, seed: int, warn_threshold: float, restore_device: str):
    """Compare FP32 logits on a small fixed batch on CPU vs CUDA."""
    import torch

    report = {
        "batch": 2,
        "seq": 64,
        "seed": seed,
        "warn_threshold": warn_threshold,
    }
    if not torch.cuda.is_available():
        report.update(skipped=True, reason="CUDA not available")
        return report
    torch.manual_seed(seed)
    ids = torch.randint(0, 256, (report["batch"], report["seq"]), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        model.to("cuda")
        gpu_logits, _ = model(ids.to("cuda"))
        gpu_logits = gpu_logits.detach().float().cpu()
        model.to("cpu")
        cpu_logits, _ = model(ids)
        cpu_logits = cpu_logits.detach().float()
        model.to(restore_device)
    # Second forward with identical random targets so CPU/GPU bpc can be compared.
    import torch.nn.functional as F
    targets = ids
    gpu_loss = F.cross_entropy(
        gpu_logits.reshape(-1, gpu_logits.size(-1)), targets.reshape(-1)
    )
    cpu_loss = F.cross_entropy(
        cpu_logits.reshape(-1, cpu_logits.size(-1)), targets.reshape(-1)
    )
    import math
    gpu_bpc = float(gpu_loss.item()) / math.log(2)
    cpu_bpc = float(cpu_loss.item()) / math.log(2)
    diff = (gpu_logits - cpu_logits).abs()
    rel = diff / cpu_logits.abs().clamp_min(1e-12)
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    cpu_logit_abs_max = float(cpu_logits.abs().max().item())
    relative_error = max_abs / cpu_logit_abs_max if cpu_logit_abs_max > 0 else float("inf")
    report.update(
        skipped=False,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        mean_abs_diff=float(diff.mean().item()),
        cpu_logit_abs_max=cpu_logit_abs_max,
        relative_error=relative_error,
        parity_batch_bpc_cpu=cpu_bpc,
        parity_batch_bpc_gpu=gpu_bpc,
        parity_batch_bpc_delta=gpu_bpc - cpu_bpc,
        warn=max_abs > warn_threshold,
    )
    print(
        f"  cpu/gpu parity max_abs={max_abs:.6g} relative_error={relative_error:.6g} "
        f"(max_abs/max|cpu_logit|) parity_batch_bpc_delta={gpu_bpc - cpu_bpc:.6g}",
        flush=True,
    )
    if report["warn"]:
        print(
            f"WARNING: CPU/GPU logit max abs diff {max_abs:.6g} "
            f"exceeds informational threshold {warn_threshold}",
            flush=True,
        )
    return report


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    load_env_file(root / ".env")
    load_env_file(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path(os.environ.get("HF_DIR", root / "data/hf/phasemesh-text8-comparison")),
                        help="HF snapshot root containing transformer/ and ssm/")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(os.environ.get("RESULTS_DIR", root / "results")),
                        help="Directory for scores.json")
    parser.add_argument("--data", type=Path,
                        default=Path(os.environ.get("TEXT8_PATH", root / "data/text8")),
                        help="Extracted text8 file")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--microbatch", type=int, default=int(os.environ.get("MICROBATCH", "4")))
    parser.add_argument("--seed", type=int, default=1, help="Seed for the sampled validation protocol")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=0,
                        help="If >0, score only this many full-split windows (dry-run / smoke).")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-sampled", action="store_true")
    parser.add_argument("--only", action="append", default=[],
                        help="Checkpoint relative path, e.g. transformer/best.pt (repeatable)")
    parser.add_argument("--allow-missing-data", action="store_true",
                        help="Load checkpoints and skip corpus evaluations if text8 is absent")
    parser.add_argument("--skip-parity", action="store_true",
                        help="Skip the CPU vs GPU logit comparison")
    parser.add_argument("--parity-warn-threshold", type=float, default=1e-3,
                        help="Warn if max absolute CPU/GPU logit difference exceeds this value (1e-3 with TF32 off)")
    parser.add_argument("--parity-seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import torch
    except ImportError:
        print("torch is required to score checkpoints", file=sys.stderr)
        return 2

    from device_check import log_device_report

    ckpt_dir = args.ckpt_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(CLAIMS)
    max_windows = args.max_windows if args.max_windows > 0 else None
    # Disable TF32 before any CUDA work (capability query, load, or eval).
    device_report = log_device_report(args.device, tf32_policy="disabled_for_scoring")
    (out_dir / "device_report.json").write_text(json.dumps(device_report, indent=2) + "\n", encoding="utf-8")

    data_obj = None
    data_info = None
    if args.data.is_file():
        raw = args.data.read_bytes()
        digest = sha256_file(args.data)
        if len(raw) != 100_000_000:
            raise SystemExit(f"Expected text8 to be 100000000 bytes, got {len(raw)}")
        if digest != EXPECTED_TEXT8_SHA256:
            print(f"warning: text8 sha256 {digest} != published {EXPECTED_TEXT8_SHA256}", file=sys.stderr)
        # Import ByteData from either vendor copy; they are byte-identical.
        vendor = ckpt_dir / "transformer" / "vendor"
        if not vendor.is_dir():
            vendor = ckpt_dir / "ssm" / "vendor"
        data_mod, _, _ = import_vendor(vendor)
        data_obj = data_mod.ByteData(raw)
        data_info = {
            "path": str(args.data),
            "sha256": digest,
            "bytes": len(raw),
            "split": "contiguous 90/5/5",
            "train_bytes": int(data_obj.train.numel()),
            "val_bytes": int(data_obj.val.numel()),
            "test_bytes": int(data_obj.test.numel()),
            "matches_published_sha256": digest == EXPECTED_TEXT8_SHA256,
        }
        del raw
    elif not args.allow_missing_data:
        raise SystemExit(f"text8 not found at {args.data}; fetch it or pass --allow-missing-data")

    results = []
    for rel in wanted:
        path = ckpt_dir / rel
        kind_dir = rel.split("/", 1)[0]
        vendor_dir = ckpt_dir / kind_dir / "vendor"
        record = {
            "name": rel,
            "path": str(path),
            "exists": path.is_file(),
            "claimed_sampled_val_bpc": CLAIMS.get(rel, {}).get("claimed_sampled_val_bpc"),
            "claimed_step": CLAIMS.get(rel, {}).get("claimed_step"),
        }
        if not path.is_file():
            record["error"] = "checkpoint file not found"
            results.append(record)
            continue
        record["sha256"] = sha256_file(path)
        record["bytes"] = path.stat().st_size
        print(f"Loading {rel} from {path}", flush=True)
        t0 = time.monotonic()
        loaded = load_checkpoint(path, vendor_dir, kind_dir, args.device)
        record["load"] = {
            "weights_only": True,
            "strict": True,
            "finite_tensors": True,
            "seconds": time.monotonic() - t0,
            "device": args.device,
            "dtype": "float32",
        }
        record["model_type"] = loaded["model_type"]
        record["step"] = loaded["step"]
        record["checkpoint_val_bpc"] = loaded["checkpoint_val_bpc"]
        record["params"] = loaded["params"]
        record["cfg"] = loaded["cfg"].__dict__ if hasattr(loaded["cfg"], "__dict__") else None
        if not args.skip_parity:
            record["cpu_gpu_parity"] = logit_parity(
                loaded["model"], args.parity_seed, args.parity_warn_threshold, args.device,
            )
        else:
            record["cpu_gpu_parity"] = {"skipped": True, "reason": "--skip-parity"}

        if data_obj is None:
            record["skipped_eval"] = "text8 not available"
            results.append(record)
            del loaded
            continue
        if not args.skip_sampled:
            t1 = time.monotonic()
            record["sampled_val_bpc"] = sampled_bpc(
                loaded["model"], data_obj, "val", args.eval_iters, args.eval_batch,
                args.seq, args.microbatch, args.device, args.seed,
            )
            record["sampled"] = {
                "split": "val",
                "iters": args.eval_iters,
                "batch": args.eval_batch,
                "seq": args.seq,
                "microbatch": args.microbatch,
                "seed": args.seed,
                "seconds": time.monotonic() - t1,
                "note": (
                    "Fresh torch.manual_seed(seed) then ByteData.get_batch, matching "
                    "the published sampling procedure. Not a replay of in-training RNG."
                ),
            }
            print(f"  sampled val bpc={record['sampled_val_bpc']}", flush=True)
        if not args.skip_full:
            for split_name, tensor in (("val", data_obj.val), ("test", data_obj.test)):
                t2 = time.monotonic()
                scored = full_bpc(
                    loaded["model"], tensor, args.seq, args.microbatch, args.device, max_windows
                )
                record[f"full_{split_name}_bpc"] = scored["bpc"]
                record[f"full_{split_name}"] = {**scored, "seconds": time.monotonic() - t2}
                print(f"  full {split_name} bpc={scored['bpc']} tokens={scored['n_tokens']}", flush=True)
        results.append(record)
        del loaded
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "pins": {
            "hf_repo": os.environ.get("HF_REPO", "MarxistLeninist/phasemesh-text8-comparison"),
            "hf_revision": os.environ.get("HF_REVISION", HF_REVISION_DEFAULT),
            "source_commit": os.environ.get("UPSTREAM_COMMIT", SOURCE_COMMIT_DEFAULT),
            "ckpt_dir": str(ckpt_dir),
        },
        "device_report": device_report,
        "tf32": device_report.get("tf32"),
        "score_gate": {
            "metric": "abs(sampled_val_bpc - claimed) and abs(full_val_bpc - claimed)",
            "threshold": 0.01,
            "full_test": "info-only, but must be present and complete",
            "parity_warn_threshold": args.parity_warn_threshold,
            "note": "Training should not start until scripts/check_score_gate.py passes.",
        },
        "dataset": data_info,
        "eval": {
            "dtype": "float32",
            "seq": args.seq,
            "full_windows": "non-overlapping, padded remainder included",
            "full_windows_note": (
                "Each window is scored independently with a causal model. "
                "The first token of each window has no prior context from earlier "
                "bytes in the split, so full-split bpc is not directly comparable "
                "to a sliding-window (stride-1) number."
            ),
            "sampled": {
                "iters": args.eval_iters,
                "batch": args.eval_batch,
                "seed": args.seed,
            },
            "device": args.device,
            "microbatch": args.microbatch,
            "skip_full": args.skip_full,
            "skip_sampled": args.skip_sampled,
            "skip_parity": args.skip_parity,
            "allow_missing_data": args.allow_missing_data,
            "max_windows": max_windows,
            "parity_warn_threshold": args.parity_warn_threshold,
        },
        "checkpoints": results,
    }
    out_path = out_dir / "scores.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_dir / 'device_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
