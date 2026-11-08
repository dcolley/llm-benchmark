#!/usr/bin/env python3
"""Export latest.pt -> score.py model-only checkpoint, with bit-identical logits."""
from __future__ import annotations

import dataclasses
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from export_model_checkpoint import (  # noqa: E402
    ExportError,
    export_checkpoint,
    sha256_file,
)
from score import load_checkpoint  # noqa: E402


HF_REV = "2eb1bef95b39e9745035044679f9d18b8ac3513a"
HF_FILE = (
    "https://huggingface.co/MarxistLeninist/phasemesh-text8-comparison/"
    f"resolve/{HF_REV}/transformer/vendor/phase_ssm/{{name}}"
)
LOCAL_HINTS = {
    "data.py": [
        Path("/tmp/phasemesh-review/hf-transformer/data.py"),
        Path("/tmp/phasemesh-review/hf-ssm/data.py"),
        Path("/tmp/phasemesh-review/upstream/data.py"),
    ],
    "model.py": [
        Path("/tmp/phasemesh-review/hf-ssm/model.py"),
        Path("/tmp/phasemesh-review/upstream/model.py"),
    ],
    "transformer.py": [
        Path("/tmp/phasemesh-review/hf-transformer/transformer.py"),
        Path("/tmp/phasemesh-review/upstream/transformer.py"),
    ],
}

_VENDOR_CACHE: dict[str, bytes] = {}


def _read_vendor_file(name: str) -> bytes:
    if name in _VENDOR_CACHE:
        return _VENDOR_CACHE[name]
    for candidate in LOCAL_HINTS.get(name, []):
        if candidate.is_file():
            _VENDOR_CACHE[name] = candidate.read_bytes()
            return _VENDOR_CACHE[name]
    url = HF_FILE.format(name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "text8-repro-tests"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            _VENDOR_CACHE[name] = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"could not load vendor file {name} from local cache or {url}: {exc}"
        ) from exc
    return _VENDOR_CACHE[name]


def prepare_vendor(root: Path, *, include_transformer: bool) -> Path:
    vendor = root / "vendor"
    pkg = vendor / "phase_ssm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "data.py").write_bytes(_read_vendor_file("data.py"))
    (pkg / "model.py").write_bytes(_read_vendor_file("model.py"))
    if include_transformer:
        (pkg / "transformer.py").write_bytes(_read_vendor_file("transformer.py"))
    return vendor


def save_latest_pt(path: Path, model, cfg, *, step: int, val_bpc: float) -> None:
    import torch

    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=0.0003, betas=(0.9, 0.95))
    spec = {
        "seed": 1,
        "seq": 16,
        "d_model": cfg.d_model,
        "layers": cfg.n_layers,
        "microbatch": 4,
        "effective_batch": 64,
        "eval_iters": 50,
        "dataset_sha256": "00" * 32,
        "model": dataclasses.asdict(cfg),
        "params": model.num_params(),
        "source_commit": "3ae43ffac6c0699857cada0865f156405fb95956",
    }
    payload = {
        "spec": spec,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best": val_bpc,
        "history": [
            {"step": 0, "val_bpc": 8.0},
            {"step": step, "val_bpc": val_bpc},
        ],
        "last_eval": step,
        "trained_seconds": 12.5,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": [],
    }
    torch.save(payload, path)


def _fixed_ids():
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261108)
    return torch.randint(0, 256, (2, 16), generator=generator)


def _source_logits(model, ids):
    import torch

    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
    return logits.detach().cpu()


def _assert_exported_keys(payload: dict, src: Path, expected_type: str, step: int) -> None:
    assert sorted(payload) == [
        "cfg",
        "model_state",
        "model_type",
        "source_basename",
        "source_sha256",
        "step",
        "torch_version",
        "val_bpc",
    ]
    assert "optimizer" not in payload
    assert "rng_cpu" not in payload
    assert "rng_cuda" not in payload
    assert "model" not in payload
    assert payload["model_type"] == expected_type
    assert payload["step"] == step
    assert payload["source_basename"] == src.name
    assert payload["source_sha256"] == sha256_file(src)
    assert isinstance(payload["cfg"], dict)
    assert isinstance(payload["torch_version"], str) and payload["torch_version"]


def test_export_transformer_logits_bit_identical(tmp_path: Path) -> None:
    import torch

    from score import import_vendor

    vendor = prepare_vendor(tmp_path, include_transformer=True)
    _data_mod, _model_mod, transformer_mod = import_vendor(vendor)
    assert transformer_mod is not None
    cfg = transformer_mod.TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff_mult=2,
        block_size=16,
        dropout=0.0,
    )
    torch.manual_seed(11)
    model = transformer_mod.TransformerLM(cfg).to(device="cpu", dtype=torch.float32)
    ids = _fixed_ids()
    source_logits = _source_logits(model, ids)

    src = tmp_path / "latest.pt"
    dest = tmp_path / "exported.pt"
    save_latest_pt(src, model, cfg, step=5000, val_bpc=1.257564388922874)
    info = export_checkpoint(src, dest, model="transformer")
    assert info["loader"] == "weights_only=True"
    assert info["safe_globals"] == []

    exported = torch.load(dest, map_location="cpu", weights_only=True)
    _assert_exported_keys(exported, src, "transformer", 5000)
    assert exported["val_bpc"] == 1.257564388922874

    loaded = load_checkpoint(dest, vendor, "transformer", "cpu")
    with torch.no_grad():
        loaded_logits, _ = loaded["model"](ids)
    assert torch.equal(source_logits, loaded_logits.cpu())


def test_export_ssm_logits_bit_identical(tmp_path: Path) -> None:
    import torch

    from score import import_vendor

    vendor = prepare_vendor(tmp_path, include_transformer=False)
    _data_mod, model_mod, transformer_mod = import_vendor(vendor)
    assert transformer_mod is None
    cfg = model_mod.PhaseSSMConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        state_dim=8,
        expand=1,
        d_ff_mult=2,
        ssm_backend="fft",
        dropout=0.0,
        use_phase_memory=False,
    )
    torch.manual_seed(13)
    model = model_mod.PhaseSSMLM(cfg).to(device="cpu", dtype=torch.float32)
    ids = _fixed_ids()
    source_logits = _source_logits(model, ids)

    src = tmp_path / "latest.pt"
    dest = tmp_path / "exported.pt"
    save_latest_pt(src, model, cfg, step=5000, val_bpc=1.268647525903973)
    info = export_checkpoint(src, dest, model="ssm")
    assert info["loader"] == "weights_only=True"
    assert info["safe_globals"] == []

    exported = torch.load(dest, map_location="cpu", weights_only=True)
    _assert_exported_keys(exported, src, "phasessm", 5000)
    assert exported["val_bpc"] == 1.268647525903973

    loaded = load_checkpoint(dest, vendor, "ssm", "cpu")
    with torch.no_grad():
        loaded_logits, _ = loaded["model"](ids)
    assert torch.equal(source_logits, loaded_logits.cpu())


def test_sha_mismatch_refusal(tmp_path: Path) -> None:
    import torch

    from score import import_vendor

    vendor = prepare_vendor(tmp_path, include_transformer=True)
    _data_mod, _model_mod, transformer_mod = import_vendor(vendor)
    cfg = transformer_mod.TransformerConfig(
        d_model=32, n_layers=1, n_heads=4, d_ff_mult=2, block_size=16
    )
    torch.manual_seed(17)
    model = transformer_mod.TransformerLM(cfg)
    src = tmp_path / "latest.pt"
    dest = tmp_path / "exported.pt"
    save_latest_pt(src, model, cfg, step=100, val_bpc=2.0)
    try:
        export_checkpoint(src, dest, model="transformer", expect_sha256="0" * 64)
    except ExportError as exc:
        message = str(exc)
        assert "sha256 mismatch" in message
        assert sha256_file(src) in message
    else:
        raise AssertionError("expected SHA256 mismatch to refuse export")
    assert not dest.exists()


def test_overwrite_refusal(tmp_path: Path) -> None:
    import torch

    from score import import_vendor

    vendor = prepare_vendor(tmp_path, include_transformer=True)
    _data_mod, _model_mod, transformer_mod = import_vendor(vendor)
    cfg = transformer_mod.TransformerConfig(
        d_model=32, n_layers=1, n_heads=4, d_ff_mult=2, block_size=16
    )
    torch.manual_seed(19)
    model = transformer_mod.TransformerLM(cfg)
    src = tmp_path / "latest.pt"
    dest = tmp_path / "exported.pt"
    save_latest_pt(src, model, cfg, step=200, val_bpc=2.0)
    export_checkpoint(src, dest, model="transformer")
    original = dest.read_bytes()
    try:
        export_checkpoint(src, dest, model="transformer")
    except ExportError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("expected overwrite refusal")
    assert dest.read_bytes() == original
