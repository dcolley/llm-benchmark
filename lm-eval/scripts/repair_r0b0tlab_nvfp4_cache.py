#!/usr/bin/env python3
"""Wire complete NVFP4 weight blobs into the active HF snapshot."""
from __future__ import annotations

import json
from pathlib import Path

CACHE = Path(
    "/root/.cache/huggingface/hub/models--r0b0tlab--Qwen3.8-27B-NVFP4-MTP-sm121"
)
LOCKS = Path(
    "/root/.cache/huggingface/hub/.locks/models--r0b0tlab--Qwen3.8-27B-NVFP4-MTP-sm121"
)
SHARDS = {
    "model-00001-of-00004.safetensors": "4208cd3b45f605d9f67e29e2939c35f4bdc255d9baf72c951a6aa77c560e25f5",
    "model-00002-of-00004.safetensors": "024111b980be5bfa93616777e567110b5f3b1340578614165c51c78917457296",
    "model-00003-of-00004.safetensors": "927ee34337d1cc4419b97b82370229128003a5f2a17bb45670cdaaab8374d169",
    "model-00004-of-00004.safetensors": "47202b11daf026e3a472c030b18c4b7e6021c475d1f31ff06f1877d16639912d",
}


def main() -> None:
    if LOCKS.exists():
        for lock in LOCKS.glob("*.lock"):
            lock.unlink(missing_ok=True)

    for name, digest in SHARDS.items():
        blob = CACHE / "blobs" / digest
        if not blob.exists():
            raise SystemExit(f"missing blob for {name}: {digest}")
        size = blob.stat().st_size
        print(f"{name} {digest} {size}")
        if size < 100_000_000:
            raise SystemExit(f"blob too small for {name}: {size}")

    snaps = list((CACHE / "snapshots").iterdir())
    print("snapshots", [s.name for s in snaps])
    snap = next(
        (s for s in snaps if (s / "model.safetensors.index.json").exists()),
        None,
    )
    if snap is None:
        raise SystemExit(f"no snapshot with index.json among {snaps}")

    print("using", snap)
    for name, digest in SHARDS.items():
        link = snap / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("../../blobs") / digest)

    (CACHE / "refs" / "main").write_text(snap.name + "\n")
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    for filename in sorted(set(index["weight_map"].values())):
        path = snap / filename
        size = path.stat().st_size if path.exists() else None
        print("check", filename, path.exists(), size)
        if not path.exists() or size is None or size < 100_000_000:
            raise SystemExit(f"incomplete shard link: {filename}")
    print("SNAPSHOT_OK", snap.name)


if __name__ == "__main__":
    main()
