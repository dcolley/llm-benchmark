#!/usr/bin/env python3
"""CPU tests for replay_eval_batches and score.py replay averaging."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from replay_eval_batches import (  # noqa: E402
    ambiguity_reasons,
    offsets_sha256,
    replay_val_offsets,
    train_eval_iters,
)
from score import replay_evaluate  # noqa: E402


TOY = {
    "seed": 7,
    "eval_interval": 2,
    "eval_iters": 2,
    "eval_batch": 3,
    "seq": 4,
    "effective_batch": 5,
    "steps": 4,
    "val_bytes": 40,
    "train_bytes": 60,
}


def simulate_loop_val_draws(cfg: dict, target_step: int) -> list[list[int]]:
    """Direct reimplementation of the published while-loop draw order."""
    import torch

    def fake_get_batch(split: str, batch_size: int, seq_len: int, device: str):
        n_src = cfg["val_bytes"] if split == "val" else cfg["train_bytes"]
        high = n_src - seq_len - 1
        return torch.randint(0, high, (batch_size,))

    torch.manual_seed(cfg["seed"])
    step = 0
    last_eval = -1
    captured = None
    while True:
        if step % cfg["eval_interval"] == 0 and last_eval != step:
            val_ixs = []
            for _ in range(cfg["eval_iters"]):
                ix = fake_get_batch("val", cfg["eval_batch"], cfg["seq"], "cpu")
                val_ixs.append(ix.tolist())
            for _ in range(train_eval_iters(cfg["eval_iters"])):
                fake_get_batch("train", cfg["eval_batch"], cfg["seq"], "cpu")
            last_eval = step
            if step == target_step:
                captured = val_ixs
        if step == cfg["steps"]:
            break
        fake_get_batch("train", cfg["effective_batch"], cfg["seq"], "cpu")
        step += 1
    if captured is None:
        raise AssertionError(f"did not capture eval at step {target_step}")
    return captured


def test_replay_matches_simulated_loop():
    for step in (0, 2, 4):
        replayed = replay_val_offsets(
            seed=TOY["seed"],
            step=step,
            eval_interval=TOY["eval_interval"],
            eval_iters=TOY["eval_iters"],
            eval_batch=TOY["eval_batch"],
            seq=TOY["seq"],
            effective_batch=TOY["effective_batch"],
            steps=TOY["steps"],
            val_bytes=TOY["val_bytes"],
            train_bytes=TOY["train_bytes"],
        )
        simulated = simulate_loop_val_draws(TOY, step)
        assert replayed == simulated, (step, replayed, simulated)


def test_offsets_in_range_and_deterministic_sha():
    batches = replay_val_offsets(
        seed=TOY["seed"], step=2,
        eval_interval=TOY["eval_interval"], eval_iters=TOY["eval_iters"],
        eval_batch=TOY["eval_batch"], seq=TOY["seq"],
        effective_batch=TOY["effective_batch"], steps=TOY["steps"],
        val_bytes=TOY["val_bytes"], train_bytes=TOY["train_bytes"],
    )
    val_high = TOY["val_bytes"] - TOY["seq"] - 1
    assert len(batches) == TOY["eval_iters"]
    for batch in batches:
        assert len(batch) == TOY["eval_batch"]
        assert all(0 <= x < val_high for x in batch)
    again = replay_val_offsets(
        seed=TOY["seed"], step=2,
        eval_interval=TOY["eval_interval"], eval_iters=TOY["eval_iters"],
        eval_batch=TOY["eval_batch"], seq=TOY["seq"],
        effective_batch=TOY["effective_batch"], steps=TOY["steps"],
        val_bytes=TOY["val_bytes"], train_bytes=TOY["train_bytes"],
    )
    assert offsets_sha256(batches) == offsets_sha256(again)
    assert batches == again


def test_refuses_non_eval_step():
    try:
        replay_val_offsets(
            seed=1, step=3, eval_interval=2, eval_iters=1, eval_batch=1,
            seq=2, effective_batch=1, steps=4, val_bytes=20, train_bytes=20,
        )
        raise AssertionError("non-eval step should have been refused")
    except SystemExit as exc:
        assert "not an eval step" in str(exc)


def test_ambiguity_detects_history_gap():
    cfg = {"eval_interval": 250, "resume": False}
    progress = {"history": [{"step": 0}, {"step": 250}, {"step": 750}], "complete": True}
    reasons = ambiguity_reasons(cfg, progress, 750)
    assert any("contiguous" in r for r in reasons)


def test_score_mean_of_batch_means_not_token_mean():
    import torch
    import torch.nn as nn

    class Dummy(nn.Module):
        def forward(self, x, y=None):
            return None, x[:, 0].float().mean()

    # src[i] = i so the first token of a window starting at i is i.
    src = torch.arange(0, 32, dtype=torch.long)
    batches = [[1, 3], [10]]
    mean_of_means = (((1 + 3) / 2) + 10) / 2
    token_mean = (1 + 3 + 10) / 3
    assert abs(mean_of_means - token_mean) > 1.0
    bpc = replay_evaluate(Dummy(), src, batches, seq_len=4, microbatch=2, device="cpu")
    expected = mean_of_means / math.log(2)
    assert abs(bpc - expected) < 1e-6, (bpc, expected, token_mean / math.log(2))


def test_microbatch_invariance():
    import torch
    import torch.nn as nn

    class Dummy(nn.Module):
        def forward(self, x, y=None):
            return None, x[:, 0].float().mean()

    src = torch.arange(0, 32, dtype=torch.long)
    batches = [[2, 5, 8]]
    bpcs = [
        replay_evaluate(Dummy(), src, batches, seq_len=4, microbatch=mb, device="cpu")
        for mb in (1, 2, 3)
    ]
    assert max(bpcs) - min(bpcs) < 1e-6, bpcs


def main() -> int:
    tests = [
        test_replay_matches_simulated_loop,
        test_offsets_in_range_and_deterministic_sha,
        test_refuses_non_eval_step,
        test_ambiguity_detects_history_gap,
        test_score_mean_of_batch_means_not_token_mean,
        test_microbatch_invariance,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failed:
        print(f"{failed} test(s) failed")
        return 1
    print(f"All {len(tests)} replay tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
