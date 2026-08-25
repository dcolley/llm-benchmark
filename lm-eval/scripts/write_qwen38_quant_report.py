#!/usr/bin/env python3
"""Compare MMLU results for the three Qwen3.8-27B quant/runtime variants."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/derek/llm-benchmark/lm-eval/results")
OUT = ROOT / "QWEN3.8_27B_QUANT_REPORT.md"
MODELS = [
    (
        "Qwen/Qwen3.8-27B-FP8",
        ROOT / "qwen-qwen3.8-27b-fp8",
        "FP8",
        "vLLM 0.27.1",
    ),
    (
        "r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121",
        ROOT / "r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121",
        "NVFP4 + FP8 KV",
        "vLLM 0.27.2rc0-sm121, MTP K=3",
    ),
    (
        "unsloth/Qwen3.8-27B-GGUF",
        ROOT / "unsloth-qwen3.8-27b-gguf-ud-q8-k-xl",
        "UD-Q8_K_XL GGUF",
        "llama.cpp CUDA 13, MTP",
    ),
]
DOMAINS = [
    ("Social sciences", "mmlu_llama_ds4_social_sciences"),
    ("Other", "mmlu_llama_ds4_other"),
    ("STEM", "mmlu_llama_ds4_stem"),
    ("Humanities", "mmlu_llama_ds4_humanities"),
]


def latest_result(root: Path) -> Path:
    matches = sorted(root.glob("**/results_*.json"))
    if not matches:
        raise SystemExit(f"No results_*.json under {root}")
    return matches[-1]


def score(results: dict, key: str) -> tuple[float, float]:
    row = results[key]
    return (
        100 * float(row["exact_match,strict_match"]),
        100 * float(row["exact_match_stderr,strict_match"]),
    )


def validity(root: Path) -> tuple[int, int]:
    valid = total = 0
    for path in root.glob("**/samples_*.jsonl"):
        for line in path.open():
            sample = json.loads(line)
            total += 1
            extracted = (sample.get("filtered_resps") or ["[invalid]"])[0]
            valid += extracted in {"A", "B", "C", "D"}
    return valid, total


def main() -> None:
    loaded = []
    for name, root, quant, runtime in MODELS:
        result_path = latest_result(root)
        data = json.loads(result_path.read_text())
        valid, total = validity(root)
        loaded.append(
            {
                "name": name,
                "root": root,
                "quant": quant,
                "runtime": runtime,
                "results": data["results"],
                "result_path": result_path,
                "overall": score(data["results"], "mmlu_llama_ds4"),
                "valid": valid,
                "total": total,
            }
        )

    best = max(loaded, key=lambda item: item["overall"][0])
    lines = [
        "# Qwen3.8-27B quant comparison on DGX Spark",
        "",
        "Generative MMLU, five-shot, 14,042 samples, identical extraction task and "
        "non-thinking sampling. Higher is better.",
        "",
        "| Model | Quant | Runtime | MMLU | Valid extractions |",
        "|-------|-------|---------|-----:|------------------:|",
    ]
    for item in loaded:
        value, stderr = item["overall"]
        valid_text = (
            f"{item['valid']:,}/{item['total']:,}"
            if item["total"]
            else "samples unavailable"
        )
        lines.append(
            f"| `{item['name']}` | {item['quant']} | {item['runtime']} | "
            f"**{value:.2f}% ± {stderr:.2f}%** | {valid_text} |"
        )

    lines.extend(
        [
            "",
            f"**Best MMLU result:** `{best['name']}` at **{best['overall'][0]:.2f}%**.",
            "",
            "## Domain comparison",
            "",
            "| Domain | Qwen FP8 | r0b0tlab NVFP4 | Unsloth GGUF Q8 |",
            "|--------|----------:|-----------------:|------------------:|",
        ]
    )
    for label, key in DOMAINS:
        values = [score(item["results"], key)[0] for item in loaded]
        lines.append(
            f"| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |"
        )
    values = [item["overall"][0] for item in loaded]
    lines.append(
        f"| **Overall** | **{values[0]:.2f}%** | **{values[1]:.2f}%** | "
        f"**{values[2]:.2f}%** |"
    )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Task: `mmlu_llama_ds4`; stock questions, targets, and five-shot examples.",
            "- Prompt: concise `/no_think` answer-only instruction.",
            "- Sampling: temperature 0.7, top-p 0.8, top-k 20, max 64 tokens.",
            "- API concurrency: 8.",
            "- The serving stacks differ, so this primarily compares end-to-end accuracy. "
            "Runtime throughput should be reported separately.",
            "",
            "## Result artifacts",
            "",
        ]
    )
    for item in loaded:
        lines.append(
            f"- `{item['name']}`: `{item['result_path'].relative_to(ROOT.parent)}`"
        )
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
