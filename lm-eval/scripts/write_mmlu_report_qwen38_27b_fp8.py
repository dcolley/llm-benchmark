#!/usr/bin/env python3
"""Write the Qwen3.8-27B-FP8 MMLU report versus Qwen3.6-35B-A3B-MTP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


QWEN36 = {
    "overall": (84.76, 0.29),
    "social sciences": (91.55, 0.49),
    "other": (87.22, 0.57),
    "stem": (83.44, 0.64),
    "humanities": (79.57, 0.57),
}


def pct(value: float) -> float:
    return 100.0 * float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/home/derek/llm-benchmark/lm-eval/results/qwen-qwen3.8-27b-fp8",
    )
    parser.add_argument(
        "--out",
        default="/home/derek/llm-benchmark/lm-eval/results/qwen-qwen3.8-27b-fp8/MMLU_LLAMA_REPORT.md",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    matches = sorted(root.glob("**/results_*.json"))
    if not matches:
        raise SystemExit(f"no results_*.json under {root}")
    result_path = matches[-1]
    data = json.loads(result_path.read_text())
    results = data["results"]

    def score(key: str) -> tuple[float, float]:
        row = results[key]
        return (
            pct(row["exact_match,strict_match"]),
            pct(row["exact_match_stderr,strict_match"]),
        )

    overall = score("mmlu_llama_ds4")
    domains = {
        "social sciences": score("mmlu_llama_ds4_social_sciences"),
        "other": score("mmlu_llama_ds4_other"),
        "stem": score("mmlu_llama_ds4_stem"),
        "humanities": score("mmlu_llama_ds4_humanities"),
    }
    delta = overall[0] - QWEN36["overall"][0]

    aggregate_keys = {
        "mmlu_llama_ds4",
        "mmlu_llama_ds4_stem",
        "mmlu_llama_ds4_other",
        "mmlu_llama_ds4_social_sciences",
        "mmlu_llama_ds4_humanities",
    }
    leaves: list[tuple[float, str]] = []
    for key, row in results.items():
        if key in aggregate_keys or not key.startswith("mmlu_llama_ds4_"):
            continue
        if "exact_match,strict_match" not in row:
            continue
        alias = row.get(
            "alias", key.replace("mmlu_llama_ds4_", "").replace("_", " ")
        )
        leaves.append((pct(row["exact_match,strict_match"]), alias))
    leaves.sort()

    sample_count = 14_042
    invalid = 0
    logged = 0
    for sample_file in root.glob("**/samples_*.jsonl"):
        for line in sample_file.open():
            sample = json.loads(line)
            logged += 1
            extracted = (sample.get("filtered_resps") or ["[invalid]"])[0]
            if extracted not in {"A", "B", "C", "D"}:
                invalid += 1

    date = result_path.name.removeprefix("results_").split("T", 1)[0]
    verb = "leads" if delta >= 0 else "trails"
    lines = [
        "# mmlu_llama comparison: Qwen3.8-27B-FP8 vs Qwen3.6-35B-A3B-MTP",
        "",
        "| Qwen3.8-27B-FP8 | Qwen3.6-35B-A3B-MTP | Δ |",
        "|------------------:|--------------------:|--:|",
        f"| **{overall[0]:.2f}%** | **84.76%** | **{delta:+.2f} pp** |",
        "",
        "Generative MMLU, five-shot, `strict_match` / `exact_match`, 14,042 samples.",
        "",
        "| | **Qwen3.8-27B-FP8** | **Qwen3.6-35B-A3B-MTP** |",
        "|--|--:|--:|",
        f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | **84.76% ± 0.29%** |",
        f"| Δ vs Qwen3.6 | **{delta:+.2f} pp** | baseline |",
        f"| Samples | {sample_count:,} | 14,042 |",
        "| Architecture | Dense 27B, block FP8, MTP-3 | MoE 35B-A3B, MTP |",
        "| Serving | vLLM 0.27.1 on DGX Spark | LM Studio |",
        f"| Date | {date} | ~2026-06-16 |",
        "",
        f"**Headline:** Qwen3.8-27B-FP8 {verb} Qwen3.6-35B-A3B-MTP by "
        f"**{abs(delta):.2f} points** overall ({overall[0]:.2f}% vs 84.76%).",
        "",
        "> The questions, five-shot examples, targets, and strict-match metric are shared, "
        "but answer extraction and serving stacks differ. Treat small deltas as directional.",
        "",
        "---",
        "",
        "## Domain scores",
        "",
        "| Domain | Qwen3.8-27B-FP8 | Qwen3.6 MTP | Δ (3.8 − 3.6) |",
        "|--------|-----------------:|------------:|--------------:|",
    ]
    for name in ["social sciences", "other", "stem", "humanities"]:
        current, stderr = domains[name]
        baseline, baseline_stderr = QWEN36[name]
        label = "STEM" if name == "stem" else name.capitalize()
        lines.append(
            f"| {label} | {current:.2f}% ± {stderr:.2f}% | "
            f"{baseline:.2f}% ± {baseline_stderr:.2f}% | {current-baseline:+.2f} pp |"
        )
    lines.extend(
        [
            f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | "
            f"**84.76% ± 0.29%** | **{delta:+.2f} pp** |",
            "",
            "---",
            "",
            "## Setup differences",
            "",
            "### Qwen3.8-27B-FP8 (this run)",
            "",
            "```bash",
            "./scripts/run_mmlu_llama_qwen38_27b_fp8.sh --foreground",
            "```",
            "",
            "- Local vLLM 0.27.1, `qwen3.8-27b-fp8`, FP8 weights and FP8 KV cache.",
            "- MTP speculative decoding: 3 tokens; eight concurrent requests.",
            "- `mmlu_llama_ds4`: stock MMLU questions, targets, and five-shot examples, "
            "but full assistant turns with regex extraction of `The best answer is [A-D]`.",
            "- Thinking disabled using `chat_template_kwargs.enable_thinking=false`; concise "
            "answer-only system instruction; Qwen-recommended non-thinking sampling "
            "(`temperature=0.7`, `top_p=0.8`, `top_k=20`).",
            f"- Extraction validity: {logged-invalid:,}/{logged:,} valid "
            f"({invalid} invalid)." if logged else "- Sample logs unavailable.",
            f"- Results: `{result_path.relative_to(root.parent.parent)}`.",
            "",
            "Stock `mmlu_llama` was rejected during smoke testing because Qwen3.8 ignored "
            "its partial assistant continuation (`The best answer is`) and produced a fresh "
            "prose answer, falsely scoring 0%. The extraction task avoids that protocol artifact.",
            "",
            "### Qwen3.6-35B-A3B-MTP (prior baseline)",
            "",
            "- Stock `mmlu_llama` partial-assistant continuation under LM Studio.",
            "- Thinking disabled with `/no_think`; `max_tokens=512`, `temperature=0.7`, "
            "`top_p=0.8`, `top_k=20`, `min_p=0.0`.",
            "",
            "---",
            "",
            "## Weakest and strongest subjects (Qwen3.8)",
            "",
            "Lowest:",
            "",
            "| Subject | Score |",
            "|---------|------:|",
        ]
    )
    for value, alias in leaves[:8]:
        lines.append(f"| {alias} | {value:.2f}% |")
    lines.extend(["", "Highest:", "", "| Subject | Score |", "|---------|------:|"])
    for value, alias in reversed(leaves[-8:]):
        lines.append(f"| {alias} | {value:.2f}% |")
    lines.extend(
        [
            "",
            "---",
            "",
            "## Takeaways",
            "",
            f"1. Overall difference versus Qwen3.6 is **{delta:+.2f} pp**.",
            "2. Domain-level differences are more informative than tiny subject-level changes.",
            "3. For a strict apples-to-apples comparison, rerun Qwen3.6 with the same "
            "vLLM server and extraction task.",
            "",
            f"*Generated from `{result_path.name}` and the published "
            "Qwen3.6-35B-A3B-MTP table.*",
            "",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print(f"overall {overall[0]:.2f}% delta_vs_qwen36 {delta:+.2f}pp")


if __name__ == "__main__":
    main()
