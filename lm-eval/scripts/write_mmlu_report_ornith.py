#!/usr/bin/env python3
"""Write Ornith-1.5-35B-A3B-NVFP4 MMLU report versus Qwen3.6-35B-A3B-MTP."""
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

QWEN36_TASKS = {
    "formal logic": 73.81,
    "high school european history": 88.48,
    "high school us history": 93.63,
    "high school world history": 93.25,
    "international law": 92.56,
    "jurisprudence": 89.81,
    "logical fallacies": 89.57,
    "moral disputes": 85.84,
    "moral scenarios": 69.72,
    "philosophy": 89.71,
    "prehistory": 91.67,
    "professional law": 70.93,
    "world religions": 89.47,
    "business ethics": 84.00,
    "clinical knowledge": 90.19,
    "college medicine": 84.39,
    "global facts": 56.00,
    "human aging": 86.10,
    "management": 90.29,
    "marketing": 94.02,
    "medical genetics": 95.00,
    "miscellaneous": 95.02,
    "nutrition": 90.20,
    "professional accounting": 75.53,
    "professional medicine": 93.38,
    "virology": 59.04,
    "econometrics": 83.33,
    "high school geography": 94.44,
    "high school government and politics": 98.45,
    "high school macroeconomics": 89.49,
    "high school microeconomics": 95.80,
    "high school psychology": 95.96,
    "human sexuality": 94.66,
    "professional psychology": 90.03,
    "public relations": 78.18,
    "security studies": 82.86,
    "sociology": 92.54,
    "us foreign policy": 95.00,
    "abstract algebra": 70.00,
    "anatomy": 88.15,
    "astronomy": 92.76,
    "college biology": 93.06,
    "college chemistry": 67.00,
    "college computer science": 78.00,
    "college mathematics": 70.00,
    "college physics": 69.61,
    "computer security": 89.00,
    "conceptual physics": 95.74,
    "electrical engineering": 86.21,
    "elementary mathematics": 80.95,
    "high school biology": 95.48,
    "high school chemistry": 83.74,
    "high school computer science": 94.00,
    "high school mathematics": 65.19,
    "high school physics": 84.11,
    "high school statistics": 84.72,
    "machine learning": 80.36,
}


def pct(value: float) -> float:
    return 100.0 * float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/home/derek/llm-benchmark/lm-eval/results/ornith-1.5-35b-a3b-nvfp4",
    )
    parser.add_argument(
        "--out",
        default="/home/derek/llm-benchmark/lm-eval/results/ornith-1.5-35b-a3b-nvfp4/MMLU_LLAMA_REPORT.md",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    matches = sorted(
        p for p in root.glob("**/results_*.json") if "smoke" not in p.parts
    )
    if not matches:
        raise SystemExit(f"no results_*.json under {root}")
    result_path = matches[-1]
    data = json.loads(result_path.read_text())
    results = data["results"]
    groups = data.get("groups") or results

    def score(key: str) -> tuple[float, float]:
        row = groups.get(key) or results[key]
        return (
            pct(row["exact_match,strict_match"]),
            pct(row["exact_match_stderr,strict_match"]),
        )

    overall = score("mmlu_llama")
    domains = {
        "social sciences": score("mmlu_llama_social_sciences"),
        "other": score("mmlu_llama_other"),
        "stem": score("mmlu_llama_stem"),
        "humanities": score("mmlu_llama_humanities"),
    }
    delta = overall[0] - QWEN36["overall"][0]

    skip = {
        "mmlu_llama",
        "mmlu_llama_stem",
        "mmlu_llama_other",
        "mmlu_llama_social_sciences",
        "mmlu_llama_humanities",
    }
    leaves: list[tuple[str, float, float, float]] = []
    for key, row in sorted(results.items()):
        if key in skip or "exact_match,strict_match" not in row:
            continue
        alias = row.get("alias", key.replace("mmlu_llama_", "").replace("_", " "))
        ov = pct(row["exact_match,strict_match"])
        se = pct(row.get("exact_match_stderr,strict_match") or 0.0)
        qv = QWEN36_TASKS.get(alias)
        dlt = (ov - qv) if qv is not None else float("nan")
        leaves.append((alias, ov, se, dlt))

    big_gaps = sorted(
        [x for x in leaves if x[3] == x[3] and abs(x[3]) >= 4.0],
        key=lambda t: t[3],
    )
    ornith_ahead = [x for x in big_gaps if x[3] > 0][-8:][::-1]
    qwen_ahead = [x for x in big_gaps if x[3] < 0][:12]
    weaks = [x for x in leaves if x[1] < 75.0]

    lines: list[str] = []
    a = lines.append
    a("# mmlu_llama comparison: Ornith-1.5-35B-A3B-NVFP4 vs Qwen3.6-35B-A3B-MTP")
    a("")
    a(
        f"| Ornith-1.5-35B-A3B-NVFP4 | Qwen3.6-35B-A3B-MTP |\n"
        f"|------------------------:|--------------------:|\n"
        f"| **{overall[0]:.2f}%** | **{QWEN36['overall'][0]:.2f}%** |"
    )
    a("")
    a(
        "Side-by-side `mmlu_llama` (5-shot, `strict_match` / `exact_match`) "
        "on local OpenAI-compatible serving."
    )
    a("")
    a("| | **ornith-ai/Ornith-1.5-35B-A3B-NVFP4** | **unsloth/qwen3.6-35b-a3b-mtp** |")
    a("|--|--|--|")
    a(f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | **{QWEN36['overall'][0]:.2f}% ± {QWEN36['overall'][1]:.2f}%** |")
    a(f"| Δ vs Qwen | **{delta:+.2f} pp** | baseline |")
    a("| Samples | 14,042 | 14,042 |")
    a("| Serving | vLLM Docker (`:8000`) | LM Studio (`:1234`) |")
    a("| Quant | NVFP4 | Unsloth MTP |")
    a("")
    a(
        f"**Headline:** Qwen3.6-35B-A3B-MTP is **{abs(delta):.1f} pp** "
        f"{'ahead' if delta < 0 else 'behind'} overall "
        f"({QWEN36['overall'][0]:.2f}% vs {overall[0]:.2f}%)."
    )
    a("")
    a("> Methodologies are not identical (decode settings / stack). Treat small subtask gaps as noise.")
    a("")
    a("## Domain scores")
    a("")
    a("| Domain | Ornith | Qwen MTP | Δ (Ornith − Qwen) |")
    a("|--------|-------:|---------:|------------------:|")
    for name in ("social sciences", "other", "stem", "humanities"):
        o, ose = domains[name]
        q, qse = QWEN36[name]
        a(f"| {name.capitalize()} | {o:.2f}% ± {ose:.2f}% | {q:.2f}% ± {qse:.2f}% | {o - q:+.2f} pp |")
    a(
        f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | "
        f"**{QWEN36['overall'][0]:.2f}% ± {QWEN36['overall'][1]:.2f}%** | "
        f"**{delta:+.2f} pp** |"
    )
    a("")
    a("## Setup (Ornith)")
    a("")
    a("```bash")
    a("lm-eval --model local-chat-completions \\")
    a('  --model_args "base_url=http://127.0.0.1:8000/v1/chat/completions,model=ornith-1.5-35b-a3b-nvfp4,..." \\')
    a("  --tasks mmlu_llama --apply_chat_template \\")
    a("  --gen_kwargs 'max_tokens=10,temperature=0,continue_final_message=true,add_generation_prompt=false'")
    a("```")
    a("")
    a(f"Results JSON: `{result_path}`")
    a("")
    a("Thinking disabled server-side (`enable_thinking: false`). Recipe: `vllm/recipes/ornith-1.5-35b-a3b-nvfp4`.")
    a("")
    a("## Largest gaps (|Δ| ≥ 4 pp)")
    a("")
    if ornith_ahead:
        a("### Ornith ahead")
        a("")
        a("| Subtask | Ornith | Qwen | Δ |")
        a("|---------|-------:|-----:|--:|")
        for alias, ov, _se, dlt in ornith_ahead:
            a(f"| {alias} | {ov:.1f}% | {QWEN36_TASKS[alias]:.1f}% | **{dlt:+.1f}** |")
        a("")
    else:
        a("Ornith has no subtasks ≥4 pp ahead of Qwen.")
        a("")
    if qwen_ahead:
        a("### Qwen ahead (biggest)")
        a("")
        a("| Subtask | Ornith | Qwen | Δ |")
        a("|---------|-------:|-----:|--:|")
        for alias, ov, _se, dlt in qwen_ahead:
            a(f"| {alias} | {ov:.1f}% | {QWEN36_TASKS[alias]:.1f}% | **{dlt:+.1f}** |")
        a("")
    a("## Weak spots (Ornith < 75%)")
    a("")
    a("| Subtask | Ornith | Qwen |")
    a("|---------|-------:|-----:|")
    for alias, ov, _se, _dlt in sorted(weaks, key=lambda t: t[1]):
        qv = QWEN36_TASKS.get(alias)
        qtxt = f"{qv:.1f}%" if qv is not None else "—"
        a(f"| {alias} | {ov:.1f}% | {qtxt} |")
    a("")
    a("## Full subtask table")
    a("")
    a("| Tasks | Ornith | ± | Qwen | Δ pp |")
    a("|-------|-------:|--:|-----:|-----:|")
    a(
        f"| **mmlu_llama** | **{overall[0]/100:.4f}** | {overall[1]/100:.4f} | "
        f"**{QWEN36['overall'][0]/100:.4f}** | **{delta:.2f}** |"
    )
    for alias, ov, se, dlt in leaves:
        qv = QWEN36_TASKS.get(alias)
        if qv is None:
            a(f"| {alias} | {ov/100:.4f} | {se/100:.4f} | — | — |")
        else:
            a(f"| {alias} | {ov/100:.4f} | {se/100:.4f} | {qv/100:.4f} | {dlt:+.2f} |")
    a("")
    a("## Interpretation")
    a("")
    a(
        f"1. Overall gap vs Qwen MTP: **{delta:+.2f} pp** on the same `mmlu_llama` harness."
    )
    a("2. Ornith is a coding/agentic-focused MoE; MMLU letter accuracy is a different axis than Terminal-Bench / SWE-bench.")
    a("3. This run used greedy short continuations with thinking off — closer to Puzzle/Fuse methodology than Qwen’s sampling `/no_think` setup.")
    a("")
    a(f"*Generated from `{result_path.name}` vs Derek’s Qwen3.6-35B-A3B-MTP table.*")
    a("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"overall {overall[0]:.2f}% delta_vs_qwen {delta:+.2f}pp")


if __name__ == "__main__":
    main()
