#!/usr/bin/env python3
"""Write MMLU_LLAMA_REPORT.md for DeepSeek-V4-Flash vs Qwen3.6-35B-A3B-MTP."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

QWEN = {
    "overall": (84.76, 0.29),
    "social sciences": (91.55, 0.49),
    "other": (87.22, 0.57),
    "stem": (83.44, 0.64),
    "humanities": (79.57, 0.57),
}


def pct(x: float) -> float:
    return 100.0 * float(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="/home/derek/llm-benchmark/lm-eval/results/deepseek-v4-flash-0731",
    )
    ap.add_argument(
        "--out",
        default="/home/derek/llm-benchmark/lm-eval/results/deepseek-v4-flash-0731/MMLU_LLAMA_REPORT.md",
    )
    args = ap.parse_args()
    root = Path(args.results_dir)
    matches = sorted(root.glob("**/results_*.json"))
    matches = [p for p in matches if "/smoke" not in str(p)]
    if not matches:
        raise SystemExit(f"no results_*.json under {root}")
    path = matches[-1]
    data = json.loads(path.read_text())
    res = data["results"]

    def score(key: str) -> tuple[float, float]:
        row = res[key]
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
    delta = overall[0] - QWEN["overall"][0]

    # weakest / strongest leaf tasks
    leaves = []
    for k, v in res.items():
        if not k.startswith("mmlu_llama_ds4_"):
            continue
        if k in {
            "mmlu_llama_ds4",
            "mmlu_llama_ds4_stem",
            "mmlu_llama_ds4_other",
            "mmlu_llama_ds4_social_sciences",
            "mmlu_llama_ds4_humanities",
        }:
            continue
        if "exact_match,strict_match" not in v:
            continue
        alias = v.get("alias", k.replace("mmlu_llama_ds4_", "").replace("_", " "))
        leaves.append((pct(v["exact_match,strict_match"]), alias))
    leaves.sort()
    weakest = leaves[:8]
    strongest = list(reversed(leaves[-8:]))

    n = data.get("n-samples") or {}
    # fallback sample count
    sample_n = 14042
    for v in n.values() if isinstance(n, dict) else []:
        if isinstance(v, dict) and "effective" in v:
            sample_n = max(sample_n, int(v["effective"]))

    lines = []
    lines.append("# mmlu_llama comparison: DeepSeek-V4-Flash-0731 vs Qwen3.6-35B-A3B-MTP")
    lines.append("")
    lines.append(
        "| DeepSeek-V4-Flash-0731 | Qwen3.6-35B-A3B-MTP | Ling INT4 (context) | Muse Glimmer (context) |"
    )
    lines.append("|-----------------------:|--------------------:|--------------------:|-----------------------:|")
    lines.append(
        f"| **{overall[0]:.2f}%** | **84.76%** | 83.61% | 82.13% |"
    )
    lines.append("")
    lines.append(
        "Side-by-side generative MMLU (`mmlu_llama` family, 5-shot, `strict_match` / `exact_match`)."
    )
    lines.append("")
    lines.append("| | **DeepSeek-V4-Flash-0731 (ds4 IQ2XXS/Q2K)** | **unsloth/qwen3.6-35b-a3b-mtp** |")
    lines.append("|--|--|--|")
    lines.append(
        f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | **84.76% ± 0.29%** |"
    )
    lines.append(f"| Δ vs Qwen | **{delta:+.2f} pp** | baseline |")
    lines.append(f"| Samples | {sample_n} | 14,042 |")
    lines.append("| Model | DeepSeek-V4-Flash-0731 (GGUF IQ2XXS/Q2K + DSpark) | 35B-A3B (Unsloth MTP) |")
    lines.append("| Serving | local ds4-server (`:8000`, ctx 8192 for eval) | LM Studio (`:1234`) |")
    lines.append(f"| Date | {path.name} | ~2026-06-16 |")
    lines.append("")
    verb = "trails" if delta < 0 else "leads"
    lines.append(
        f"**Headline:** DeepSeek-V4-Flash-0731 {verb} Qwen3.6-35B-A3B-MTP by about "
        f"**{abs(delta):.1f} points** overall ({overall[0]:.2f}% vs 84.76%) under this harness."
    )
    lines.append("")
    lines.append(
        "> Methodologies are **not identical** (see [Setup differences](#setup-differences)). "
        "Treat domain gaps as directional."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Domain scores")
    lines.append("")
    lines.append("| Domain | DeepSeek V4 Flash | Qwen3.6 MTP | Δ (DS − Qwen) |")
    lines.append("|--------|------------------:|------------:|--------------:|")
    for name in ["social sciences", "other", "stem", "humanities"]:
        d, e = domains[name]
        qd, _ = QWEN[name]
        lines.append(f"| {name.capitalize() if name != 'stem' else 'STEM'} | {d:.2f}% ± {e:.2f}% | {qd:.2f}% ± {QWEN[name][1]:.2f}% | {d-qd:+.2f} pp |")
    lines.append(
        f"| **Overall** | **{overall[0]:.2f}% ± {overall[1]:.2f}%** | **84.76% ± 0.29%** | **{delta:+.2f} pp** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Setup differences")
    lines.append("")
    lines.append("### DeepSeek-V4-Flash-0731 (this run)")
    lines.append("")
    lines.append("```bash")
    lines.append("# Serve (short ctx is enough for MMLU and frees KV for concurrency)")
    lines.append("ds4-server --cuda -m $HOME/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-....gguf -c 8192 --port 8000")
    lines.append("")
    lines.append("# Eval")
    lines.append("./scripts/run_mmlu_llama_deepseek_v4_flash.sh")
    lines.append("nohup ./scripts/mmlu_watchdog_deepseek_v4_flash.sh >/dev/null 2>&1 &")
    lines.append("```")
    lines.append("")
    lines.append("Harness notes:")
    lines.append("")
    lines.append("- Task pack `mmlu_llama_ds4`: same prompts/targets as `mmlu_llama`, but **no `gen_prefix` continuation**.")
    lines.append(
        "- Stock `mmlu_llama` uses `until: [\".\"]` after prefacing `The best answer is`; "
        "this quantised DeepSeek stack often emits `.` first under continuation, which yields empty answers."
    )
    lines.append(
        "- Instead: full assistant turns, `max_tokens=512`, `temperature=0`, `reasoning_effort=low`, "
        "regex extract of `The best answer is [A-D]` or a bare letter reply."
    )
    lines.append("- `num_concurrent=4` against local ds4-server.")
    lines.append(f"- Results JSON: `{path}`")
    lines.append("")
    lines.append("### Qwen3.6-35B-A3B-MTP (Derek’s prior run)")
    lines.append("")
    lines.append("- Thinking disabled via `/no_think` system instruction; LM Studio sampling (`max_tokens=512`).")
    lines.append("- Shared: generative MMLU letter answers, chat template, `strict_match` exact match.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Notable weak / strong subjects (DeepSeek)")
    lines.append("")
    lines.append("Lowest:")
    lines.append("")
    lines.append("| Subtask | Score |")
    lines.append("|---------|------:|")
    for s, a in weakest:
        lines.append(f"| {a} | {s:.2f}% |")
    lines.append("")
    lines.append("Highest:")
    lines.append("")
    lines.append("| Subtask | Score |")
    lines.append("|---------|------:|")
    for s, a in strongest:
        lines.append(f"| {a} | {s:.2f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    lines.append(
        f"1. **Overall Δ vs Qwen: {delta:+.2f} pp** on this generative letter-answer harness."
    )
    lines.append(
        "2. Domain rank order should be compared carefully given the DeepSeek-specific extraction path."
    )
    lines.append(
        "3. Cross-stack caveats: DeepSeek used greedy decode on ds4 GGUF; Qwen used LM Studio sampling with `/no_think`."
    )
    lines.append("")
    lines.append(
        f"*Generated from `{path.name}` and Derek Colley’s published Qwen3.6-35B-A3B-MTP mmlu_llama table (2026-06-16).*"
    )
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print(f"overall {overall[0]:.2f}% delta_vs_qwen {delta:+.2f}pp")


if __name__ == "__main__":
    main()
