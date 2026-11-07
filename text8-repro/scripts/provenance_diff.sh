#!/usr/bin/env bash
# Diff both vendored phase_ssm trees from the HF snapshot against upstream
# PhaseMesh at the pinned commit. Writes results/provenance.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
load_text8_env "$ROOT"

HF_DIR="${HF_DIR:-$ROOT/data/hf/phasemesh-text8-comparison}"
UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/ashhart/PhaseMesh}"
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-3ae43ffac6c0699857cada0865f156405fb95956}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$ROOT/data/upstream/PhaseMesh}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
REPORT="$RESULTS_DIR/provenance.md"

mkdir -p "$RESULTS_DIR"

if [[ ! -d "$HF_DIR/transformer/vendor/phase_ssm" || ! -d "$HF_DIR/ssm/vendor/phase_ssm" ]]; then
  echo "Vendored sources not found under $HF_DIR. Run scripts/fetch_hf.sh first." >&2
  exit 1
fi

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --tags origin "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_COMMIT"

UPSTREAM_SSM="$UPSTREAM_DIR/phase_ssm"
if [[ ! -d "$UPSTREAM_SSM" ]]; then
  echo "Upstream tree has no phase_ssm/ directory at $UPSTREAM_COMMIT" >&2
  exit 1
fi

resolved_upstream="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
date_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python3 - "$HF_DIR" "$UPSTREAM_SSM" "$resolved_upstream" "$UPSTREAM_COMMIT" "$REPORT" "$date_utc" << 'PY'
import hashlib, os, subprocess, sys
from pathlib import Path

hf_dir, upstream, resolved, pinned, report, date_utc = sys.argv[1:7]
hf_dir = Path(hf_dir)
upstream = Path(upstream)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def files(root: Path):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}

def diff_pair(left: Path, right: Path):
    proc = subprocess.run(
        ["diff", "-u", str(left), str(right)],
        check=False, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout

upstream_files = files(upstream)
sides = {
    "transformer": files(hf_dir / "transformer" / "vendor" / "phase_ssm"),
    "ssm": files(hf_dir / "ssm" / "vendor" / "phase_ssm"),
}

# Compare the two vendor copies with each other.
shared = set(sides["transformer"]) & set(sides["ssm"])
only_tf = sorted(set(sides["transformer"]) - set(sides["ssm"]))
only_ssm = sorted(set(sides["ssm"]) - set(sides["transformer"]))
identical = []
vendor_mismatch = []
for rel in sorted(shared):
    if sha256(sides["transformer"][rel]) == sha256(sides["ssm"][rel]):
        identical.append(rel)
    else:
        vendor_mismatch.append(rel)

lines = []
lines.append("# PhaseMesh vendored-code provenance")
lines.append("")
lines.append(f"- Generated (UTC): `{date_utc}`")
lines.append(f"- Upstream pin requested: `{pinned}`")
lines.append(f"- Upstream checkout resolved: `{resolved}`")
lines.append(f"- Upstream tree: `{upstream}`")
lines.append(f"- Hugging Face snapshot: `{hf_dir}`")
lines.append("")
lines.append("This report diffs each published `vendor/phase_ssm` tree against upstream `phase_ssm/` at the pinned commit. It does not judge training quality.")
lines.append("")
lines.append("## Vendor copies compared with each other")
lines.append("")
lines.append(f"- Files present in both copies: {len(shared)}")
lines.append(f"- Byte-identical in both copies: {len(identical)}")
lines.append(f"- Present only under transformer/vendor/phase_ssm: {', '.join(only_tf) if only_tf else '(none)'}")
lines.append(f"- Present only under ssm/vendor/phase_ssm: {', '.join(only_ssm) if only_ssm else '(none)'}")
if vendor_mismatch:
    lines.append("- Hash mismatches between the two copies:")
    for rel in vendor_mismatch:
        lines.append(f"  - `{rel}`")
else:
    lines.append("- No hash mismatches among files present in both copies.")
lines.append("")

for name, vendor_files in sides.items():
    lines.append(f"## {name}/vendor/phase_ssm vs upstream phase_ssm")
    lines.append("")
    only_vendor = sorted(set(vendor_files) - set(upstream_files))
    only_up = sorted(set(upstream_files) - set(vendor_files))
    common = sorted(set(vendor_files) & set(upstream_files))
    changed = []
    unchanged = []
    diffs = []
    for rel in common:
        code, text = diff_pair(upstream_files[rel], vendor_files[rel])
        if code == 0:
            unchanged.append(rel)
        else:
            changed.append(rel)
            diffs.append((rel, text))
    lines.append(f"- Overlapping files: {len(common)}")
    lines.append(f"- Identical to upstream: {len(unchanged)}")
    lines.append(f"- Different from upstream: {len(changed)}")
    lines.append(f"- Only in vendor: {', '.join(only_vendor) if only_vendor else '(none)'}")
    lines.append("- Only in upstream (not vendored):")
    if only_up:
        for rel in only_up:
            lines.append(f"  - `{rel}`")
    else:
        lines.append("  - (none)")
    lines.append("")
    if unchanged:
        lines.append("Identical files:")
        for rel in unchanged:
            lines.append(f"- `{rel}` (`{sha256(vendor_files[rel])}`)")
        lines.append("")
    if changed:
        lines.append("Changed files (unified diff, upstream then vendor):")
        lines.append("")
        for rel, text in diffs:
            lines.append(f"### `{rel}`")
            lines.append("")
            lines.append("```diff")
            # Keep the report bounded if a file is huge.
            body = text if len(text) <= 200_000 else text[:200_000] + "\n... [diff truncated]\n"
            lines.append(body.rstrip())
            lines.append("```")
            lines.append("")
    else:
        lines.append("No content differences in overlapping files.")
        lines.append("")

import re
KEYWORDS = (
    "triton", "torch.compile", "cuda", "load_inline", "cpp_extension",
    "mamba", "causal_conv1d", "causal-conv1d", "nvcc", "cubin",
    "get_arch_list", "sm_90", "sm_100", "sm_120", "sm_121",
)
keyword_re = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)

def scan_tree(label, root: Path, filemap):
    hits = []
    for rel, path in sorted(filemap.items()):
        if path.suffix not in {".py", ".cu", ".cuh", ".cpp", ".h", ".md"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if keyword_re.search(line):
                hits.append(f"{label}:{rel}:{i}:{line.strip()}")
    return hits

lines.append("## Kernel and architecture-specific code")
lines.append("")
lines.append("Scanned vendored and upstream `phase_ssm` trees for Triton/CUDA kernels, `torch.compile`, mamba/causal-conv1d extensions, and explicit arch lists. GB10 is `sm_121` and is not named in this source.")
lines.append("")
lines.append("Known findings from the published vendor copies (independent of this scan):")
lines.append("- Published SSM training uses `ssm_backend=\"fft\"` (PyTorch `torch.fft.rfft` / `irfft`), not a custom CUDA kernel.")
lines.append("- `model.py` also implements `fixed_triton` via `_FixedKernelSSM`, which imports `.triton_scan.ssm_recurrent_triton` when the tensor is on CUDA. `triton_scan.py` is **not** present in either vendor tree.")
lines.append("- Backend `auto` selects `fixed_triton` when sequence length exceeds `ssm_auto_threshold` (default 32768). Published `seq` is 1024, so `auto` would stay on FFT.")
lines.append("- Short conv is `nn.Conv1d` depthwise with right-pad trimmed; not the `causal-conv1d` / Mamba package.")
lines.append("- `train_core.py` can `torch.compile(model)`, but the published runners (`run_matched.py`, `run_ssm.py`) only import `lr_at` and use their own loop.")
lines.append("- Transformer attention uses `F.scaled_dot_product_attention`, whose backend (flash/mem-efficient/math) is architecture-dependent and should be confirmed on sm_121.")
lines.append("")
all_hits = []
all_hits.extend(scan_tree("upstream", upstream, upstream_files))
for name, vendor_files in sides.items():
    all_hits.extend(scan_tree(f"{name}-vendor", hf_dir / name / "vendor" / "phase_ssm", vendor_files))
if all_hits:
    lines.append("Keyword hits:")
    lines.append("")
    lines.append("```")
    for hit in all_hits:
        lines.append(hit)
    lines.append("```")
    lines.append("")
else:
    lines.append("No keyword hits.")
    lines.append("")

report_path = Path(report)
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {report_path}")
PY
