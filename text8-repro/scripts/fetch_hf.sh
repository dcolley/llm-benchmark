#!/usr/bin/env bash
# Download the pinned Hugging Face PhaseMesh text8 comparison snapshot and
# verify sha256 hashes against MANIFEST.json. If MANIFEST.json is absent, a
# generated hash list is written under results/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
load_text8_env "$ROOT"

HF_REPO="${HF_REPO:-MarxistLeninist/phasemesh-text8-comparison}"
HF_REVISION="${HF_REVISION:-2eb1bef95b39e9745035044679f9d18b8ac3513a}"
HF_DIR="${HF_DIR:-$ROOT/data/hf/phasemesh-text8-comparison}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
HF_BASE_URL="${HF_BASE_URL:-https://huggingface.co}"

mkdir -p "$HF_DIR" "$RESULTS_DIR"

manifest_url="${HF_BASE_URL}/${HF_REPO}/resolve/${HF_REVISION}/MANIFEST.json"
generated_manifest=0
if curl -fsIL --retry 3 "$manifest_url" >/dev/null 2>&1; then
  download_url "$manifest_url" "$HF_DIR/MANIFEST.json"
else
  echo "MANIFEST.json was not found at the pinned revision; generating hashes after download."
  generated_manifest=1
fi

python3 - "$HF_REPO" "$HF_REVISION" "$HF_DIR" "$RESULTS_DIR" "$HF_BASE_URL" "$generated_manifest" << 'PY'
import hashlib, json, os, sys, urllib.request

repo, rev, dest, results_dir, base, generated = sys.argv[1:7]
generated = generated == "1"
dest = os.path.abspath(dest)
os.makedirs(dest, exist_ok=True)

def urlopen(url):
    req = urllib.request.Request(url, headers={"User-Agent": "text8-repro"})
    return urllib.request.urlretrieve if False else urllib.request.urlopen(req)

def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "text8-repro"})
    with urllib.request.urlopen(req) as resp, open(path + ".tmp", "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(path + ".tmp", path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def list_repo_paths():
    api = f"{base}/api/models/{repo}/tree/{rev}?recursive=1"
    req = urllib.request.Request(api, headers={"User-Agent": "text8-repro"})
    with urllib.request.urlopen(req) as resp:
        entries = json.load(resp)
    paths = []
    for entry in entries:
        if entry.get("type") == "file":
            paths.append(entry["path"])
    if not paths:
        raise SystemExit(f"HF API returned no files for {repo}@{rev}")
    return paths

manifest_path = os.path.join(dest, "MANIFEST.json")
if generated or not os.path.exists(manifest_path):
    paths = list_repo_paths()
    manifest = None
else:
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Support {path: {sha256, bytes}} or {path: "sha256"}
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("MANIFEST.json is empty or not an object")
    sample = next(iter(raw.values()))
    if isinstance(sample, dict):
        manifest = raw
        paths = list(raw)
    elif isinstance(sample, str):
        manifest = {k: {"sha256": v} for k, v in raw.items()}
        paths = list(raw)
    else:
        raise SystemExit("Unrecognised MANIFEST.json format")

errors = []
computed = {}
for rel in paths:
    if rel == "MANIFEST.json" and os.path.exists(manifest_path) and not generated:
        digest = sha256(manifest_path)
        size = os.path.getsize(manifest_path)
        computed[rel] = {"bytes": size, "sha256": digest}
        continue
    url = f"{base}/{repo}/resolve/{rev}/{rel}"
    path = os.path.join(dest, rel)
    download(url, path)
    digest = sha256(path)
    size = os.path.getsize(path)
    computed[rel] = {"bytes": size, "sha256": digest}
    if not generated:
        expected = manifest[rel]
        exp_hash = expected["sha256"] if isinstance(expected, dict) else expected
        exp_bytes = expected.get("bytes") if isinstance(expected, dict) else None
        if digest != exp_hash:
            errors.append(f"{rel}: sha256 mismatch expected {exp_hash} got {digest}")
        if exp_bytes is not None and size != exp_bytes:
            errors.append(f"{rel}: size mismatch expected {exp_bytes} got {size}")

if generated:
    out = os.path.join(results_dir, "generated_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(computed, f, indent=2)
        f.write("\n")
    # Also place a copy next to the snapshot so later verify passes.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(computed, f, indent=2)
        f.write("\n")
    print(f"Wrote generated manifest to {out} and {manifest_path}")
else:
    # Record verification summary even when the published manifest is used.
    summary = {
        "repo": repo,
        "revision": rev,
        "manifest_source": "published",
        "files_checked": len(computed),
        "ok": not errors,
        "errors": errors,
    }
    out = os.path.join(results_dir, "hf_verify.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {len(computed)} files against MANIFEST.json")
PY

echo "HF snapshot is at $HF_DIR (revision $HF_REVISION)"
