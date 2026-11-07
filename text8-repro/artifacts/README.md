# Reviewed artifacts

This folder is the only place generated scoring outputs should be committed.

Copy them with:

```
python scripts/export_artifacts.py --src "$RESULTS_DIR" --runs-src "$RUNS_DIR" --dest artifacts/reviewed
```

Allowlist (nothing else is copied):

- `scores.json`
- `device_report.json`
- `provenance.md`
- `gate.json`
- `gate.txt`
- `transformer/progress.json` and `transformer/wrapper_timing.json` if present
- `ssm/progress.json` and `ssm/wrapper_timing.json` if present

The exporter refuses weights, datasets, `.env` files, and other secrets. Run it
only after reviewing the ignored `results/` outputs.
