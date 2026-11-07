# Reviewed artifacts

This folder is the only place generated scoring and training-record outputs
should be committed. Destinations must be fresh subdirectories of `artifacts/`
(not `artifacts/` itself). The exporter refuses a dest that exists and is
non-empty, that is or contains a symlink, or that has a symlink anywhere in
its ancestor path (every component is lstat'd from the filesystem root).

Use a separate destination for each stage:

```
python scripts/export_artifacts.py --stage scoring --src "$RESULTS_DIR" --dest artifacts/reviewed-scoring
python scripts/export_artifacts.py --stage complete --src "$RESULTS_DIR" --runs-src "$RUNS_DIR" --dest artifacts/reviewed-complete
```

`--stage scoring` copies only:

- `scores.json`
- `device_report.json`
- `provenance.md`
- `gate.json`
- `gate.txt`

It does not read `RUNS_DIR` and rejects `--runs-src`.

`--stage complete` copies those scoring files plus these required training
records from `--runs-src` (or `RUNS_DIR`):

- `runs/transformer/progress.json`
- `runs/transformer/wrapper_timing.json`
- `runs/ssm/progress.json`
- `runs/ssm/wrapper_timing.json`

A missing required file fails the export. The exporter refuses weights,
datasets, `.env` files, and other secrets. Run it only after reviewing the
ignored `results/` outputs (and `runs/` for the complete stage).
