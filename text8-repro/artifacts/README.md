# Reviewed artifacts

This folder is the only place generated scoring outputs should be committed.

Copy them with:

```
python scripts/export_artifacts.py --src results --dest artifacts
```

Allowlist (nothing else is copied):

- `scores.json`
- `device_report.json`
- `provenance.md`
- `gate.json`
- `gate.txt`

The exporter refuses weights, datasets, `.env` files, and other secrets. Run it
only after reviewing the ignored `results/` outputs.
