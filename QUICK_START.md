# Quick Start

This repository is in recovery mode. Use the smoke pipeline first, then proceed to
audited benchmarks only after recovery gates pass.

## Smoke run (safe)

1) Build a tiny `offers_core_smoke.parquet` (see RUNBOOK).
2) Run a minimal benchmark:
   - `label_horizon=3`
   - `label_goal_min=50`
   - `limit_rows=5000`
   - `models=dlinear patchtst`
3) Run Gates A–D and `summarize_horizon_audit.py`.

## Recovery checkpoints

- `docs/ESSENTIAL_FILES.md` defines required files.
- `runs/backups/` contains snapshots and checksums.
