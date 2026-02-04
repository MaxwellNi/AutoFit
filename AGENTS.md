# Agent Context

## Mission
Upgrade `TRAIN_WIDE_FINAL` to a true wide feature freeze with highest-tier audits, so future modeling does not require backfilling features. This is the last data freeze before Block 3 training.

## Hard Constraints
- Only commit/push: `scripts/`, `configs/`, `docs/audits/`, and `docs/audits/MANIFEST.json`.
- Never commit anything under `runs/`.
- Never create/track files whose **names** contain: `cursor`, `prompt`, `recovery`, `runbook`, `decision`, `transcript` (case-insensitive).
- All tracked files must be **English** only.
- Block 3 must read **only** `docs/audits/FULL_SCALE_POINTER.yaml`.

## Current State (as of 2026-02-03)
- 4090 storage is insufficient; execution migrated to shared iris/aion storage.
- aion job failed because `deltalake` was missing. `scripts/run_wide_freeze_full.sh` now auto-installs `deltalake` if absent.
- Slurm QoS errors on iris/aion; `scripts/slurm/run_wide_freeze_iris.sbatch` now uses `batch` partition with no QoS directive.
- Wide snapshot/daily pipeline hardened:
  - Robust time column fallback (`crawled_date_day` → `crawled_date` → `snapshot_ts`).
  - Nested columns are routed to derived `__json/__len/__hash` features.
  - SQLite temp DB location is configurable (`--sqlite_dir`) to avoid `/tmp` overflow.
  - Parquet output schema is stabilized to prevent chunk type mismatch errors.
  - Daily aggregation preserves key columns and avoids groupby key loss.
- Smoke tests for snapshot/daily succeed; full wide run is still pending on iris/aion.
- This local repo currently shows many deletions (uncommitted). **Do not stage or commit those deletions.**

## Key Scripts
- Full pipeline: `scripts/run_wide_freeze_full.sh`
- Slurm wrapper: `scripts/slurm/run_wide_freeze_iris.sbatch`
- Wide snapshot build: `scripts/build_offers_core_daily_full.py` (called via `build_offers_core_full_snapshot.py`)
- Wide daily build: `scripts/build_offers_core_full_daily.py`
- Raw inventory: `scripts/profile_raw_delta_columns.py`
- Audits: `scripts/audit_column_manifest.py`, `scripts/audit_raw_cardinality_coverage.py`
- Pointer & candidates: `scripts/update_wide_pointer.py`, `scripts/inspect_freeze_pointer.py`, `scripts/freeze_candidates.py`

## Execution Notes
- Use `SQLITE_DIR` on a large filesystem; `/tmp` will overflow on full scans.
- The full pipeline writes all outputs under `runs/` and audits under `docs/audits/`.
- The slurm wrapper calls `run_wide_freeze_full.sh` end-to-end (Steps A–D).

## Pending Work
- Run the full wide pipeline on iris/aion and verify all gates PASS.
- Update `FULL_SCALE_POINTER.yaml` to the new WIDE2 stamp and ensure all audit anchors are recorded.
- Produce final summary tables (selected artifacts and audit gates) for Block 3 readiness.
