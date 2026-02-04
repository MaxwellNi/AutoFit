# Execution Plan (Wide Freeze WIDE2)

This plan is updated with the actual execution history in this thread and the current handoff path (cluster -> Mac -> 4090).

## Current Progress Snapshot
- Completed: Steps 1-3 for `WIDE_STAMP=20260203_225620`.
- Not completed: Steps 4-10.
- Known successful artifacts:
  - `runs/offers_core_full_snapshot_wide_20260203_225620/offers_core_snapshot.parquet`
  - `runs/offers_core_full_snapshot_wide_20260203_225620/MANIFEST.json`
- Known failed attempts:
  - `11029034`: OOM at Step 4.
  - `11029184`: failed before Step 4 due to missing `bash` in `srun`.

## Work Already Done (Code/Scripts)
1. Added robust resume pipeline:
   - `scripts/run_wide_freeze_from_daily.sh` (Step 4-10)
   - `scripts/slurm/run_wide_freeze_aion_from_daily.sbatch`
2. Added OOM protection:
   - `DUCKDB_MEMORY_LIMIT_GB` and `DUCKDB_THREADS` in `scripts/build_offers_core_full_daily.py`
   - default limits exported in sbatch wrappers.
3. Added runtime hardening:
   - `BASH_BIN` detection in sbatch wrappers.
   - dependency preflight (`duckdb`, `deltalake`) in resume sbatch.
4. Added monitor automation:
   - `scripts/monitor_wide_freeze_resume.sh`
   - `scripts/slurm/monitor_wide_freeze_aion.sbatch`
5. Added 4090 local adapters:
   - `scripts/ssh_4090/*.sh`
6. Added sync path:
   - cluster push Mac: `scripts/sync_outputs_to_mac.sh`
   - Mac push 4090: `scripts/sync_outputs_to_4090.sh`
   - Mac pull cluster (preferred): `scripts/pull_outputs_from_cluster.sh`

## Active Queue Status (Cluster)
- `5168794` pending (`wide_freeze_aion_resume`)
- `5168809` pending (`wide_freeze_aion_resume`)
- `5168800` pending (`monitor_wide_freeze_aion`)
- Reason: `Priority` (no running job yet).

## Next Action Plan

### A. Finish data transfer chain
1. On Mac, authenticate to ULHPC using the intended key (`id_ed25519_iris`).
2. Pull cluster outputs to Mac:
   - `bash scripts/pull_outputs_from_cluster.sh`
3. Push pulled outputs from Mac to 4090:
   - `bash scripts/sync_outputs_to_4090.sh`

### B. Execute remaining steps on 4090
Run Step 4-10 directly on 4090 (non-SLURM):
```bash
WIDE_STAMP=20260203_225620 bash scripts/ssh_4090/run_wide_freeze_aion_from_daily_4090.sh
```

### C. Validate completion gates
Required for completion:
- Step 4: `runs/offers_core_full_daily_wide_20260203_225620/offers_core_daily.parquet`, `MANIFEST.json`
- Step 5: `runs/offers_core_full_daily_wide_20260203_225620/snapshots_index/`
- Step 6: `runs/edgar_feature_store_full_daily_wide_20260203_225620/edgar_features/`
- Step 7: `runs/multiscale_full_wide_20260203_225620/MANIFEST.json`
- Step 8: `analysis/wide_20260203_225620/column_manifest.{json,md}`
- Step 9: `analysis/wide_20260203_225620/raw_cardinality_coverage_wide_20260203_225620.{json,md}`
- Step 10: `analysis/wide_20260203_225620/freeze_candidates.{json,md}` and updated `docs/audits/FULL_SCALE_POINTER.yaml`

## Fallback Rules (if 4090 run fails)
1. OOM in Step 4:
   - `DUCKDB_THREADS=4`
   - `DUCKDB_MEMORY_LIMIT_GB=<~60% of RAM>`
2. Missing dependency:
   - install in active env, rerun same command.
3. Missing input paths:
   - re-run sync scripts for the missing directories only.

## Block 3 Exit Criteria
- All Step 4-10 artifacts exist and are internally consistent.
- Audit JSON/MD outputs present and PASS thresholds met.
- `FULL_SCALE_POINTER.yaml` points only to final wide paths for `WIDE_STAMP=20260203_225620`.
