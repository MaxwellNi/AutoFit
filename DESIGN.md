# Wide Freeze Design

## Pipeline Data Products
1. **Wide Snapshot (Step 3)**
   - Source: `data/raw/offers` (Delta)
   - Output: `runs/offers_core_full_snapshot_wide_${STAMP}/offers_core_snapshot.parquet`
   - Nested columns route to extras as:
     - `<col>__json`, `<col>__len`, `<col>__hash`

2. **Wide Daily (Step 4)**
   - Source: wide snapshot
   - Output: `runs/offers_core_full_daily_wide_${STAMP}/offers_core_daily.parquet`
   - Aggregation: last-non-null by `entity_id` and day
   - Derived columns include delta and pct change

3. **Snapshots Index (Step 5)**
   - `snapshots_offer_day.parquet`
   - `snapshots_cik_day.parquet`

4. **EDGAR Store (Step 6)**
   - Output: `runs/edgar_feature_store_full_daily_wide_${STAMP}/edgar_features/**`
   - EDGAR aligned to snapshot day and cik

5. **Multiscale Views (Step 7)**
   - Output: `runs/multiscale_full_wide_${STAMP}/`

6. **Audits + Freeze Selection (Steps 8-10)**
   - `column_manifest.{json,md}`
   - `raw_cardinality_coverage_wide_${STAMP}.{json,md}`
   - `freeze_candidates.{json,md}`
   - updated `docs/audits/FULL_SCALE_POINTER.yaml`

## Runtime Topologies

### Topology A: Cluster SLURM (iris/aion)
- Full: `scripts/slurm/run_wide_freeze_aion.sbatch`
- Resume: `scripts/slurm/run_wide_freeze_aion_from_daily.sbatch`
- Monitor: `scripts/slurm/monitor_wide_freeze_aion.sbatch` + `scripts/monitor_wide_freeze_resume.sh`

### Topology B: 4090 Local (no SLURM)
- Full: `scripts/ssh_4090/run_wide_freeze_aion_4090.sh`
- Resume: `scripts/ssh_4090/run_wide_freeze_aion_from_daily_4090.sh`
- Local monitor: `scripts/ssh_4090/monitor_wide_freeze_aion_4090.sh`

## Hardening Added During This Thread
- Daily builder now supports DuckDB runtime controls via env:
  - `DUCKDB_MEMORY_LIMIT_GB`
  - `DUCKDB_THREADS`
- Sbatch wrappers detect explicit bash path via `BASH_BIN` fallback chain.
- Resume sbatch preflights and auto-installs `duckdb` and `deltalake` when missing.
- Full/Resume wrappers export OpenMP/BLAS thread limits for reproducibility.

## Data Mobility Design (Implemented)
- Cluster -> Mac pull:
  - `scripts/pull_outputs_from_cluster.sh`
  - defaults set for ULHPC access: `access-iris.uni.lu:8022`
- Mac -> 4090 push:
  - `scripts/sync_outputs_to_4090.sh`
- Optional Cluster -> Mac push:
  - `scripts/sync_outputs_to_mac.sh`
- Required synced sets:
  - `runs/offers_core_full_snapshot_wide_${STAMP}`
  - `runs/offers_core_full_daily_wide_${STAMP}` (if exists)
  - `runs/edgar_feature_store_full_daily_wide_${STAMP}` (if exists)
  - `runs/multiscale_full_wide_${STAMP}` (if exists)
  - `runs/orchestrator/20260129_073037/analysis/wide_${STAMP}`
  - `runs/selections/b11_v2_canonical`
  - `runs/offers_text_v1_20260129_073037_full` (if referenced)
  - `configs/column_contract_wide.yaml`
  - `docs/audits/*`

## Failure Modes and Applied Fixes
1. Step 4 OOM (observed in job `11029034`)
   - Fix: move daily build to DuckDB backend and export memory/thread limits.
2. `bash` missing within `srun` (observed in job `11029184`)
   - Fix: `BASH_BIN` detection and explicit executable path in sbatch.
3. Invalid CPU shape at submit time
   - Fix: `--ntasks-per-node=1`, `-c 16`.
4. Queue stagnation due priority + whole-node requests
   - Parallel submission strategy added (`mem=64G` variant).

## Manifest and Audit Requirements
Every product directory must include `MANIFEST.json` with:
- source version metadata
- rows scanned/emitted
- cmd args and git hash
- schema/output columns
- grain and partition strategy

Audit PASS gates:
- `column_manifest.json`: contract integrity and EDGAR recompute checks
- `raw_cardinality_coverage_wide_${STAMP}.json`: full-scan coverage integrity
- pointer update in `docs/audits/FULL_SCALE_POINTER.yaml`

## Single Source of Truth
`docs/audits/FULL_SCALE_POINTER.yaml` remains the only approved Block 3 input pointer.
