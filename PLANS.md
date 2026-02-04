# Execution Plan (Wide Freeze WIDE2)

This plan mirrors the required Step A–D sequence and maps to the full pipeline script.

## Step A — Wide Build (Snapshot/Daily + Extras + MANIFEST)
**Objective:** Rebuild `offers_core_snapshot/daily` with full wide columns and nested derived features, including complete MANIFEST fields.

Run via the pipeline script:
- `scripts/run_wide_freeze_full.sh` (Steps 3–7)

Key outputs:
- `runs/offers_core_full_snapshot_wide_${WIDE_STAMP}/offers_core_snapshot.parquet`
- `runs/offers_core_full_snapshot_wide_${WIDE_STAMP}/offers_extras_snapshot.parquet`
- `runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet`
- `runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_extras_daily.parquet`
- `MANIFEST.json` in each directory (with output_columns, dropped_columns, schema, rows, versions)

## Step B — Raw Inventory (10M Samples)
**Objective:** Produce reproducible raw inventory parquet and public audit anchors.

Outputs (under analysis):
- `raw_offers_profile.json/.md`
- `raw_edgar_profile.json/.md`
- `raw_*_inventory.parquet`
- `docs/audits/raw_*_profile_${WIDE_STAMP}.md` + MANIFEST update

## Step C — Highest-Tier Audits (No Fallbacks)
**Objective:** Ensure full-scan coverage and strict value-level checks.

Audits:
- `scripts/audit_column_manifest.py`
  - Uses `configs/column_contract_wide.yaml`
  - EDGAR recompute requires `total_compared >= 200` and `diff_count == 0`
- `scripts/audit_raw_cardinality_coverage.py`
  - `raw_scan_limit=0`
  - Computes raw entity-day totals, structured-signal totals, text coverage by keys, pair-level snapshots→edgar coverage
  - Writes debug samples under `analysis/**/debug`

## Step D — Pointer & Candidate Selection
**Objective:** Ensure single source of truth for Block 3.

Outputs:
- `runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}/freeze_candidates.{json,md}`
- Updated `docs/audits/FULL_SCALE_POINTER.yaml`
- `freeze_pointer_status.{json,md}`
- `docs/audits/MANIFEST.json` updated with anchor hashes

---

# Recommended Execution (Iris/Aion Slurm)

## Update code
```bash
cd ~/repo_root
git pull --ff-only
```

## Submit job (iris or aion)
```bash
sbatch -M iris scripts/slurm/run_wide_freeze_iris.sbatch
# or
sbatch -M aion scripts/slurm/run_wide_freeze_iris.sbatch
```

## Optional resource overrides
```bash
sbatch -M iris --partition=bigmem --time=2-00:00:00 scripts/slurm/run_wide_freeze_iris.sbatch
```

## Environment overrides
```bash
sbatch -M iris --export=ALL,WIDE_STAMP=20260203_120000,SQLITE_DIR=/work/$USER/sqlite_wide2,MIN_FREE_GB=200 \
  scripts/slurm/run_wide_freeze_iris.sbatch
```

---

# Resource Guidance (for full run)
- **CPU:** 1 node, 28 cores
- **RAM:** 120–150 GB
- **Disk:** > 200 GB free for SQLite + intermediates
- **Walltime:** 1–2 days (I/O bound)

---

# Gate Success Criteria (Block 3)
- `column_manifest.json`: `must_keep_missing == 0`, `edgar_recompute.total_compared >= 200`, `diff_count == 0`
- `raw_cardinality_coverage.json`: full-scan counts, pair-level snapshots→edgar coverage, no fallback
- `FULL_SCALE_POINTER.yaml` points to the **WIDE2** paths only
