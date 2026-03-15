# BLOCK3 READY FULL-SCALE (20260131_235740)

Full-scale data freeze completed. All audit gates PASS.

## Base Data MANIFEST Summary

### offers_core_full_daily
- Path: runs/offers_core_full_daily_20260131_235740/
- rows_scanned: 112,608,525 (raw offers)
- rows_emitted: 5,774,931
- n_unique_entities: 22,569
- date_range: 2022-04-18 to 2026-01-19
- raw delta version: 86604, active_files: 2832

### offers_text_full
- Path: runs/offers_text_v1_20260129_073037_full/
- raw_rows_scanned: 112,608,525
- rows_emitted: 5,774,931
- n_unique_entity_id: 22,569

### edgar_store_full_daily
- Path: runs/edgar_feature_store_full_daily_20260131_235740/edgar_features
- MANIFEST: runs/edgar_feature_store_full_daily_20260131_235740/MANIFEST.json
- row_count: 1,325,222 (aligned to snapshots_cik_day)
- raw edgar version: 1639, active_files: 46
- partition_strategy: snapshot_year

## Key Audit Evidences

### 1. EDGAR MANIFEST (required)
- Path: runs/edgar_feature_store_full_daily_20260131_235740/MANIFEST.json
- Fields: raw_edgar_delta_version, raw_edgar_active_files, snapshots_index_rows, output_rows, output_columns, partition_strategy, git_head, built_at

### 2. EDGAR Recompute (value-level gate)
- column_manifest.json: edgar_recompute.total_compared >= 200, diff_count == 0
- Must use actual edgar_store_full_daily; no fallback for PASS
- Granularity: offer-day or cik-day (matches store)

### 3. Full Coverage / Structured-Signal Audit
- raw_cardinality_coverage.json: raw_vs_core_coverage, text_coverage, snapshots_to_edgar_coverage
- raw_cardinality_coverage_full_daily.json: raw_offers_total_rows, raw_structured_signal_entity_day_count, offers_core_entity_day_count, structured_signal_covered_by_core
- gap_reasons_top_k for snapshots→edgar alignment

## Raw Profiles / Inventory
- raw_offers_profile: 148 columns, delta_log stats
- raw_edgar_profile: delta_log stats
- raw_offers_column_inventory.parquet: scan_sample (10M rows) with std/top_k when available
- raw_edgar_column_inventory.parquet: idem

## Column Contracts
- configs/column_contract_v3.yaml: coverage_min=0.05
- configs/column_contract_wide.yaml: non_null_min=0.001, wide-table principle
- offers_core_snapshot/daily: must_keep, should_add, can_drop
- offers_text: must_keep
- edgar_store: must_keep (27 last/mean/ema features)

## Audit Gate Status
| Audit | Gate | Path |
|-------|------|------|
| column_manifest | PASS | runs/orchestrator/20260129_073037/analysis/column_manifest.json |
| raw_cardinality_coverage | PASS | runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage.json |
| raw_cardinality_coverage_full_daily | PASS | runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage_full_daily.json |
| distillation_fidelity | PASS | runs/orchestrator/20260129_073037/analysis/distillation_fidelity_report.json |

## Multi-Scale Products
- runs/multiscale_full_20260131_235740/offers_weekly.parquet
- runs/multiscale_full_20260131_235740/offers_monthly.parquet
- snapshots_index: snapshots_offer_day.parquet, snapshots_cik_day.parquet

## Reproducible Commands (4090)
```bash
# Profile
PYTHONUNBUFFERED=1 python scripts/profile_raw_delta_columns.py \
  --raw_delta data/raw/offers --mode offers \
  --output_json runs/orchestrator/20260129_073037/analysis/raw_offers_profile.json \
  --output_md runs/orchestrator/20260129_073037/analysis/raw_offers_profile.md

PYTHONUNBUFFERED=1 python scripts/profile_raw_delta_columns.py \
  --raw_delta data/raw/edgar/accessions --mode edgar \
  --output_json runs/orchestrator/20260129_073037/analysis/raw_edgar_profile.json \
  --output_md runs/orchestrator/20260129_073037/analysis/raw_edgar_profile.md

# Contract v3
python scripts/build_column_contract_v3.py \
  --raw_offers_profile runs/orchestrator/20260129_073037/analysis/raw_offers_profile.json \
  --raw_edgar_profile runs/orchestrator/20260129_073037/analysis/raw_edgar_profile.json \
  --output_yaml configs/column_contract_v3.yaml

# Column manifest audit (no fallback; recompute on actual edgar_store_full_daily)
HOST_TAG=4090 python scripts/audit_column_manifest.py \
  --offers_core runs/offers_core_full_daily_20260131_235740/offers_core_daily.parquet \
  --offers_static runs/offers_core_full_daily_20260131_235740/offers_static.parquet \
  --offers_text runs/offers_text_v1_20260129_073037_full/offers_text.parquet \
  --edgar_dir runs/edgar_feature_store_full_daily_20260131_235740/edgar_features \
  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --edgar_min_compared 200 --require_deltalake 1 --text_required_mode 1 \
  --contract_path configs/column_contract_v3.yaml

# Raw cardinality coverage (full_daily focus)
HOST_TAG=4090 python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet runs/offers_core_full_daily_20260131_235740/offers_core_daily.parquet \
  --offers_core_manifest runs/offers_core_full_daily_20260131_235740/MANIFEST.json \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir runs/edgar_feature_store_full_daily_20260131_235740/edgar_features \
  --snapshots_index_parquet runs/offers_core_full_daily_20260131_235740/snapshots_index/snapshots_offer_day.parquet \
  --output_dir runs/orchestrator/20260129_073037/analysis --docs_audits_dir docs/audits
```
