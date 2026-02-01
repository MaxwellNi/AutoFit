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
- row_count: ~26M (aligned to snapshots)
- raw edgar version: 1639, active_files: 46

## Raw Profiles
- raw_offers_profile: 148 columns, delta_log stats
- raw_edgar_profile: delta_log stats

## v3 Column Contract
- configs/column_contract_v3.yaml
- offers_core_snapshot/daily: must_keep, should_add, can_drop
- offers_text: must_keep
- edgar_store: must_keep (27 last/mean/ema features)

## Audit Gate Status
| Audit | Gate | Path |
|-------|------|------|
| column_manifest | PASS | runs/orchestrator/20260129_073037/analysis/column_manifest.json |
| raw_cardinality_coverage | PASS | runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage.json |
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

# Column manifest audit (use edgar_recompute_dir when full_daily store lacks offer_id)
HOST_TAG=4090 python scripts/audit_column_manifest.py \
  --offers_core runs/offers_core_full_daily_20260131_235740/offers_core_daily.parquet \
  --offers_static runs/offers_core_full_daily_20260131_235740/offers_static.parquet \
  --offers_text runs/offers_text_v1_20260129_073037_full/offers_text.parquet \
  --edgar_dir runs/edgar_feature_store_full_daily_20260131_235740/edgar_features \
  --edgar_recompute_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --require_deltalake 1 --text_required_mode 1 \
  --contract_path configs/column_contract_v3.yaml
```
