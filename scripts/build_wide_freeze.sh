#!/usr/bin/env bash
# Wide Feature Freeze: full 112M raw scan, wide contract, edgar, multiscale.
# Run on 4090 with sufficient RAM. No OOM; never filter rows for empty text/array.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WIDE_STAMP="${WIDE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_BASE=runs/orchestrator/20260129_073037/analysis
WIDE_ANALYSIS="${ANALYSIS_BASE}/wide_${WIDE_STAMP}"
LOGS="${WIDE_ANALYSIS}/logs"
mkdir -p "$LOGS" "${WIDE_ANALYSIS}/debug"

echo "WIDE_STAMP=${WIDE_STAMP}"
echo "WIDE_ANALYSIS=${WIDE_ANALYSIS}"

# 1) Raw inventory (offers 10M, edgar 5M)
PYTHONUNBUFFERED=1 python scripts/profile_raw_delta_columns.py \
  --raw_delta data/raw/offers --mode offers \
  --scan_sample_rows 10000000 \
  --output_json "${WIDE_ANALYSIS}/raw_offers_profile.json" \
  --output_md   "${WIDE_ANALYSIS}/raw_offers_profile.md" \
  --inventory_out "${WIDE_ANALYSIS}/raw_offers_inventory.parquet" \
  2>&1 | tee "${LOGS}/raw_inventory_offers_${WIDE_STAMP}.log"

PYTHONUNBUFFERED=1 python scripts/profile_raw_delta_columns.py \
  --raw_delta data/raw/edgar/accessions --mode edgar \
  --scan_sample_rows 5000000 \
  --output_json "${WIDE_ANALYSIS}/raw_edgar_profile.json" \
  --output_md   "${WIDE_ANALYSIS}/raw_edgar_profile.md" \
  --inventory_out "${WIDE_ANALYSIS}/raw_edgar_inventory.parquet" \
  2>&1 | tee "${LOGS}/raw_inventory_edgar_${WIDE_STAMP}.log"

# 2) Wide contract v2
python scripts/build_column_contract_wide.py \
  --inventory_offers "${WIDE_ANALYSIS}/raw_offers_inventory.parquet" \
  --inventory_edgar "${WIDE_ANALYSIS}/raw_edgar_inventory.parquet" \
  --profile_offers "${WIDE_ANALYSIS}/raw_offers_profile.json" \
  --profile_edgar "${WIDE_ANALYSIS}/raw_edgar_profile.json" \
  --output_yaml configs/column_contract_wide.yaml \
  --output_md "docs/audits/column_contract_wide_${WIDE_STAMP}.md"

# 3) offers_core_full_snapshot_wide (112M scan)
python scripts/build_offers_core_full_snapshot.py \
  --raw_offers_delta data/raw/offers \
  --contract configs/column_contract_wide.yaml \
  --output_dir "runs/offers_core_full_snapshot_wide_${WIDE_STAMP}" \
  --overwrite 1 \
  2>&1 | tee "${LOGS}/build_snapshot_wide_${WIDE_STAMP}.log"

# 4) offers_core_full_daily_wide (from snapshot + derived)
python scripts/build_offers_core_full_daily.py \
  --snapshot_dir "runs/offers_core_full_snapshot_wide_${WIDE_STAMP}" \
  --output_dir "runs/offers_core_full_daily_wide_${WIDE_STAMP}" \
  --overwrite 1

# 5) snapshots_index
python scripts/make_snapshots_index_full.py \
  --offers_core_daily "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --output_dir "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index"

# 6) edgar_store_full_daily_wide
python scripts/build_edgar_features.py \
  --edgar_path data/raw/edgar/accessions \
  --snapshots_index_parquet "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index/snapshots_cik_day.parquet" \
  --output_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" \
  --align_to_snapshots \
  --partition_by_year \
  2>&1 | tee "${LOGS}/build_edgar_wide_${WIDE_STAMP}.log"

# 7) multiscale_full_wide
python scripts/build_multiscale_views.py \
  --offers_core_daily "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --edgar_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" \
  --output_dir runs \
  --stamp "wide_${WIDE_STAMP}"

echo "Wide freeze complete. WIDE_STAMP=${WIDE_STAMP}"
