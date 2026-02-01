#!/usr/bin/env bash
# Wide Feature Freeze: full 112M raw scan. Run manually after WIDE_STAMP is set.
# Array/map columns excluded from core (see configs/column_contract_wide.yaml exclude_nested).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
WIDE_STAMP="${WIDE_STAMP:?Set WIDE_STAMP (e.g. 20260201_211317)}"
ANALYSIS_BASE=runs/orchestrator/20260129_073037/analysis
WIDE_ANALYSIS="${ANALYSIS_BASE}/wide_${WIDE_STAMP}"
LOGS="${WIDE_ANALYSIS}/logs"
mkdir -p "$LOGS"

echo "WIDE_STAMP=${WIDE_STAMP}"

# 3) offers_core_full_snapshot_wide (112M scan, ~1-2h)
PYTHONUNBUFFERED=1 python scripts/build_offers_core_full_snapshot.py \
  --raw_offers_delta data/raw/offers \
  --contract configs/column_contract_wide.yaml \
  --output_dir "runs/offers_core_full_snapshot_wide_${WIDE_STAMP}" \
  --overwrite 1 \
  2>&1 | tee "${LOGS}/build_snapshot_wide_${WIDE_STAMP}.log"

# 4) offers_core_full_daily_wide
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

echo "Wide build complete. WIDE_STAMP=${WIDE_STAMP}"
python scripts/update_wide_pointer.py --wide_stamp "${WIDE_STAMP}"
