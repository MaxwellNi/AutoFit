#!/usr/bin/env bash
# Resume wide-freeze pipeline from daily build (steps 4-10).
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WIDE_STAMP="${WIDE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ANALYSIS_BASE="runs/orchestrator/20260129_073037/analysis"
WIDE_ANALYSIS="${ANALYSIS_BASE}/wide_${WIDE_STAMP}"
LOGS="${WIDE_ANALYSIS}/logs"
mkdir -p "$LOGS" "${WIDE_ANALYSIS}/debug"

SQLITE_DIR="${SQLITE_DIR:-${SLURM_TMPDIR:-runs/_sqlite_wide_${WIDE_STAMP}}}"
MIN_FREE_GB="${MIN_FREE_GB:-200}"

SNAPSHOT_DIR="runs/offers_core_full_snapshot_wide_${WIDE_STAMP}"
if [ ! -f "${SNAPSHOT_DIR}/offers_core_snapshot.parquet" ]; then
  echo "ERROR: snapshot not found at ${SNAPSHOT_DIR}/offers_core_snapshot.parquet" >&2
  exit 2
fi

echo "WIDE_STAMP=${WIDE_STAMP}"
echo "WIDE_ANALYSIS=${WIDE_ANALYSIS}"
echo "SNAPSHOT_DIR=${SNAPSHOT_DIR}"
echo "SQLITE_DIR=${SQLITE_DIR}"

# 4) offers_core_full_daily_wide
python scripts/build_offers_core_full_daily.py \
  --snapshot_dir "${SNAPSHOT_DIR}" \
  --output_dir "runs/offers_core_full_daily_wide_${WIDE_STAMP}" \
  --overwrite 1 \
  --backend duckdb \
  --duckdb_tmp_dir "${SQLITE_DIR}" \
  2>&1 | tee "${LOGS}/build_daily_wide_${WIDE_STAMP}.log"

# 5) snapshots_index
python scripts/make_snapshots_index_full.py \
  --offers_core_daily "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --output_dir "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index" \
  2>&1 | tee "${LOGS}/snapshots_index_wide_${WIDE_STAMP}.log"

# 6) edgar_store_full_daily_wide
python scripts/build_edgar_features.py \
  --raw_edgar_delta data/raw/edgar/accessions \
  --snapshots_index_parquet "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index/snapshots_cik_day.parquet" \
  --output_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" \
  --align_to_snapshots \
  --partition_by_year \
  --snapshot_time_col crawled_date_day \
  --ema_alpha 0.2 \
  2>&1 | tee "${LOGS}/build_edgar_wide_${WIDE_STAMP}.log"

# 7) multiscale_full_wide
python scripts/build_multiscale_views.py \
  --offers_core_daily "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --edgar_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" \
  --output_dir runs \
  --stamp "wide_${WIDE_STAMP}" \
  2>&1 | tee "${LOGS}/multiscale_wide_${WIDE_STAMP}.log"

# 8) Column manifest (wide)
HOST_TAG="${HOST_TAG:-iris}" python scripts/audit_column_manifest.py \
  --offers_core "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --offers_static "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_static.parquet" \
  --offers_extras "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_extras_daily.parquet" \
  --offers_text "runs/offers_text_v1_20260129_073037_full/offers_text.parquet" \
  --edgar_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --output_dir "${WIDE_ANALYSIS}" \
  --seed 42 --sample_entities 1000 \
  --edgar_recompute_pairs 50 --edgar_min_compared 200 \
  --require_deltalake 1 --text_required_mode 1 \
  --contract_path configs/column_contract_wide.yaml \
  2>&1 | tee "${LOGS}/column_manifest_wide_${WIDE_STAMP}.log"

# 9) Raw cardinality coverage (full scan)
HOST_TAG="${HOST_TAG:-iris}" python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" \
  --offers_core_manifest "runs/offers_core_full_daily_wide_${WIDE_STAMP}/MANIFEST.json" \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" \
  --snapshots_index_parquet "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index/snapshots_offer_day.parquet" \
  --output_dir "${WIDE_ANALYSIS}" \
  --raw_scan_limit 0 \
  --contract_wide configs/column_contract_wide.yaml \
  --docs_audits_dir docs/audits \
  --output_basename "raw_cardinality_coverage_wide_${WIDE_STAMP}" \
  2>&1 | tee "${LOGS}/raw_cardinality_coverage_wide_${WIDE_STAMP}.log"

# 10) Freeze candidates + pointer + status
python scripts/freeze_candidates.py \
  --output_dir "${WIDE_ANALYSIS}" \
  2>&1 | tee "${LOGS}/freeze_candidates_${WIDE_STAMP}.log"

python scripts/update_wide_pointer.py --wide_stamp "${WIDE_STAMP}"

python scripts/inspect_freeze_pointer.py \
  --pointer docs/audits/FULL_SCALE_POINTER.yaml \
  --output_dir "${WIDE_ANALYSIS}" \
  2>&1 | tee "${LOGS}/freeze_pointer_status_${WIDE_STAMP}.log"

echo "Wide freeze resume complete. WIDE_STAMP=${WIDE_STAMP}"
