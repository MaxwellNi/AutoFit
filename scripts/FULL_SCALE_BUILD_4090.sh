#!/usr/bin/env bash
# FULL_SCALE_RUNBOOK_4090.sh
# Block 3 data freeze runbook for 4090 (airisklab).
# Execute steps in order. All gates must PASS before Block 3 training.
#
# PASS CONDITIONS:
# - A. profile_raw_delta_columns: outputs raw_offers_profile.json, raw_edgar_profile.json
# - A. build_column_contract_v3: outputs column_contract_v3.yaml
# - B. build_offers_core_full_snapshot: outputs offers_core_snapshot.parquet (rows > 5M expected)
# - B. build_offers_core_full_daily: outputs offers_core_daily.parquet (rows > 5M expected)
# - C. make_snapshots_index: outputs snapshots_index/snapshots.parquet
# - C. build_edgar_features: outputs edgar_features/** (rows > 1M expected)
# - D. build_multiscale_views: outputs weekly/monthly/stage parquets
# - E. audit_column_manifest: Gate PASS
# - E. audit_raw_cardinality_coverage: Gate PASS
# - E. audit_benchmark_matrix_truthfulness: Gate PASS
# - E. make_paper_tables_v2: outputs main_results.csv
#
# FAIL: Any gate FAIL -> stop and fix before proceeding.

set -euo pipefail
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FULL-SCALE RUNBOOK START (stamp=$STAMP) ==="

# Directories
ANALYSIS_DIR="runs/orchestrator/20260129_073037/analysis"
LOGS_DIR="${ANALYSIS_DIR}/logs"
mkdir -p "$LOGS_DIR"

# ============================================================
# A. Raw Profile + Column Contract v3
# ============================================================
echo "[A.1] profile_raw_delta_columns..."
python scripts/profile_raw_delta_columns.py \
  --raw_offers data/raw/offers \
  --raw_edgar data/raw/edgar/accessions \
  --output_dir "$ANALYSIS_DIR" \
  2>&1 | tee "${LOGS_DIR}/profile_raw_delta_columns_${STAMP}.log"

echo "[A.2] build_column_contract_v3..."
python scripts/build_column_contract_v3.py \
  --raw_offers_profile "${ANALYSIS_DIR}/raw_offers_profile.json" \
  --raw_edgar_profile "${ANALYSIS_DIR}/raw_edgar_profile.json" \
  --stamp "$STAMP" \
  2>&1 | tee "${LOGS_DIR}/build_column_contract_v3_${STAMP}.log"

# ============================================================
# B. Build offers_core_full_snapshot + daily
# ============================================================
SNAPSHOT_DIR="runs/offers_core_full_snapshot_${STAMP}"
DAILY_DIR="runs/offers_core_full_daily_${STAMP}"

echo "[B.1] build_offers_core_full_snapshot..."
python scripts/build_offers_core_full_snapshot.py \
  --raw_offers_delta data/raw/offers \
  --output_dir "$SNAPSHOT_DIR" \
  --overwrite 0 \
  2>&1 | tee "${LOGS_DIR}/build_offers_core_full_snapshot_${STAMP}.log"

echo "[B.2] build_offers_core_full_daily (from snapshot)..."
python scripts/build_offers_core_full_daily.py \
  --snapshot_dir "$SNAPSHOT_DIR" \
  --output_dir "$DAILY_DIR" \
  --overwrite 0 \
  2>&1 | tee "${LOGS_DIR}/build_offers_core_full_daily_${STAMP}.log"

# ============================================================
# C. Snapshots index + EDGAR store
# ============================================================
EDGAR_DIR="runs/edgar_feature_store_full_daily_${STAMP}"

echo "[C.1] make_snapshots_index..."
python scripts/make_snapshots_index.py \
  --offers_core_parquet "${DAILY_DIR}/offers_core_daily.parquet" \
  --output_dir "$DAILY_DIR" \
  --dedup_cik 0 \
  2>&1 | tee "${LOGS_DIR}/make_snapshots_index_${STAMP}.log"

echo "[C.2] build_edgar_features..."
python scripts/build_edgar_features.py \
  --raw_edgar_delta data/raw/edgar/accessions \
  --snapshots_index_parquet "${DAILY_DIR}/snapshots_index/snapshots.parquet" \
  --output_dir "$EDGAR_DIR" \
  2>&1 | tee "${LOGS_DIR}/build_edgar_features_${STAMP}.log"

# ============================================================
# D. Multi-scale views
# ============================================================
echo "[D] build_multiscale_views..."
python scripts/build_multiscale_views.py \
  --offers_core_daily "${DAILY_DIR}/offers_core_daily.parquet" \
  --edgar_dir "${EDGAR_DIR}/edgar_features" \
  --output_dir "runs" \
  --stamp "$STAMP" \
  2>&1 | tee "${LOGS_DIR}/build_multiscale_views_${STAMP}.log"

# ============================================================
# E. Audit gates
# ============================================================
echo "[E.1] audit_column_manifest (v3 contract)..."
HOST_TAG=4090 python scripts/audit_column_manifest.py \
  --offers_core "${DAILY_DIR}/offers_core_daily.parquet" \
  --offers_static "${DAILY_DIR}/offers_static.parquet" \
  --offers_text "runs/offers_text_v1_20260129_073037_full/offers_text.parquet" \
  --edgar_dir "${EDGAR_DIR}/edgar_features" \
  --edgar_recompute_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --output_dir "$ANALYSIS_DIR" \
  --seed 42 --sample_entities 2000 --edgar_recompute_pairs 50 \
  --require_deltalake 1 --text_required_mode 1 \
  --contract_path configs/column_contract_v3.yaml \
  2>&1 | tee "${LOGS_DIR}/audit_column_manifest_${STAMP}.log"

echo "[E.2] audit_raw_cardinality_coverage..."
HOST_TAG=4090 python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet "${DAILY_DIR}/offers_core_daily.parquet" \
  --offers_core_manifest "${DAILY_DIR}/MANIFEST.json" \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir "${EDGAR_DIR}/edgar_features" \
  --snapshots_index_parquet "${DAILY_DIR}/snapshots_index/snapshots.parquet" \
  --output_dir "$ANALYSIS_DIR" \
  --docs_audits_dir docs/audits \
  2>&1 | tee "${LOGS_DIR}/audit_raw_cardinality_coverage_${STAMP}.log"

echo "[E.3] audit_benchmark_matrix_truthfulness..."
HOST_TAG=4090 python scripts/audit_benchmark_matrix_truthfulness.py \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir "$ANALYSIS_DIR" \
  2>&1 | tee "${LOGS_DIR}/audit_benchmark_truthfulness_${STAMP}.log"

echo "[E.4] make_paper_tables_v2..."
python scripts/make_paper_tables_v2.py \
  --truthfulness_json "${ANALYSIS_DIR}/benchmark_truthfulness.json" \
  --output_dir runs/orchestrator/20260129_073037/paper_tables \
  2>&1 | tee "${LOGS_DIR}/make_paper_tables_v2_${STAMP}.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== FULL-SCALE RUNBOOK COMPLETE (stamp=$STAMP) ==="
echo "Check logs in: ${LOGS_DIR}"
echo ""
echo "Outputs:"
echo "  - offers_core_snapshot: ${SNAPSHOT_DIR}/offers_core_snapshot.parquet"
echo "  - offers_core_daily: ${DAILY_DIR}/offers_core_daily.parquet"
echo "  - edgar_store: ${EDGAR_DIR}/edgar_features/"
echo "  - multiscale: runs/multiscale_full_${STAMP}/"
echo "  - column_contract_v3: configs/column_contract_v3.yaml"
echo "  - profiles: ${ANALYSIS_DIR}/raw_*_profile.json"
echo ""
echo "If all gates PASS, proceed to Block 3 training."
