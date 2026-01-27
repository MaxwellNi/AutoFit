#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate insider
export PYTHONPATH="$BASE_DIR/src"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
BENCH_ON_DIR="runs/benchmarks/paper_min_matrix_edgar_on_v1_${RUN_TS}"
BENCH_OFF_DIR="runs/benchmarks/paper_min_matrix_edgar_off_v1_${RUN_TS}"
VALIDATION_DIR="runs/benchmarks_validation/${RUN_TS}"
SANITY_DIR="runs/sanity_${RUN_TS}"
AUDIT_DIR="runs/audit_${RUN_TS}"
DOCS_DIR="runs/docs_validation/${RUN_TS}"
PAPER_TABLES_DIR="runs/paper_tables"

declare -A STATUS

run_step() {
  local name="$1"
  shift
  echo ">>> ${name}"
  set +e
  "$@"
  local code=$?
  set -e
  if [ $code -ne 0 ]; then
    STATUS["$name"]="FAIL(${code})"
  else
    STATUS["$name"]="PASS"
  fi
  return 0
}

run_step "paper_min_on" \
  python scripts/run_full_benchmark.py \
    --offers_core runs/offers_core/20260125_162348/offers_core.parquet \
    --edgar_features runs/edgar_feature_store/20260125_163720_smoke/edgar_features \
    --limit_rows 100000 \
    --plan paper_min \
    --models dlinear patchtst itransformer timesnet \
    --fusion_types none film \
    --module_variants base nonstat multiscale \
    --seeds 42 43 \
    --use_edgar 1 \
    --strict_matrix 1 \
    --exp_name paper_min_matrix_edgar_on_v1 \
    --output_dir "${BENCH_ON_DIR}"

run_step "paper_min_off" \
  python scripts/run_full_benchmark.py \
    --offers_core runs/offers_core/20260125_162348/offers_core.parquet \
    --limit_rows 100000 \
    --plan paper_min \
    --models dlinear patchtst itransformer timesnet \
    --fusion_types none film \
    --module_variants base nonstat multiscale \
    --seeds 42 43 \
    --use_edgar 0 \
    --strict_matrix 1 \
    --exp_name paper_min_matrix_edgar_off_v1 \
    --output_dir "${BENCH_OFF_DIR}"

run_step "validate_matrix" \
  python scripts/validate_benchmark_matrix.py \
    --bench_dirs \
      "${BENCH_ON_DIR}" \
      "${BENCH_OFF_DIR}" \
    --expected_runs 48 \
    --require_exact 1 \
    --require_backbones dlinear patchtst itransformer timesnet \
    --require_fusions none film \
    --require_module_variants base nonstat multiscale \
    --output_dir "${VALIDATION_DIR}"

run_step "make_paper_tables" \
  python scripts/make_paper_tables.py \
    --bench_root runs/benchmarks \
    --include_prefix paper_min_matrix_ \
    --output_dir "${PAPER_TABLES_DIR}"

run_step "verify_docs" \
  python scripts/verify_docs_against_runs.py \
    --paper_tables_dir "${PAPER_TABLES_DIR}" \
    --doc_paths PROJECT_SUMMARY.md docs/RESEARCH_PIPELINE_IMPLEMENTATION.md \
    --output_dir "${DOCS_DIR}"

run_step "audit_repo" \
  python scripts/audit_repo_consistency.py \
    --output_dir "${AUDIT_DIR}"

run_step "sanity_check_metrics" \
  python scripts/sanity_check_metrics.py \
    --bench_dirs \
      "${BENCH_ON_DIR}" \
      "${BENCH_OFF_DIR}" \
    --output_dir "${SANITY_DIR}"

run_step "check_label_leakage" \
  python scripts/check_label_leakage.py \
    --offers_core runs/offers_core/20260125_162348/offers_core.parquet \
    --edgar_features runs/edgar_feature_store/20260125_163720_smoke/edgar_features \
    --limit_rows 100000 \
    --output_dir "${SANITY_DIR}"

echo "========================================"
echo "B10 Paper-Min Reproduction Summary"
echo "Timestamp: ${RUN_TS}"
echo "----------------------------------------"
for step in paper_min_on paper_min_off validate_matrix make_paper_tables verify_docs audit_repo sanity_check_metrics check_label_leakage; do
  echo "${step}: ${STATUS[$step]:-SKIPPED}"
done
echo "========================================"

if printf '%s\n' "${STATUS[@]}" | grep -q "FAIL"; then
  exit 1
fi
exit 0
