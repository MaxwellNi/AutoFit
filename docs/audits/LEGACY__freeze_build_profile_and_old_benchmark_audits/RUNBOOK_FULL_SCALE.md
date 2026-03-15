# Full-Scale Build Runbook

Execute in order on 4090 (`~/projects/repo_root`).

## Preflight
```bash
git status --porcelain   # must be clean
git pull --ff-only
ls -la data/raw/offers/_delta_log | head
ls -la data/raw/edgar/accessions/_delta_log | head
ls -lah runs/offers_text_v1_20260129_073037_full/offers_text.parquet
```

## Stamp
```bash
export FULL_STAMP=$(date +%Y%m%d_%H%M%S)
```

## B. Build offers_core_full_daily
```bash
mkdir -p runs/orchestrator/20260129_073037/analysis/logs
PYTHONUNBUFFERED=1 python scripts/build_offers_core_daily_full.py \
  --raw_offers_delta data/raw/offers \
  --output_dir runs/offers_core_full_daily_${FULL_STAMP} \
  --overwrite 0 \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_offers_core_full_daily_$(hostname -s).log
```
Verify: `ls -lah runs/offers_core_full_daily_${FULL_STAMP}/`

## C. Snapshots index + edgar_store_full_daily
```bash
python scripts/make_snapshots_index_from_offers_core.py \
  --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet \
  --output_path runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day/snapshots.parquet \
  --dedup_cik 0

PYTHONUNBUFFERED=1 python scripts/build_edgar_features.py \
  --raw_edgar_delta data/raw/edgar/accessions \
  --snapshots_index_parquet runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day \
  --output_dir runs/edgar_feature_store_full_daily_${FULL_STAMP} \
  --align_to_snapshots \
  --partition_by_year \
  --snapshot_time_col crawled_date_day \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_edgar_store_full_daily_$(hostname -s).log
```

## D. Audits
```bash
HOST_TAG=4090 python scripts/audit_column_manifest.py \
  --offers_core runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet \
  --offers_static runs/offers_core_full_daily_${FULL_STAMP}/offers_static.parquet \
  --offers_text runs/offers_text_v1_20260129_073037_full/offers_text.parquet \
  --edgar_dir runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --seed 42 --sample_entities 2000 --edgar_recompute_pairs 50 \
  --require_deltalake 1 --text_required_mode 1 \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/column_manifest_full_daily_4090.log

HOST_TAG=4090 python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet \
  --offers_core_manifest runs/offers_core_full_daily_${FULL_STAMP}/MANIFEST.json \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --docs_audits_dir docs/audits \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/raw_cardinality_coverage_4090.log

HOST_TAG=4090 python scripts/audit_benchmark_matrix_truthfulness.py \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis

python scripts/make_paper_tables_v2.py \
  --truthfulness_json runs/orchestrator/20260129_073037/analysis/benchmark_truthfulness.json \
  --output_dir runs/orchestrator/20260129_073037/paper_tables
```
