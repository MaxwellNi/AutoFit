# Full-Scale Build: Execution Commands

**Target:** Full scan of `data/raw/offers` and `data/raw/edgar/accessions`, outputs pass Block 3 audits.

**Data semantics:**
- **offers_core_full_daily:** Full raw offers scan (112M rows) → dedup by (entity_id, snapshot_ts) → 5.77M rows. 21MB parquet (numeric cols only; offers_text 20GB = same rows + text).
- **edgar_store_full_daily:** Full raw edgar scan, aligned to ALL offer snapshots with cik (one row per platform_name, offer_id, cik, snapshot_ts) → ~400MB+.

**Memory:** Scripts auto-tune from `MemAvailable`. Ensure ≥32GB free for edgar (loads full raw into RAM during aggregation).

---

## Step 0: Preflight

```bash
cd ~/projects/repo_root
git status --porcelain   # must be clean
git pull --ff-only
ls -la data/raw/offers/_delta_log | head -3
ls -la data/raw/edgar/accessions/_delta_log | head -3
```

---

## Step 1: Stamp

```bash
export FULL_STAMP=$(date +%Y%m%d_%H%M%S)
echo "FULL_STAMP=$FULL_STAMP"
```

---

## Step 2: offers_core_full_daily (full raw scan)

Scans **all** raw offers via DeltaTable. Output ~5.77M rows, ~21MB (numeric only).

```bash
mkdir -p runs/orchestrator/20260129_073037/analysis/logs

PYTHONUNBUFFERED=1 python scripts/build_offers_core_daily_full.py \
  --raw_offers_delta data/raw/offers \
  --output_dir runs/offers_core_full_daily_${FULL_STAMP} \
  --overwrite 1 \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_offers_core_full_daily_$(hostname -s).log
```

**Verify:**
```bash
cat runs/offers_core_full_daily_${FULL_STAMP}/MANIFEST.json
# Expect: rows_scanned ~112608525, rows_emitted ~5774931, n_unique_entities ~22569
```

---

## Step 3: Snapshots index (FULL, no cik dedup)

**Critical:** `--dedup_cik 0` keeps one row per (platform_name, offer_id, cik, snapshot_ts) so edgar can join to each offer. Default changed to 0.

```bash
python scripts/make_snapshots_index_from_offers_core.py \
  --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet \
  --output_path runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day/snapshots.parquet \
  --dedup_cik 0
```

**Verify:** Row count ≈ offers_core rows with valid cik (~5M+).

---

## Step 4: edgar_store_full_daily (full raw scan + full alignment)

Loads **all** raw edgar, aligns to **all** snapshots. Expect ~400MB output, 10–30 min.

```bash
PYTHONUNBUFFERED=1 python scripts/build_edgar_features.py \
  --raw_edgar_delta data/raw/edgar/accessions \
  --snapshots_index_parquet runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day \
  --output_dir runs/edgar_feature_store_full_daily_${FULL_STAMP} \
  --align_to_snapshots \
  --partition_by_year \
  --snapshot_time_col crawled_date_day \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_edgar_store_full_daily_$(hostname -s).log
```

**Verify:**
```bash
du -sh runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features/
# Expect ~400MB (similar to runs/edgar_feature_store/20260127_133511)
```

---

## Step 5: Block 3 audits

```bash
# Column manifest (must PASS)
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

# Raw cardinality
HOST_TAG=4090 python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet \
  --offers_core_manifest runs/offers_core_full_daily_${FULL_STAMP}/MANIFEST.json \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --docs_audits_dir docs/audits \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/raw_cardinality_coverage_4090.log

# Truthfulness + paper tables
HOST_TAG=4090 python scripts/audit_benchmark_matrix_truthfulness.py \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis

python scripts/make_paper_tables_v2.py \
  --truthfulness_json runs/orchestrator/20260129_073037/analysis/benchmark_truthfulness.json \
  --output_dir runs/orchestrator/20260129_073037/paper_tables
```

---

## Memory tuning (optional)

- **offers_core:** `--chunk_rows N` (default auto from RAM, cap 25M).
- **edgar:** No streaming; requires sufficient RAM for full raw load. If OOM, reduce concurrency or use a machine with more RAM.

---

## One-shot copy-paste (replace ${FULL_STAMP} after Step 1)

```bash
# After: export FULL_STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p runs/orchestrator/20260129_073037/analysis/logs

PYTHONUNBUFFERED=1 python scripts/build_offers_core_daily_full.py --raw_offers_delta data/raw/offers --output_dir runs/offers_core_full_daily_${FULL_STAMP} --overwrite 1 2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_offers_core_full_daily_$(hostname -s).log

python scripts/make_snapshots_index_from_offers_core.py --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet --output_path runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day/snapshots.parquet --dedup_cik 0

PYTHONUNBUFFERED=1 python scripts/build_edgar_features.py --raw_edgar_delta data/raw/edgar/accessions --snapshots_index_parquet runs/offers_core_full_daily_${FULL_STAMP}/snapshots_cik_day --output_dir runs/edgar_feature_store_full_daily_${FULL_STAMP} --align_to_snapshots --partition_by_year --snapshot_time_col crawled_date_day 2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/build_edgar_store_full_daily_$(hostname -s).log

HOST_TAG=4090 python scripts/audit_column_manifest.py --offers_core runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet --offers_static runs/offers_core_full_daily_${FULL_STAMP}/offers_static.parquet --offers_text runs/offers_text_v1_20260129_073037_full/offers_text.parquet --edgar_dir runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions --selection_json runs/selections/b11_v2_canonical/sampled_entities.json --output_dir runs/orchestrator/20260129_073037/analysis --seed 42 --sample_entities 2000 --edgar_recompute_pairs 50 --require_deltalake 1 --text_required_mode 1 2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/column_manifest_full_daily_4090.log

HOST_TAG=4090 python scripts/audit_raw_cardinality_coverage.py --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions --offers_core_parquet runs/offers_core_full_daily_${FULL_STAMP}/offers_core_daily.parquet --offers_core_manifest runs/offers_core_full_daily_${FULL_STAMP}/MANIFEST.json --offers_text_full_dir runs/offers_text_v1_20260129_073037_full --edgar_store_dir runs/edgar_feature_store_full_daily_${FULL_STAMP}/edgar_features --output_dir runs/orchestrator/20260129_073037/analysis --docs_audits_dir docs/audits 2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/raw_cardinality_coverage_4090.log

HOST_TAG=4090 python scripts/audit_benchmark_matrix_truthfulness.py --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt --output_dir runs/orchestrator/20260129_073037/analysis

python scripts/make_paper_tables_v2.py --truthfulness_json runs/orchestrator/20260129_073037/analysis/benchmark_truthfulness.json --output_dir runs/orchestrator/20260129_073037/paper_tables
```
