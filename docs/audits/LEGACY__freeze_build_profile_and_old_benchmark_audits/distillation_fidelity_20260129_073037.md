# Distillation Fidelity Audit Anchor (20260129_073037)

Public audit anchor for raw-to-processed distillation fidelity. No sensitive paths or internal hostnames. Host names used: **3090**, **4090** only.

## Stamp and Hashes

- **Stamp:** 20260129_073037
- **Git head (at audit run):** 2454607 (audit script commit)
- **offers_core_sha256:** 43029f8369c2b57e5eb470dfc7b2854ed437bcaf14383fcc8b8f872e471fbd34
- **selection_hash:** 8cc078efff1abfaad304625a8f77fb424b2618a46bcddf162953e22fff2ac1a5
- **edgar_col_hash:** 48f2c0100b60fc94285c4cd86c0a6cde1b6aca895bb2e5259805574a5fd9d97a

## Raw Delta Versions (MANIFEST alignment)

- **raw_offers (data/raw/offers):** version 86604, active_files 2832 — matches MANIFEST `offers_snapshots_version` 86604.
- **raw_edgar (data/raw/edgar/accessions):** version 1639, active_files 46 — matches MANIFEST `edgar_accessions_version` 1639.

When `--require_deltalake 1` is set, any unreadable raw Delta or version mismatch yields gate FAIL (exit code 2).

## Scope

- **Delta version / active-files consistency:** Raw offers and EDGAR Delta versions and file counts are compared against `runs/edgar_feature_store/20260127_133511/MANIFEST.json`. Mismatch or unreadable raw when `--require_deltalake 1` → gate FAIL.
- **Offers raw vs offers_core:** Entity selection from `runs/selections/b11_v2_canonical/sampled_entities.json`; coverage by year; fidelity entities chosen from offers_core pool with raw row counts; row-count ratio tolerance 0.5% (`--row_count_ratio_tolerance 0.005`); value fidelity with diff_ratio tolerance 1% (`--fidelity_diff_ratio_tolerance 0.01`). Entity IDs use normalized `platform_name|offer_id` (single pipe) and PyArrow filter cast to string for raw Delta.
- **EDGAR raw vs feature store:** Recompute of EDGAR features from raw accessions (same aggregation logic as the feature store) for a small sample of (offer_id, cik); comparison with store values (float tolerance).
- **Column rationale:** Report documents why selected columns are essential for (i) outcome prediction, (ii) trajectory forecasting, and (iii) concept bottleneck (NBI/NCI) hooks, from `scripts/build_offers_core.py` and `src/narrative/data_preprocessing/edgar_feature_store.py`.

## Gate Rule

- Skipped checks (no deltalake, raw unreadable, timeout), version mismatch, row/value inconsistency above thresholds, or EDGAR recompute mismatch → **gate FAIL** (exit code 2).
- Outputs: `distillation_fidelity_report.json`, `distillation_fidelity_report.md`, and `logs/distillation_fidelity_{host}.log` under the run analysis dir. Key hashes are written into the report.

## Artifacts

- Script: `scripts/audit_distillation_fidelity.py`
- Run analysis dir: `runs/orchestrator/20260129_073037/analysis/`
- EDGAR MANIFEST: `runs/edgar_feature_store/20260127_133511/MANIFEST.json`

## Reproducibility

**On 3090** (log file: `distillation_fidelity_3090.log`):

```bash
HOST_TAG=3090 conda run -n insider python scripts/audit_distillation_fidelity.py \
  --offers_core runs/offers_core_v2_20260127_043052/offers_core.parquet \
  --edgar_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --seed 42 --sample_entities 1000 --fidelity_entities 10 --require_deltalake 1 \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/distillation_fidelity_3090.log
```

**On 4090** (log file: `distillation_fidelity_4090.log`):

```bash
HOST_TAG=4090 conda run -n insider python scripts/audit_distillation_fidelity.py \
  --offers_core runs/offers_core_v2_20260127_043052/offers_core.parquet \
  --edgar_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --seed 42 --sample_entities 1000 --fidelity_entities 10 --require_deltalake 1 \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/distillation_fidelity_4090.log
```

Requires `deltalake` in the `insider` environment; raw Delta paths must exist and match MANIFEST versions for PASS.
