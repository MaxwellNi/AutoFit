# Raw Cardinality Coverage Audit Anchor (20260129_073037)

Public audit anchor for raw vs processed cardinality coverage.

## Data Morphology

- **offers_core_v2:** Entity subset full trajectory (limit_entities from MANIFEST)
- **offers_text_full:** Full raw scan filtered/deduped text panel (manifest counters prove)
- **edgar_store:** Full raw aggregation aligned to snapshots

## Two-Machine Consistency

Run on 3090 first, then on 4090 with `--reference_json runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage.json`.
raw_offers_version, raw_edgar_version, active_files must match; else FAIL.

## Reproducibility

```bash
HOST_TAG=3090 python scripts/audit_raw_cardinality_coverage.py \
  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \
  --offers_core_parquet runs/offers_core_v2_20260127_043052/offers_core.parquet \
  --offers_core_manifest runs/offers_core_v2_20260127_043052/MANIFEST.json \
  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \
  --edgar_store_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --docs_audits_dir docs/audits
```
