# Distillation Fidelity Audit Anchor (20260129_073037)

Public audit anchor for raw-to-processed distillation fidelity. No sensitive paths or internal hostnames.

## Scope

- **Delta version / active-files consistency**: Raw offers Delta and raw EDGAR Delta versions and active file counts are compared against the EDGAR feature-store MANIFEST (e.g. `offers_snapshots_version`, `edgar_accessions_version`). Any mismatch or unreadable raw Delta when `--require_deltalake 1` yields gate FAIL.
- **Offers raw vs offers_core**: Entity selection from `runs/selections/b11_v2_canonical/sampled_entities.json`; coverage statistics over a sample of entities; fidelity check over a subset of entities for row-count ratio, time-key consistency, and key numerical column tolerance. Entity IDs use `platform_name|offer_id` (single pipe) to align with offers_core and selection.
- **EDGAR raw vs feature store**: Recompute of EDGAR features from raw accessions (same aggregation logic as the feature store) for a small sample of (offer_id, cik) pairs that have data in the store; comparison with store values (float tolerance).
- **Column rationale**: Report documents why selected columns are essential for (i) outcome prediction, (ii) trajectory forecasting, and (iii) concept bottleneck (NBI/NCI) hooks, drawn from `scripts/build_offers_core.py` and `src/narrative/data_preprocessing/edgar_feature_store.py`.

## Gate Rule

- Skipped checks (no deltalake, raw unreadable, timeout), version mismatch, row/value inconsistency, or EDGAR recompute mismatch → **gate FAIL** (exit code 2).
- Outputs: `distillation_fidelity_report.json`, `distillation_fidelity_report.md`, and `logs/distillation_fidelity_{host}.log` under the run analysis dir. Key hashes (raw Delta versions, offers_core SHA256, selection hash, script SHA256) are written into the report.

## Artifacts

- Script: `scripts/audit_distillation_fidelity.py`
- Run analysis dir: `runs/orchestrator/20260129_073037/analysis/`
- EDGAR MANIFEST (expected Delta versions): `runs/edgar_feature_store/20260127_133511/MANIFEST.json`

## Reproducibility

Run with:

```bash
conda run -n insider python scripts/audit_distillation_fidelity.py \
  --offers_core runs/offers_core_v2_20260127_043052/offers_core.parquet \
  --edgar_dir runs/edgar_feature_store/20260127_133511/edgar_features \
  --raw_offers_delta data/raw/offers \
  --raw_edgar_delta data/raw/edgar/accessions \
  --selection_json runs/selections/b11_v2_canonical/sampled_entities.json \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --seed 42 --sample_entities 500 --fidelity_entities 10 --require_deltalake 1
```

Requires `deltalake` in the `insider` environment; raw Delta paths must exist and match MANIFEST versions for PASS.
