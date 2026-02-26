# Block 3 Delivery Plan (Current)

## Plan A: Full SOTA Baseline Maintenance

1. Keep strict-comparable full benchmark frozen at `104/104`.
2. Regenerate the canonical table after each truth-pack refresh:
   - `python scripts/build_block3_full_sota_benchmark.py`
3. Keep result summary docs synchronized:
   - `docs/BLOCK3_RESULTS.md`
   - `docs/BLOCK3_MODEL_STATUS.md`
   - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Plan B: V7.2 Completion

1. Close remaining missing keys from `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv`.
2. Submit only missing key jobs with completion controller.
3. Avoid duplicate submissions against already materialized keys.

## Plan C: V7.3 Development

1. Reuse strict benchmark evidence as fixed baseline.
2. Build routing/HPO policy artifacts from existing runs:
   - `scripts/build_v72_policy_dataset.py`
   - `scripts/train_v72_offline_policy.py`
3. Progress sequence: smoke, pilot, full.
