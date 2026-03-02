# Investors Count Stability Audit

- generated_at_utc: **2026-02-23T12:42:41.424304+00:00**
- overall_pass: **False**

## Split Distribution

| split | n | mean | std | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2416660 | 508.68631210017134 | 11767.858107809607 | 141.0 | 1715.0 | 4902.0 | 3462351.0 |
| val | 264399 | 351.8037624953196 | 1240.972634877779 | 143.0 | 1125.0 | 3928.0 | 53472.0 |
| test | 271792 | 355.38079119326545 | 1396.0148435972017 | 129.0 | 1176.0 | 4021.0 | 53472.0 |

## Pairwise Drift Metrics

| pair | ks | psi | wasserstein |
|---|---:|---:|---:|
| train_vs_val | 0.03840753786996798 | 0.019544712778755753 | 177.25514090446634 |
| train_vs_test | 0.052591118766738054 | 0.018979606851450247 | 181.05473671042563 |
| val_vs_test | 0.03764020082481234 | 0.007293239361858082 | 26.493511700119896 |

## Telemetry

- strict_records: **2260**
- catastrophic_spikes: **4**
- repeatability_groups: **1441**

## Gate Checks

| check | pass |
|---|---|
| distribution_available | True |
| no_catastrophic_spike | False |
| strict_rows_present | True |
| ks_train_vs_test_lt_0_25 | True |
| psi_train_vs_test_lt_0_30 | True |
