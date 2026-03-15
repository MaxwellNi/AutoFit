# Investors Count Stability Audit

- generated_at_utc: **2026-02-18T15:20:54.540707+00:00**
- overall_pass: **False**

## Split Distribution

| split | n | mean | std | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0 | None | None | None | None | None | None |
| val | 0 | None | None | None | None | None | None |
| test | 0 | None | None | None | None | None | None |

## Pairwise Drift Metrics

| pair | ks | psi | wasserstein |
|---|---:|---:|---:|
| train_vs_val | None | None | None |
| train_vs_test | None | None | None |
| val_vs_test | None | None | None |

## Telemetry

- strict_records: **1595**
- catastrophic_spikes: **4**
- repeatability_groups: **1015**

## Gate Checks

| check | pass |
|---|---|
| distribution_available | False |
| no_catastrophic_spike | False |
| strict_rows_present | True |
| ks_train_vs_test_lt_0_25 | False |
| psi_train_vs_test_lt_0_30 | False |
