# V7.2 Pilot Gate Report

- generated_at_utc: **2026-02-21T22:21:03.656509+00:00**
- scope: **strict_comparable_only**
- overall_pass: **False**

## Counts

| metric | value |
|---|---:|
| rows_total | 13871 |
| rows_strict | 4489 |
| v72_rows_raw | 4 |
| overlap_keys_v7_v72_non_autofit | 4 |

## Metrics

| metric | value |
|---|---:|
| v72_vs_v7_win_rate_pct | 0.0 |
| global_normalized_mae_v7 | 1.7875382892516192 |
| global_normalized_mae_v72 | 1.8121863675997423 |
| global_normalized_mae_improvement_pct | -1.3788839375542778 |
| investors_count_median_gap_v7 | 4.97532483477165 |
| investors_count_median_gap_v72 | 5.124429864117575 |
| investors_count_gap_reduction_pct | -2.996890339779567 |

## Checks

| check | pass |
|---|---|
| fairness_pass_100 | true |
| investors_count_gap_reduction_ge_50pct | false |
| global_normalized_mae_improvement_ge_8pct | false |

## Evidence

- metrics: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
- full_rows: `runs/benchmarks/*/metrics.json`
