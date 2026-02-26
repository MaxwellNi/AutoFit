# V7.2 Pilot Gate Report

- generated_at_utc: **2026-02-26T00:08:29.980975+00:00**
- scope: **strict_comparable_only**
- overall_pass: **False**

## Counts

| metric | value |
|---|---:|
| rows_total | 14928 |
| rows_strict | 5546 |
| v72_rows_raw | 92 |
| overlap_keys_v7_v72_non_autofit | 88 |

## Metrics

| metric | value |
|---|---:|
| v72_vs_v7_win_rate_pct | 50.0 |
| global_normalized_mae_v7 | 0.7913173496142921 |
| global_normalized_mae_v72 | 0.7942795312307287 |
| global_normalized_mae_improvement_pct | -0.37433548220324225 |
| investors_count_median_gap_v7 | 4.976170796812308 |
| investors_count_median_gap_v72 | 5.123765827509727 |
| investors_count_gap_reduction_pct | -2.9660362701370064 |

## Checks

| check | pass |
|---|---|
| fairness_pass_100 | true |
| investors_count_gap_reduction_ge_50pct | false |
| global_normalized_mae_improvement_ge_8pct | false |

## Evidence

- metrics: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
- full_rows: `runs/benchmarks/*/metrics.json`
