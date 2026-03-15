# V7.2 Pilot Gate Report

- generated_at_utc: **2026-03-01T14:49:18.495962+00:00**
- scope: **strict_comparable_only**
- overall_pass: **False**

## Counts

| metric | value |
|---|---:|
| rows_total | 14944 |
| rows_strict | 5562 |
| v72_rows_raw | 108 |
| overlap_keys_v7_v72_non_autofit | 104 |

## Metrics

| metric | value |
|---|---:|
| v72_vs_v7_win_rate_pct | 42.30769230769231 |
| global_normalized_mae_v7 | 0.9470888300520848 |
| global_normalized_mae_v72 | 0.9645341947664067 |
| global_normalized_mae_improvement_pct | -1.8419987820321455 |
| investors_count_median_gap_v7 | 4.976657176255181 |
| investors_count_median_gap_v72 | 5.126292900448616 |
| investors_count_gap_reduction_pct | -3.006751698858897 |

## Checks

| check | pass |
|---|---|
| fairness_pass_100 | true |
| investors_count_gap_reduction_ge_50pct | false |
| global_normalized_mae_improvement_ge_8pct | false |

## Evidence

- metrics: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
- full_rows: `runs/benchmarks/*/metrics.json`
