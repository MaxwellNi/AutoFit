# V7.2 Pilot Gate Report

- generated_at_utc: **2026-02-18T23:47:03.306668+00:00**
- scope: **strict_comparable_only**
- overall_pass: **False**

## Counts

| metric | value |
|---|---:|
| rows_total | 13416 |
| rows_strict | 4034 |
| v72_rows_raw | 0 |
| overlap_keys_v7_v72_non_autofit | 0 |

## Metrics

| metric | value |
|---|---:|
| v72_vs_v7_win_rate_pct | None |
| global_normalized_mae_v7 | None |
| global_normalized_mae_v72 | None |
| global_normalized_mae_improvement_pct | None |
| investors_count_median_gap_v7 | None |
| investors_count_median_gap_v72 | None |
| investors_count_gap_reduction_pct | None |

## Checks

| check | pass |
|---|---|
| fairness_pass_100 | false |
| investors_count_gap_reduction_ge_50pct | false |
| global_normalized_mae_improvement_ge_8pct | false |

## Evidence

- metrics: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
- full_rows: `runs/benchmarks/*/metrics.json`
