# Current Queue Progress

> Snapshot time: 2026-03-29 14:36 CEST
> Source: live `squeue -u npin`, `squeue --start -u npin`, and `sacct -u npin -S 2026-03-26T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 41 |
| Running | 7 |
| Pending | 34 |
| gpu running | 6 |
| gpu pending | 1 |
| bigmem running | 0 |
| l40s running | 1 |
| l40s pending | 16 |
| hopper running | 0 |
| hopper pending | 17 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5290112 | af739_t2_e2 | gpu | 1-19:45:28 | 2-00:00:00 | 91.2% | 4h15m | iris-196 |
| 5290110 | af739_t1_e2 | gpu | 1-19:48:28 | 2-00:00:00 | 91.3% | 4h12m | iris-172 |
| 5290366 | af739_t3_e2 | gpu | 1-19:38:28 | 2-00:00:00 | 90.9% | 4h22m | iris-179 |
| 5290365 | gpu_cos2_t2 | gpu | 1-19:44:28 | 2-00:00:00 | 91.1% | 4h16m | iris-170 |
| 5298048 | af739_t2_s2 | gpu | 0:23:44 | 2-00:00:00 | 0.8% | 1d23h36m | iris-186 |
| 5298049 | af739_t1_s2 | gpu | 0:23:44 | 2-00:00:00 | 0.8% | 1d23h36m | iris-170 |
| 5269761 | l2_ac_t3_co | l40s | 1-13:13:18 | 2-00:00:00 | 77.5% | 10h47m | iris-198 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5298059 | v740_lgts_clr | gpu | N/A | Priority | first LightGTS narrow benchmark-clear probe |
| 5269748 | l2_ac_t1_ce | l40s | 2026-03-30 01:25:41 | Resources | first scheduled l40s restart slot |
| 5269758 | l2_ac_t2_fu | l40s | 2026-03-30 14:24:32 | Priority | next l40s backlog item with a concrete start time |
| 5294241 | l2_ac_t2_s2 | l40s | 2026-04-04 14:20:00 | Priority | resumed overflow copy after 2-day TIMEOUT on `5269759` |
| 5269763 | l2_ac_t3_e2 | l40s | 2026-04-03 01:30:00 | Priority | later l40s restart already scheduled |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269771 | h2_ac_t2_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269780 | h2_ac_t3_e2 | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 1 | canonical gpu critical path is fully running; the only gpu pending job is the new `LightGTS` local-clear probe |
| l40s | 16 | accel_v2 overflow backlog; current live state has only one active `l40s` runner |
| hopper | 17 | pure overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The gpu critical path remains fully live: all five `af739_*` gap-fill jobs plus `gpu_cos2_t2` are running again.
2. `af739_t1_s2` and `af739_t2_s2` failed once more at `189G` (`MaxRSS≈198.18G`) and were resubmitted as `5298049` / `5298048` at `200G / 8 CPU`.
3. `l2_ac_t2_s2` is the confirmed latest `l40s` timeout casualty on 2026-03-28; log evidence shows it progressed deep into `investors_count` before the 2-day wall, so it was resubmitted as `5294241`.
4. The short-lived `l40s` burst observed in the previous check has ended. At this snapshot only `l2_ac_t3_co` remains RUNNING and the rest of the `l40s` lane is back in PENDING state, so the queue has shifted to `7R+33PD`.
5. Local-only V740 model-clear status has advanced:
   - `5294242` `v740_samf_clr` → completed successfully
   - `5294243` `v740_prop_clr` → failed before model execution because `quick` preset cannot run `h=30`
   - `5294254` `v740_prop_std` → corrected `Prophet` resubmission, completed successfully
   - `5294255` `v740_tpfn26c` → first `TabPFN 2.6` narrow clear, completed successfully
   - `5294259` `v740_tpfn26r_fu` → completed as the first `TabPFNRegressor` funding clear (`fairness_pass=true`, quality weak)
   - `5294260` `v740_tpfn26r_inv` → completed as the first `TabPFNRegressor` investors-count clear, but with `fairness_pass=false`
6. These local-only jobs use separate output roots and do not count as canonical benchmark results.
7. `5298059` `v740_lgts_clr` has now been added as the first `LightGTS` narrow benchmark-clear probe after the wrapper passed a non-fallback tiny real-data smoke on `task2_forecast/core_edgar/funding_raised_usd/h=30`.
