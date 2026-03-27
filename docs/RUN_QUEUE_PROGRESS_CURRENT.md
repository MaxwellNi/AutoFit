# Current Queue Progress

> Snapshot time: 2026-03-27 17:10:30 CET
> Source: live `squeue -u npin`, `squeue --start -u npin`, and `sacct -u npin -S 2026-03-26T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 40 |
| Running | 5 |
| Pending | 35 |
| gpu running | 0 |
| gpu pending | 6 |
| l40s running | 3 |
| l40s pending | 14 |
| hopper running | 2 |
| hopper pending | 15 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5269779 | h2_ac_t3_ct | hopper | 00:25:49 | 2-00:00:00 | 0.9% | 1d23h34m | iris-197 |
| 5269766 | h2_ac_t1_co | hopper | 00:21:49 | 2-00:00:00 | 0.8% | 1d23h38m | iris-197 |
| 5269762 | l2_ac_t3_ct | l40s | 17:43:35 | 2-00:00:00 | 37.0% | 1d6h16m | iris-199 |
| 5269759 | l2_ac_t2_s2 | l40s | 1-06:28:11 | 2-00:00:00 | 63.5% | 17h32m | iris-198 |
| 5269757 | l2_ac_t2_e2 | l40s | 17:43:35 | 2-00:00:00 | 37.0% | 1d6h16m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5290110 | af739_t1_e2 | gpu | 2026-03-27 21:42:41 | Priority | V739 task1 `core_edgar_seed2` resubmission after TIMEOUT |
| 5290111 | af739_t1_s2 | gpu | 2026-03-28 00:02:49 | Priority | V739 task1 `core_only_seed2` resubmission, memory raised to `189G` after OOM |
| 5290112 | af739_t2_e2 | gpu | 2026-03-28 03:08:59 | Priority | V739 task2 `core_edgar_seed2` resubmission after TIMEOUT |
| 5290113 | af739_t2_s2 | gpu | 2026-03-28 04:44:03 | Priority | V739 task2 `core_only_seed2` resubmission, memory raised to `189G` after OOM |
| 5290366 | af739_t3_e2 | gpu | N/A | Priority | V739 task3 `core_edgar_seed2` resubmission after fresh TIMEOUT |
| 5290365 | gpu_cos2_t2 | gpu | N/A | Priority | canonical task2 `core_only_seed2` trimmed 23-model resubmission after fresh TIMEOUT |
| 5269763 | l2_ac_t3_e2 | l40s | 2026-03-28 10:40:07 | Resources | l40s overflow backlog |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-29 13:46:19 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 6 | five mandatory V739 / AutoFit gap-fill jobs plus `gpu_cos2_t2` |
| l40s | 14 | accel_v2 overflow backlog only |
| hopper | 15 | accel_v2 overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The previous queue snapshot that showed five live `af739_*` jobs as RUNNING is now stale.
2. `af739_t1_e2` and `af739_t2_e2` both hit `TIMEOUT` at the 2-day wall.
3. `af739_t1_s2` and `af739_t2_s2` both hit `OUT_OF_MEMORY` at `150G`, with observed peaks around `157.3G`.
4. `af739_t3_e2` and `gpu_cos2_t2` also hit `TIMEOUT` in their most recent copies and were resubmitted as `5290366` and `5290365`.
5. There are currently no live local-only V740 jobs in the queue; the latest high-value V740 local jobs already completed and their results have been harvested into the research docs.
