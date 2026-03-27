# Current Queue Progress

> Snapshot time: 2026-03-27 14:12:54 CET
> Source: live `squeue -u npin`, `squeue --start -u npin`, and `sacct -u npin -S 2026-03-26T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 40 |
| Running | 7 |
| Pending | 33 |
| gpu running | 2 |
| gpu pending | 4 |
| l40s running | 3 |
| l40s pending | 14 |
| hopper running | 2 |
| hopper pending | 15 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5284506 | af739_t3_e2 | gpu | 1-23:37:04 | 2-00:00:00 | 99.2% | 22m | iris-179 |
| 5284505 | gpu_cos2_t2 | gpu | 1-23:39:04 | 2-00:00:00 | 99.3% | 20m | iris-169 |
| 5269769 | h2_ac_t1_fu | hopper | 1-23:31:03 | 2-00:00:00 | 99.0% | 28m | iris-197 |
| 5269770 | h2_ac_t1_s2 | hopper | 1-23:31:03 | 2-00:00:00 | 99.0% | 28m | iris-197 |
| 5269762 | l2_ac_t3_ct | l40s | 14:47:30 | 2-00:00:00 | 30.8% | 1d9h12m | iris-199 |
| 5269759 | l2_ac_t2_s2 | l40s | 1-03:32:06 | 2-00:00:00 | 57.4% | 20h27m | iris-198 |
| 5269757 | l2_ac_t2_e2 | l40s | 14:47:30 | 2-00:00:00 | 30.8% | 1d9h12m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5290110 | af739_t1_e2 | gpu | 2026-03-28 21:44:42 | Priority | V739 task1 `core_edgar_seed2` resubmission after TIMEOUT |
| 5290111 | af739_t1_s2 | gpu | 2026-03-28 21:50:42 | Priority | V739 task1 `core_only_seed2` resubmission, memory raised to `189G` after OOM |
| 5290112 | af739_t2_e2 | gpu | 2026-03-28 22:20:00 | Priority | V739 task2 `core_edgar_seed2` resubmission after TIMEOUT |
| 5290113 | af739_t2_s2 | gpu | 2026-03-28 22:59:50 | Priority | V739 task2 `core_only_seed2` resubmission, memory raised to `189G` after OOM |
| 5269763 | l2_ac_t3_e2 | l40s | 2026-03-28 10:40:07 | Resources | l40s overflow backlog |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-27 14:41:10 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 4 | all four are mandatory V739 gap-fill resubmissions |
| l40s | 14 | accel_v2 overflow backlog only |
| hopper | 15 | accel_v2 overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The previous queue snapshot that showed five live `af739_*` jobs as RUNNING is now stale.
2. `af739_t1_e2` and `af739_t2_e2` both hit `TIMEOUT` at the 2-day wall.
3. `af739_t1_s2` and `af739_t2_s2` both hit `OUT_OF_MEMORY` at `150G`, with observed peaks around `157.3G`.
4. All four missing V739 jobs have now been resubmitted and are safely back in queue.
5. There are currently no live local-only V740 jobs in the queue; the latest high-value V740 local jobs already completed and their results have been harvested into the research docs.
