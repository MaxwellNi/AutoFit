# Current Queue Progress

> Snapshot time: 2026-03-28 17:10 CET
> Source: live `squeue -u npin`, `squeue --start -u npin`, and `sacct -u npin -S 2026-03-26T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 40 |
| Running | 9 |
| Pending | 31 |
| gpu running | 6 |
| gpu pending | 0 |
| bigmem running | 0 |
| l40s running | 3 |
| l40s pending | 14 |
| hopper running | 0 |
| hopper pending | 17 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5290113 | af739_t2_s2 | gpu | 21:45:29 | 2-00:00:00 | 45.3% | 1d2h14m | iris-170 |
| 5290112 | af739_t2_e2 | gpu | 21:46:29 | 2-00:00:00 | 45.4% | 1d2h13m | iris-196 |
| 5290111 | af739_t1_s2 | gpu | 21:48:29 | 2-00:00:00 | 45.4% | 1d2h11m | iris-172 |
| 5290110 | af739_t1_e2 | gpu | 21:49:29 | 2-00:00:00 | 45.5% | 1d2h10m | iris-172 |
| 5290366 | af739_t3_e2 | gpu | 21:39:29 | 2-00:00:00 | 45.1% | 1d2h20m | iris-179 |
| 5290365 | gpu_cos2_t2 | gpu | 21:45:29 | 2-00:00:00 | 45.3% | 1d2h14m | iris-170 |
| 5269761 | l2_ac_t3_co | l40s | 15:14:19 | 2-00:00:00 | 31.7% | 1d8h45m | iris-198 |
| 5269757 | l2_ac_t2_e2 | l40s | 15:12:21 | 2-00:00:00 | 31.7% | 1d8h47m | iris-199 |
| 5269762 | l2_ac_t3_ct | l40s | 15:12:21 | 2-00:00:00 | 31.7% | 1d8h47m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5294241 | l2_ac_t2_s2 | l40s | N/A | Priority | resumed overflow copy after 2-day TIMEOUT on `5269759` |
| 5269748 | l2_ac_t1_ce | l40s | 2026-03-30 01:25:41 | Resources | l40s overflow backlog |
| 5269758 | l2_ac_t2_fu | l40s | 2026-03-30 01:27:39 | Priority | l40s overflow backlog |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 0 | canonical gpu critical path only; no live local-only jobs remain |
| l40s | 14 | accel_v2 overflow backlog plus the requeued `l2_ac_t2_s2` |
| hopper | 17 | pure overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The gpu critical path has flipped back from all-pending to fully running: all five `af739_*` gap-fill jobs plus `gpu_cos2_t2` are live again.
2. `l2_ac_t2_s2` is the new confirmed timeout casualty on 2026-03-28; log evidence shows it progressed deep into `investors_count` before the 2-day wall, so it was resubmitted as `5294241`.
3. Local-only V740 model-clear status has advanced:
   - `5294242` `v740_samf_clr` → completed successfully
   - `5294243` `v740_prop_clr` → failed before model execution because `quick` preset cannot run `h=30`
   - `5294254` `v740_prop_std` → corrected `Prophet` resubmission, completed successfully
   - `5294255` `v740_tpfn26c` → first `TabPFN 2.6` narrow clear, completed successfully
4. These local-only jobs use separate output roots and do not count as canonical benchmark results.
