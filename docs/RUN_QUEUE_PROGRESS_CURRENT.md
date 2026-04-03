# Current Queue Progress

> Snapshot time: 2026-04-03 09:10 UTC
> Source: live `squeue -u npin`, targeted `squeue --start`, and targeted `sacct`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 36 |
| Running | 8 |
| Pending | 28 |
| gpu running | 0 |
| gpu pending | 5 |
| l40s running | 5 |
| l40s pending | 9 |
| hopper running | 3 |
| hopper pending | 14 |
| cfisch pending | 0 |
| pending by priority | 27 |
| pending by resources | 1 |

## Running Jobs

| jobid | job | partition | elapsed | time_left | node |
| --- | --- | --- | --- | --- | --- |
| 5269777 | `h2_ac_t3_ce` | hopper | `19:16:57` | `1-04:43:03` | `iris-197` |
| 5269768 | `h2_ac_t1_e2` | hopper | `19:42:57` | `1-04:17:03` | `iris-197` |
| 5269776 | `h2_ac_t2_s2` | hopper | `19:46:58` | `1-04:13:02` | `iris-197` |
| 5269756 | `l2_ac_t2_ct` | l40s | `50:28` | `1-23:09:32` | `iris-198` |
| 5269763 | `l2_ac_t3_e2` | l40s | `6:39:38` | `1-17:20:22` | `iris-198` |
| 5269755 | `l2_ac_t2_co` | l40s | `1-01:23:01` | `22:36:59` | `iris-199` |
| 5269753 | `l2_ac_t1_s2` | l40s | `1-09:23:21` | `14:36:39` | `iris-199` |
| 5269762 | `l2_ac_t3_ct` | l40s | `1-12:35:02` | `11:24:58` | `iris-199` |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5304393 | `v740_repr_pa` | gpu | `2026-04-11T21:20:00` | Priority | full post-audit rerun for binary and investors local surfaces |
| 5305468 | `v740_112_inv` | gpu | `2026-04-12T11:20:00` | Priority | full routed shared112 investors loop (`0/48` landed today) |
| 5305469 | `v740_112_bin` | gpu | `2026-04-12T12:10:00` | Priority | full routed shared112 binary loop (`0/16` landed today) |
| 5305472 | `v740_112_invh1` | gpu | `2026-04-12T15:30:00` | Priority | faster routed investors h1 probe (`0/12` landed today) |
| 5305473 | `v740_112_binh1` | gpu | `2026-04-12T15:40:00` | Priority | faster routed binary h1 probe (`0/4` landed today) |
| 5298506 | `l2_ac_t3_co` | l40s | `2026-04-03T22:33:25` | Resources | resume-safe accel_v2 continuation |
| 5269769 / 5269770 / 5269778 | hopper backlog | hopper | `2026-04-04T15:21:29` | Priority | next visible hopper starts |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 5 | one V740 post-audit rerun plus four formal routed V740 jobs |
| l40s | 9 | accel_v2 overflow backlog with one current resources block |
| hopper | 14 | priority-limited overflow backlog |

## Current Interpretation

1. The canonical benchmark is currently advancing only on `l40s` and `hopper`; there is no active gpu runner.
2. `V739` has fallen out of the live queue entirely. The latest repair wave should be read as a failure ledger, not as active progress.
3. The immediate blocker for formal V740 routed evidence is queue delay, not missing implementation.
4. If the current scheduler estimates hold, the earliest new V740 gpu evidence window is `5304393` on `2026-04-11`, and the earliest formal routed window is `5305468` on `2026-04-12`.
