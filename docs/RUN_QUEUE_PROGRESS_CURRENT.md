# Current Queue Progress

> Snapshot time: 2026-04-03 14:13 UTC
> Source: live `squeue -u npin`, live `squeue -u npin -t running`, and live `squeue --start -u npin`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 34 |
| Running | 8 |
| Pending | 26 |
| gpu running | 0 |
| gpu pending | 3 |
| l40s running | 5 |
| l40s pending | 9 |
| hopper running | 3 |
| hopper pending | 14 |
| cfisch pending | 0 |
| pending by priority | 25 |
| pending by resources | 1 |

## Running Jobs

| jobid | job | partition | elapsed | time_left | node |
| --- | --- | --- | --- | --- | --- |
| 5269778 | `h2_ac_t3_co` | hopper | `4:44:38` | `1-19:15:22` | `iris-197` |
| 5269773 | `h2_ac_t2_ct` | hopper | `4:10:34` | `1-19:49:26` | `iris-197` |
| 5269779 | `h2_ac_t3_ct` | hopper | `4:10:34` | `1-19:49:26` | `iris-197` |
| 5269756 | `l2_ac_t2_ct` | l40s | `5:55:41` | `1-18:04:19` | `iris-198` |
| 5269763 | `l2_ac_t3_e2` | l40s | `11:44:51` | `1-12:15:09` | `iris-198` |
| 5269755 | `l2_ac_t2_co` | l40s | `1-06:28:14` | `17:31:46` | `iris-199` |
| 5269753 | `l2_ac_t1_s2` | l40s | `1-14:28:34` | `9:31:26` | `iris-199` |
| 5269762 | `l2_ac_t3_ct` | l40s | `1-17:40:15` | `6:19:45` | `iris-199` |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5305468 | `v740_112_inv` | gpu | `2026-04-09T22:50:00` | Priority | full routed shared112 investors loop (`0/48` full-loop cells landed; post-audit rerun already landed separately) |
| 5305469 | `v740_112_bin` | gpu | `2026-04-09T22:50:00` | Priority | full routed shared112 binary loop (`0/16` full-loop cells landed; binary routed h1 `4/4` already landed separately) |
| 5305472 | `v740_112_invh1` | gpu | `2026-04-09T22:50:00` | Priority | faster routed investors h1 probe (`0/12` landed today) |
| 5298506 | `l2_ac_t3_co` | l40s | `2026-04-03T22:33:25` | Resources | resume-safe accel_v2 continuation |
| 5269760 | `l2_ac_t3_ce` | l40s | `2026-04-05T04:28:49` | Priority | next visible l40s resume-safe accel_v2 continuation |
| 5269765 / 5269766 / 5269767 | hopper backlog | hopper | `2026-04-05T11:29:02` | Priority | first visible hopper start wave |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 3 | all remaining gpu pending jobs are formal routed V740 jobs |
| l40s | 9 | accel_v2 overflow backlog with one current resources block |
| hopper | 14 | priority-limited overflow backlog |

## Current Interpretation

1. The canonical benchmark is currently advancing only on `l40s` and `hopper`; there is no active gpu runner.
2. `V739` has fallen out of the live queue entirely. The latest repair wave should be read as a failure ledger, not as active progress.
3. The immediate blocker is now extending routed evidence beyond the already-landed binary h1 probe; that blocker is queue delay, not missing implementation.
4. If the current scheduler estimates hold, the earliest remaining formal routed V740 full-loop window is `5305468` on `2026-04-09`.
