# Current Queue Progress

> Snapshot time: 2026-03-25 14:16:30 
> Source: live `squeue -u npin` after repairing `gpu_cos2_t2` and resubmitting `af739_t3_e2`.

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 57 |
| Running | 25 |
| Pending | 32 |
| gpu running | 21 |
| gpu pending | 2 |
| l40s running | 4 |
| l40s pending | 13 |
| hopper running | 0 |
| hopper pending | 17 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time left | node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5269757 | l2_ac_t2_e2 | l40s | 21:12:09 | 2-00:00:00 | 44.2% | 1d02h47m | iris-198 |
| 5269763 | l2_ac_t3_e2 | l40s | 21:12:09 | 2-00:00:00 | 44.2% | 1d02h47m | iris-198 |
| 5279065 | g2_ac_t1_ce | gpu | 18:53:55 | 2-00:00:00 | 39.4% | 1d05h06m | iris-179 |
| 5279066 | g2_ac_t1_co | gpu | 18:48:55 | 2-00:00:00 | 39.2% | 1d05h11m | iris-179 |
| 5279067 | g2_ac_t1_ct | gpu | 18:37:55 | 2-00:00:00 | 38.8% | 1d05h22m | iris-185 |
| 5279068 | g2_ac_t1_e2 | gpu | 17:58:53 | 2-00:00:00 | 37.5% | 1d06h01m | iris-175 |
| 5279069 | g2_ac_t1_fu | gpu | 17:58:53 | 2-00:00:00 | 37.5% | 1d06h01m | iris-185 |
| 5279070 | g2_ac_t1_s2 | gpu | 17:58:53 | 2-00:00:00 | 37.5% | 1d06h01m | iris-185 |
| 5279071 | g2_ac_t2_ce | gpu | 17:32:52 | 2-00:00:00 | 36.6% | 1d06h27m | iris-184 |
| 5279072 | g2_ac_t2_co | gpu | 16:59:51 | 2-00:00:00 | 35.4% | 1d07h00m | iris-183 |
| 5279073 | g2_ac_t2_ct | gpu | 16:37:50 | 2-00:00:00 | 34.6% | 1d07h22m | iris-175 |
| 5279074 | g2_ac_t2_e2 | gpu | 15:16:46 | 2-00:00:00 | 31.8% | 1d08h43m | iris-178 |
| 5279075 | g2_ac_t2_fu | gpu | 14:12:44 | 2-00:00:00 | 29.6% | 1d09h47m | iris-196 |
| 5279076 | g2_ac_t2_s2 | gpu | 13:24:42 | 2-00:00:00 | 27.9% | 1d10h35m | iris-171 |
| 5279077 | g2_ac_t3_ce | gpu | 12:16:40 | 2-00:00:00 | 25.6% | 1d11h43m | iris-191 |
| 5279078 | g2_ac_t3_co | gpu | 12:16:40 | 2-00:00:00 | 25.6% | 1d11h43m | iris-186 |
| 5279079 | g2_ac_t3_ct | gpu | 11:08:38 | 2-00:00:00 | 23.2% | 1d12h51m | iris-186 |
| 5279080 | g2_ac_t3_e2 | gpu | 9:33:36 | 2-00:00:00 | 19.9% | 1d14h26m | iris-195 |
| 5279081 | g2_ac_t3_fu | gpu | 9:20:36 | 2-00:00:00 | 19.5% | 1d14h39m | iris-178 |
| 5279082 | af739_t1_e2 | gpu | 8:41:35 | 2-00:00:00 | 18.1% | 1d15h18m | iris-177 |
| 5280104 | af739_t1_s2 | gpu | 7:03:32 | 2-00:00:00 | 14.7% | 1d16h56m | iris-175 |
| 5280105 | af739_t2_s2 | gpu | 6:57:32 | 2-00:00:00 | 14.5% | 1d17h02m | iris-186 |
| 5280106 | af739_t2_e2 | gpu | 6:54:32 | 2-00:00:00 | 14.4% | 1d17h05m | iris-177 |
| 5279084 | l2_ac_t1_ct | l40s | 1:45:21 | 2-00:00:00 | 3.7% | 1d22h14m | iris-199 |
| 5269762 | l2_ac_t3_ct | l40s | 1:32:20 | 2-00:00:00 | 3.2% | 1d22h27m | iris-199 |

## Pending Jobs

| jobid | job | partition | state | limit | reason |
| --- | --- | --- | --- | --- | --- |
| 5284506 | af739_t3_e2 | gpu | PENDING | 2-00:00:00 | (Priority) |
| 5284505 | gpu_cos2_t2 | gpu | PENDING | 2-00:00:00 | (Priority) |
| 5269765 | h2_ac_t1_ce | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269766 | h2_ac_t1_co | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269767 | h2_ac_t1_ct | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269768 | h2_ac_t1_e2 | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269769 | h2_ac_t1_fu | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269770 | h2_ac_t1_s2 | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269771 | h2_ac_t2_ce | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269772 | h2_ac_t2_co | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269773 | h2_ac_t2_ct | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269774 | h2_ac_t2_e2 | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269775 | h2_ac_t2_fu | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269776 | h2_ac_t2_s2 | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269777 | h2_ac_t3_ce | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269778 | h2_ac_t3_co | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269779 | h2_ac_t3_ct | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269780 | h2_ac_t3_e2 | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269781 | h2_ac_t3_fu | hopper | PENDING | 2-00:00:00 | (Priority) |
| 5269748 | l2_ac_t1_ce | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269749 | l2_ac_t1_co | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5279085 | l2_ac_t1_e2 | l40s | PENDING | 2-00:00:00 | (Resources) |
| 5269752 | l2_ac_t1_fu | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269753 | l2_ac_t1_s2 | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269754 | l2_ac_t2_ce | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269755 | l2_ac_t2_co | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269756 | l2_ac_t2_ct | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269758 | l2_ac_t2_fu | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269759 | l2_ac_t2_s2 | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269760 | l2_ac_t3_ce | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269761 | l2_ac_t3_co | l40s | PENDING | 2-00:00:00 | (Priority) |
| 5269764 | l2_ac_t3_fu | l40s | PENDING | 2-00:00:00 | (Priority) |

## Operational Notes

1. `af739_t3_e2` is back in queue and no longer missing from the V739 gap-fill surface.
2. `gpu_cos2_t2` is the repaired resubmission that now points at the trimmed 23-model working list.
3. `hopper` is currently pure overflow backlog: 17 pending, 0 running.
4. The longest-running active jobs remain the l40s `e2` slices and the earliest-started `g2_ac_t1_*` GPU slices.
