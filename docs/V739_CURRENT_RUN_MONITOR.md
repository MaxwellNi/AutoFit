# V739 Current Run Monitor and Landing Checklist

> Last verified: 2026-03-27 14:12 CET
> Scope: current AutoFit V739 execution reality on the canonical clean benchmark line only.

## Current Verified Reality

1. **V739 is the only valid AutoFit baseline.** V734-V738 remain retired because of oracle test-set leakage.
2. **V739 is not fully landed yet.** The current landed surface is **132/160** conditions on `runs/benchmarks/block3_phase9_fair/`.
3. **V739 quality remains clean** on the landed surface:
   - 0 NaN/Inf
   - 0 fallback
   - 100% fairness pass
4. **Current live V739 queue state is still 5 jobs total, but no longer 5 RUNNING**:
   - 1 RUNNING: `af739_t3_e2`
   - 4 PENDING: `af739_t1_e2`, `af739_t1_s2`, `af739_t2_s2`, `af739_t2_e2`
5. `af739_t3_e2` remains the only actively running V739 copy at the moment, now in its final ~20 minutes before the 2-day limit.

## Coverage Audit

| Metric | Value |
| --- | --- |
| Landed conditions | 132 / 160 |
| Missing total | 28 |
| Missing `task1_outcome / core_edgar_seed2` | 9 |
| Missing `task1_outcome / core_only_seed2` | 8 |
| Missing `task2_forecast / core_edgar_seed2` | 4 |
| Missing `task2_forecast / core_only_seed2` | 4 |
| Missing `task3_risk_adjust / core_edgar_seed2` | 3 |
| NaN/Inf in landed metrics | 0 |
| Fallback fraction | 0.0 |
| Fairness pass | 131 / 131 (100%) |

### Exact Missing Cells

| Task | Ablation | Missing targets / horizons |
| --- | --- | --- |
| `task1_outcome` | `core_edgar_seed2` | `funding_raised_usd@30`, `investors_count@{1,7,14,30}`, `is_funded@{1,7,14,30}` |
| `task1_outcome` | `core_only_seed2` | `investors_count@{1,7,14,30}`, `is_funded@{1,7,14,30}` |
| `task2_forecast` | `core_edgar_seed2` | `investors_count@{1,7,14,30}` |
| `task2_forecast` | `core_only_seed2` | `investors_count@{1,7,14,30}` |
| `task3_risk_adjust` | `core_edgar_seed2` | `investors_count@{1,7,14,30}` |

## Current Queue Monitor

| Job | State | Partition | Notes |
| --- | --- | --- | --- |
| `af739_t1_e2` | PENDING | `gpu` | task1 `core_edgar_seed2` gap-fill, resubmitted as `5290110` |
| `af739_t1_s2` | PENDING | `gpu` | task1 `core_only_seed2` gap-fill, resubmitted as `5290111` at `189G` |
| `af739_t2_s2` | PENDING | `gpu` | task2 `core_only_seed2` gap-fill, resubmitted as `5290113` at `189G` |
| `af739_t2_e2` | PENDING | `gpu` | task2 `core_edgar_seed2` gap-fill, resubmitted as `5290112` |
| `af739_t3_e2` | RUNNING | `gpu` | task3 `core_edgar_seed2` gap-fill, repaired 2026-03-25 |

## What Changed on 2026-03-27

1. `sacct` shows that the earlier “all 5 running” state has already ended:
   - `5279082 af739_t1_e2` → `TIMEOUT`
   - `5280104 af739_t1_s2` → `OUT_OF_MEMORY` (`MaxRSS=157290792K`)
   - `5280105 af739_t2_s2` → `OUT_OF_MEMORY` (`MaxRSS=157284604K`)
   - `5280106 af739_t2_e2` → `TIMEOUT`
2. All four missing jobs were resubmitted immediately:
   - `5290110 af739_t1_e2`
   - `5290111 af739_t1_s2`
   - `5290112 af739_t2_e2`
   - `5290113 af739_t2_s2`
3. The two seed2 jobs were escalated from `150G` to `189G` because the failed copies exceeded the original memory budget by ~7G.
4. The queue gap for `af739_t3_e2` remains repaired: the current live replacement `5284506` is still RUNNING.

## Operational Interpretation

- V739 is now in a **pure seed2/e2 gap-fill phase**. The original co/ce/ct/fu surface is complete.
- The remaining 28 missing conditions are all known and localized; there is no longer any ambiguity about where the gaps are.
- The dominant risk is now **throughput plus queue churn**, not correctness. The key bottlenecks are long validation-based candidate selection in AutoFit, the shared queue pressure from ModernTCN-heavy accel jobs, and the need to resubmit long-running seed2/e2 jobs cleanly after timeout/OOM.
- There is currently **no uncovered mandatory V739 work outside the queue**.

## Benchmark Position (Keep Using This Carefully)

The last valid computed summary still places V739 at roughly **#13** on the stable shared comparison slice. That remains useful context, but any paper-facing statement must continue to describe V739 as the **current valid baseline under active gap-fill**, not as a fully finished 160/160 line.

## Monitoring Commands

### Live queue view
```bash
squeue -u npin -o '%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R'
```

### V739-only queue view
```bash
squeue -u npin -h -o '%i|%j|%T|%P|%R' | rg 'af739|v739'
```

### Rebuild current snapshot
```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/build_phase9_current_snapshot.py
```

### Rebuild aggregated result table
```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

## Landing Checklist

1. Confirm all 5 V739 gap-fill jobs are still present or safely requeued.
2. Confirm newly landed rows increase `AutoFitV739` from `132` upward in `all_results.csv`.
3. Confirm no NaN/Inf, no fallback, and fairness remains 100%.
4. Only then update `docs/CURRENT_SOURCE_OF_TRUTH.md`, `docs/BLOCK3_MODEL_STATUS.md`, and paper-facing status notes.
