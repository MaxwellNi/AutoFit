# Block 3 Model Benchmark Status

> Last updated: 2026-04-03 14:13 UTC
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: regenerated `docs/BLOCK3_RESULTS.md`, regenerated `docs/benchmarks/phase9_current_snapshot.{md,json}`, live `squeue`, live `squeue --start`, targeted `sacct`, and executed V740 local reference notes.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16299** | regenerated snapshot |
| raw models (all) | 137 | 114 current-surface models + 23 archived AutoFit-family lines |
| raw complete @160 | 75 | regenerated snapshot |
| active leaderboard models | **90** | 114 current-surface raw models - 24 audit-excluded |
| active complete @160 | **62** | 75 raw complete - 13 excluded complete models |
| incomplete active models | **28** | 90 - 62 |
| post-filter records in `all_results.csv` | **11976** | regenerated `docs/BLOCK3_RESULTS.md` |
| post-filter distinct models | 84 | archived AutoFit-family lines purged from current surface |
| post-filter non-retired models | 84 | regenerated snapshot |
| clean full comparable frontier | **55** | post-filter non-retired models at shared 160/160 |
| archived AutoFit cleanup | **23 models / 460 rows removed** | executed aggregate rebuild log |
| live jobs | **34** | 8R + 26PD |
| V739 landed | **132/160** | raw metrics scan |
| V739 live jobs | **0** | live `squeue -u npin` |
| V740 shared112 local line | **112/112**, **15/2/95** | executed local reference docs |
| V740 formal routed outputs | **4 routed cells / 8 JSON** | binary h1 routed probe landed; full routed loops still pending |

## Canonical Backlog Tiers

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known structural exceptions |
| AutoFit gap | `AutoFitV739@132` | valid baseline, currently stalled by repeated TIMEOUT/OOM |
| foundation partials | `Chronos2@114`, `TTM@114` | partial real coverage, not complete |
| old TSLib partials | `Crossformer@107`, `MSGNet@107`, `MambaSimple@107`, `PAttn@107` | remaining accel_v2 backlog |
| older TSLib e2-limited partials | `ETSformer@94`, `LightTS@94`, `Pyraformer@94`, `Reformer@94` | remaining accel_v2 backlog |
| valid Phase 15 entrants | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | each at `92/160` |
| excluded Finding H entrants | `CFPT`, `DeformableTST`, `MICN`, `PathFormer`, `SEMPO`, `SparseTSF`, `TimeBridge`, `TimePerceiver` | each at `69/160`, excluded for constant predictions |
| excluded structural partial | `NegativeBinomialGLM@21` | excluded structural failure |

## V739 Status

| Fact | Value | Notes |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | current registry and leaderboard export only this line |
| landed conditions | `132/160` | unchanged |
| missing total | `28` | no new landed conditions from the latest repair wave |
| live queue jobs | `0` | there are no `af739_*` or `v739_*` jobs in live `squeue` |
| archived AutoFit lines in current surface | `0` | rebuilt `all_results.csv` no longer carries retired / invalid AutoFit-family rows |
| current state | **stalled** | latest repair copies all ended in TIMEOUT or OOM |

### Latest Failed Repair Wave

| jobs | lane | final state | reqmem | note |
| --- | --- | --- | --- | --- |
| `5298285`, `5298286`, `5298287` | `t1_e2`, `t2_e2`, `t3_e2` | `TIMEOUT` | `189G` | first repaired e2 wave timed out |
| `5302271`, `5302272`, `5302273` | `t1_e2`, `t2_e2`, `t3_e2` | `TIMEOUT` | `189G` | second repaired e2 wave also timed out |
| `5299888` | `t1_s2` | `OUT_OF_MEMORY` | `224G` | `MaxRSS ~= 234.9G` |
| `5300059` | `t2_s2` | `OUT_OF_MEMORY` | `224G` | `MaxRSS ~= 234.9G` |
| `5302274` | `t1_s2` | `OUT_OF_MEMORY` | `280G` | `MaxRSS ~= 293.6G` |
| `5302275` | `gpu_cos2_t2` | `TIMEOUT` | `150G` | second repaired cos2 copy timed out |

## Live Queue Reality

Detailed per-job progress and ETA live in `docs/RUN_QUEUE_PROGRESS_CURRENT.md`.

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 0 | no active gpu job at verification time |
| gpu PENDING | 3 | all remaining gpu pending jobs are formal routed V740 jobs |
| l40s RUNNING | 5 | active accel_v2 throughput lane |
| l40s PENDING | 9 | one resource-blocked restart plus backlog |
| hopper RUNNING | 3 | active accel_v2 overflow lane |
| hopper PENDING | 14 | priority-limited overflow backlog |
| **total** | **34** | **8 RUNNING + 26 PENDING** |

### Current Queue Interpretation

- Canonical benchmark throughput is currently coming only from `l40s` and `hopper`.
- The earliest scheduled non-gpu release now visible is `5298506 l2_ac_t3_co` at `2026-04-03T22:33:25` on `l40s`.
- All remaining gpu pending jobs are formal routed V740 jobs.
- The earliest remaining formal routed V740 job is `5305468 v740_112_inv` at `2026-04-09T22:50:00`.

## V740 Local Research Status

| Lane | Current evidence | Honest read |
| --- | --- | --- |
| binary | shared112 non-routed `7/2/7`; full post-audit `10/0/6`; routed h1 `2/0/2` | real binary competitiveness exists, and routed h1 has first landed proof, but it is not yet dominant |
| funding | shared112 non-routed `8/0/40`; widened best-branch duel gives `20/28` for both strongest no-log branches | partially rescuable, but still far from championship level |
| investors | shared112 non-routed `0/0/48`; full post-audit `0/0/48`; h1 post-audit `0/0/12` | still a structural failure lane |
| routed formal evidence | binary h1 landed at `2/0/2`; full routed loops still `0/64` | target-routed code is now partially executed, but the full routed verdict is still queue-blocked |

## Current Operational Conclusion

1. The filtered public surface is no longer the old `12300`-row snapshot; after regeneration plus AutoFit cleanup it is `11976`, so any doc still quoting `12300` or `12422` as the current surface is stale.
2. The canonical benchmark backlog still exists, but the immediate limiting factor today is queue placement and repeated V739 failure, not missing harness code.
3. For V740 specifically, the next missing artifact is no longer the first routed output. That has landed on binary h1. The next missing artifact is routed coverage beyond that first 4-cell proof.
