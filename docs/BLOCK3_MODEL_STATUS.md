# Block 3 Model Benchmark Status

> Last updated: 2026-04-03 09:15 UTC
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: regenerated `docs/BLOCK3_RESULTS.md`, regenerated `docs/benchmarks/phase9_current_snapshot.{md,json}`, live `squeue`, targeted `sacct`, and executed V740 local reference notes.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16284** | regenerated snapshot |
| raw models (all) | 137 | 116 non-retired + 21 retired AutoFit legacy lines |
| raw complete @160 | 75 | regenerated snapshot |
| active leaderboard models | **92** | 116 non-retired raw models - 24 audit-excluded |
| active complete @160 | **62** | 75 raw complete - 13 excluded complete models |
| incomplete active models | **30** | 92 - 62 |
| post-filter records in `all_results.csv` | **12422** | regenerated `docs/BLOCK3_RESULTS.md` |
| post-filter distinct models | 107 | includes 21 retired AutoFit legacy lines |
| post-filter non-retired models | 86 | regenerated snapshot |
| clean full comparable frontier | **55** | post-filter non-retired models at shared 160/160 |
| live jobs | **36** | 8R + 28PD |
| V739 landed | **132/160** | raw metrics scan |
| V739 live jobs | **0** | live `squeue -u npin` |
| V740 shared112 local line | **112/112**, **15/2/95** | executed local reference docs |
| V740 formal routed outputs | **0** | no landed routed docs or JSON outputs |

## Canonical Backlog Tiers

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known structural exceptions |
| AutoFit gap | `AutoFitV739@132` | valid baseline, currently stalled by repeated TIMEOUT/OOM |
| foundation partials | `Chronos2@114`, `TTM@114` | partial real coverage, not complete |
| old TSLib partials | `Crossformer@107`, `MSGNet@107`, `MambaSimple@107`, `PAttn@107` | remaining accel_v2 backlog |
| older TSLib e2-limited partials | `ETSformer@94`, `LightTS@94`, `Pyraformer@94`, `Reformer@94` | remaining accel_v2 backlog |
| valid Phase 15 entrants | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | each at `91/160` |
| excluded Finding H entrants | `CFPT`, `DeformableTST`, `MICN`, `PathFormer`, `SEMPO`, `SparseTSF`, `TimeBridge`, `TimePerceiver` | each at `69/160`, excluded for constant predictions |
| excluded structural partial | `NegativeBinomialGLM@21` | excluded structural failure |

## V739 Status

| Fact | Value | Notes |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | only valid AutoFit baseline |
| landed conditions | `132/160` | unchanged |
| missing total | `28` | no new landed conditions from the latest repair wave |
| live queue jobs | `0` | there are no `af739_*` or `v739_*` jobs in live `squeue` |
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
| gpu PENDING | 5 | one V740 post-audit rerun + four routed V740 jobs |
| l40s RUNNING | 5 | active accel_v2 throughput lane |
| l40s PENDING | 9 | one resource-blocked restart plus backlog |
| hopper RUNNING | 3 | active accel_v2 overflow lane |
| hopper PENDING | 14 | priority-limited overflow backlog |
| **total** | **36** | **8 RUNNING + 28 PENDING** |

### Current Queue Interpretation

- Canonical benchmark throughput is currently coming only from `l40s` and `hopper`.
- The earliest scheduled non-gpu release now visible is `5298506 l2_ac_t3_co` at `2026-04-03T22:33:25` on `l40s`.
- The earliest pending V740 gpu rerun is `5304393 v740_repr_pa` at `2026-04-11T21:20:00`.
- The earliest formal routed V740 job is `5305468 v740_112_inv` at `2026-04-12T11:20:00`.

## V740 Local Research Status

| Lane | Current evidence | Honest read |
| --- | --- | --- |
| binary | shared112 non-routed `7/2/7`; h1 post-audit `2/0/2` | real binary competitiveness exists, but it is not yet dominant |
| funding | shared112 non-routed `8/0/40`; widened best-branch duel gives `20/28` for both strongest no-log branches | partially rescuable, but still far from championship level |
| investors | shared112 non-routed `0/0/48`; h1 post-audit `0/0/12` | still a structural failure lane |
| routed formal evidence | `0 landed` | target-routed code exists, but queue-blocked formal proof has not started landing |

## Current Operational Conclusion

1. The filtered public surface is no longer the old `12300`-row snapshot; after regeneration it is `12422`, so any doc still quoting `12300` is stale.
2. The canonical benchmark backlog still exists, but the immediate limiting factor today is queue placement and repeated V739 failure, not missing harness code.
3. For V740 specifically, the next missing artifact is not another design note. It is the first landed routed head-to-head output.
