# Current Source of Truth

> Last verified: 2026-04-03 14:13 UTC
> Verification basis: direct raw metrics scan, regenerated `all_results.csv`, regenerated `docs/benchmarks/phase9_current_snapshot.{json,md}`, live `squeue -u npin`, live `squeue --start -u npin`, targeted `sacct`, and executed V740 local reference notes including landed post-audit and routed-h1 outputs.

This file is the authoritative entry point for the current Block 3 project state.
If another status document disagrees with this file, prefer this file plus the evidence paths named here.

## Authoritative Sources (Read in This Order)

1. Root `AGENTS.md`
2. `.local_mandatory_preexec.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/BLOCK3_RESULTS.md`
7. `docs/benchmarks/phase9_current_snapshot.md`
8. `docs/references/MODEL_REGISTRY.md`
9. `docs/references/AUTOFIT_RETIREMENT_LEDGER.md`
10. `docs/references/V740_ROADMAP_STATUS_20260330.md`
11. `docs/references/V740_REPR_POSTAUDIT_GATE_20260402.md`
12. `docs/references/V740_EDGAR_TEXT_ROOTCAUSE_AUDIT_20260402.md`

## Execution Planning Companion

For forward-looking execution sequencing, use:

- `docs/V740_MASTER_EXECUTION_PLAN.md`

This file remains the current-state truth pack.
`docs/V740_MASTER_EXECUTION_PLAN.md` is the operational roadmap and should not
be treated as a source of current benchmark facts unless those facts are also
reflected here or in the evidence files it cites.

## Verified Current Facts

| Fact | Current value | Evidence |
| --- | --- | --- |
| Canonical benchmark directory | `runs/benchmarks/block3_phase9_fair/` | direct scan |
| Raw metric records | `16299` | raw metrics scan 2026-04-03 |
| Raw models materialized | `137` | regenerated snapshot |
| Raw non-retired models | `114` | regenerated snapshot |
| Audit-excluded models | `24` | `AUDIT_EXCLUDED_MODELS` in `scripts/aggregate_block3_results.py` |
| Active leaderboard models | `90` | `114 - 24` |
| Raw complete models (`@160`) | `75` | regenerated snapshot |
| Active complete models (`@160`) | `62` | `75 - 13` excluded-complete models |
| Incomplete active models | `28` | `90 - 62` |
| Filtered records in `all_results.csv` | `11976` | regenerated `scripts/aggregate_block3_results.py` output |
| Filtered distinct models | `84` | regenerated `all_results.csv` |
| Filtered non-retired models | `84` | regenerated snapshot |
| Clean full comparable frontier (`160/160`, non-retired, post-filter) | `55` | regenerated snapshot |
| Archived AutoFit-family cleanup | `23 models / 460 rows purged from current surface` | executed aggregate rebuild log 2026-04-03 |
| Current valid AutoFit baseline | `AutoFitV739` only | root `AGENTS.md` |
| V739 landed conditions | `132/160` | raw metrics scan |
| V739 live jobs | `0` | live `squeue -u npin` |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/` plus regenerated snapshot |
| Phase 12 text reruns | `48/48 completed` | existing phase-12 execution notes plus active artifacts |
| V740 shared112 non-routed local line | `112/112 complete`, `15 win / 2 tie / 95 loss` | executed local reference docs |
| V740 formal routed outputs | `binary h1 routed probe landed` (`4` cells / `8` JSON) | `docs/references/V740_SHARED112_BINARY_ROUTED_H1_PROBE_20260403.md` plus `runs/benchmarks/v740_localclear_20260403/v740_shared112_binary_routed_h1_probe_20260403/` |

## What Changed Since The 2026-04-01 Snapshot

- Raw benchmark records increased from `16182` to `16299` (`+117`).
- The filtered public surface moved in two steps during the 2026-04-03 refresh cycle: `12300 -> 12422 -> 11976`.
- The latest AutoFit cleanup purged `23` archived AutoFit-family models / aliases (`460` rows) from the active registry and current public leaderboard surface.
- `V739` no longer has live queue presence; the latest repair wave ended in repeated `TIMEOUT` and `OUT_OF_MEMORY`, not fresh landings.
- `5304393 v740_repr_pa` completed and landed full post-audit reruns: binary `10/0/6` (`32` JSON) and investors `0/0/48` (`96` JSON).
- `5305473 v740_112_binh1` completed and landed the first routed V740 evidence: binary h1 `2/0/2` across `4` cells (`8` JSON).
- The remaining queued routed jobs are now all placed at `2026-04-09T22:50:00` in live `squeue --start`.

## Current Execution Reality

1. Live queue snapshot at verification time:
   - `34` total jobs
   - `8 RUNNING = 5 l40s + 3 hopper`
   - `26 PENDING = 3 gpu + 9 l40s + 14 hopper`
   - there are **no gpu RUNNING jobs**
2. AutoFit current-surface cleanup is now enforced in code and artifacts:
   - the active `autofit` registry exports `AutoFitV739` only
   - `AutoFitV1-V738`, `FusedChampion`, and `NFAdaptiveChampion` remain archived in raw artifacts / source for auditability only
   - rebuilt `all_results.csv` now contains `0` archived AutoFit-family lines
3. `V739` status is currently stalled, not actively progressing:
   - `5298285`, `5298286`, `5298287` timed out at `189G`
   - `5302271`, `5302272`, `5302273` timed out again at `189G`
   - `5299888` OOMed at `224G` with `MaxRSS ~= 234.9G`
   - `5300059` OOMed at `224G` with `MaxRSS ~= 234.9G`
   - `5302274` OOMed again at `280G` with `MaxRSS ~= 293.6G`
   - `5302275 gpu_cos2_t2` timed out at `150G`
   - honest state: `V739` remains the only valid AutoFit baseline, but there is currently no live successful gap-fill job in queue
4. Current canonical backlog surface from the regenerated snapshot:
   - `XGBoost` at `159/160` and `XGBoostPoisson` at `157/160` remain the known structural OOM exceptions
   - `AutoFitV739` remains at `132/160`
   - `Chronos2` and `TTM` are at `114/160`
   - `Crossformer`, `MSGNet`, `MambaSimple`, and `PAttn` are at `107/160`
   - `ETSformer`, `LightTS`, `Pyraformer`, and `Reformer` are at `94/160`
   - the 15 valid Phase 15 TSLib entrants are currently at `92/160`
5. V740 local truth that is settled enough to cite today:
   - shared112 non-routed aggregate is complete at `112/112` with `15/2/95`
   - binary target split: `7/2/7`
   - funding target split: `8/0/40`
   - investors target split: `0/0/48`
   - binary full post-audit rerun: `10/0/6`
   - investors full post-audit rerun: `0/0/48`
   - binary h1 post-audit rerun: `2/0/2`
   - investors h1 post-audit rerun: `0/0/12`
   - binary routed h1 probe: `2/0/2`
   - funding widened best-branch duel `5304260` is completed:
     - `anchor_only_no_log_a085`: `20 wins / 28 losses`
     - `scale_anchor_no_log_a085`: `20 wins / 28 losses`
     - `full` funding cells remain `0 wins / 12 losses`
   - EDGAR exact-day vs as-of misalignment is no longer a live root-cause hypothesis
   - missing or failed text embeddings are no longer a live root-cause hypothesis
6. V740 formal routed evidence is now partial, not absent:
   - `5305473 v740_112_binh1` completed at `2026-04-03T11:57:14` and landed `docs/references/V740_SHARED112_BINARY_ROUTED_H1_PROBE_20260403.md`
   - landed routed surface so far: binary h1 only = `2 wins / 0 ties / 2 losses` across `4` cells (`8` JSON)
   - remaining pending routed jobs:
   - `5305468 v740_112_inv` -> ETA `2026-04-09T22:50:00`
   - `5305469 v740_112_bin` -> ETA `2026-04-09T22:50:00`
   - `5305472 v740_112_invh1` -> ETA `2026-04-09T22:50:00`
   - honest state: the target-routed code path has first executed proof on binary h1, but the full routed shared112 verdict is still pending

## Interpretation Rules

1. Phase 7 and Phase 8 are historical only.
2. `AutoFitV739` is the only active AutoFit baseline in the current registry and public surface.
3. `V734` through `V738` are invalid due to oracle leakage, and `V1` through `V733` plus `FusedChampion` / `NFAdaptiveChampion` are archived historical lines only.
4. The canonical benchmark remains the read-only Phase 9 fair freeze under `runs/benchmarks/block3_phase9_fair/`.
5. `docs/BLOCK3_RESULTS.md` is the filtered leaderboard view, `docs/benchmarks/phase9_current_snapshot.md` is the live artifact snapshot, and this file is the project-wide truth pack that reconciles them.
6. All `V740_*` local notes are research evidence only unless and until a result is explicitly landed into the canonical benchmark.
