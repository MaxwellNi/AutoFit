# Current Source of Truth

> Last verified: 2026-04-03 09:15 UTC
> Verification basis: direct raw metrics scan, regenerated `all_results.csv`, regenerated `docs/benchmarks/phase9_current_snapshot.{json,md}`, live `squeue -u npin`, targeted `squeue --start`, targeted `sacct`, and executed V740 local reference notes.

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
8. `docs/references/V740_ROADMAP_STATUS_20260330.md`
9. `docs/references/V740_REPR_POSTAUDIT_GATE_20260402.md`
10. `docs/references/V740_EDGAR_TEXT_ROOTCAUSE_AUDIT_20260402.md`

## Verified Current Facts

| Fact | Current value | Evidence |
| --- | --- | --- |
| Canonical benchmark directory | `runs/benchmarks/block3_phase9_fair/` | direct scan |
| Raw metric records | `16284` | raw metrics scan 2026-04-03 |
| Raw models materialized | `137` | regenerated snapshot |
| Raw non-retired models | `116` | regenerated snapshot |
| Audit-excluded models | `24` | `AUDIT_EXCLUDED_MODELS` in `scripts/aggregate_block3_results.py` |
| Active leaderboard models | `92` | `116 - 24` |
| Raw complete models (`@160`) | `75` | regenerated snapshot |
| Active complete models (`@160`) | `62` | `75 - 13` excluded-complete models |
| Incomplete active models | `30` | `92 - 62` |
| Filtered records in `all_results.csv` | `12422` | regenerated `scripts/aggregate_block3_results.py` output |
| Filtered distinct models | `107` | regenerated `all_results.csv` |
| Filtered non-retired models | `86` | regenerated snapshot |
| Clean full comparable frontier (`160/160`, non-retired, post-filter) | `55` | regenerated snapshot |
| Current valid AutoFit baseline | `AutoFitV739` only | root `AGENTS.md` |
| V739 landed conditions | `132/160` | raw metrics scan |
| V739 live jobs | `0` | live `squeue -u npin` |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/` plus regenerated snapshot |
| Phase 12 text reruns | `48/48 completed` | existing phase-12 execution notes plus active artifacts |
| V740 shared112 non-routed local line | `112/112 complete`, `15 win / 2 tie / 95 loss` | executed local reference docs |
| V740 formal routed outputs | `0 landed` | no routed JSON outputs and no routed summary docs found under 20260402/20260403 roots |

## What Changed Since The 2026-04-01 Snapshot

- Raw benchmark records increased from `16182` to `16284` (`+102`).
- The filtered public surface also moved after a fresh aggregation rebuild: `12300 -> 12422`.
- `V739` no longer has live queue presence; the latest repair wave ended in repeated `TIMEOUT` and `OUT_OF_MEMORY`, not fresh landings.
- Formal routed V740 evidence is still `0 landed`, and the scheduler now places the queued routed jobs around `2026-04-12`.

## Current Execution Reality

1. Live queue snapshot at verification time:
   - `36` total jobs
   - `8 RUNNING = 5 l40s + 3 hopper`
   - `28 PENDING = 5 gpu + 9 l40s + 14 hopper`
   - there are **no gpu RUNNING jobs**
2. `V739` status is currently stalled, not actively progressing:
   - `5298285`, `5298286`, `5298287` timed out at `189G`
   - `5302271`, `5302272`, `5302273` timed out again at `189G`
   - `5299888` OOMed at `224G` with `MaxRSS ~= 234.9G`
   - `5300059` OOMed at `224G` with `MaxRSS ~= 234.9G`
   - `5302274` OOMed again at `280G` with `MaxRSS ~= 293.6G`
   - `5302275 gpu_cos2_t2` timed out at `150G`
   - honest state: `V739` remains the only valid AutoFit baseline, but there is currently no live successful gap-fill job in queue
3. Current canonical backlog surface from the regenerated snapshot:
   - `XGBoost` at `159/160` and `XGBoostPoisson` at `157/160` remain the known structural OOM exceptions
   - `AutoFitV739` remains at `132/160`
   - `Chronos2` and `TTM` are at `114/160`
   - `Crossformer`, `MSGNet`, `MambaSimple`, and `PAttn` are at `107/160`
   - `ETSformer`, `LightTS`, `Pyraformer`, and `Reformer` are at `94/160`
   - the 15 valid Phase 15 TSLib entrants are currently at `91/160`
4. V740 local truth that is settled enough to cite today:
   - shared112 non-routed aggregate is complete at `112/112` with `15/2/95`
   - binary target split: `7/2/7`
   - funding target split: `8/0/40`
   - investors target split: `0/0/48`
   - binary h1 post-audit rerun: `2/0/2`
   - investors h1 post-audit rerun: `0/0/12`
   - funding widened best-branch duel `5304260` is completed:
     - `anchor_only_no_log_a085`: `20 wins / 28 losses`
     - `scale_anchor_no_log_a085`: `20 wins / 28 losses`
     - `full` funding cells remain `0 wins / 12 losses`
   - EDGAR exact-day vs as-of misalignment is no longer a live root-cause hypothesis
   - missing or failed text embeddings are no longer a live root-cause hypothesis
5. V740 formal routed evidence is still absent:
   - no routed summary markdown landed under `docs/references/V740_SHARED112_*ROUTED*_2026040*.md`
   - no routed JSON outputs landed under `runs/benchmarks/v740_localclear_20260402/` or `runs/benchmarks/v740_localclear_20260403/`
   - current pending routed jobs:
     - `5305468 v740_112_inv` -> ETA `2026-04-12T11:20:00`
     - `5305469 v740_112_bin` -> ETA `2026-04-12T12:10:00`
     - `5305472 v740_112_invh1` -> ETA `2026-04-12T15:30:00`
     - `5305473 v740_112_binh1` -> ETA `2026-04-12T15:40:00`
   - related non-routed post-audit rerun job `5304393 v740_repr_pa` is also still pending, ETA `2026-04-11T21:20:00`
   - honest state: the target-routed code path is real, but the first formal routed head-to-head results have not landed yet

## Interpretation Rules

1. Phase 7 and Phase 8 are historical only.
2. `V734` through `V738` are retired due to oracle leakage and must not be treated as current baselines.
3. The canonical benchmark remains the read-only Phase 9 fair freeze under `runs/benchmarks/block3_phase9_fair/`.
4. `docs/BLOCK3_RESULTS.md` is the filtered leaderboard view, `docs/benchmarks/phase9_current_snapshot.md` is the live artifact snapshot, and this file is the project-wide truth pack that reconciles them.
5. All `V740_*` local notes are research evidence only unless and until a result is explicitly landed into the canonical benchmark.
