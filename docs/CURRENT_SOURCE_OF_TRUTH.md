# Current Source of Truth

> Last verified: 2026-04-20 UTC
> Verification basis: direct raw metrics scan, regenerated `all_results.csv`, regenerated `docs/benchmarks/phase9_current_snapshot.{json,md}`, live `squeue -u npin,cfisch`, targeted `sacct`, P3/P4 gate results audit, P5 adaptive shrinkage gate implementation and 12/12 test pass.

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
10. `docs/references/V740_V745_TESF_STATUS_20260412.md`
11. `docs/references/single_model_true_champion/SINGLE_MODEL_TRUE_CHAMPION_HANDOFF_ZH_20260411.md`
12. `docs/references/V740_REPR_POSTAUDIT_GATE_20260402.md`
13. `docs/references/V740_EDGAR_TEXT_ROOTCAUSE_AUDIT_20260402.md`

## Execution Planning Companion

For forward-looking execution sequencing, use:

- `docs/V740_MASTER_EXECUTION_PLAN.md`

This file remains the current-state truth pack.
`docs/V740_MASTER_EXECUTION_PLAN.md` is the operational roadmap and should not
be treated as a source of current benchmark facts unless those facts are also
reflected here or in the evidence files it cites.

## Active Mainline Research Contract

For the current single-model mainline, the design correction is now fixed as a
hard contract rather than a loose preference:

1. one model still means one efficient forward path and one shared
   event-state / business-evolution geometry trunk
2. it does **not** mean one shared target process, one shared output law, or
   one common financing residual after the trunk
3. the trunk should be pushed toward an event-state-atomic representation of
   what is truly shared: event boundary, progression phase, source-activation
   topology, persistence, and transition pressure
4. `is_funded`, `funding_raised_usd`, and `investors_count` must each read that
   shared geometry through target-isolated observation / generative processes
5. cross-target consistency must come from shared-state supervision, routing
   contracts, calibration, and geometry audits rather than a forced common
   output process
6. near-term mainline work should therefore prioritize trunk-level event-state
   atomization; funding and binary now mainly serve as non-regression guard
   rails while investors remains the hardest proof surface

## 2026-04-15 Mainline Implementation Update

The architecture contract above is no longer only a design statement.

It is now implemented in the native mainline runtime as follows:

1. `src/narrative/block3/models/single_model_mainline/backbone.py` now
   declares an explicit `event_state_v2` trunk schema and atom list
2. `src/narrative/block3/models/single_model_mainline/wrapper.py` now emits a
   first-class `event_state_trunk` card through `get_regime_info()`
3. the trunk card now exposes deeper audited atom groups:
   - financing phase
   - event boundary
   - funded flip
   - funding jump severity
   - investor jump process
   - goal crossing
   - financing persistence
   - source topology
   - source arrival / decay
   - shared-state energy
4. the deeper event-state atoms are no longer telemetry-only: the native
   runtime can now append them into the investors auxiliary path when
   `mainline_event_state_boundary_guard` is active
5. investors-first geometry audit now has a dedicated executable surface:
   - `scripts/analyze_mainline_event_state_geometry.py`
   - `scripts/slurm/single_model_mainline/cf_mainline_investors_event_state_wave_bigmem.sh`
   - `scripts/slurm/single_model_mainline/cf_mainline_investors_event_state_track_bigmem.sh`
6. the first serious investors-first event-state wave completed successfully as
   job `5316199` and wrote
   `runs/benchmarks/single_model_mainline_localclear_20260415/mainline_investors_event_state_wave_cf_bigmem_20260415_serious/report.json`
7. honest first verdict from that wave:
   - the trunk-level geometry instrumentation is real and stable
   - it can now measure dynamic entities, phase activation, transition pressure,
     source heterogeneity, and shared-state energy on real slices
   - but it did **not** yet produce a decisive investors uplift from the current
     source-read / transition variants
   - the only positive aggregate movement was a tiny `+0.0135%` mean MAE delta
     from `source_read_transition_guard_plus_sparsity` on `8` eligible cases,
     which is instrumentation evidence rather than champion evidence
8. 2026-04-15 22:54:21 CEST follow-through after that first verdict:
   - a new serious profile, `mainline_event_state_boundary_guard`, is now wired
     into the native runtime, official track gate, and event-state geometry
     surfaces
   - the old `source_policy_transition_guard` family is now explicitly treated
     as a demoted comparison route rather than the active generation focus
    - the first bounded generation has now fully landed on all four submitted
       surfaces:
       - funding guard `5316359`: `12/12` no-leak pass, `12/12` anchor fallback,
          `12/12` tie vs anchor, incumbent comparison still `incomplete`
       - binary guard `5316360`: `4/4` no-leak pass, `0` collapse cases, metric
          comparison still `incomplete`
       - investors full track gate `5316361`: `event_state_boundary_guard`
          failed official promotion with `mean_mae_delta_pct = -0.3693%`,
          `3` positive / `9` negative cases, while the first dynamic read looked
          positive on `1` dynamic case
       - investors geometry wave `5316362`: `event_state_boundary_guard` was
          negative on all `16/16` evaluated cells with
          `mean_mae_delta_pct = -0.0280%`
    - during result readback, one evaluation-surface bug was confirmed:
       the dynamic multisource track-gate path was not actually feeding
       event-state auxiliary features into the investors lane, so the apparent
       dynamic positive for `event_state_boundary_guard` was not a clean reading
    - that bug is now fixed in code, and the corrected full-gate rerun
       `5316454` has now completed successfully
    - corrected rerun verdict:
       - official gate stayed unchanged and still failed:
          `mean_mae_delta_pct = -0.3693%`, `3` positive / `9` negative
       - corrected dynamic gate is now genuinely positive for
          `event_state_boundary_guard`, but only on the single dynamic case it
          evaluates: `mean_mae_delta_pct = +0.7238%`
       - final gate status is still **not promotable** because official-track
          failure dominates dynamic-only positivity under the current contract
9. current honest reading therefore remains:
   - the shared trunk is now implemented one level deeper than before
   - the geometry-audit loop is now executable end-to-end
    - the second bounded investors-first generation is now fully executed
    - it shows a real dynamic-only signal for the event-state candidate after
       evaluation repair, but still fails official promotion and therefore does
       not clear the mainline gate
   - the investors lane is still not solved
   - no oral-grade or benchmark-champion claim is justified yet
10. 2026-04-16 11:54:51 CEST generation-3 follow-through:
    - the next active candidate is now `mainline_selective_event_state_guard`
    - generation-3 rule is explicit in code: keep the official-positive
       `guarded_jump` base route, disable event-state on `h1`, and disable it on
       source-rich official slices while leaving it available on source-light
       `h>1` slices
      - focused validation for this generation passed `22/22`
      - the first generation-3 submission `5316515` failed fast with a
         fit/predict feature-shape mismatch (`100` vs `126` features)
      - that bug is now fixed in code by composing investors aux features only
         after selective event-state activation is resolved during `fit()`
      - corrected generation-3 investors full gate `5316577` has now completed
      - corrected generation-3 verdict:
         - official mean `+0.0138%`, worst `0.0%`, `9` positive / `0` negative
         - dynamic mean `+0.5548%` on the evaluated dynamic case
         - final gate status is now **promotable_on_current_track = true**
      - official case pattern matches the intended selective contract:
         - all `h1` official slices stay neutral at `0.0%`
         - all source-rich official slices stay event-state-blocked and remain
           non-negative
         - only source-light `h>1` official slices actually open event-state
         - because investors promotion passed, funding and binary guard waves were
            conditionally triggered as `5317300` and `5317301` and have now both
            completed without breaking the current line:
             - funding guard `5317300`: `12/12` no-leak runtime pass, `12/12` jump-
                hurdle active, `12/12` tie vs anchor, incumbent comparison still
                `incomplete`
             - binary guard `5317301`: `4/4` no-leak runtime pass, `0` probability-
                collapse cases, `0` severe-calibration cases, incumbent metric panel
                still `incomplete`
         - because all three heads are now alive on the current bounded local
            contract, the next contractual step has been launched: a full routed
            shared112 integrated package for the same promotable variant is now
            submitted as `5317315` (investors), `5317316` (binary), and `5317317`
            (funding); current queue state is `PENDING` on `QOSGrpNodeLimit`

   ## 2026-04-16 Mainline Understanding Status

   The current evidence is now strong enough to separate what is already locked
   from what is still open.

   ### What Is Now Locked Enough To Use As Mainline Fact

   1. the correct single-model factorization is one shared event-state /
      business-evolution geometry trunk plus hard target-isolated observation /
      generative processes after the trunk
   2. the shared object is financing progression state, not one common target
      process or one shared residual law
   3. the key cross-head asymmetry is real enough to treat as design truth:
      investor-count jumps are mostly nested inside broader funding progression,
      while many funding jumps happen without investor jumps
   4. the current investors horizon law is stable enough to guide architecture:
      `h1` behaves more like immediate occurrence / local readout, while `h>1`
      behaves more like hurdle / transition / positive-count intensity
   5. the current selective event-state law is also locked enough to guide
      execution: `h1` official slices should keep event-state off, source-rich
      official slices should keep event-state blocked, and source-light `h>1`
      slices are the current best place to open event-state

   ### What Is Still Not Proved Yet

   1. `event_state_v2` is not yet proved to be the minimum sufficient atom set;
      it is the current best audited atomization, not the final irreducible one
   2. the full data-law atlas is not complete yet; the current mainline has the
      important financing-progression asymmetry, but it does not yet have a fully
      settled marked-event picture for institutional / large-investor subtypes
   3. the funding head is not yet fully solved on the hardest strong-anchor
      family; current guard evidence shows survival rather than decisive uplift
   4. the binary head is not yet fully solved in benchmark-comparable terms;
      current guard evidence shows no-leak survival and no-collapse rather than a
      finished superiority claim
   5. three-head integrated champion status is not yet proved; that claim should
      stay blocked until the routed shared112 package is materially above the
      historical `15/2/95` line under one runtime

   ### Current Integrated Execution State

   1. investors gate `5316577` is promotable on the current local track
   2. funding guard `5317300` survived cleanly as `12/12` tie vs anchor
   3. binary guard `5317301` survived cleanly with `4/4` no-leak, `0` collapse,
      and `0` severe-calibration cases
   4. the routed shared112 integrated package has now fully landed:
      - shared112 investors `5317315` completed at `0` wins / `0` ties /
        `48` losses with `0` errors and `0` constant-prediction failures
      - shared112 binary `5317316` completed at `3` wins / `1` tie /
        `12` losses with `0` errors and `0` constant-prediction failures
      - shared112 funding `5317317` completed at `24` wins / `0` ties /
        `24` losses with `0` errors and `0` constant-prediction failures
      - this means local gate promotion still does **not** imply integrated
        contender transfer; the current route shows a split picture where
        investors still fails transfer, binary has limited local life, and
        funding has real but highly ablation-sensitive transfer
   5. the institutional / large-investor marked-event subproblem is now being
      tracked in a dedicated evolving note:
      `docs/references/single_model_true_champion/SINGLE_MODEL_TRUE_CHAMPION_MARKED_INVESTOR_EVENT_RESEARCH_ZH_20260416.md`

## Verified Current Facts

| Fact | Current value | Evidence |
| --- | --- | --- |
| Canonical benchmark directory | `runs/benchmarks/block3_phase9_fair/` | direct scan |
| Raw metric records | `16902` | raw metrics scan 2026-04-12 |
| Raw models materialized | `137` | regenerated snapshot |
| Raw non-retired models | `114` | regenerated snapshot |
| Audit-excluded models | `24` | `AUDIT_EXCLUDED_MODELS` in `scripts/aggregate_block3_results.py` |
| Active leaderboard models | `90` | `114 - 24` |
| Raw complete models (`@160`) | `76` | regenerated snapshot |
| Active complete models (`@160`) | `63` | `76 - 13` excluded-complete models |
| Incomplete active models | `27` | `90 - 63` |
| Filtered records in `all_results.csv` | `12514` | regenerated `scripts/aggregate_block3_results.py` output |
| Filtered distinct models | `84` | regenerated `all_results.csv` |
| Filtered non-retired models | `84` | regenerated snapshot |
| Clean full comparable frontier (`160/160`, non-retired, post-filter) | `56` | regenerated snapshot |
| Archived AutoFit-family cleanup | `23 models / 460 rows purged from current surface` | executed aggregate rebuild log 2026-04-03 |
| Current valid AutoFit baseline | `AutoFitV739` only | root `AGENTS.md` |
| V739 landed conditions | `160/160` | raw metrics scan |
| V739 live jobs | `0` | live `squeue -u npin,cfisch` |
| Live queue footprint | `20` total = `7 RUNNING` + `13 PENDING` | direct live `squeue -u npin,cfisch` after `g2_ac_t3_fu` dispatch |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/` plus regenerated snapshot |
| Phase 12 text reruns | `48/48 completed` | existing phase-12 execution notes plus active artifacts |
| Active research runtime owner | `src/narrative/block3/models/single_model_mainline/` | direct code review 2026-04-12 |
| Mainline runtime mode | `mainline_alpha` primary, `mainline_delegate_alpha` compatibility fallback | direct code review 2026-04-12 |
| Live `V740+` / `TESF` jobs visible in queue | `0` matching `v74|mainline|delegate|tesf` | direct live `squeue` name scan 2026-04-12 |
| V741-Lite investors h1 gate | `0 win / 0 tie / 12 loss`, `stop_h1_threshold` | `docs/references/ARCHIVED_V741_LITE_SHARED112_INVESTORS_H1_GATE_20260406.md` plus ladder status |
| V740 shared112 non-routed local line | `112/112 complete`, `15 win / 2 tie / 95 loss` | executed local reference docs |
| V740-V745 local branch verdict | shared event-state / business-evolution geometry trunk survives; shared target process does not | `docs/references/V740_V745_TESF_STATUS_20260412.md` |

## What Changed Since The 2026-04-07 Snapshot

- Raw benchmark records increased from `16314` to `16902` (`+588`).
- The filtered public surface increased from `11990` to `12514` (`+524`) after the 2026-04-12 regeneration pass.
- `AutoFitV739` advanced from `132/160` to `160/160` and has fully left the live queue.
- The live queue shrank from `57` jobs to `20` jobs.
- `AutoFitV739` remains off-queue at `0` live jobs.
- The one uncovered accel_v2 primary-partition lane, `g2_ac_t3_fu`, was resubmitted as `5314271` and has already started running.

## Current Execution Reality

1. Live queue snapshot at verification time:
   - `20` total jobs
   - `7 RUNNING = 1 gpu + 4 l40s + 2 hopper`
   - `13 PENDING = 5 l40s + 8 hopper`
   - pending reasons are `12 x (Priority)` and `1 x (Resources)`
2. AutoFit current-surface cleanup is now enforced in code and artifacts:
   - the active `autofit` registry exports `AutoFitV739` only
   - `AutoFitV1-V738`, `FusedChampion`, and `NFAdaptiveChampion` remain archived in raw artifacts / source for auditability only
   - rebuilt `all_results.csv` now contains `0` archived AutoFit-family lines
3. `V739` is now complete on the canonical clean line:
   - landed baseline is `160/160`
   - live canonical `af739_*` jobs are `0`
   - regenerated snapshot reports `v739_jobs_live = 0`
   - the earlier `TIMEOUT` / `OUT_OF_MEMORY` repair waves are now historical failure ledger only, not current operational state
4. Current canonical backlog surface from the regenerated snapshot:
   - `XGBoost` at `159/160` and `XGBoostPoisson` at `157/160` remain the known structural OOM exceptions
   - `Chronos2` and `TTM` are at `114/160`
   - valid Phase 15 entrants currently span `117-128/160`
   - older partial TSLib lines still occupy the main non-structural backlog (`Crossformer`, `MSGNet`, `MambaSimple`, `PAttn`, `ETSformer`, `LightTS`, `Pyraformer`, `Reformer`)
5. `V741-Lite` local truth that is settled enough to cite today as historical branch evidence:
   - the shared112 investors h1 gate is complete at `0 wins / 0 ties / 12 losses`
   - there were `0` execution failures and `0` constant-prediction collapses, so this is a quality failure, not a harness failure
   - ladder state is `stop_h1_threshold`, which means `full_investors`, `binary_guard`, and `funding_guard` should stay unsubmitted for this revision
6. Earlier `V740` local evidence remains research-only:
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
7. V740 formal routed evidence is no longer partial in the old sense:
   - routed binary guard is executed positive at `10 wins / 0 ties / 6 losses`
   - routed investors h1 is executed negative at `0 wins / 0 ties / 12 losses`
   - routed investors full loop is executed negative at `0 wins / 0 ties / 48 losses`
   - honest state: the routed code path is real, but the shared-process family is still structurally non-viable on investors
8. The local research line is no longer well-described as just `V740`:
   - concrete local variants in code are `v740_alpha`, `v741_lite`, `v742_unified`, `v743_factorized`, `v744_guarded_phase`, and `v745_evidence_residual`
   - the active runtime owner is now `single_model_mainline/`
   - `mainline_alpha` is the primary runtime, with `mainline_delegate_alpha` retained only as a compatibility fallback
   - no live `v74|mainline|delegate|tesf` job names were visible in the 2026-04-12 queue scan, so current research status should be read from executed evidence plus the current handoff docs, not from an assumed live queue
8. Immediate canonical execution priority is no longer AutoFit repair:
   - the live queue currently covers only a subset of accel_v2 lanes on `l40s` and `hopper`
   - a direct raw scan against the accel_v2 23-model list shows that the absent families `t1_s2`, `t2_co`, `t2_s2`, and `t3_co` are already complete and should not be requeued
   - the only uncovered non-colliding primary-partition backlog surface was `g2_ac_t3_fu`, and it has now been resubmitted as `5314271` and dispatched onto `gpu`
   - after that submission, there is no remaining known uncovered accel_v2 family on the 23-model canonical list

## 2026-04-20 Mainline Track Gate P3/P4 Results and P5 Implementation

> Last updated: 2026-04-20

### Gate Results Summary (P3 and P4)

1. **P3 TPP Intensity Baseline — PASSED (promotable)**
   - official: `+0.0138%` mean, `0.0%` worst, `9` positive / `0` negative
   - dynamic: `+0.555%` on evaluated dynamic case
   - `selective_event_state_guard` remains the only promotable candidate

2. **P4 Jump Neural ODE — FAILED (not promotable)**
   - official: `-1.04%` mean, `-17.8%` worst, `5` positive / `7` negative
   - structural failure pattern: `task1_core_only` (no EDGAR) all positive (`+0.4~3.3%`), `task2_core_edgar` all negative (`h1 -17.8%`)
   - root cause: z-score normalization of ODE features fails on EDGAR surfaces (distribution shift)
   - same root cause as Gen-8 Hawkes (`-173%`): trunk-level feature expansion does not generalize across heterogeneous entity surfaces

3. **P4 gate independently cross-validated selective**: numbers identical to P3 gate (`+0.014% mean, 9+/0-`)

### Strategic Decision (2026-04-20)

- Trunk-level feature expansion direction abandoned after two consecutive failures (Hawkes -173%, Jump ODE -1.04%)
- All future innovation moves to lane-level improvements
- P5 Adaptive Shrinkage Gate implemented and passing 12/12 tests

### P5 Implementation: Adaptive Shrinkage Gate

- **Design**: James-Stein inspired per-sample oracle alpha, HGBR prediction, lane-level only
- **Formula**: `alpha = clip((learned - target)(learned - anchor) / (learned - anchor)^2, 0, 1)`
- **New files**: `src/narrative/block3/models/single_model_mainline/shrinkage_utils.py`
- **Modified files**: `investors_lane.py`, `wrapper.py`, `objectives.py`, `variant_profiles.py`
- **Variant profile**: `mainline_shrinkage_gate_guard` (Gen-10/P5)
- **Gate scripts ready**: `np_mainline_track_gate_p5_official_bigmem.sh`, `np_mainline_track_gate_p5_dynamic_bigmem.sh`
- **Test status**: `12/12` passed (unit + integration)
- **Gate status**: awaiting full regression pass → commit → sbatch

## Interpretation Rules

1. Phase 7 and Phase 8 are historical only.
2. `AutoFitV739` is the only active AutoFit baseline in the current registry and public surface.
3. `V734` through `V738` are invalid due to oracle leakage, and `V1` through `V733` plus `FusedChampion` / `NFAdaptiveChampion` are archived historical lines only.
4. The canonical benchmark remains the read-only Phase 9 fair freeze under `runs/benchmarks/block3_phase9_fair/`.
5. `docs/BLOCK3_RESULTS.md` is the filtered leaderboard view, `docs/benchmarks/phase9_current_snapshot.md` is the live artifact snapshot, and this file is the project-wide truth pack that reconciles them.
6. All `V740_*` through `V745_*` local notes are research evidence only unless and until a result is explicitly landed into the canonical benchmark.
7. Archived `V741` / `V742` branch notes are negative-result or lesson-pack documents, not active promotion candidates, even when their original dates or older names suggest otherwise.
8. The active research runtime now lives under `single_model_mainline/`; older `v740_alpha.py` branches remain audited prototype surfaces, not the final ownership boundary.
