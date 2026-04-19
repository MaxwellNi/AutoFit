# Block 3 Model Benchmark Status

> Last updated: 2026-04-20 UTC
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: regenerated `docs/BLOCK3_RESULTS.md`, regenerated `docs/benchmarks/phase9_current_snapshot.{md,json}`, live `squeue`, targeted `sacct`, P3/P4 gate results, P5 shrinkage gate implementation.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16902** | regenerated snapshot |
| raw models (all) | 137 | 114 current-surface models + 23 archived AutoFit-family lines |
| raw complete @160 | 76 | regenerated snapshot |
| active leaderboard models | **90** | 114 current-surface raw models - 24 audit-excluded |
| active complete @160 | **63** | 76 raw complete - 13 excluded complete models |
| incomplete active models | **27** | 90 - 63 |
| post-filter records in `all_results.csv` | **12514** | regenerated `docs/BLOCK3_RESULTS.md` |
| post-filter distinct models | 84 | archived AutoFit-family lines purged from current surface |
| post-filter non-retired models | 84 | regenerated snapshot |
| clean full comparable frontier | **56** | post-filter non-retired models at shared 160/160 |
| archived AutoFit cleanup | **23 models / 460 rows removed** | executed aggregate rebuild log |
| live jobs | **20** | 7R + 13PD |
| V739 landed | **160/160** | raw metrics scan |
| V739 live jobs | **0** | live `squeue -u npin,cfisch` |
| V741-Lite investors h1 gate | **0/0/12**, stop | landed h1 gate plus ladder status |
| V740 shared112 local line | **112/112**, **15/2/95** | executed local reference docs |
| V740 formal routed evidence | **binary guard executed positive, investors full/h1 negative** | executed local routed verdict exists: binary `10/0/6`, investors h1 `0/0/12`, investors full `0/0/48` |
| V743 factorized key-slice result | **local breakthrough, not shared112 champion** | key investors slice `0.473` vs `V740 3.637`; shared112 quick suite still negative |
| current TESF runtime owner | **`single_model_mainline/`** | `mainline_alpha` primary, `mainline_delegate_alpha` compatibility fallback |

## 2026-04-15 Mainline Event-State Update

1. The active mainline research contract is no longer only documented; it is
   now partially materialized in code through a native `event_state_trunk`
   regime card.
2. Current implementation entry points:
   - `src/narrative/block3/models/single_model_mainline/backbone.py`
   - `src/narrative/block3/models/single_model_mainline/wrapper.py`
   - `src/narrative/block3/models/single_model_mainline/variant_profiles.py`
3. Current executable audit entry points:
   - `scripts/analyze_mainline_event_state_geometry.py`
   - `scripts/slurm/single_model_mainline/cf_mainline_investors_event_state_wave_bigmem.sh`
   - `scripts/analyze_mainline_investors_track_gate.py`
   - `scripts/slurm/single_model_mainline/cf_mainline_investors_event_state_track_bigmem.sh`
4. The first investors-first event-state wave (`5316199`) completed cleanly and
   produced a real report under:
   - `runs/benchmarks/single_model_mainline_localclear_20260415/mainline_investors_event_state_wave_cf_bigmem_20260415_serious/report.json`
5. Honest verdict from that wave:
   - the shared trunk can now expose dynamic-entity, transition, source-surface,
     and shared-state-energy geometry on real investors slices
   - this is real progress in trunk observability and architecture discipline
   - it is **not** yet a proof that current source-read / transition variants
     have solved investors
   - aggregate uplift is still negligible, with only a tiny positive move from
     `source_read_transition_guard_plus_sparsity`
6. 2026-04-15 22:54:21 CEST execution update:
    - the trunk schema is now `event_state_v2`
    - deeper atoms are wired into the investors auxiliary path through
       `mainline_event_state_boundary_guard`
    - the source-read / transition family is now demoted to a comparison route
       in the official track gate contract
      - landed execution status:
          - `5316359` funding postfix guard rail: COMPLETED, `12/12` tie vs anchor,
             incumbent panel still `incomplete`
          - `5316360` binary postfix guard rail: COMPLETED, `4/4` no-leak runtime
             pass, `0` collapse cases, metric panel still `incomplete`
          - `5316361` investors full track gate: COMPLETED,
             `event_state_boundary_guard` official mean `-0.3693%`, `3` positive /
             `9` negative, not promotable
          - `5316362` investors event-state geometry wave: COMPLETED,
             `event_state_boundary_guard` mean `-0.0280%`, `0` positive /
             `16` negative
      - result-interpretation correction:
          - the first dynamic positive read for `event_state_boundary_guard` was not
             trustworthy because the dynamic multisource surface was not yet carrying
             event-state aux features
          - that evaluation bug is now fixed in code
          - corrected investors full-gate rerun `5316454` has now completed
          - corrected rerun kept official results unchanged and negative for
             `event_state_boundary_guard`, but upgraded its dynamic signal to a real
             `+0.7238%` on the one dynamic case
          - final gate verdict still remains `not promotable`

### 2026-04-15 22:54:21 CEST Real-Time Bounded Program Bar

| Step | Status | Honest read |
| --- | --- | --- |
| deeper trunk atoms | done | `event_state_v2` landed in native runtime |
| investors-first candidate wiring | done | `mainline_event_state_boundary_guard` now active |
| focused validation | done | targeted tests passed `21/21` |
| funding guard rail wave | done | job `5316359`, pure tie vs anchor |
| binary guard rail wave | done | job `5316360`, clean runtime, metric panel incomplete |
| investors serious wave first read | done | `5316361` official negative, `5316362` geometry negative |
| corrected investors full gate rerun | done | `5316454`, dynamic fixed and still not promotable |

Bounded generation status: `6/6` steps completed, `0/6` actively running,
`0/6` blocked.

### 2026-04-16 11:54:51 CEST Generation-3 Delta

| Step | Status | Honest read |
| --- | --- | --- |
| selective event-state design | done | `h1` off, source-rich official slices off |
| generation-3 candidate implementation | done | `mainline_selective_event_state_guard` landed |
| gate/tests update | done | focused validation `22/22` passed |
| generation-3 first submit | done | `5316515` failed fast on feature-shape mismatch, now fixed |
| generation-3 corrected investors gate | done | `5316577`, official and dynamic both positive |
| conditional funding/binary guards | done | `5317300`, `5317301`, no regression signal |
| shared112 integrated contender package | done | investors `0/0/48`, binary `3/1/12`, funding `24/0/24` |

Current active generation bar: `7/7` complete, `0/7` in-progress, `0/7` blocked.

Generation-3 investors verdict:
- promotable on current track
- official mean `+0.0138%`, worst `0.0%`, `9/12` positive and `0/12` negative
- dynamic mean `+0.5548%` on the evaluated dynamic case
- selective gating behaved as designed: `h1` neutral, source-rich slices blocked,
  source-light `h>1` slices opened

Guard-rail verdict after promotion:
- funding survived cleanly but did not yet improve: `12/12` tie vs anchor
- binary survived cleanly: `4/4` no-leak, `0` collapse, `0` severe calibration
- incumbent comparison panels remain `incomplete`, so these are still local
   contract survival signals rather than final benchmark claims

First shared112 integrated read:
- investors shared112 full `5317315` is now landed and decisively negative:
   `0` wins / `0` ties / `48` losses
- there were `0` execution errors and `0` constant-prediction failures
- honest interpretation: the current selective-event-state route clears the
   audited local gate but still does not transfer to full shared112 contender
   status
- binary shared112 full `5317316` is also now landed at `3` wins / `1` tie /
   `12` losses with `0` execution errors and `0` constant-prediction failures
- honest interpretation: binary shows limited local life, with transfer signal
   concentrated in a narrow subset rather than broad shared112 superiority
- funding shared112 full `5317317` is now landed at `24` wins / `0` ties /
   `24` losses with `0` execution errors and `0` constant-prediction failures
- honest interpretation: funding has real transfer on some ablation faces but
   remains sharply split, especially with the `full` face still clearly losing

## Canonical Backlog Tiers

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known structural exceptions |
| landed AutoFit baseline | `AutoFitV739@160` | valid baseline, canonical line complete and clean |
| foundation partials | `Chronos2@114`, `TTM@114` | partial real coverage, not complete |
| old TSLib partials | `Crossformer@125`, `MSGNet@122`, `MambaSimple@121`, `PAttn@119` | remaining accel_v2 backlog |
| intermediate TSLib partials | `MultiPatchFormer@105`, `TimeFilter@105` | remaining excluded/partial backlog outside the clean frontier |
| older TSLib e2-limited partials | `ETSformer@108`, `LightTS@107`, `Pyraformer@102`, `Reformer@102` | remaining accel_v2 backlog |
| valid Phase 15 entrants | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | currently span `117-128/160` |
| excluded Finding H entrants | `CFPT`, `DeformableTST`, `MICN`, `PathFormer`, `SEMPO`, `SparseTSF`, `TimeBridge`, `TimePerceiver` | each at `69/160`, excluded for constant predictions |
| excluded structural partial | `NegativeBinomialGLM@21` | excluded structural failure |

### Current accel_v2 uncovered queue surface

- not currently represented in live queue: `t1_s2`, `t2_co`, `t2_s2`, `t3_co`, `t3_fu`
- direct raw scanning against the accel_v2 23-model list shows `t1_s2`, `t2_co`, `t2_s2`, and `t3_co` are already complete and should not be requeued
- the only still-incomplete absent family was `t3_fu`, and it has now been resubmitted as `g2_ac_t3_fu` (`5314271`)
- after that submission, there is no remaining known uncovered accel_v2 family on the canonical 23-model list

## V739 Status

| Fact | Value | Notes |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | current registry and leaderboard export only this line |
| landed conditions | `160/160` | canonical clean line complete |
| missing total | `0` | no remaining canonical coverage gap |
| live queue jobs | `0` | no active `af739_*` work remains in queue |
| archived AutoFit lines in current surface | `0` | rebuilt `all_results.csv` no longer carries retired / invalid AutoFit-family rows |
| current state | **landed complete** | failure history is now historical only |

### Historical Repair Wave

| jobs | lane | final state | reqmem | note |
| --- | --- | --- | --- | --- |
| `5298285`, `5298286`, `5298287` | `t1_e2`, `t2_e2`, `t3_e2` | `TIMEOUT` | `189G` | first repaired e2 wave timed out |
| `5302271`, `5302272`, `5302273` | `t1_e2`, `t2_e2`, `t3_e2` | `TIMEOUT` | `189G` | second repaired e2 wave also timed out |
| `5299888` | `t1_s2` | `OUT_OF_MEMORY` | `224G` | `MaxRSS ~= 234.9G` |
| `5300059` | `t2_s2` | `OUT_OF_MEMORY` | `224G` | `MaxRSS ~= 234.9G` |
| `5302274` | `t1_s2` | `OUT_OF_MEMORY` | `280G` | `MaxRSS ~= 293.6G` |
| `5302275` | `gpu_cos2_t2` | `TIMEOUT` | `150G` | second repaired cos2 copy timed out |

These rows are now historical ledger only. The regenerated 2026-04-12 snapshot reports `v739_conditions_landed = 160` and `v739_jobs_live = 0`.

## Live Queue Reality

Detailed per-job progress and ETA live in `docs/RUN_QUEUE_PROGRESS_CURRENT.md`.

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 1 | `g2_ac_t3_fu` is now live on the primary partition |
| gpu PENDING | 0 | no gpu shard remains queued but not dispatched |
| l40s RUNNING | 4 | accel_v2 overflow lane is active |
| l40s PENDING | 5 | current l40s backlog |
| hopper RUNNING | 2 | opportunistic overflow lane is active |
| hopper PENDING | 8 | current hopper backlog |
| **total** | **20** | **7 RUNNING + 13 PENDING** |

### Current Queue Interpretation

- Canonical benchmark throughput is currently active on all three partitions, with `g2_ac_t3_fu` now running on `gpu`.
- `V739` is no longer a live queue concern; its canonical coverage gap is closed.
- The remaining live canonical work is accel_v2 continuation.
- The immediate uncovered primary-partition hole is closed: `t3_fu` is now represented by running `g2_ac_t3_fu` (`5314271`).

## V741-Lite Status

| Lane | Current evidence | Honest read |
| --- | --- | --- |
| investors h1 gate | shared112 investors h1 `0/0/12`, `0` failures, `0` constant predictions | hard negative result; ladder correctly stopped at h1 |
| binary guard | not submitted | should remain blocked until investors h1 materially improves |
| funding guard | not submitted | should remain blocked until investors h1 materially improves |
| overall | ladder state `stop_h1_threshold` | `V741-Lite` is not ready for championship claims |

## V740 Local Research Status

| Lane | Current evidence | Honest read |
| --- | --- | --- |
| binary | shared112 non-routed `7/2/7`; full post-audit `10/0/6`; routed h1 `2/0/2` | real binary competitiveness exists, and routed h1 has first landed proof, but it is not yet dominant |
| funding | shared112 non-routed `8/0/40`; widened best-branch duel gives `20/28` for both strongest no-log branches | partially rescuable, but still far from championship level |
| investors | shared112 non-routed `0/0/48`; full post-audit `0/0/48`; h1 post-audit `0/0/12` | still a structural failure lane |
| routed formal evidence | binary guard executed positive; investors h1 and full routed surfaces executed negative | target-routed code is real, and the executed verdict now says the shared-process family is still not investors-competitive |

## V742-V745 Route-Correction Status

| Line | Current evidence | Honest read |
| --- | --- | --- |
| `V742-unified` | protected-blend key investors slice stayed worse than `V740`; hard blend was catastrophic | common financing process is not the right end-state |
| `V743-factorized` | key investors slice improved strongly, but shared112 quick suite still lost on investors/binary/funding | factorized investors state helps, but does not by itself create a champion package |
| `V744-guarded_phase` | matched-slice regressions versus `V743` on investors, binary, and funding | guarded common-process coupling further falsified the shared-process thesis |
| `V745-evidence_residual` | investors improved again only after lane isolation, while binary/funding still trailed `V740` guard rails | hard target isolation is necessary, but final full-surface dominance is still unproven |

## Current Operational Conclusion

1. The filtered public surface is now `12514`, so any doc still quoting `11990`, `11976`, `12300`, or `12422` as the current surface is stale.
2. The canonical benchmark backlog is no longer blocked by `V739`; the immediate limiting factor is accel_v2 throughput, not a known missing queue family.
3. For `V741-Lite`, the first hard gate is now settled and negative: investors h1 is `0/0/12`, so this line is not close to an internal champion package, let alone an external champion or oral-level package.
4. For the single-model mainline, one model remains the goal but one shared target process no longer does; the active design direction is now one shared event-state / business-evolution geometry trunk with target-isolated observation / generative processes after it.
5. A direct 2026-04-12 live queue scan did not show any `v74|mainline|delegate|tesf` job names, so current `TESF` status should be read from executed evidence and current handoff docs, not from an assumed live queue wave.
6. As of 2026-04-15 22:54:21 CEST, the mainline has crossed from design-only into audited trunk instrumentation plus a fully executed second bounded investors-first generation. The corrected rerun removed the dynamic evaluation bug and confirmed a real dynamic-only positive for `event_state_boundary_guard`, but official promotion still failed, so this generation closes as informative but non-promotable.

## 2026-04-20 P3/P4 Gate Results and P5 Implementation

1. **P3 TPP Intensity Baseline gate — PASSED**: official `+0.0138%` mean, `9+/0-`; dynamic `+0.555%`. `selective_event_state_guard` remains the only promotable candidate.
2. **P4 Jump Neural ODE gate — FAILED**: official `-1.04%` mean, `-17.8%` worst, `5+/7-`. Structural failure: `task1_core_only` (no EDGAR) all positive, `task2_core_edgar` all negative. Root cause: z-score normalization fails on EDGAR surfaces.
3. **Strategic decision**: trunk-level feature expansion abandoned after two consecutive failures (Hawkes `-173%`, Jump ODE `-1.04%`). All future innovation moves to lane-level.
4. **P5 Adaptive Shrinkage Gate implemented**: James-Stein inspired per-sample oracle alpha, HGBR prediction, lane-level only. `12/12` tests passed. Gate scripts ready for sbatch.
5. **P4 dynamic gate bug fixed**: parameter naming mismatch `enable_investors_intensity_baseline` → `enable_intensity_baseline` in gate script.
6. **Current mainline track candidates**: `selective_event_state_guard` (promotable, Gen-6), `shrinkage_gate_guard` (P5 candidate, Gen-10, pending gate).
