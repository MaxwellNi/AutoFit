# V740-alpha Real Smoke Results (2026-03-25)

> Date: 2026-03-25
> Status: first real-data narrow-slice smoke
> Scope: V740-alpha pre-benchmark validation only
> Code path: `scripts/run_v740_alpha_smoke_slice.py`
>
> 2026-03-25 superseding update:
> the first two 4-entity smoke runs below were intentionally tiny and useful
> for plumbing validation, but they are no longer the best summary of current
> V740-alpha behavior. After binary anti-collapse loss updates, entity
> selection with train/test target coverage, and a timezone fix in the
> source-native EDGAR memory path, the audited six-case smoke matrix in
> `docs/references/V740_ALPHA_SMOKE_MATRIX_20260325.md` now passes end-to-end
> on all six cases. In particular, the two binary slices no longer collapse to
> constant predictions, although their current predictive quality remains weak
> and far from benchmark-ready.
>
> The matrix was then expanded further to include representative `task2`,
> `task3`, and `seed2` slices. The current 10-case audited matrix also passes
> end-to-end. Binary behavior remains non-constant on both the main and seed2
> EDGAR slices, while the `funding_raised_usd` task1/task2/task3 EDGAR cases
> remain numerically very close to each other, suggesting that task conditioning
> is wired correctly but is not yet inducing meaningfully differentiated
> behavior.
>
> A later 2026-03-25 update added a stronger offline binary teacher
> (`logistic + histogram-gradient`) plus explicit task-specific modulation and
> residual biases inside the network. Under the same 10-case audited matrix,
> `core_edgar + is_funded + h=14` improved further to `MAE = 0.6881`, while
> `full + investors_count + h=14` improved materially to `MAE = 535.35`. At the
> same time, `full + is_funded + h=14` regressed slightly relative to the prior
> round, which is a useful signal that the current teacher/modulation design is
> helping some regimes more than others rather than acting as a uniformly
> beneficial upgrade.
>
> A subsequent component-ablation pass in
> `docs/references/V740_ALPHA_COMPONENT_ABLATION_20260325.md` then showed a
> sharper operational conclusion: on the currently tested binary slices,
> disabling task modulation improves MAE more than disabling teacher or event
> supervision, while on `full + investors_count` task modulation remains
> beneficial. That evidence was strong enough to change the default prototype
> behavior: binary targets now bypass task modulation by default, while
> continuous/count targets keep it enabled.
>
> A later 2026-03-25 audit found that the local smoke path was still using a
> global `train.tail(max_rows)` truncation rule. On source-sparse regimes such
> as `core_edgar`, this silently erased much of the EDGAR-bearing history from
> the local train slice and made `edgar_source_density` appear to be `0.0` even
> though the join itself was working. That truncation bug has now been replaced
> by deterministic time-preserving downsampling, and the audited 10-case matrix
> in `docs/references/V740_ALPHA_SMOKE_MATRIX_20260325.md` should be treated as
> superseding the earlier narrow-slice interpretation. Under the corrected
> local slice construction, the binary EDGAR cells improved dramatically:
> `core_edgar + is_funded + h=14` now reaches `MAE = 0.1063`, `full +
> is_funded + h=14` reaches `MAE = 0.1061`, and the corresponding seed2 EDGAR
> binary slice remains essentially identical at `MAE = 0.1063`. The stronger
> conclusion is therefore not only that the multisource path is wired
> correctly, but that local audit slices must preserve source coverage or they
> will materially understate V740-alpha's behavior.
>
> Under that corrected slice construction, a refreshed binary-only component
> ablation in `docs/references/V740_ALPHA_COMPONENT_ABLATION_20260325_binary_refresh.md`
> shows a different priority ordering from the earlier tail-truncated view.
> On the currently audited binary slices, teacher distillation and the event
> head now behave as second-order refinements: disabling either one changes MAE
> only in the fourth decimal place, whereas the corrected source coverage is
> what unlocked the large performance jump. Meanwhile, the binary bypass of
> task modulation remains the right default, but the new ablation makes clear
> that on these corrected binary slices the task-modulation switch is now
> mostly a regime-safety default rather than the main source of gain.
>
> The first benchmark-like local mini-benchmark run
> (`docs/references/V740_ALPHA_MINIBENCHMARK_20260325.md`) also confirms that
> harder local slices remain substantially more difficult than the smallest
> audited smoke cells: on a larger `16 entity / 1600 row` local
> `core_edgar + is_funded + h=14` slice, V740-alpha currently records
> `MAE = 0.3426`. This is a healthier estimate of where the prototype stands on
> harder local binary conditions.

## 1. Purpose

This note records the first **real freeze-backed** smoke results for
`V740AlphaPrototypeWrapper` after the addition of:

- compact temporal memory
- value-space auxiliary branch
- source-aware EDGAR event memory
- source-aware text event memory
- lean smoke loader that avoids full-table OOM for narrow-slice experiments

These runs do **not** belong to the active benchmark line. They are
implementation-audit smoke checks only.

## 2. What Was Validated

The smoke runner now uses:

1. canonical freeze pointer
2. same strict temporal split semantics as Block 3
3. same leakage-safe target preparation logic
4. a lean loader:
   - read first `N` entities only
   - filter core/text/EDGAR at load time
   - avoid full-table memory blow-up

This matters because the earlier smoke path inherited the benchmark harness's
full-surface loading behavior and was killed with exit `137` even on tiny
slices.

## 3. Result A: `core_edgar`, `task1_outcome`, `is_funded`, `h=14`

Command shape:

- task: `task1_outcome`
- ablation: `core_edgar`
- target: `is_funded`
- horizon: `14`
- max_entities: `4`
- max_rows: `300`

Observed result:

- train rows: `300`
- val rows: `47`
- test rows: `103`
- feature count: `105`
- EDGAR columns used internally: `39`
- text columns used internally: `0`
- coverage ratio: `1.0`
- wall time: `16.021s`

Metric snapshot:

- `MAE = 0.3107`
- `RMSE = 0.5574`
- `sMAPE = 62.14`

Critical behavior:

- **constant_prediction = true**
- `prediction_std = 0.0`

Interpretation:

The narrow real-data EDGAR path now runs end-to-end without OOM and without
fairness-shape failure, which is a real implementation milestone. However, the
model still collapses to a constant solution on this tiny binary slice. That is
not a loader problem anymore; it is a current modeling/training limitation.

## 4. Result B: `full`, `task1_outcome`, `is_funded`, `h=14`

Command shape:

- task: `task1_outcome`
- ablation: `full`
- target: `is_funded`
- horizon: `14`
- max_entities: `4`
- max_rows: `300`

Observed result:

- train rows: `300`
- val rows: `47`
- test rows: `103`
- feature count: `169`
- EDGAR columns used internally: `39`
- text columns used internally: `64`
- coverage ratio: `1.0`
- wall time: `14.013s`

Metric snapshot:

- `MAE = 0.3107`
- `RMSE = 0.5574`
- `sMAPE = 62.14`

Critical behavior:

- **constant_prediction = true**
- `prediction_std = 0.0`

Interpretation:

This is the first real smoke proving that the `full` path truly reaches the
prototype with:

- EDGAR event memory
- text event memory
- target-specific binary head

So the multisource path is now operational in code. The remaining issue is
model quality, not wiring.

## 5. Coverage Failure on Tiny Slices

Two additional tiny-slice attempts on the first 4 entities failed before
training because the train split had zero valid target rows after leakage-safe
preparation:

- `full` + `funding_raised_usd` + `h=30`
- `full` + `investors_count` + `h=14`

This is not a code bug. It is a reminder that very small entity slices can be
pathologically sparse for some targets under strict temporal validity.

## 6. Source-Native EDGAR Sanity Check

The new `build_source_native_edgar_memory()` path was also sanity-checked on a
toy repeated-event frame.

Observed behavior:

- repeated daily carry-forward rows were collapsed into the true filing events
- recent token output shape: `(4, 5)`
- bucket counts reflected two deduplicated filing events rather than four daily
  rows

This confirms that the new EDGAR path is strictly better than treating repeated
daily rows as independent events.

## 7. What This Means

The main conclusion from this round is very clear:

1. the **smoke infrastructure problem is solved**
2. the **multisource wiring problem is largely solved**
3. the next bottleneck is **binary/event learning quality**

So the next V740-alpha priority should not be “more plumbing.” It should be:

- stronger binary calibration
- harder anti-collapse regularization
- better use of teacher signals on `is_funded`
- and then wider real-slice smoke expansion beyond 4 entities

## 8. Current Best Reading of the Smoke State

As of the latest 2026-03-25 update, the most accurate interpretation is:

1. the multisource prototype path is operational on real freeze-backed slices
2. the earlier binary constant-collapse issue has been softened enough to
   produce non-constant predictions on the audited binary slices
3. the next bottleneck is no longer basic wiring or timezone correctness, but
   predictive quality and calibration quality
4. seed2-conditioned smoke behavior is currently stable on the tested slices
5. task conditioning is present, but current task-specific differentiation on
   shared funding slices is still weak
6. stronger teacher-guided calibration helps some hard regimes, but current
   gains are uneven and still need sharper task-aware targeting
7. local smoke/mini-benchmark slices must preserve temporal/source coverage;
   naive tail truncation is not acceptable for source-sparse EDGAR/text audits
8. the first clean regime-specific default discovered so far is: keep task
   modulation for continuous/count targets, but bypass it for binary targets

That is the right place to be. We have moved from “does the prototype really
work at all?” to “which modeling upgrades most improve quality on the hardest
cells?”
