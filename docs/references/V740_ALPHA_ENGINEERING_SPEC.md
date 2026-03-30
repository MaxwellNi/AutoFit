# V740-alpha Engineering Specification

> Date: 2026-03-23
> Status: design-ready, pre-benchmark
> Scope: first implementation-stage single-model V740 prototype
> Depends on: `docs/CURRENT_SOURCE_OF_TRUTH.md`, `docs/references/V740_DESIGN_SPECIFICATION.md`, `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`, `docs/references/V740_MULTISOURCE_ALIGNMENT_SPEC.md`
>
> 2026-03-24 implementation update:
> the local prototype now includes a compact temporal memory branch, a
> value-space auxiliary branch, and source-specific EDGAR/text event-memory
> encoders wired through the wrapper. This remains pre-benchmark and should be
> treated as an audited prototype path, not as an active leaderboard model.
>
> 2026-03-25 smoke update:
> first real freeze-backed narrow-slice smoke results are recorded in
> `docs/references/V740_ALPHA_REAL_SMOKE_RESULTS_20260325.md`. The core
> multisource wiring is now operational. Those first tiny binary slices did
> collapse to constant predictions, but the newer audited matrix in
> `docs/references/V740_ALPHA_SMOKE_MATRIX_20260325.md` now passes six
> end-to-end cases and no longer shows constant prediction on the two binary
> slices. The matrix has since been expanded to 10 audited cases including
> representative `task2`, `task3`, and `seed2` slices, all of which now pass.
> The remaining issue is model quality, not basic execution integrity, and the
> latest smoke evidence suggests that task conditioning is stable but not yet
> expressive enough to induce materially different behavior on shared funding
> slices. A subsequent 2026-03-25 update added a stronger offline binary
> teacher and explicit task-specific modulation/residual biases. That update
> improved some hard slices further, especially `core_edgar + is_funded` and
> `full + investors_count`, but the gains remain uneven, which reinforces that
> alpha is now in a regime-quality optimization phase rather than a plumbing
> phase. A follow-up component ablation then showed that task modulation is
> currently helpful for the tested count slice but harmful on the tested binary
> slices; accordingly, the prototype now bypasses task modulation for binary
> targets by default while keeping it for continuous/count targets.
>
> A second 2026-03-25 audit then corrected an important local-evaluation bug:
> the smoke path had been capping train rows with a global tail truncation,
> which is inappropriate for source-sparse EDGAR/text regimes because it can
> erase historical source coverage while leaving the join code itself intact.
> The smoke runner now uses deterministic time-preserving downsampling instead.
> After that correction, the 10-case audited matrix changed materially:
> `core_edgar + is_funded + h=14` improved to `MAE = 0.1063`, `full +
> is_funded + h=14` improved to `MAE = 0.1061`, `core_edgar + funding_raised`
> task1 improved to `MAE = 308.5K`, and the task2/task3 EDGAR funding slices
> also improved. This is a strong reminder that V740-alpha's local audit tools
> must preserve source coverage and temporal diversity if they are to provide
> trustworthy engineering feedback.

> The pre-benchmark tooling has also been expanded with a benchmark-like local
> comparison script, `scripts/run_v740_alpha_minibenchmark.py`, which compares
> V740-alpha and V739 on identical freeze-backed local slices without touching
> the live benchmark harness or writing under `runs/`.
>
> A corrected binary-only component ablation now also exists at
> `docs/references/V740_ALPHA_COMPONENT_ABLATION_20260325_binary_refresh.md`.
> After the downsampling fix, the key binary conclusion is sharper:
> preserving source coverage drives the main gain, while teacher distillation
> and the event head are currently second-order refinements on the audited
> binary slices. The binary task-modulation bypass remains the safest default,
> but it is no longer the dominant explanation for the corrected local gains.
>
> The first larger local mini-benchmark slice also provides an important
> caution: scaling a binary EDGAR slice from the smallest audited smoke setting
> to a `16 entity / 1600 row` local comparison initially pushed `core_edgar +
> is_funded + h=14` back to `MAE = 0.3426`. After the corrected binary
> representative-sampling path was wired into the mini-benchmark runner, the
> current V740-only result on the same case improved to `MAE = 0.2016`; the
> paired corrected `V739` comparison is still pending because that workload was
> moved off the iris access server and into a resumable SLURM path. The right
> takeaway is that corrected local comparison is now more credible, but the
> head-to-head evidence is not complete yet.
>
> 2026-03-26 long-horizon local-only update:
> V740-alpha now has a real exercised `h>30` prototype path. The horizon
> conditioning path has been upgraded from a tiny discrete lookup to a hybrid
> continuous-plus-bucket horizon encoder, so local research is no longer
> limited to a fixed `60/90` extension. The first verified `h=60` smoke result
> is recorded in
> `docs/references/V740_ALPHA_LONGH_SMOKE_20260326.md`. On a local
> `task2_forecast / core_only / funding_raised_usd / h=60` slice
> (`12 entities / 1500 train rows / 2 epochs`), the prototype runs end-to-end
> without constant collapse and reaches `MAE = 360123.84` in `42.14s`.
> A same-slice `h=30` control on the same day is actually slightly worse and
> collapses to a constant prediction (`MAE = 361820.06`), which is weak but
> still important evidence that `h>30` should not be dismissed a priori on the
> current V740 prototype.
> This is still local-only evidence and must not be interpreted as an official
> benchmark expansion or as proof that `h>30` is already beneficial.
> A second same-day `investors_count` check now also exists and points in the
> opposite direction: on `task2_forecast / core_only / investors_count`,
> `h=30` remains healthier (`MAE = 48.64`, non-constant) while `h=60`
> collapses to a constant prediction (`MAE = 50.08`). So `h>30` is now a real
> exercised path, but it is not yet uniformly helpful across targets.
> A same-day `h=90` probe is now also recorded in
> `docs/references/V740_ALPHA_LONGH_SMOKE_20260326.md`. On the tested narrow
> local slices, `core_only` still falls back while the first viable `h=90`
> behavior now appears on `core_edgar / funding_raised_usd / input_size=120`.
> A new `h=45 / input_size=90` funding probe is also non-constant, while a
> first `h=180 / input_size=180` funding probe remains fallback-only. These
> results sharpen the implementation lesson: longer-horizon behavior depends on
> joint scaling of horizon, context length, target type, and slice
> representativeness, not just on adding more horizon IDs.
>
> One more caution is now explicit after a density-audit rerun: the earlier
> `edgar_source_density = 0.0` / `text_source_density = 0.0` readings on
> funding/count long-h slices were caused by a logging bug, not by genuine
> source absence. Representative reruns now show `core_edgar` funding at
> `h=60` with `edgar_source_density = 1.0`, and `full` funding at `h=60` with
> both `edgar_source_density = 1.0` and `text_source_density = 1.0`.
> Even so, the present long-h improvements should still be interpreted
> carefully: they now show source-covered slices plus improved
> horizon/context behavior, but they do not yet prove that source-native
> sparse-event memory is already carrying the longer-horizon load.
>
> 2026-03-27 local-compare update:
> the first corrected local head-to-head is no longer pending.
> On `mb_t1_core_edgar_is_funded_h14`, `V739` currently beats `V740-alpha`
> (`MAE = 0.1623` vs `0.2016`) and does so by selecting `PatchTST`.
> That result is important because it sharpens the current engineering target:
> V740-alpha is already dramatically cheaper on fit time (`38.9s` vs `301.3s`)
> but still lags the valid baseline on an audited binary EDGAR slice.
> The same day also added a larger-slice `full / funding_raised_usd / h=90 /
> input_size=120 / 24 entities / 3000 train rows` audit, which completed
> cleanly and non-constantly (`MAE = 397737.19`, `edgar_source_density ≈ 1.0`,
> `text_source_density = 1.0`) but should not be over-read as proof that text
> already improves long-horizon behavior. It mainly confirms that the bigger
> `h=90` path now executes cleanly on a richer local slice.
>
> 2026-03-30 implementation update:
> alpha now includes a first lightweight scaffold for two previously
> design-only mechanisms:
> 
> 1. **selective-learning sample weighting**
>    - windows are now assigned mild per-sample weights from target magnitude
>      and history volatility,
>    - the weighting is intentionally conservative (`0.6–1.3`) so the first
>      implementation can be audited without destabilizing the prototype,
>    - this currently affects the training loss only and is not yet claimed as
>      a benchmark-ready curriculum policy.
> 2. **DistDF-style multistep distribution alignment**
>    - non-binary targets now receive an auxiliary per-window alignment on
>      prediction-vs-target horizon mean and horizon dispersion,
>    - this is a first approximation of the intended long-horizon objective,
>      not the full paper-faithful DistDF implementation.
>
> The same day also moved `ElasTST` out of the docs-only bucket:
> a local vendor-import wrapper now exists, the first funding smoke is
> non-fallback, and a shorter-context investors smoke also runs end-to-end.
> `ElasTST` is therefore now a real local-clear candidate rather than only a
> literature reference.
> A post-change V740 sanity check on
> `task2_forecast / core_edgar / funding_raised_usd / h=30` also completed
> cleanly (`docs/references/v740_alpha_selective_smoke_20260330/`), which is
> enough to say the first selective-learning / DistDF scaffold does not break
> the current alpha execution path.
>
> A later same-day refinement tightened that scaffold without changing the
> architecture surface:
>
> - selective-learning weights now use per-window source activity from the
>   EDGAR/text memories rather than only global source-density scalars,
> - the non-binary DistDF-style auxiliary now also aligns horizon endpoint and
>   step-difference shape, not just horizon mean/dispersion.
>
> On the same audited funding slice
> (`task2_forecast / core_edgar / funding_raised_usd / h=30`), the second-pass
> sanity smoke remains clean and slightly improves from `MAE = 441572.68` to
> `MAE = 441156.71`. This is still far from benchmark-ready, but it confirms
> that the stronger objective scaffold remains stable.
>
> `ElasTST` also now has a second narrow benchmark-clear result on the
> `investors_count` slice (`5298437`). That run completes cleanly at the system
> level but returns `fairness_pass = false`, which is an important reminder
> that varied-horizon models still need count-side tuning before they are ready
> for broader benchmark promotion.
>
> A dedicated summary of the second-pass objective audit now exists at:
> `docs/references/V740_ALPHA_SELECTIVE_DISTDF_AUDIT_20260330.md`.
> The short version is:
>
> - same-slice funding smoke improves slightly (`441572.68 -> 441156.71`),
> - a hard binary EDGAR smoke remains non-degenerate (`MAE = 0.3089`),
> - and a source-covered `h=60` funding smoke also remains non-degenerate
>   (`MAE = 232810.09`).
>
> This is still not benchmark-ready evidence, but it is enough to justify
> continuing with the strengthened objective scaffold rather than reverting it.
>
> A third same-day mechanism pass has now also landed in code:
>
> - a lightweight `CASALocalContextBlock` now sits after the patch/context
>   mixer as a cheap condition-aware local-context refinement path,
> - a `StaticDynamicTimeFusion` gate now performs a first explicit
>   TimeEmb-inspired static/dynamic/source fusion before the final pooled
>   combine step.
>
> This remains intentionally lightweight and is not a faithful reimplementation
> of either paper family. It is an engineering-first synthesis intended to
> answer whether these mechanism directions help on Block 3 without breaking
> the one-model cost profile.
>
> The first audited results from this third pass are now recorded in
> `docs/references/V740_ALPHA_SELECTIVE_DISTDF_AUDIT_20260330.md` and show:
>
> - `task2_forecast / core_edgar / funding_raised_usd / h=30`
>   improves to `MAE = 182493.85`,
> - `task1_outcome / core_edgar / is_funded / h=14`
>   improves to `MAE = 0.2853`,
> - `task2_forecast / core_edgar / funding_raised_usd / h=60`
>   improves to `MAE = 176137.83`,
> - `task1_outcome / full / is_funded / h=14`
>   improves further to `MAE = 0.2758`,
> - and `task2_forecast / full / funding_raised_usd / h=60`
>   stays effectively tied with the matching `core_edgar` slice
>   (`MAE = 176137.07` vs `176137.83`),
>
> all while remaining non-constant and source-covered on the audited EDGAR
> slices. This is still local-only evidence, but it is the strongest direct
> sign so far that V740 benefits from moving beyond objective-only refinement
> and into disciplined mechanism synthesis. The current honest reading is:
>
> - fuller-source binary behavior is slightly stronger than `core_edgar`,
> - fuller-source long-h funding is now stable and source-covered,
> - but text is not yet proven to be the primary gain source on the current
>   audited funding slice.

## 1. Purpose

V740-alpha is the first engineering realization of the new V740 direction:
**one condition-aware model, one inference path, no runtime ensemble**.

Alpha is not supposed to solve the entire project by itself. Its job is to
validate the architecture thesis that emerged from the current clean benchmark:

- champion behavior is concentrated in a small set of reusable mechanisms,
- those mechanisms can plausibly be synthesized into one shared model,
- and doing so should be cheaper than V739-style per-cell multi-candidate
  validation selection.

Alpha therefore exists to answer one question rigorously:

> Can a single conditional forecasting model absorb the right champion
> mechanisms and immediately outperform the current validation-selector baseline
> in the places where the selector is weakest, especially on binary and
> EDGAR-sensitive cells, without giving back its strength on funding and count
> targets?

This is also an operational question. V739 is valid, but it is slow because it
trains and evaluates an 8-model candidate pool per condition. Alpha only
matters if it preserves the one-model cost profile that V739 does not have.

One design clarification is now explicit:

> Alpha should be understood as a **N-BEATS-like decomposition-first skeleton
> with only the minimum additional branches needed to absorb the benchmark's
> other repeated champion mechanisms**.

This matters because Block 3 no longer needs a vague “unified model.” It needs
a model with a clean forecasting backbone and a disciplined policy for where
extra complexity is allowed.

## 2. Non-Negotiable Constraints

Alpha must satisfy all of the following.

### 2.1 Fairness constraints

1. identical train/val/test chronology to the current benchmark
2. no test-derived routing or tuning
3. no hidden ensemble at inference
4. no fallback path that silently changes the effective model
5. identical feature-regime semantics (`co`, `s2`, `ct`, `ce`, `e2`, `fu`)

### 2.2 Efficiency constraints

1. single forward path
2. materially cheaper than V739 multi-candidate selection
3. parameter budget in the rough range of a normal forecasting model, not a
   heavyweight foundation model
4. feasible on the current benchmark hardware allocations without special-case
   infrastructure
5. no hidden reintroduction of candidate tournaments inside the training loop
6. longer-horizon support must remain single-model and must not rely on
   degenerate fallback behavior as the default explanation for `h>30` results

### 2.3 Modeling constraints

1. explicit support for all three target types
2. explicit internal conditioning on task, target, horizon, and ablation
3. robust behavior on heavy-tailed funding targets
4. calibrated behavior on binary funding success
5. ability to exploit EDGAR when it helps without forcing EDGAR behavior on all
   cells

## 3. What Alpha Must Learn from the Benchmark

Alpha should not imitate one paper. It should absorb the useful parts of the
current champion frontier.

### 3.1 Mechanisms to absorb

- `NBEATS` / `NBEATSx`: basis decomposition, residual refinement, cheap short-
  horizon structure
- `NHITS`: multi-resolution pooling and interpolation
- `Chronos` / `ChronosBolt`: scale-robust long-horizon priors
- `KAN`: nonlinear short-horizon response for counts
- `DeepNPTS`: binary/event calibration behavior
- `GRU`: compact temporal-state regularization
- `PatchTST`: local patch/context modeling for EDGAR-sensitive cells
- `SAMformer` / `CASA`: efficient channel-wise attention-style mixing
- `OLinear`: strong cheap low-cost branch for easy continuous/count cells
- `DIAN` / `TimeEmb`: invariant/variant and static/dynamic disentanglement
- `UniTS` / `ElasTST`: one-model multi-task and multi-horizon conditioning

### 3.2 Mechanisms Alpha should not copy blindly

- giant full-self-attention stacks
- heavyweight foundation-model pretraining assumptions
- runtime model selection among multiple trained candidates
- naive text concatenation as the default multimodal strategy

## 4. Input / Output Contract

### 4.1 Required inputs

For each instance, alpha receives:

1. **target history**
   - recent target trajectory for one entity
   - default context length: `60`
   - alpha ablations may test `96` and `128`

2. **numeric covariates**
   - same numeric surface available under the active benchmark ablation
   - no private or future-aware features

3. **static summary vector**
   - compact entity/context summary derived from currently available covariates
   - should be small and stable, not a huge side network

4. **event memory inputs**
   - optional `edgar` event tokens
   - optional `text` event tokens
   - both follow the dual-clock tokenization in
     `docs/references/V740_MULTISOURCE_ALIGNMENT_SPEC.md`

5. **condition tokens**
   - task id
   - target id
   - horizon id
   - ablation id

In the current alpha code path, these event memories are constructed from the
joined daily panel as source-aware event proxies:

- EDGAR memory is built from daily rows where EDGAR-derived as-of features are
  present.
- text memory is built from daily rows where numeric text embeddings are
  present.

This is already useful and leakage-safe, but it should still be viewed as the
first serious implementation step rather than the final multisource design.

### 4.2 Outputs

Alpha should produce:

1. point forecast
2. optional distributional or interval auxiliary output
3. target-specific auxiliary outputs:
   - calibrated probability for `is_funded`
   - nonnegative/count-aware output for `investors_count`

## 5. Architecture Blueprint

Alpha should be a seven-stage model:

1. lightweight adapters
2. decomposition-first shared trunk
3. multi-resolution branch
4. local-context efficiency branch
5. compact temporal memory branch
6. value-space auxiliary branch
7. invariant/variant fusion under condition tokens
8. target-specific heads

### 5.1 Lightweight adapters

Five small adapters are enough.

1. **history adapter**
   - projects raw target history into latent temporal channels

2. **covariate adapter**
   - projects numeric exogenous features into a compact context vector

3. **event-memory adapter**
   - projects EDGAR and text token sequences into compact source-specific event
     memories

4. **value-space adapter**
   - projects bucketized target values into a small auxiliary latent space

5. **condition adapter**
   - embeds task / target / horizon / ablation into a small condition vector

These adapters should remain intentionally modest. Alpha should win by the right
structure, not by inflating embeddings.

### 5.2 Decomposition-first trunk

This is the architectural anchor.

The trunk should begin with a decomposition-style block that separates and
recombines:

- trend
- oscillatory / seasonal behavior
- residual structure

The purpose is to absorb the strongest parts of:

- `NBEATS`
- `NBEATSx`
- `OLinear`
- decomposition-oriented 2024–2025 literature

This trunk is not just one branch among many. It is the **default explanation
path** for the model. In practical terms, alpha should first try to solve a
cell with:

1. coefficient generation,
2. basis-style forecast construction,
3. residual refinement,
4. and direct horizon-wise prediction.

Only after this backbone is in place should the model rely on the more
specialized side branches for:

- multi-resolution structure (`NHITS`),
- long-horizon value robustness (`Chronos` / `ChronosBolt`),
- compact temporal memory (`GRU`),
- binary/event calibration (`DeepNPTS`),
- local EDGAR-sensitive context (`PatchTST`),
- and local short-horizon count nonlinearity (`KAN`).

Implementation target:

- moving-average or learnable smoothing trend extractor
- residual residualization path
- shared hidden representation after recomposition

### 5.3 Multi-resolution block

This block should explicitly absorb `NHITS`-style advantages.

Required behaviors:

- pooled views at several scales
- interpolation or learned upsampling back to forecast space
- support for weekly and multi-week structure

This block is especially important for h=7 and h=14 cells.

### 5.4 Lightweight local-context branch

This branch exists because some cells need richer local-context modeling,
especially:

- EDGAR-sensitive cells
- binary/event-style cells
- cells where local transitions matter more than long smooth extrapolation

This branch should borrow ideas from:

- `PatchTST`
- `SAMformer`
- `CASA`
- `LiPFormer`

but should remain lightweight.

Suggested implementation:

- patch or channel slice extraction
- depthwise or channel-wise mixing
- efficient score or gate computation
- fusion back into the trunk without a large Transformer tower

### 5.5 Compact temporal memory branch

Alpha should explicitly include a **small gated memory path** to preserve the
useful part of the `GRU` advantage:

- compact hidden state
- low-variance temporal regularization
- cheap recurrent bias for cells where compact memory beats pure decomposition

Implementation note:
the current local prototype uses a small GRU-based summary path rather than a
full recurrent backbone.

### 5.6 Value-space auxiliary branch

Alpha should explicitly include a **small value-space branch** to preserve the
useful part of the `Chronos` advantage without turning alpha into a full TSFM:

- bucketized target-value view
- scale-insensitive auxiliary representation
- auxiliary supervision for heavy-tailed and long-horizon targets

Implementation note:
the current local prototype uses a lightweight bucketized value encoder and a
log-alignment auxiliary loss rather than a large tokenized TSFM path.

### 5.7 Invariant / variant fusion

Alpha should explicitly split two representations:

1. **invariant dynamics**
   - temporal structure shared across many entities and conditions

2. **variant dynamics**
   - behavior modulated by entity context, EDGAR, or local regime

This is the simplest serious approximation to the `DIAN` / `TimeEmb` direction.
A practical implementation is:

- invariant branch
- variant branch
- condition-driven fusion gate
- optional source gate for `core` / `edgar` / `text` balance

### 5.8 Condition modulation

Condition tokens are not just metadata. They should actively modulate the model.

Required conditioning axes:

- task
- target
- horizon
- ablation

Recommended mechanism:

- FiLM-style affine transforms,
- small residual adapters,
- or conditional gates.

Alpha should not use separate submodels for separate horizons or targets.

### 5.9 Event-memory policy

Alpha should not treat `edgar` and `text` as ordinary dense covariates by
default. Instead:

- `core` remains the main daily state stream
- `edgar` is encoded as sparse filing-event memory
- `text` is encoded as sparse semantic-event memory

This is mandatory if alpha is expected to become the basis for a serious V740
multisource model rather than just a stronger `core_only` baseline.

The current alpha prototype now instantiates this policy concretely:

- a source-specific EDGAR memory encoder
- a source-specific text memory encoder
- each consumes:
  - recent raw event tokens
  - recency-bucket summary tokens
- both feed the shared fused representation through a single inference path

## 6. Target-Specific Heads

### 6.1 Continuous head

Used for `funding_raised_usd`.

Requirements:

- robust to heavy tails
- stable under long horizon extrapolation
- compatible with transformed-target auxiliary losses

### 6.2 Count-aware head

Used for `investors_count`.

Requirements:

- nonnegative output behavior
- tolerance for jumps and local nonlinearities
- compatibility with a count-sensitive auxiliary penalty

### 6.3 Binary calibrated head

Used for `is_funded`.

Requirements:

- Bernoulli/logit output
- explicit calibration regularization
- must not be a trivial reuse of the continuous head
- should be capable of mimicking `DeepNPTS`-style behavior under internal
  conditioning

### 6.4 Optional uncertainty head

Not mandatory for alpha.0, but strongly recommended by alpha.1 or alpha.2.

Purpose:

- prediction intervals
- better analysis of ambiguous cells
- compatibility with conformal or `JAPAN`-style post-calibration

## 7. Objective Stack

### 7.1 Primary losses

- continuous targets: `Huber` or `MAE`
- count target: `MAE` plus nonnegative / count-sensitive regularization
- binary target: `BCE` plus `Brier` regularization

### 7.2 Auxiliary losses to phase in

#### Time-o1-style alignment

Use transformed-label alignment for heavy-tailed `funding_raised_usd`.
This should enter by alpha.1.

#### DistDF-style multistep objective

Use a direct distributional multistep auxiliary loss for h=14 and h=30.
This should enter by alpha.2.

#### QDF-style hard-cell weighting

Use structure-aware weighting so hard windows matter more than trivial ones.
This should enter by alpha.2.

#### Selective-learning regularization

Downweight windows that are persistently noisy, unstable, or misleading.
This should enter by alpha.2.

#### Consistency regularization

Encourage stable latent geometry across seed-equivalent or redundant task slices.
This should enter by alpha.1.

## 8. Distillation Strategy

Distillation is allowed only during training.

### 8.1 Teacher pool

The teacher pool should be fixed to the strongest current champion families:

- `NBEATS`
- `NHITS`
- `Chronos` / `ChronosBolt`
- `KAN`
- `DeepNPTS`
- `GRU`
- `PatchTST`

### 8.2 Distillation policy

For each condition, alpha may receive:

1. ground truth
2. one primary teacher prediction
3. optional secondary teacher prediction for ambiguous cells

Important:

- no runtime teacher ensemble
- no inference-time teacher dependence
- no test-derived teacher choice
- no source-specific teacher cheating that leaks unavailable EDGAR/text context

## 9. Efficiency Budget

Alpha should feel like a normal model, not an infrastructure project.

Target budget:

1. **parameters**: roughly `2M–8M` for early versions
2. **VRAM**: comfortable on current benchmark GPUs
3. **training time**: materially closer to one NeuralForecast-scale model than
   to V739 multi-candidate selection
4. **inference**: one forward pass only

## 10. Implementation Phases

### 10.1 Alpha.0

Purpose:
prove the single-model synthesis direction is viable.

Required modules:

- decomposition block
- multi-resolution block
- condition tokens
- target-specific heads
- compact temporal memory branch (minimal)
- value-space auxiliary branch (minimal)
- first source-aware event-memory encoders on the joined daily surface

No distillation yet.
No uncertainty head required.

### 10.2 Alpha.1

Purpose:
make alpha structurally credible against the frontier.

Add:

- invariant / variant split
- lightweight local-context branch
- event-memory branch for `edgar`
- champion distillation
- Time-o1-style alignment
- consistency regularization

### 10.3 Alpha.2

Purpose:
make alpha competitive on the hardest current regimes.

Add:

- DistDF-style multistep objective
- QDF-style hard-cell weighting
- selective-learning regularization
- optional uncertainty head
- longer-context ablation (`96` / `128`)
- `text` event-memory branch after EDGAR path is stable

## 11. Acceptance Criteria

Alpha should be considered successful only if it improves in the right way.

1. clearly better than V739 on its clean comparable slice
2. materially stronger on `is_funded`
3. no collapse on `funding_raised_usd`
4. no large penalty on `investors_count`
5. materially cheaper than multi-candidate V739-style selection
6. no leakage, no hidden special-casing, no fairness regression

## 12. Immediate Next Engineering Tasks

1. finish the first **real** `SAMformer` benchmark-smoke path
2. tighten the local benchmark memory path so efficiency-first additions can be
   verified without loading unnecessary full-surface data interactively
3. harden `V740AlphaPrototypeWrapper` around the exact condition-token contract
4. move from current event-proxy memory to a more audited source-native EDGAR
   event path
5. add a first objective switch for binary calibration and heavy-tail handling
6. prepare the first champion-distillation scaffold
7. benchmark the current alpha prototype on narrow audited smoke slices only

## 13. Non-Goals

Alpha is not:

1. a large foundation model
2. an online ensemble
3. a retrieval-heavy production system
4. a text-first multimodal model
5. the final V740 production method

Its purpose is to validate the single-model synthesis thesis under the current
benchmark.
