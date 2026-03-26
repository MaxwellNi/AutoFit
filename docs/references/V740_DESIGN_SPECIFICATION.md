# V740 AutoFit Design Specification

> Date: 2026-03-23
> Status: superseding design draft
> Scope: V740 methodology only
> Depends on: `docs/CURRENT_SOURCE_OF_TRUTH.md`, `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`, `docs/references/V740_ALPHA_ENGINEERING_SPEC.md`, `docs/references/V740_MULTISOURCE_ALIGNMENT_SPEC.md`
>
> 2026-03-24 implementation note:
> the local V740-alpha prototype now contains explicit compact-memory,
> value-space, and source-aware EDGAR/text event-memory paths. The design is
> therefore no longer only conceptual; the remaining work is to harden,
> benchmark, and audit those paths carefully.

## 0. Design Reset

V740 should no longer be framed as a repair of an old oracle router.
That framing was useful during the transition away from the leaked AutoFit lines,
but it is no longer ambitious enough and it is no longer faithful to the actual
project objective.

The current clean facts are already strong enough to support a more serious
design target:

- canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
- current raw benchmark surface: **16,077** records
- raw models materialized: **137**
- audit-excluded models: **24**
- active leaderboard models: **92**
- active complete models: **62 @160/160**
- only valid current AutoFit baseline: **AutoFitV739**
- V739 landed coverage: **132/160**
- V739 clean comparable slice: **132 conditions**, **56 comparable non-retired models**
- V739 current filtered-slice performance: mean rank **13.18**, rank position **14/56**,
  median gap **1.52%**, mean gap **3.91%**, wins **17**

These facts imply two things.

First, clean adaptive selection is already viable after leakage removal.
Second, the remaining problem is no longer “how to route correctly with a table,”
but “how to compress champion behavior into one efficient model that does not
need to train or run several specialists per cell.”

That is the actual V740 problem.

## 1. Objective

V740 is the project’s first attempt at a **single-model champion system** for
Block 3.

The target is not merely to be “competitive on average.” The target is:

1. one training graph
2. one inference path
3. condition-aware internal specialization
4. no test-set leakage
5. no runtime ensemble
6. no requirement to wait for several candidate models before producing a forecast
7. a realistic path to challenging the current benchmark champions across all
   **160** evaluation cells

In short, V740 should pursue the following Pareto point:

- accuracy close to or above the best current specialists,
- wall-clock closer to a single model than to V739-style validation selection,
- and fairness identical to the current benchmark protocol.

### 1.1 Why V739 is not the end-state

V739 is scientifically valid, but operationally it is still too expensive to be
the final AutoFit answer.

The reason is structural. V739 does not train one model per condition. It
trains and evaluates an 8-model candidate pool on the temporal validation split
before committing to a winner. That is exactly why it remains a strong clean
baseline and exactly why it remains slow in practice.

So the V740 objective is not "make V739 route a bit better." The objective is:

- preserve the leakage-free selection lesson,
- preserve the useful champion mechanisms,
- remove the runtime multi-candidate cost profile.

If V740 ever regresses into hidden runtime model tournaments, then it has
missed the point.

## 2. What the Current Benchmark Actually Teaches

The benchmark is now large enough and stable enough that the core structural
lessons are no longer ambiguous.

### 2.1 The frontier is concentrated, not chaotic

The active frontier is not controlled by dozens of unrelated models. The true
champion set is compact:

- `NBEATS`
- `Chronos`
- `NHITS`
- `KAN`
- `DeepNPTS`
- `GRU`
- `PatchTST`
- with `NBEATSx` and `ChronosBolt` acting as useful close variants

This means V740 does not need to absorb every benchmarked model. It needs to
absorb the mechanisms that repeatedly produce wins.

### 2.2 The benchmark is regime-dependent

The same model family does not dominate all targets, horizons, and feature
regimes.

- `NBEATS` dominates many short-horizon continuous and cross-task cells.
- `Chronos` is strongest in several longer-horizon funding regimes.
- `NHITS` is strongest where multi-resolution structure matters.
- `KAN` is uniquely strong on short-horizon investor-count prediction.
- `DeepNPTS` is a binary specialist, but only on a narrow task slice.
- `PatchTST` wins few cells but has the best mean rank, making it the cleanest
  generalist sanity anchor.

### 2.3 The biggest remaining weakness is binary/event behavior

The current V739 line is already close to the local champion on
`funding_raised_usd` and `investors_count`, but it remains clearly weaker on
`is_funded`.

That is not mainly a routing failure anymore. It is a mechanism gap.
Binary/event-style behavior must be built into the model, not treated as a side
case of regression.

### 2.3b The current design still needs three additional mechanism lifts

If V740 is pushed toward a truly strong end-state, three mechanism classes need
to be made more explicit than they are in most generic single-model designs:

1. **value-space robustness**
   - to preserve the useful part of the `Chronos` advantage without requiring a
     heavyweight foundation model

2. **compact temporal memory**
   - to preserve the useful part of the `GRU` advantage without turning the
     whole model into a recurrent forecaster

3. **binary/event calibration**
   - to preserve the useful part of the `DeepNPTS` advantage without collapsing
     binary behavior into a regression head

### 2.4 Text is not yet a first-class strength

The current PCA text path still hurts more often than it helps.
That does not mean text is useless. It means the current text pathway is not yet
the right one.

Therefore V740 should treat text as:

- a secondary branch,
- a late-stage optional module,
- or a retrieval/context source,

rather than as a mandatory early fusion branch.

### 2.5 Seed effects are small, but calibration and objective choice still matter

Seed2 stability is strong overall. This is good news: once V740 finds the right
inductive biases, its gains should be real rather than just stochastic. It also
means future gains are more likely to come from:

- better objective design,
- better calibration,
- better condition-aware modulation,
- and better handling of heterogeneous covariate structure,

than from brute-force architecture inflation.

## 3. Research Delta from 2024–2026 Top Venues

The new literature does not suggest “build a bigger controller.” It points in a
more disciplined direction: **single-model conditional forecasting with stronger
internal structure, better objectives, and tighter efficiency discipline**.

### 3.1 Most important architecture signals

#### SAMformer (ICML 2024 Oral)

Key signal:
lightweight attention with channel-wise structure and sharpness-aware
optimization can outperform heavier alternatives without requiring a giant
foundation-model budget.

V740 implication:
use efficient attention sparingly and locally, not as the whole model.
This supports a small EDGAR-sensitive context mixer rather than a full
Transformer trunk.

#### UniTS (NeurIPS 2024)

Key signal:
one model can cover multiple time-series tasks if task conditioning is treated
as a first-class design principle instead of an afterthought.

V740 implication:
condition tokens for task, target, horizon, and ablation are not decorative.
They should be central to the architecture.

#### ElasTST (NeurIPS 2024)

Key signal:
varied-horizon forecasting should not require one separate model per horizon.

V740 implication:
shared horizon-aware conditioning is preferable to training four unrelated
horizon-specific systems.

#### CASA (IJCAI 2025)

Key signal:
efficient attention can be improved by better score computation and lighter
cross-dimensional interaction, reducing memory and saturation.

V740 implication:
if V740 keeps an attention-like submodule, it should be closer to CASA/SAMformer
style efficiency than to generic full self-attention.

#### LightGTS (ICML 2025)

Key signal:
lightweight general forecasting can remain highly competitive if multi-level and
multi-scale collaboration are built carefully.

V740 implication:
this is one of the strongest external confirmations that a lightweight,
mechanism-rich V740 is a credible path.

#### OLinear (NeurIPS 2025)

Key signal:
low-cost orthogonal-domain linear modeling can be much stronger than naive
linear baselines.

V740 implication:
V740 needs a serious cheap branch, not only a “deep” branch. A strong linear or
near-linear decomposition path can cover many stable continuous/count cells at
very low cost.

#### LiPFormer (ICDE 2025)

Key signal:
patch-based local structure can be exploited without paying the full PatchTST
cost.

V740 implication:
retain patch-style local modeling, but do it in a lightweight way.

### 3.2 Most important representation signals

#### DIAN (IJCAI 2024)

Key signal:
invariant and variant structure should be decoupled in heterogeneous temporal
settings.

V740 implication:
entity-invariant temporal dynamics and entity/context-specific behavior should be
modeled separately and fused later.

#### TimeEmb (NeurIPS 2025)

Key signal:
static-dynamic disentanglement can be done with lightweight machinery.

V740 implication:
Block 3 is exactly the kind of data where this matters: static startup context,
dynamic financing trajectory, partial EDGAR, and noisy text side information.

### 3.3 Most important objective and reliability signals

#### Time-o1 (2025)

Key signal:
transformed-label alignment helps when the raw target geometry is hostile to
standard point losses.

V740 implication:
heavy-tailed `funding_raised_usd` should not rely on plain MAE alone.

#### DistDF (ICLR 2026)

Key signal:
long-horizon forecasting benefits from distribution-level direct objectives rather
than only local pointwise penalties.

V740 implication:
h=14 and h=30 cells need stronger training signals than stepwise point loss.

#### QDF (ICLR 2026)

Key signal:
quadratic-form weighting can emphasize harder structure without simply exploding
loss values.

V740 implication:
V740 should distinguish “easy stable cells” from “hard structured cells” inside
the loss, not only inside the architecture.

#### Selective Learning for Deep Time Series Forecasting (NeurIPS 2025)

Key signal:
not every temporal window should influence optimization equally.

V740 implication:
window weighting and curriculum should be part of the method, especially under
noise, regime shifts, and label instability.

#### JAPAN (ICLR 2026)

Key signal:
reliable prediction sets in structured forecasting benefit from density-aware,
not only residual-based, calibration.

V740 implication:
uncertainty and conformal calibration should become native outputs of the model,
not a post-hoc extra.

### 3.4 Emerging system-level signals

#### TimeDiT (2025)

Key signal:
a general-purpose diffusion-style time-series foundation model can unify several
downstream tasks, but at meaningful compute cost.

V740 implication:
diffusion-style expressivity is interesting, but not for alpha. It is a beta-or-
gamma-level research signal, not the first implementation target.

#### Apollo-Forecast (AAAI 2025)

Key signal:
there is still room to improve the speed-accuracy tradeoff of language-model-like
forecasting pipelines by focusing on aliasing and inference efficiency.

V740 implication:
if V740 ever revisits sequence-model scaling, efficiency must remain explicit.

#### FLAIRR-TS (Findings of EMNLP 2025)

Key signal:
retrieval and iterative refinement can strengthen forecasting systems when raw
sequence modeling alone is not enough.

V740 implication:
retrieval should be treated as an optional external memory or distillation aid,
not as the first-line architecture.

## 4. V740 Methodological Thesis

The most defensible thesis, given both the benchmark and the latest literature,
is the following:

> V740 should be a single, condition-aware forecasting model that synthesizes the
> benchmark champions’ mechanisms into one efficient architecture, and uses
> stronger target-aware objectives and calibrated uncertainty to close the final
> gap to the specialist frontier.

This is a stronger thesis than “build a better selector,” and it is also more
aligned with the final product goal.

## 5. Proposed V740 Architecture

V740 should be a **decomposition-first conditional multi-head forecaster**.

### 5.1 High-level structure

1. input adapters
2. decomposition-first shared trunk
3. multi-resolution temporal branch
4. lightweight local-context branch
5. compact temporal memory branch
6. value-space auxiliary branch
7. invariant/variant fusion block
8. condition-aware modulation
9. target-specific prediction heads
10. optional uncertainty/calibration head

### 5.2 Input adapters

Inputs should include:

- target history
- numeric covariates from the active ablation
- compact static summary of entity context
- condition tokens for task, target, horizon, and ablation

The input stage should remain intentionally small. Parameter budget should be
spent on useful internal structure, not on oversized embeddings.

### 5.2b Extended-horizon logic

The next long-horizon expansion cannot be "just add 60 and 90 everywhere."
That would confuse token support with true long-horizon competence.

As of 2026-03-26, local-only V740-alpha evidence is already mixed:

- `funding_raised_usd / core_only / task2`: `h=60` looks slightly healthier
  than `h=30` on the tested slice,
- `investors_count / core_only / task2`: `h=60` looks worse than `h=30`,
- `funding_raised_usd / core_edgar / task2`: `h=60` is also slightly healthier
  than `h=30` on the tested slice,
- `investors_count / core_edgar / task2`: `h=60` is still slightly worse than
  `h=30`,
- the first viable `h=90` path now appears only on
  `funding_raised_usd / core_edgar / input_size=120`,
- `h=180 / core_edgar / funding_raised_usd / input_size=180` is still
  fallback-only on the tested narrow slice.

There is one more important caveat: the first `0.0` source-density readings on
long-horizon funding/count slices were caused by a logging bug in
`V740AlphaPrototypeWrapper.fit()`, not by genuine source absence. After fixing
that bug and rerunning representative funding slices, `core_edgar` now records
`edgar_source_density = 1.0`, and `full` records both
`edgar_source_density = 1.0` and `text_source_density = 1.0`.

That correction matters, but it still does not justify overclaiming. The
refreshed artifacts show that source-covered regimes are present on the tested
long-h slices; they do **not** yet prove that source-native sparse event-memory
is already the dominant reason those slices improve. The strongest supported
interpretation remains a joint one about horizon conditioning, context length,
and richer feature regimes.

So the correct V740 design principle is:

> longer entrepreneurial-finance horizons must be logic-driven and jointly
> engineered with context length, usable windows, and representative sampling.

The candidate future research horizons worth auditing are:

- `45` days: campaign follow-up / traction window
- `60` days: medium fundraising evolution
- `90` days: quarter-scale financing development
- `180` days: half-year financing progression
- `365` days: annual financing state change

These are design targets, not active benchmark claims. They should remain
local-only until the prototype demonstrates non-degenerate behavior and stable
benefit across more than one target family.

Recent long-horizon work on maximum effective window (MEW) reinforces why this
discipline matters. Longer lookback is only useful when the model can actually
convert it into a larger usable window instead of more redundancy and noise.
That matches the current V740 evidence well: `h=90` only becomes viable after
context scaling on a source-richer funding slice, while `h=180` still fails
under a narrow effective-window budget.

The engineering implication is straightforward:

> V740 should not use a tiny hard-coded horizon lookup table. It should use a
> mixed continuous-plus-bucket horizon encoding that can express both the
> benchmark horizons and the longer entrepreneurial-finance research horizons
> under one single-model path.

### 5.3 Decomposition-first trunk

The shared trunk should start with a decomposition-first block rather than a
plain Transformer encoder.

Desired behavior:

- `NBEATS` / `NBEATSx` style basis refinement for short and medium horizons
- `OLinear` / decomposition-style cheap coverage for stable cells
- `Chronos`-compatible scale robustness through normalization and residual
  projection rather than through giant pretraining alone

A practical implementation target is:

- trend branch
- oscillatory/seasonal branch
- residual refinement branch
- cheap implicit decomposition without extra inference passes

### 5.4 Multi-resolution branch

This branch should absorb the useful part of `NHITS`:

- pooled temporal views
- shared interpolation back to the requested horizon
- explicit support for weekly and biweekly structure

This is likely one of the keys for h=7 and h=14 robustness.

### 5.5 Lightweight local-context branch

This branch should absorb the useful part of `PatchTST`, `SAMformer`, `CASA`,
and `LiPFormer` without paying their full cost.

Design target:

- small patch extraction or channel-wise context slices
- efficient score mixing or channel attention
- strong conditioning by EDGAR-aware covariates
- no large global self-attention stack

This branch exists because EDGAR-sensitive and binary-sensitive cells often need
local contextual structure that the decomposition branch alone will miss.

### 5.6 Compact temporal memory branch

The design should explicitly include a **small gated memory path** that absorbs
the useful part of the `GRU` behavior:

- compact state tracking
- low-variance temporal regularization
- short- to mid-range memory that is harder to replace with only decomposition
  or patch mixing

This should remain lightweight. The goal is not to turn V740 into a recurrent
model, but to give it a controlled recurrent inductive bias where needed.

### 5.7 Value-space auxiliary branch

To absorb the useful part of the `Chronos` advantage, V740 should include a
small **value-space auxiliary branch**:

- quantized or bucketized target-value representation
- scale-insensitive latent projection
- auxiliary supervision that stabilizes heavy-tailed and long-horizon behavior

This branch does not need to be large. Its purpose is to give the model a
second view of the target beyond pure time-domain regression.

### 5.8 Invariant / variant fusion

This block is where `DIAN` and `TimeEmb` become first-class citizens.

We should explicitly split:

- invariant temporal dynamics shared across many entities
- variant context behavior driven by entity/static/exogenous conditions

A simple but serious implementation is:

- invariant encoder
- variant encoder
- learned fusion gate conditioned on condition tokens and static summary

### 5.9 Condition-aware modulation

Task, target, horizon, and ablation must modulate the shared trunk.

This should be implemented through lightweight adapters or FiLM-style affine
transforms, not by spawning separate submodels.

Required modulation axes:

- task (`task1`, `task2`, `task3`)
- target (`funding_raised_usd`, `investors_count`, `is_funded`)
- horizon (official `1`, `7`, `14`, `30` plus research `45`, `60`, `90`, `180`, `365`)
- ablation (`co`, `s2`, `ct`, `ce`, `e2`, `fu`)

### 5.10 Target-specific heads

V740 must have explicit target-specific heads.

#### Continuous head

For `funding_raised_usd`:

- heavy-tail robust
- stable under long horizon drift
- compatible with transformed-label alignment

#### Count-aware head

For `investors_count`:

- nonlinear jump-aware
- numerically stable on nonnegative data
- able to mimic KAN-like local response when needed

#### Binary calibrated head

For `is_funded`:

- Bernoulli/logit output
- calibration-aware
- not a reuse of the regression head
- should mimic the useful part of `DeepNPTS` behavior rather than generic MSE
- should support an event-style or prototype-style binary auxiliary path rather
  than relying only on dense regression-style hidden states

#### Optional uncertainty head

For all targets:

- quantile or interval outputs
- compatible with conformal or JAPAN-style post-calibration
- especially useful for high-stakes downstream deployment and future paper
  generalization tests

## 6. Proposed V740 Objective Stack

Architecture alone is not enough.

### 6.1 Base objectives

- continuous: MAE or Huber
- count: MAE plus nonnegative/count-aware regularization
- binary: BCE plus Brier-style calibration regularization

### 6.2 Structure-aware auxiliary objectives

- transformed-label alignment (`Time-o1` style) for heavy-tailed funding
- direct multistep distributional loss (`DistDF` style) for h=14 and h=30
- hard-structure weighting (`QDF` style) to emphasize difficult windows
- selective-learning or curriculum weighting for noisy/nonstationary windows
- seed-consistency and representation-consistency penalties on redundant cells

### 6.3 Distillation policy

Training-time distillation is allowed; inference-time ensembling is not.

Teacher pool priority:

- `NBEATS`
- `NHITS`
- `Chronos` / `ChronosBolt`
- `KAN`
- `DeepNPTS`
- `GRU`
- `PatchTST`

Policy:

- one primary teacher for clearly dominated cells
- optional secondary teacher for ambiguous cells
- no teacher blending at inference

## 7. Multisource Alignment Policy

V740 should no longer treat `core`, `edgar`, and `text` as if they were all the
same kind of feature source.

The correct representation policy is specified in:

- `docs/references/V740_MULTISOURCE_ALIGNMENT_SPEC.md`

The short version is:

- `core` is a regular daily state stream
- `edgar` is a sparse event stream
- `text` is a sparse semantic event stream
- `edgar` and `text` should each use **dual-clock hybrid tokenization**:
  - recent raw event tokens
  - recency-bucket summary tokens
- all sources must obey strict availability-aware as-of alignment

## 8. Feature-Regime Policy

### 7.1 Text

Text should not be treated as a mandatory always-on early branch.
Current evidence says the existing PCA embeddings hurt too often.

Recommendation:

- keep text optional in alpha,
- audit retrieval-style and event-style use first,
- revisit stronger text integration only after a better representation path exists.

### 8.2 EDGAR

EDGAR is mixed, not useless.
It deserves a dedicated lightweight context pathway rather than naive early
feature concatenation.

### 8.3 Seed2 and task redundancy

High redundancy should be used as a regularizer and validation aid, not as an
excuse to collapse condition awareness.

## 9. Benchmark Addition Priorities

The benchmark should still grow, but additions must serve the V740 goal.

### 8.1 First-wave additions

1. `SAMformer`
   - already integrated locally
   - next step is a real benchmark-smoke path that does not blow up the local
     memory path

2. `LightGTS`
   - highest-priority missing efficiency-first model

3. `OLinear`
   - strong low-cost baseline
   - blocked by matrix artifact generation, so preprocessing support is needed

4. `LiPFormer`
   - lightweight patch-based competitor

### 8.2 Second-wave additions

5. `ElasTST`
6. `UniTS`
7. `CASA`
8. `DIAN`-style or invariant/variant reference baselines
9. `TimeDiT`-style heavier generalist references only after the efficient path
   is stable

### 8.3 What is not the priority

- another heavyweight foundation model by default
- online multi-model ensembles
- a new LLM controller before the single-model path is exhausted

## 10. Audit and Fairness Requirements

V740 must remain stricter than the older AutoFit lines.

Hard requirements:

1. no test-set information in training or selection
2. identical train/val/test protocol as current benchmark
3. no hidden fallback path that silently changes the effective model
4. explicit handling of NaN/Inf and constant-prediction sentinels
5. reproducible condition token semantics
6. no evaluation special-casing for easy cells
7. benchmark additions validated under the same fairness rules as existing models

## 11. Implementation Roadmap

### 10.1 V740-alpha

Goal:
validate the single-model synthesis direction.

Required modules:

- decomposition-first trunk
- multi-resolution block
- lightweight local-context mixer
- condition tokens
- target-specific heads

### 10.2 V740-beta

Goal:
close the gap to the frontier with stronger objectives and representation splits.

Add:

- invariant/variant fusion
- distillation from champion teachers
- Time-o1 / DistDF / QDF-style losses
- stronger calibration head

### 10.3 V740-gamma

Goal:
turn V740 into a paper-grade single-model champion candidate.

Add:

- robust uncertainty layer
- optional retrieval memory or auxiliary retrieval distillation
- generalization tests on mainstream public datasets from recent TS literature

## 12. Acceptance Criteria

V740 should be considered directionally successful only if it satisfies all of
the following:

1. beats V739 on its clean comparable slice in a statistically credible way
2. materially improves on `is_funded`
3. does not collapse on `funding_raised_usd`
4. stays close to single-model efficiency rather than selector-style cost
5. remains fully fair under the current benchmark protocol
6. generalizes beyond Block 3 to mainstream public TS datasets without becoming
   obviously noncompetitive

## 13. Final Recommendation

The project should stop treating V740 as a better routing table.

The strongest current route is:

- benchmark the missing efficiency-first specialists,
- use them to refine the champion mechanism atlas,
- and implement V740 as a single decomposition-first conditional forecaster with
  stronger objectives, explicit invariant/variant structure, and target-aware
  calibration.

That is the most credible path to an eventual “all 160 cells, fair and clean,
no-drama champion” system.
