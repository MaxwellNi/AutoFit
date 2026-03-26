# Research Pipeline Implementation

> Last updated: 2026-03-26
> Scope: current clean Block 3 line only
> Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
> Canonical valid AutoFit baseline: `AutoFitV739`
> Canonical data pointer: `docs/audits/FULL_SCALE_POINTER.yaml`

This document replaces older pre-V739 pipeline descriptions. It summarizes the
current research and implementation workflow that should guide both the
benchmark-expansion program and the V740 single-model research line.

## 1. Current objective

The project now has two coupled priorities:

1. keep the clean benchmark authoritative, current, and fair,
2. push V740 toward a genuinely lightweight single-model champion.

Those goals must be pursued together. A stronger V740 is not useful if it is
validated against an incomplete or stale benchmark, and a larger benchmark is
not enough if the next AutoFit generation still behaves like a slow
multi-candidate controller.

## 2. Current benchmark line

Only the current clean benchmark line is operational:

- benchmark root: `runs/benchmarks/block3_phase9_fair/`
- valid AutoFit baseline: `AutoFitV739`
- freeze pointer: `docs/audits/FULL_SCALE_POINTER.yaml`

Historical V72/V73 truth-pack material and V734-V738 AutoFit variants remain
useful for lessons learned, but they must not be used as active baselines or
paper claims.

## 3. Why AutoFitV739 is operationally slow

AutoFitV739 is slow for a structural reason, not because one monolithic model
is exceptionally large.

The current implementation in
`src/narrative/block3/models/nf_adaptive_champion.py` is a validation-based
selection controller. For each benchmark condition it:

1. receives `train_raw` and `val_raw`,
2. iterates over an 8-model candidate pool,
3. trains each candidate on the train split,
4. predicts on the validation split,
5. scores each candidate,
6. keeps the best validated model for test-time prediction.

The active candidate pool is:

- `NHITS`
- `PatchTST`
- `NBEATS`
- `NBEATSx`
- `ChronosBolt`
- `KAN`
- `Chronos`
- `TimesNet`

Operationally, V739 therefore pays multiple training costs per cell rather
than the cost of one model. Its wall-clock is driven by:

- candidate multiplicity,
- temporal validation overhead,
- per-candidate timeout handling,
- and shared-cluster queue pressure from other long-running jobs.

This is the main reason V739 remains acceptable as a clean baseline but
unacceptable as the final end-state. It is methodologically valid, but it is
not the lightweight system we ultimately want to deploy or publish as the
culminating model.

## 4. Why V740 must stay single-model and lightweight

V740 is being developed specifically to break away from the V739 cost profile.
The design target is:

- one training graph,
- one inference path,
- condition-aware internal specialization,
- no runtime ensemble,
- no runtime candidate tournament,
- fairness identical to the current benchmark.

In practice this means V740 should absorb champion mechanisms into one shared
conditional architecture rather than reproduce V739's external selection loop.
The current prototype direction is therefore:

- decomposition-first trunk,
- multi-resolution path,
- compact temporal memory,
- value-space auxiliary branch,
- source-aware EDGAR/text event memory,
- target-specific heads,
- condition tokens for task/target/horizon/ablation.

The prototype is still pre-benchmark, but the design principle is already
clear: **V740 should behave like one strong model, not like a delayed ensemble
disguised as AutoFit.**

## 5. Long-horizon policy

### 5.1 What is official today

The official comparable benchmark surface still uses:

\[
h \in \{1, 7, 14, 30\}.
\]

No public benchmark table should currently claim official results for
`h > 30`.

### 5.2 What is prototyped today

As of 2026-03-26, the local-only V740 prototype no longer relies on a tiny
fixed horizon lookup table. The current horizon-conditioning path now combines:

- continuous horizon features,
- coarse horizon buckets,
- context-to-horizon ratio features.

So the local V740 research tools can now exercise arbitrary positive integer
horizons. The horizons already probed in real local artifacts include:

- `45`
- `60`
- `90`
- `180`

Those paths remain outside the clean benchmark protocol.

### 5.3 What the first real evidence says

The current local-only evidence is mixed rather than uniformly positive:

- `task2_forecast / core_only / funding_raised_usd`
  - `h=30`: `MAE = 361820.06`, constant prediction
  - `h=60`: `MAE = 360123.84`, non-constant
- `task2_forecast / core_only / investors_count`
  - `h=30`: `MAE = 48.64`, non-constant
  - `h=60`: `MAE = 50.08`, constant prediction
- `task2_forecast / core_edgar / funding_raised_usd`
  - `h=30`: `MAE = 179797.74`, non-constant
  - `h=60`: `MAE = 178084.08`, non-constant
- `task2_forecast / core_edgar / investors_count`
  - `h=30`: `MAE = 56.51`, non-constant
  - `h=60`: `MAE = 57.06`, non-constant
- `task2_forecast / full / funding_raised_usd`
  - `h=30`: `MAE = 179797.75`, non-constant
  - `h=60`: `MAE = 178084.08`, non-constant
- `task2_forecast / full / investors_count`
  - `h=30`: `MAE = 56.47`, non-constant
  - `h=60`: `MAE = 57.07`, non-constant
- `task2_forecast / core_only / funding_raised_usd`
  - `h=90`: fallback-only on the tested local slice
- `task2_forecast / core_only / investors_count`
  - `h=90`: fallback-only on the tested local slice
- `task2_forecast / core_edgar / funding_raised_usd`
  - `h=90` with `input_size=120`: non-constant on the tested slice
- `task2_forecast / core_edgar / investors_count`
  - `h=90` with `input_size=120`: still constant on the tested slice
- `task2_forecast / core_only / funding_raised_usd`
  - `h=90` with `input_size=120`: still fallback-only on the tested slice
- `task2_forecast / core_edgar / funding_raised_usd`
  - `h=45` with `input_size=90`: non-constant on the tested slice
- `task2_forecast / core_edgar / funding_raised_usd`
  - `h=180` with `input_size=180`: fallback-only on the tested slice

The correct current conclusion is therefore:

> Longer horizon is now a real exercised V740 path, but it is not yet a free
> gain. It may help on some funding slices, it currently hurts or collapses on
> some count slices, EDGAR may stabilize some `h=60` behavior without making it
> universally better, the current text pathway is not yet showing clear
> incremental long-horizon benefit on the tested local slices, and the first
> viable `h=90` behavior currently appears to
> require both longer context and a richer source regime rather than a larger
> horizon token alone.

One more caveat is important for interpretation: the first `0.0`
`edgar_source_density` / `text_source_density` readings on non-binary long-h
artifacts were caused by a logging bug. After fixing that path and rerunning
representative funding slices, `core_edgar` now records
`edgar_source_density = 1.0`, and `full` records both
`edgar_source_density = 1.0` and `text_source_density = 1.0`.

That correction removes a false negative, but it still does not justify
overclaiming. The safer interpretation is that the current long-h artifacts now
support a joint story about:

- better horizon/context conditioning,
- genuinely source-covered `core_edgar` and `full` local slices,
- but not yet a fully isolated proof that source-native sparse event-memory is
  the main driver of the gain.

### 5.4 How longer entrepreneurial-finance horizons should be designed

The next long-horizon expansion should not be "just add 60 and 90 everywhere."
It should be guided by startup-financing time logic. The candidate future
research horizons worth auditing are:

- `45` days: campaign follow-up / near-term traction window
- `60` days: medium follow-up window
- `90` days: quarter-scale fundraising evolution
- `180` days: half-year financing progression
- `365` days: annual financing state change

These are **research candidates**, not active benchmark settings. They should
only be promoted after local-only evidence shows that:

1. the prototype has enough effective windows,
2. performance remains non-trivial across targets,
3. the longer horizon adds real scientific value instead of just noise.

The current best entrepreneurial-finance horizon ladder is therefore:

- `45`: first research extension between `30` and `60`
- `60`: first promising medium-term extension, especially for funding slices
- `90`: next target only after context and usable-window scaling
- `180`: half-year audit horizon, currently still fallback-only on tested narrow slices
- `365`: annual horizon, to be attempted only after `180` becomes non-degenerate

## 6. Public generalization benchmark program

To make V740 credible beyond our proprietary startup-financing panel, we need a
separate public benchmark program aligned with the datasets most frequently used
in 2024-2026 top-venue forecasting papers.

The recurring public benchmark families across recent papers and repos include:

### 6.1 Long-term multivariate forecasting

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `Electricity` / `ECL`
- `Traffic`
- `Weather`
- `Exchange`
- `ILI` / `Illness`

These remain the most common direct-comparison datasets in recent forecasting
papers, especially for long-horizon multivariate forecasting.

The recurring public horizon protocol on these datasets is also highly stable:

- `96 / 192 / 336 / 720` for ETT, Weather, Electricity, Traffic, and similar
  long-term multivariate benchmarks,
- `12 / 24 / 48 / 96` for common PEMS traffic subsets.

This matters for V740 because our eventual public generalization pack should
not improvise horizon settings. It should inherit the horizon ladders that the
recent literature repeatedly uses, then report our entrepreneurial-finance
research horizons separately rather than conflating the two.

### 6.2 Spatiotemporal / traffic-heavy forecasting

- `PEMS03`
- `PEMS04`
- `PEMS07`
- `PEMS08`

These are useful for testing whether a mechanism that works on our panel data
also remains competitive on structured multivariate temporal dynamics with
clear spatial correlation.

### 6.3 Exogenous / covariate-sensitive tasks

- `Solar`
- energy / load datasets with covariates
- electricity-price or market datasets used in exogenous forecasting papers

These are especially important for the EDGAR/text side of V740, because they
stress covariate usefulness and availability rather than pure endogenous trend
fitting.

### 6.4 Broader classical forecasting sanity checks

- `M4`
- `M5`
- selected Monash Forecasting Repository datasets

These are not perfect analogues of our panel problem, but they are still useful
for checking whether V740 stays robust outside the specific Block 3 setting.

## 6.5 Recurring comparator families in recent papers

Across 2024-2026 forecasting papers, the recurrent comparison packs are also
remarkably stable. The most common strong baselines still cluster into the
following groups:

- linear / decomposition:
  - `DLinear`
  - `RLinear`
  - `OLinear`
- transformer / mixer:
  - `PatchTST`
  - `iTransformer`
  - `TimesNet`
  - `TimeMixer`
  - `FEDformer`
  - `Crossformer`
  - `Informer` / `Autoformer`
- deep classical / direct forecasters:
  - `NBEATS`
  - `NHITS`
  - `TFT`
  - `TiDE`
  - `TCN`
- foundation / large pretrained references:
  - `Chronos`
  - `Moirai`
  - `TimesFM`
  - `TimerXL`
  - `TimeMoE`
  - `Sundial`

This matters because our eventual public generalization benchmark should not be
built around arbitrary model names. It should deliberately include the
comparison families that top-venue papers repeatedly treat as serious
competitive references.

## 6.6 Most relevant 2024-2026 mechanism pack for V740

The current literature audit points to a fairly compact set of mechanism
families that matter most for V740's actual goal. The important point is not
to imitate every paper. It is to identify which ideas are both repeatedly
competitive in top-venue comparisons and structurally relevant to our panel,
multisource, multi-horizon setting.

### Single-model multi-task / multi-horizon conditioning

- `UniTS`
- `ElasTST`
- `TimerXL`

These matter because V740 must eventually cover both the current 160 official
cells and a longer entrepreneurial-finance horizon ladder without turning into
one separate model per target or per horizon.

### Lightweight long-horizon and local-context modeling

- `LightGTS`
- `LiPFormer`
- `SAMformer`
- `CASA`
- `PIH` / maximum-effective-window modules

These are the most relevant efficiency-first references when the goal is to
challenge strong specialists without inheriting the cost profile of a runtime
selector or a giant foundation model.

Two especially useful current signals are:

- MEW-oriented work such as PIH, which argues that longer lookback only helps
  when the model can suppress redundancy and enlarge its effective window,
- decomposition-efficiency results showing that decomposition-first designs can
  improve both accuracy and cost rather than forcing a trade-off.

### Distribution alignment, calibration, and hard-cell optimization

- `DistDF`
- `QDF`
- `JAPAN`
- `Time-o1`
- `Selective Learning for Deep Time Series Forecasting`

### Decomposition-first efficiency and compression

- `Decomposition Delivers Both in Long-term Time Series Forecasting`

These matter because Block 3 is not only an architecture problem. It is also a
heavy-tail, multistep, calibration-sensitive forecasting problem.

### Heterogeneous covariates and mixed static/dynamic structure

- `DIAN`
- `TimeEmb`

These are especially relevant for EDGAR/text design. They are strong signals
that V740 should keep pushing toward invariant/variant and static/dynamic
factorization instead of naive early concatenation.

## 7. Research-ingestion workflow for new papers

Every new 2024-2026 top-venue paper that looks relevant should be processed
through the same minimal research audit:

1. official source only:
   - OpenReview
   - conference proceedings
   - arXiv
   - official code repository
2. capture:
   - venue and year,
   - core mechanism,
   - datasets used,
   - comparison baselines,
   - code availability,
   - relevance to V740,
   - relevance to benchmark expansion,
   - likely hyperparameter transferability to our data
3. classify:
   - benchmark-addition priority,
   - V740-mechanism priority,
   - track-only / too-heavy / not-aligned

This avoids the two failure modes we already know well:

- adding papers by name without mechanism fit,
- copying hyperparameters or evaluation claims without verifying their domain fit.

## 8. Hyperparameter adaptation policy

No paper configuration should be copied blindly into Block 3.

Hyperparameters must be adapted to:

- target type:
  - heavy-tailed continuous
  - count-like
  - binary/event
- feature regime:
  - `core_only`
  - `core_edgar`
  - `core_text`
  - `full`
- horizon regime:
  - short
  - medium
  - extended long horizon
- compute budget:
  - must remain compatible with fair, resumable benchmarking

The correct standard is not "same settings as the paper," but "settings that
are fair and appropriate for our data regime."

## 9. HPC execution policy

The HPC admin has explicitly asked that heavy processes stop running on the
iris access/login servers.

Effective policy:

- access server:
  - lightweight inspection
  - aggregation
  - tiny smoke checks
  - orchestration
- meaningful prototype training or local compare:
  - resumable SLURM only
  - `--requeue`
  - signal-based checkpoint handling
  - logged outputs

This policy is now part of the research pipeline itself, not an operational
afterthought. A V740 experiment path that cannot be resumed safely is not ready
for serious use.

## 10. Immediate next implementation priorities

1. complete the first corrected local `V739 vs V740` head-to-head result,
2. expand the local mini-benchmark to a broader representative slice set,
3. continue long-horizon local-only experiments with proper window/context
   scaling before any protocol expansion,
4. keep expanding the benchmark with high-value missing models that are both
   scientifically relevant and feasible to run fairly.
