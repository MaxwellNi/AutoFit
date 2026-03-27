# V740 Champion Mechanism Digest

> Date: 2026-03-27
> Status: public research synthesis for V740 design
> Scope: current repeated champion mechanisms on the clean 160-cell Block 3 surface
> Depends on: `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`, `docs/references/V740_DESIGN_SPECIFICATION.md`, `docs/references/V740_ALPHA_ENGINEERING_SPEC.md`, `docs/references/SOTA_MODELS_REFERENCE_CATALOG.md`, `docs/references/RESEARCH_PIPELINE_IMPLEMENTATION_EN.md`

## 0. Why this document exists

The current benchmark is now strong enough that V740 should no longer be driven
by a vague idea of “absorbing many good models.” It should be driven by a much
cleaner question:

> Which mechanisms repeatedly produce wins on the current clean surface, and how
> can those mechanisms be compressed into one lightweight single-model system?

This document is therefore not a leaderboard recap. It is a design digest.
Its goal is to separate:

1. **primary repeated champion-mechanism sources**,
2. **close variants / secondary signals**,
3. **useful public-paper comparators that are not yet central mechanism anchors**.

That separation is essential if V740 is to remain elegant, efficient, and
scientifically defensible.

## 1. Primary vs secondary mechanism sources

### 1.1 Primary repeated champion-mechanism sources

The current clean 160-cell surface points to the following models as the main
sources of mechanisms that V740 must absorb directly:

- `NBEATS`
- `Chronos`
- `NHITS`
- `KAN`
- `DeepNPTS`
- `GRU`
- `PatchTST`

These models are not all equally strong in mean rank, but they repeatedly win
meaningful slices of the benchmark and therefore define the real design target.

### 1.2 Close variants and secondary anchors

Some models are important, but should not currently be treated as separate
first-order mechanism sources:

- `NBEATSx`
  - best understood as an exogenous-support extension of the `NBEATS` family
- `ChronosBolt`
  - best understood as a stronger long-horizon rank signal and a close
    `Chronos`-family variant

These matter for V740, but mainly by refining how we absorb the parent family.

### 1.3 Candidate-pool members and literature references

Some models remain useful in V739's candidate pool or as public-paper
comparators, but current Block 3 evidence does not yet justify making them core
V740 mechanism anchors:

- `TimesNet`

This does not mean `TimesNet` is weak. It means that on the current surface we
have stronger evidence for the mechanisms above than for treating
`TimesNet`-style temporal-2D modeling as a primary design pillar.

## 2. The deepest N-BEATS lesson

`NBEATS` matters for V740 for a deeper reason than “it wins 65/160 cells.”

Its most important lesson is that forecasting strength can come from a very
clean decomposition of the problem into:

1. a shared coefficient generator,
2. a structured basis space,
3. iterative residual refinement,
4. and direct horizon-wise forecasting.

This is the backbone lesson for V740.

### 2.1 What V740 should copy from N-BEATS

- decomposition-first forecasting
- additive/hierarchical forecast construction
- residual cleanup and refinement across blocks
- simple and cheap core trunk
- direct forecasting rather than hidden runtime selection logic

### 2.2 What V740 should not copy blindly from N-BEATS

- competition-style ensemble packaging
- the assumption that one basis family alone is sufficient for every regime
- the assumption that binary/event behavior can be handled by the same head as
  continuous/count forecasting

The most important takeaway is therefore:

> V740 should be built around a N-BEATS-like skeleton, not around a giant
> controller, a runtime tournament, or a kitchen-sink multimodal stack.

## 3. Mechanism-by-mechanism design digest

### 3.1 `NBEATS`

**What it contributes:**
- basis expansion
- direct multi-horizon forecasting
- double residual refinement
- robust short/medium-horizon behavior on continuous/count targets

**Why it matters on Block 3:**
- dominant on short-horizon funding and many count cells
- remains the hardest single champion family to beat

**V740 implication:**
- the shared trunk should be decomposition-first and residual-refinement-first
- this is the model's default reasoning path, not an optional side branch

### 3.2 `NHITS`

**What it contributes:**
- multi-resolution processing
- pooled views at several scales
- interpolation-based reconstruction/forecast stabilization

**Why it matters on Block 3:**
- strongest around weekly-scale structure and some EDGAR-sensitive binary cells
- captures h=7 structure better than a pure basis model alone

**V740 implication:**
- add one explicit multi-resolution branch
- keep it lightweight and complementary to the trunk rather than making the
  whole model hierarchical by default

### 3.3 `Chronos` / `ChronosBolt`

**What they contribute:**
- scale-robust value-space behavior
- stronger longer-horizon priors
- better uncertainty anchoring on hard multi-step funding regimes

**Why they matter on Block 3:**
- longer-horizon funding cells increasingly stop being about local pattern
  extrapolation and start becoming about staying stable under uncertainty
- `ChronosBolt` sharpens the longer-horizon signal even when it is not the main
  win-count source

**V740 implication:**
- do not copy the full TSFM stack
- do add a small value-space / bucketized auxiliary branch
- longer-horizon robustness should come from the model's internal value
  representation, not from making V740 a large foundation model

### 3.4 `GRU`

**What it contributes:**
- compact hidden state
- low-variance temporal regularization
- a cheap recurrent sufficient-statistic path

**Why it matters on Block 3:**
- strongest in the h=14 transition zone on simpler source regimes
- shows that some cells benefit from compact memory more than from broad
  pretraining or large context mixers

**V740 implication:**
- include a small compact-memory path
- do not convert the whole model into a recurrent backbone

### 3.5 `DeepNPTS`

**What it contributes:**
- binary/event-style calibration
- distribution-free behavior
- context-conditioned weighting over historical outcomes

**Why it matters on Block 3:**
- pure temporal binary cells still prefer this kind of calibration-aware logic
- it is a narrow but real specialist signal

**V740 implication:**
- binary/event forecasting must have an explicit head and calibration path
- it must not be treated as a side effect of the continuous/count loss

### 3.6 `PatchTST`

**What it contributes:**
- patch-level local context modeling
- strong generalist behavior by mean rank
- EDGAR-sensitive context handling on some binary cells

**Why it matters on Block 3:**
- few outright wins, but excellent rank stability
- a strong sanity anchor for local context modeling without overcommitting to a
  giant architecture

**V740 implication:**
- keep a small local-context mixer
- use it where EDGAR-sensitive transitions matter
- do not let it replace the decomposition-first trunk

### 3.7 `KAN`

**What it contributes:**
- strong local nonlinear response
- flexible short-horizon approximation for count-like targets

**Why it matters on Block 3:**
- uniquely strong on `investors_count h=1`
- evidence that count forecasting needs a more locally adaptive micro-response
  than smooth polynomial or interpolation bias alone

**V740 implication:**
- add a local nonlinear count-sensitive block or micro-head
- keep it narrow and target-aware rather than making the whole trunk spline-like

### 3.8 `NBEATSx`

**What it contributes:**
- exogenous-support extension of the decomposition-first family
- direct conditioning of basis coefficients on richer covariates

**Why it matters on Block 3:**
- strongest signal is not “new family,” but “how to let exogenous context alter
  basis behavior cleanly”

**V740 implication:**
- V740 should modulate decomposition/basis behavior with covariate-aware gates
  rather than treating exogenous features as a completely separate forecasting
  stack

### 3.9 `TimesNet`

**What it contributes:**
- temporal 2D / periodic-structure modeling
- a strong public-paper comparator family

**Why it matters on Block 3 right now:**
- still useful in candidate pools and literature comparisons
- but current clean Block 3 evidence does not yet justify promoting it to a
  primary mechanism anchor

**V740 implication:**
- keep it in the research horizon
- do not let it displace decomposition, multi-resolution, compact memory,
  binary calibration, or local context as the main design priorities

## 4. What this means for the V740 skeleton

The cleanest current summary is:

> V740 should be a N-BEATS-like decomposition-first skeleton plus only the
> minimum branches required by the benchmark's other repeated champion
> mechanisms.

That gives the following design hierarchy.

### 4.1 Non-negotiable core

- shared coefficient generator / decomposition trunk
- residual refinement
- direct horizon-wise forecasting

### 4.2 First-order attachments

- `NHITS`-style multi-resolution branch
- `Chronos`/`ChronosBolt`-style value-space branch
- `GRU`-style compact memory branch
- `DeepNPTS`-style binary/event calibration head
- `PatchTST`-style local context mixer
- `KAN`-style local nonlinear count block

### 4.3 Secondary refinements

- exogenous-aware modulation (`NBEATSx`-style lesson)
- selective weighting / calibration / hard-cell objectives
- source-aware EDGAR/text event memory

### 4.4 What to avoid

- runtime model tournaments
- hidden ensemble logic
- giant all-purpose Transformer trunks
- early mandatory text fusion
- treating every literature model as a first-order design requirement

## 5. Public-paper generalization priorities

The current repo-wide research synthesis already points to the most recurring
external dataset families for eventual V740 generalization testing:

### 5.1 Canonical long-horizon multivariate pack

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `Electricity` / `ECL`
- `Traffic`
- `Weather`
- `Exchange`
- `ILI`

### 5.2 Graph / traffic pack

- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`

### 5.3 Robustness / broad-transfer pack

- `M4`
- `M5`
- selected Monash datasets

For public-paper credibility, the recurring comparator families remain:

- linear/decomposition: `DLinear`, `RLinear`, `OLinear`
- transformer/mixer: `PatchTST`, `iTransformer`, `TimesNet`, `TimeMixer`,
  `FEDformer`, `Crossformer`, `Informer`, `Autoformer`
- deep classical/direct: `NBEATS`, `NHITS`, `TFT`, `TiDE`, `TCN`
- foundation: `Chronos`, `Moirai`, `TimesFM`, `TimerXL`, `TimeMoE`, `Sundial`

## 6. Immediate design consequence

The next V740 iteration should not start by adding more branches at random.
It should start by answering one sharper engineering question:

> Does the current V740-alpha backbone behave like a decomposition-first direct
> forecaster that only calls on the specialized branches when the regime truly
> requires them?

If the answer is “not yet,” then the next gain is more likely to come from a
cleaner skeleton and stricter branch discipline than from adding yet another
paper mechanism.
