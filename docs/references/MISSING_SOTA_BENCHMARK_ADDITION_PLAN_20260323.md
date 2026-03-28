# Missing SOTA Benchmark Addition Plan (2026-03-23)

> Date: 2026-03-23
> Status: implementation planning (updated after first local SAMformer integration; queue state refreshed 2026-03-28)
> Scope: high-priority missing models for Block 3 benchmark expansion
>
> 2026-03-27 superseding note:
> for the latest source-verification, code/package availability, and proposed
> search-space map, read
> `docs/references/BLOCK3_MISSING_SOTA_SOURCE_MAP_20260327.md` first.
>
> 2026-03-24 note:
> V740-alpha now has a first source-aware EDGAR/text event-memory path.
> This slightly raises the value of efficient single-model additions that can be
> compared against, or distilled into, a multisource one-path design without
> requiring runtime ensembles.

## 1. Goal

This document converts the updated missing-model list into an implementation order that is realistic for the current Block 3 codebase.

The point is not to add every missing paper immediately. The point is to identify which additions are:

1. benchmark-relevant
2. compatible with the existing wrapper structure
3. worth the engineering effort relative to the V740 goal

## 2. Current Codebase Reality

The benchmark currently has three practical integration lanes:

1. **NeuralForecast lane**
   - easiest when the upstream model already exists in NF
   - current entry point: `src/narrative/block3/models/deep_models.py`

2. **TSLib lane**
   - easiest when the upstream code already follows Time-Series-Library conventions
   - current entry point: `src/narrative/block3/models/tslib_models.py`

3. **Custom lightweight wrapper lane**
   - required for models with standalone PyTorch repos
   - best option for compact models with minimal dependencies

## 3. High-Priority Candidates

## 3.1 SAMformer

### Value

- lightweight
- strong efficiency/accuracy story
- directly relevant to the V740 single-model target

### Code availability

- official PyTorch implementation exists
- repository structure is simple
- dependency surface is small

### Integration assessment

**Best path**: custom lightweight wrapper

Why:
- the official implementation is standalone PyTorch, not TSLib-native
- architecture is small enough to re-implement or vendor locally
- no exotic preprocessing artifacts are required

### Difficulty

**LOW to MEDIUM**

### Recommended action

1. ~~implement a `SAMformerWrapper`~~ ✅ local wrapper added on 2026-03-23
2. ~~narrow benchmark-clear probe now queued as `5294242` (`v740_samf_clr`)~~ ✅ completed on 2026-03-28
3. ~~train on panel windows extracted from the current entity histories~~ ✅ synthetic panel smoke passed
4. next step is no longer “first clear”, but “decide whether to promote SAMformer into a wider canonical benchmark lane”

### Current local status

- code path: `src/narrative/block3/models/samformer_model.py`
- registry hook: `src/narrative/block3/models/deep_models.py`
- verified so far:
  - insider env `py_compile` passed
  - registry import succeeded
  - synthetic entity-panel `fit/predict` smoke succeeded
  - freeze-backed micro-smoke on a tiny real subset ran end-to-end without crash
  - new generic local smoke runner `scripts/run_block3_local_model_smoke.py` passed on:
    - `task1_outcome / core_edgar / is_funded / h=14`
    - `MAE = 0.3863`
    - `fit_seconds = 11.6`
    - `constant_prediction = false`
- real-benchmark status:
  - first two direct harness smoke attempts were **killed with exit 137**
  - current evidence suggests the blocker is the local benchmark memory path, not an immediate model-logic failure
  - earlier tiny freeze-backed micro-smokes had shown near-constant behavior, but the new generic smoke confirms the wrapper can produce non-constant real outputs on a narrow audited slice
  - the model is still **not yet canonical-benchmark-cleared**
- not yet done:
  - canonical Block 3 smoke benchmark completion
  - fairness / constant-prediction audit on real benchmark outputs

## 3.5 Prophet

### Value

- cheap reviewer-recognizable sanity baseline
- especially useful for business / finance forecasting narratives
- low engineering burden once dependency is available

### Current local status

- wrapper now exists in `src/narrative/block3/models/statistical.py`
- registered through `STATISTICAL_MODELS`
- user-local vendor install path works:
  - `~/.cache/block3_optional_pydeps/py312`
- generic local smoke now passes on:
  - `task2_forecast / core_only / funding_raised_usd / h=30`
  - `MAE = 5300.86`
  - `fit_seconds = 5.24`
  - `constant_prediction = false`

### Recommended action

1. keep Prophet on the user-local vendor path instead of mutating the shared env
2. first narrow benchmark-clear probe (`5294243`) failed before model execution because `quick` preset does not support `h=30`
3. corrected resubmission `5294254` (`v740_prop_std`) on `bigmem` with `standard` preset has now completed successfully
4. next decision is no longer “can Prophet clear the harness?”, but whether its weak narrow-slice quality still justifies a canonical sanity-baseline lane

## 3.6 TabPFN-TS

### Value

- very high information-gain efficient baseline
- directly relevant to the claim that V740 should remain single-model and lightweight

### Current local status

- local generic wrappers already exist in `src/narrative/block3/models/traditional_ml.py`
- runtime compatibility is now fixable in-code by preloading the insider env
  `libstdc++.so.6.0.34`
- the real situation is now more specific:
  - HF auth works and official `Prior-Labs/tabpfn_2_6` checkpoints are
    locally cached
  - the shared `tabpfn 6.3.2` runtime was too old for 2.6 and failed with
    `KeyError: 'tabpfn_v2_6'`
  - a latest-source vendor runtime (`tabpfn 7.0.1`) is now installed in
    `~/.cache/block3_optional_pydeps/py312_tabpfn_latest`
- wrapper now supports:
  - local checkpoint paths via `model_path` / `BLOCK3_TABPFN_MODEL_PATH`
  - latest-source vendor override via `BLOCK3_TABPFN_VENDOR`

### Recommended action

1. use the official latest 2.6 line rather than the older 2.5 default
2. the first narrow benchmark-harness probe (`5294255`, `v740_tpfn26c`) has now completed successfully
3. the next two narrow probes have now also completed:
   - `5294259` `v740_tpfn26r_fu` (`TabPFNRegressor`, funding, `task2/core_edgar/h=30`) — clean harness path, weak quality
   - `5294260` `v740_tpfn26r_inv` (`TabPFNRegressor`, investors, `task2/core_edgar/h=14`) — red-flag `fairness_pass=false`
4. the current decision is therefore more specific:
   - `TabPFNClassifier 2.6` is the strongest first expansion candidate
   - `TabPFNRegressor 2.6` is not yet ready for blind canonical promotion

## 3.2 OLinear

### Value

- high-value missing low-cost SOTA
- directly relevant to decomposition-first / efficient forecasting
- likely useful both as a benchmark model and as a V740 design influence

### Code availability

- official repository exists
- architecture is close to Time-Series-Library style

### Integration assessment

**Best path**: TSLib-style vendor integration

### Blocker

OLinear is not plug-and-play in the current benchmark because it requires additional preprocessing artifacts:

1. orthogonal transformation matrices (`Q_mat`, `Q_out_mat`)
2. sometimes channel-correlation matrices

These artifacts are dataset-specific and are not currently generated by the Block 3 pipeline.

### Difficulty

**MEDIUM to HIGH**

### Recommended action

1. add a preprocessing design note for generating the required matrices on Block 3
2. only then attempt wrapper integration
3. do not rush this model before the matrix pipeline is specified

## 3.3 LightGTS

### Value

- highest-value missing model from the current V740 perspective
- explicitly targets lightweight general forecasting

### Code availability

- public implementation exists and has now been directly audited at
  official HEAD `36ad5bfb4c71ce11bf0372f8e4433c29e1ea5ff5`
- repo shape:
  - `finetune.py`
  - `zero_shot.py`
  - `datautils.py`
  - `src/models/LightGTS*.py`
  - released checkpoints under `checkpoints/`
- practical implication:
  - this is a **script-driven research repo**, not a clean pip package lane
  - the Block 3 path should be a custom vendor wrapper that imports the model
    class and bypasses the official dataset/CLI layer

### Integration assessment

**Best path**: custom wrapper or separate lightweight vendor import

### Current integration prep

- a first Block 3 vendor-path helper now exists in
  `src/narrative/block3/models/optional_runtime.py`
- this reduces future wrapper friction, but the actual `LightGTS` model wrapper
  still needs to be written

### Main uncertainty

- the model class itself is accessible, but the official repo hardcodes public
  benchmark dataset names and script-level CUDA settings
- heavy `requirements.txt` breadth makes “just install everything” a bad idea
  for the shared research environment
- should therefore be treated as a deliberate addition, not an opportunistic copy

### Difficulty

**MEDIUM**

### Recommended action

1. inspect upstream training interface
2. decide whether to vendor directly or re-implement core architecture
3. prioritize after SAMformer unless upstream code is unexpectedly clean

## 3.4 LiPFormer

### Value

- lightweight patch-based alternative
- useful competitor to PatchTST-style ideas

### Code availability

- public paper / code line exists, but not yet locally vendored

### Integration assessment

**Best path**: custom wrapper

### Difficulty

**MEDIUM**

### Recommended action

Treat as second-wave addition after we settle the SAMformer and OLinear paths.

## 4. Medium-Priority Candidates

## 4.1 ElasTST

### Why it matters

- directly aligned with the multi-horizon efficiency objective

### Why it is not first

- more useful as a V740 mechanism reference than as the first benchmark addition

## 4.2 DIAN

### Why it matters

- attractive for invariant/variant disentanglement in heterogeneous panels

### Why it is not first

- more architectural than benchmark-critical at this moment

## 5. Recommended Addition Order

Given current effort-to-value tradeoffs:

1. **SAMformer real smoke benchmark**
   - code now exists locally
   - best immediate information gain comes from moving it from ``implemented'' to ``benchmarked''

2. **OLinear**
   - highest-value efficient baseline after preprocessing is solved

3. **LightGTS**
   - major addition, but needs clean wrapper strategy

4. **LiPFormer**
   - useful patch baseline after the first two are stable

5. **ElasTST / DIAN**
   - second wave or V740 mechanism-guiding additions

## 6. Immediate Engineering Tasks

### Track A: SAMformer

1. ~~define wrapper location~~ ✅
2. ~~define window extraction logic~~ ✅
3. ~~define minimal production config~~ ✅
4. run one-condition smoke benchmark
5. if harness smoke still exits `137`, tighten the interactive data-loading path before full benchmark submission
6. compare against the new V740-alpha event-memory smoke path before any large submission

### Track B: OLinear

1. specify required matrix artifacts
2. design artifact generation script for Block 3
3. test whether the official code can run with current TSLib path conventions

### Track C: LightGTS / LiPFormer

## 7. Public Generalization Benchmark Pack (Research-Driven)

For the NeurIPS-2026-quality paper target, benchmark expansion should not stop
at Block 3 alone. The external public evaluation pack should be chosen from the
dataset families that recur across 2024-2026 top-venue forecasting papers.

### 7.1 First public dataset pack

These are the highest-priority public datasets to standardize first:

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `Electricity` / `ECL`
- `Traffic`
- `Weather`
- `Exchange`
- `ILI`
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`
- `Solar`

This pack is the most defensible starting point because it captures the
long-horizon multivariate core, the traffic/spatiotemporal family, and the
covariate-sensitive family that matter most for V740's projected generalization
story.

### 7.2 First public comparator pack

When those datasets are activated, the default public comparator families
should include:

- linear / decomposition:
  - `DLinear`, `RLinear`, `OLinear`
- transformer / mixer:
  - `PatchTST`, `iTransformer`, `TimesNet`, `TimeMixer`, `FEDformer`,
    `Crossformer`
- deep classical:
  - `NBEATS`, `NHITS`, `TFT`, `TiDE`
- foundation:
  - `Chronos`, `Moirai`, `TimesFM`, `TimerXL`, `TimeMoE`, `Sundial`

### 7.3 Why this matters for Block 3

This external pack should not be treated as a side quest. It serves two
purposes directly tied to V740:

1. it tells us whether the mechanisms that seem promising on Block 3 are
   general forecasting mechanisms or merely local artifacts,
2. it defines the public-paper comparator surface that a NeurIPS oral-level
   submission would be expected to address credibly.

### 7.4 Highest-value next-wave additions after the current queue stabilizes

Once the current Phase 15 / V739 queue pressure eases, the next benchmark
additions should be chosen by their information value for V740, not by paper
novelty alone. The current best next-wave order is:

1. `LightGTS`
   - strongest efficiency-first single-model reference
   - most aligned with the V740 target profile
2. `OLinear`
   - still preprocessing-blocked, but strategically important because it
     stress-tests whether a cheap decomposition/orthogonal baseline can cover a
     large fraction of easy continuous/count cells
3. `LiPFormer`
   - efficient patch-wise alternative to PatchTST
   - directly relevant to V740's local-context branch
4. `UniTS`
   - more important as a single-model conditioning reference than as a raw
     leaderboard threat
5. `CASA`
   - useful efficiency-first attention reference once the lighter additions are
     in place

This order is intentionally not "all latest papers." It is the smallest
credible next-wave pack that most directly improves both the Block 3 benchmark
and the V740 design line.

1. inspect upstream code organization
2. estimate dependency and vendoring cost
3. decide whether to vendor or re-implement core architecture

## 7. Bottom Line

The missing-model expansion should not be treated as a flat checklist.

The practical truth is:

- **SAMformer** is the best first real benchmark addition
- **OLinear** is strategically important but preprocessing-blocked
- **LightGTS** is highly relevant and should be treated as a major next addition
- **LiPFormer** is worth adding, but after the easier and higher-information models

This is the order most consistent with the current benchmark state and the V740 single-model objective.
