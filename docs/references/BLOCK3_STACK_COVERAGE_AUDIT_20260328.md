# Block 3 Stack Coverage Audit

> Last updated: 2026-03-28
> Scope: factual audit of what is and is not currently integrated for the
> Block 3 benchmark and V740 research line.
> Evidence sources:
> - local registry: `src/narrative/block3/models/registry.py`
> - current filtered benchmark: `runs/benchmarks/block3_phase9_fair/all_results.csv`
> - direct local smokes / narrow clears
> - official repo paths tracked in the current research docs and same-day GitHub checks

This document is intentionally strict about status labels:

- `active lane`: the library is genuinely used as a current integration lane
- `registered`: model name exists in the local registry
- `benchmark landed`: appears in the canonical filtered benchmark
- `local-only clear`: has passed a local or narrow benchmark-clear path, but is not yet a canonical entrant
- `docs-only`: tracked in research docs, not yet integrated
- `not integrated`: no local registry entry and no runnable benchmark path yet

## 0. Strict Verdict

As of **2026-03-28**, the answer to

> "Are these stacks and repo families already fully registered, integrated,
> tuned, and adapted to Block 3?"

is **no**.

More precisely:

1. **The current Block 3 benchmark is already large and serious, but not fully
   exhaustive across the 2024-2026 frontier.**
2. **Only a minority of the named stack families are active current lanes.**
3. **Many named foundation / official repos are only in one of these states:**
   - registered only,
   - partial benchmark,
   - local-only clear,
   - docs-only,
   - or not integrated.
4. **“Fully tuned and adapted” is not an honest claim for most of the listed
   repos.**
   For many of them there is:
   - no active wrapper,
   - no canonical benchmark row,
   - no audited hyperparameter sweep on Block 3,
   - and no clean completion proof on the shared benchmark surface.

The strongest honest statement today is:

> Block 3 already has a meaningful benchmark frontier, but the full stack /
> repo universe listed by the user has **not** been fully integrated or fully
> tuned yet.

## 1. Layer 1: Skeleton Libraries

| Stack | Current local reality | Benchmark role today |
| --- | --- | --- |
| `TSLib` | **active lane** via `src/narrative/block3/models/tslib_models.py` | major current benchmark surface |
| `BasicTS` | docs/research only | not integrated |
| `ProbTS` | docs/research only | not integrated |
| `NeuralForecast` | **active lane** via `deep_models.py` / AutoFit candidate pool | major current benchmark surface |
| `StatsForecast` | **active lane** via `statistical.py` | major current benchmark surface |
| `Darts` | no active wrapper lane | not integrated |
| `PyTorch Forecasting` | referenced historically, but not an active Block 3 lane | not integrated as a benchmark lane |
| `sktime` | no active wrapper lane | not integrated |
| `autogluon.timeseries` | no active wrapper lane | not integrated |

## 2. Layer 2: Foundation / Zero-shot Official Repos

| Direction | Local registry | Current filtered benchmark | Current status |
| --- | --- | ---: | --- |
| `Chronos` | yes | 160 rows / 160 cells | benchmark landed |
| `ChronosBolt` | yes | 160 / 160 | benchmark landed |
| `Chronos2` | yes | 160 rows / 114 cells | partial benchmark |
| `TimesFM` | yes | 160 / 160 | benchmark landed |
| `TimesFM2` | yes | 0 / 0 | registered only |
| `uni2ts / Moirai` | yes | 0 / 0 | registered only |
| `Moirai2` | yes | 0 / 0 | registered only |
| `MoiraiLarge` | yes | 0 / 0 | registered only |
| `Time-MoE` | yes | 0 / 0 | registered only |
| `Sundial` | yes | 0 / 0 | registered only |
| `MOMENT` | yes | 0 / 0 | registered only |
| `Lag-Llama` | yes | 0 / 0 | registered only |
| `TiRex` | yes | 0 / 0 | registered but not benchmark-landed |
| `Kairos` | no | 0 / 0 | not integrated |
| `IBM Granite TSFM` | no dedicated local model name today | 0 / 0 | docs-only / not integrated |
| `Timer-XL` | yes | 0 / 0 | registered only |
| `SEMPO` | yes | 0 / 0 | registered but excluded on the current benchmark line |
| `TS-RAG` | no | 0 / 0 | not integrated |
| `TabPFN-time-series` | wrappers exist as `TabPFNClassifier/Regressor` | 0 / 0 | registered; latest-source 2.6 runtime now locally ready; classifier and funding-regressor narrow clears completed successfully; investors-count regressor probe completed but failed fairness |

## 3. Layer 3: 2025-2026 Supervised Official Repos

| Direction | Local registry | Current filtered benchmark | Current status |
| --- | --- | ---: | --- |
| `TimeBridge` | yes | 0 / 0 | registered but excluded on current line |
| `DUET` | yes | 72 / 72 | partial benchmark |
| `TimeRecipe` | yes | 81 / 81 | partial benchmark |
| `TimePerceiver` | yes | 0 / 0 | registered but excluded on current line |
| `MMPD` | no | 0 / 0 | not integrated |
| `COSA` | no | 0 / 0 | not integrated |
| `interPDN` | no | 0 / 0 | not integrated |
| `TimeMosaic` | no | 0 / 0 | not integrated |
| `xPatch` | yes | 81 / 81 | partial benchmark |
| `ARMD` | no | 0 / 0 | not integrated |
| `TimePFN` | no | 0 / 0 | not integrated |

## 4. Layer 4: LLM / Multimodal / Transfer / Edge Cases

| Direction | Local status |
| --- | --- |
| `Aurora` | not integrated |
| `TimeOmni-1` | not integrated |
| `AdaPTS` | not integrated |
| `TimeCMA` | not integrated |
| `TimeKD` | not integrated |
| `FreqLLM` | not integrated |
| `LLM-TPF` | not integrated |

## 4b. Long-term Index Repos and Benchmark Infrastructure

These matter for horizon scanning and public-dataset generalization planning,
but they are **not** current benchmark entrants by themselves.

| Direction | Current local status |
| --- | --- |
| `TongjiFinLab/awesome-time-series-forecasting` | docs-only tracking reference |
| `ddz16/TSFpaper` | docs-only tracking reference |
| `hushuguo/awesome-time-series-papers` | docs-only tracking reference |
| `LLMs4TS` resource index | docs-only tracking reference |
| `SalesforceAIResearch/GIFT-Eval` | docs-only benchmark-infra reference |
| `TimeCopilot` | docs-only benchmark-infra reference |
| `TimeSeriesGym` | docs-only benchmark-infra reference |

## 5. Immediate Comparator Status for the Current Missing High-value Wave

| Model | Local status today | Best next action |
| --- | --- | --- |
| `SAMformer` | wrapper exists; generic local smoke passes; **narrow benchmark-clear completed** (`MAE=0.3485`, fairness pass true) | decide whether to promote into a wider canonical benchmark lane |
| `Prophet` | wrapper exists; user-local vendor path works; generic local smoke passes; corrected `bigmem` narrow clear completed successfully, but the audited funding slice quality is weak | keep as a cheap sanity baseline candidate, not a high-ceiling frontier entrant |
| `TabPFN-TS` | wrappers exist; official `tabpfn_2_6` checkpoints cached locally; latest-source vendor runtime (`tabpfn 7.0.1`) installed; classifier and funding-regressor narrow clears completed; investors-count regressor probe surfaced a fairness red flag | keep exploring, but do not promote blindly; binary/event slices remain the most credible first expansion path |
| `LightGTS` | official repo audited at HEAD; script-driven project with checkpoints and custom data pipeline; a first vendor-path helper now exists, but no Block 3 wrapper yet | next real benchmark entrant after the current first wave, via a custom vendor wrapper rather than a package lane |
| `OLinear` | docs-only; blocked by artifact-generation path | finish preprocessing design before wrapper work |
| `ElasTST` | docs-only | treat as a strong varied-horizon comparator after first-wave additions |
| `UniTS` | docs-only | treat as a higher-cost second-wave addition tied to the V740 design |

## 6. What This Audit Means

1. The current benchmark is already large and serious, but it does **not**
   yet cover every 2024-2026 direction the project should eventually compare.
2. “registered” is not enough. A large part of the current frontier still sits
   in one of these intermediate states:
   - registered only,
   - partial benchmark,
   - local-only clear,
   - docs-only.
3. The most credible immediate expansion path is still:
   - `SAMformer`
   - `Prophet`
   - `TabPFN-TS`
   before the project spends effort on the heavier second wave.
4. The user-provided stack/repo list is therefore **not yet fully covered**.
   The clearest current gaps are:
   - Layer 1 skeleton libraries beyond `TSLib` / `NeuralForecast` /
     `StatsForecast`
   - multiple foundation repos that are registered-only but not benchmark-landed
   - multiple 2025-2026 supervised repos that are still not integrated at all
   - the entire LLM / multimodal layer, which is still docs-only / not integrated

## 7. Frontier Beyond the User's List

The user also asked whether there are **more** advanced / high-value directions
beyond the explicit repo list. The answer is **yes**.

The most important currently tracked additions beyond the user's enumerated
repo list are:

### 7.1 Highest-value standalone missing comparators

- `LightGTS`
- `OLinear`
- `ElasTST`
- `UniTS`
- `SAMformer` (now local-clear, but still not canonical-landed)
- `TEMPO`

These are the strongest current additions because they directly address one or
more of the project's demonstrated needs:

- lightweight single-model forecasting,
- varied-horizon / arbitrary-horizon behavior,
- decomposition-first efficiency,
- or a strong non-ensemble comparator against the current V740 ambition.

### 7.2 Highest-value mechanism papers that should shape V740

- `CASA`
- `DistDF`
- `Selective Learning`
- `TimeEmb`
- `QDF`
- `JAPAN`
- `Time-o1`

These are not all best treated as standalone benchmark entrants. Several are
more valuable as:

- loss functions,
- sample weighting schemes,
- calibration layers,
- or static/dynamic fusion mechanisms inside V740.

## 8. Current Practical Answer

If the user asks:

> "Have we already deeply studied and fully integrated all these stacks and
> their latest time-series models?"

the factual answer today is:

- **No, not fully.**
- We have a **strong current benchmark core**.
- We have **partial integration** for a meaningful subset.
- We have **local-clear evidence** for `SAMformer`, `Prophet`, and `TabPFN 2.6`.
- But we do **not** yet have full integration + audited tuning + canonical
  benchmark landing for the full four-layer list the user provided.
