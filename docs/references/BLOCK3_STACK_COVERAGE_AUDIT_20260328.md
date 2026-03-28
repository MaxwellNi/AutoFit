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
| `TabPFN-time-series` | wrappers exist as `TabPFNClassifier/Regressor` | 0 / 0 | registered; latest-source 2.6 runtime now locally ready; narrow clear completed successfully |

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

## 5. Immediate Comparator Status for the Current Missing High-value Wave

| Model | Local status today | Best next action |
| --- | --- | --- |
| `SAMformer` | wrapper exists; generic local smoke passes; **narrow benchmark-clear completed** (`MAE=0.3485`, fairness pass true) | decide whether to promote into a wider canonical benchmark lane |
| `Prophet` | wrapper exists; user-local vendor path works; generic local smoke passes; corrected `bigmem` narrow clear completed successfully, but the audited funding slice quality is weak | keep as a cheap sanity baseline candidate, not a high-ceiling frontier entrant |
| `TabPFN-TS` | wrappers exist; official `tabpfn_2_6` checkpoints cached locally; latest-source vendor runtime (`tabpfn 7.0.1`) installed; first narrow clear completed successfully | decide whether to promote it into a wider canonical benchmark lane, especially for binary/event-like tasks |
| `LightGTS` | docs-only / source-map complete | next real benchmark entrant after the current first wave |
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
