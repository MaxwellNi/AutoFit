# Block 3 Missing SOTA Source Map

> Last updated: 2026-03-28
> Purpose: turn the current ``missing SOTA'' discussion into an executable map:
> which directions are truly highest-value for Block 3 and V740, what evidence
> supports that claim, where the official code lives, whether a package
> implementation exists, what the local status is, and what search space should
> be used before claiming a fair comparison.

## 1. Short Answer

No, the current canonical benchmark still does **not** contain every
highest-value 2024-2026 direction.

But the missing set is **not** an unstructured pile of paper names. It now
separates into three groups:

1. **high-value standalone benchmark entrants that should be added**
2. **models already partly integrated locally but not benchmark-cleared**
3. **mechanism / objective papers that should shape V740 rather than be treated
   as standalone benchmark entrants**

This distinction matters because otherwise we waste GPU time benchmarking
training tricks as if they were independent model families.

## 2. Why these directions are “high-value” for Block 3

The value judgment below is not based on novelty alone. A direction is treated
as high-value only if it satisfies at least one of the following:

1. it appears in recent top-venue forecasting comparisons as a serious modern
   baseline or toolkit reference;
2. it directly addresses a demonstrated Block 3 weakness:
   - V739 is too slow because it is a runtime selector rather than a single
     model,
   - V740 still trails on `is_funded`,
   - `h=60/90` remains target-dependent and context-sensitive,
   - EDGAR/text are not yet translated into stable, efficient gains;
3. it supports the eventual public-dataset generalization pack used repeatedly
   in recent papers:
   - `ETTh1`, `ETTh2`
   - `ETTm1`, `ETTm2`
   - `Electricity`
   - `Traffic`
   - `Weather`
   - `Exchange`
   - `ILI`
   - `Solar`
   - `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`
4. it offers either:
   - a strong efficient single-model baseline,
   - a varied-horizon / arbitrary-horizon capability,
   - a clean exogenous or multisource path,
   - or a distribution / calibration / effective-window mechanism that V740
     clearly still needs.

## 3. Highest-Value Standalone Additions

These are the strongest current candidates for **canonical benchmark
integration**, not just literature inspiration.

| Direction | Why high-value for Block 3 | External evidence | Official paper | Official code / package | Local status | Best integration lane |
|---|---|---|---|---|---|---|
| `SAMformer` | Directly aligned with the V740 goal: lightweight single-model forecasting that is simpler than runtime selection | ICML 2024 oral; efficiency-oriented forecaster | `https://arxiv.org/abs/2402.10198` | Official GitHub: `https://github.com/romilbert/samformer` | **Locally integrated wrapper exists** in `src/narrative/block3/models/samformer_model.py`; a freeze-backed generic local smoke passes on `task1_outcome/core_edgar/is_funded/h=14` with non-constant output (`MAE=0.3863`, `fit_seconds=11.6`), and a narrow benchmark-clear SLURM probe (`5294242`) is now queued; still not canonical-benchmark-cleared yet | Custom lightweight wrapper |
| `LightGTS` | Strongest current efficiency-first missing baseline; directly relevant to longer-horizon single-model V740 | ICML 2025 official repo claims SOTA on 9 benchmarks in zero-shot and full-shot settings with better efficiency than TSFMs | `https://arxiv.org/abs/2506.06005` | Official GitHub: `https://github.com/decisionintelligence/LightGTS` | Missing from local benchmark and registry | Custom wrapper or vendor import |
| `OLinear` | Cheap but serious decomposition/linear challenger; highly relevant because NBEATS/NHITS-style decomposition is already dominant on Block 3 | NeurIPS 2025 official repo claims strong efficiency and 24-benchmark / 140-task coverage | NeurIPS 2025 poster: `https://neurips.cc/virtual/2025/poster/119234` | Official GitHub: `https://github.com/jackyue1994/OLinear` | Missing from registry and benchmark; preprocessing artifacts blocked | TSLib-style vendor integration |
| `ElasTST` | Highest-value varied-horizon standalone comparator; directly tied to V740’s h1–365 ambition | ProbTS added an official ElasTST branch and varied-horizon scripts in Oct 2024 | OpenReview PDF: `https://openreview.net/pdf/4747a9cde6909a55d4179490101ff0d5f36daf83.pdf` | ProbTS implementation / branch: `https://github.com/microsoft/ProbTS` | Missing from registry and benchmark | ProbTS-style vendor integration or custom wrapper |
| `UniTS` | Strongest unified multi-task/multi-horizon design reference; directly relevant to replacing V739-style runtime selection | NeurIPS 2024 multi-task time-series model | `https://papers.nips.cc/paper_files/paper/2024/file/fe248e22b241ae5a9adf11493c8c12bc-Paper-Conference.pdf` | Official GitHub: `https://github.com/mims-harvard/UniTS` | Missing from registry and benchmark | Script-based vendor integration |
| `TEMPO` | Valuable TSFM comparator for prompting + decomposition; also useful because it explicitly advertises multimodal and financial/news-side signals | ICLR 2024 foundation-model line; public code and pip route exist | `https://openreview.net/forum?id=YH5w12OUuU` | Official GitHub: `https://github.com/DC-research/TEMPO`; PyPI route noted in repo via `timeagi` | Missing from registry and benchmark | Foundation-model custom wrapper |
| `TabPFN-TS` | Very high information-gain baseline because it is ultra-fast and zero-shot; ideal stress-test against “single-model V740 should be lightweight” | Official repo positions it as zero-shot TS forecasting with TabPFN | NeurIPS 2024 TRL / TSALM workshop line | Official GitHub: `https://github.com/PriorLabs/tabpfn-time-series` | Local generic `TabPFNRegressor/Classifier` wrappers already exist in `src/narrative/block3/models/traditional_ml.py`; the `GLIBCXX_3.4.31` runtime blocker is now resolved, and `hf auth login` has been tested, but local smoke still stops at the gated `Prior-Labs/tabpfn_2_5` repo because account-level terms acceptance (or a local checkpoint via `BLOCK3_TABPFN_MODEL_PATH`) is still missing; not benchmark-clear | ML-tabular style custom wrapper |
| `Prophet` | Not the newest, but still high-value as a business / finance sanity baseline that reviewers recognize immediately | Reappears in practical forecasting baselines and is still a useful business-time-series trust anchor | “Forecasting at Scale” | Official GitHub: `https://github.com/facebook/prophet` ; pip package `prophet` | Local wrapper exists in `src/narrative/block3/models/statistical.py`; a user-local vendor install now works, a generic real-data smoke passes on `task2_forecast/core_only/funding_raised_usd/h=30` (`MAE=5300.86`, non-constant), and a narrow benchmark-clear SLURM probe (`5294243`) is now queued; still not canonical-benchmark-cleared yet | Statistical baseline wrapper |

## 4. High-Value But Not Yet “Canonical Missing”

These are important, but they are not all in the same class as the table above.

| Direction | Why it matters | Current factual status |
|---|---|---|
| `TimePerceiver` | Relevant long-context / generalized encoder-decoder idea | Already in local registry and raw benchmark line (`TimePerceiver` appears in snapshot materials), but **not** in current clean frontier |
| `TimerXL` | Strong long-context reference family for public generalization | Already in local registry, but **0 rows** in current `all_results.csv` |
| `TimesFM2` | Updated foundation-model reference | Already in local registry, but **0 rows** in current `all_results.csv` |
| `TimePro` | Modern Mamba-based forecasting candidate | Public repo path appears in existing local research notes, but not yet locally integrated |
| `LiPFormer` | Lightweight PatchTST alternative | Paper-level value is real, but we have **not verified an official benchmark-ready implementation path today** |

## 5. High-Value Mechanism Papers That Should Shape V740, Not Be Treated as Standalone Entrants

These are extremely important, but benchmarking them as if they were separate
``models'' is often the wrong move.

| Mechanism | Why it matters | Official source | Best use in this project |
|---|---|---|---|
| `CASA` | Efficient score-attention; relevant to a lightweight local-context branch | IJCAI 2025 paper: `https://www.ijcai.org/proceedings/2025/619` ; official code: `https://github.com/lmh9507/CASA` | V740 local-context / patch-mixing design |
| `DistDF` | Direct multistep joint-distribution alignment; strong long-horizon signal | ICLR 2026 OpenReview: `https://openreview.net/forum?id=VrdLwUmzBy` ; official code: `https://github.com/Master-PLC/DistDF` | V740 loss / training objective |
| `Selective Learning` | Selectively optimize more generalizable timesteps; important for noisy, nonstationary panels | NeurIPS 2025; official code signaled on GestaltCog GitHub: `https://github.com/GestaltCogTeam/selective-learning` | V740 curriculum / sample weighting |
| `TimeEmb` | Lightweight static-dynamic disentanglement; especially relevant to Block 3’s core + EDGAR + text mixture | NeurIPS 2025 poster: `https://neurips.cc/virtual/2025/poster/115701` | V740 static-dynamic split and fusion |
| `QDF` / `JAPAN` / `Time-o1` | Hard-cell weighting, uncertainty, transformed-label alignment | already tracked in local research notes | V740 objective / calibration / robustness layer |

## 6. Official Code / Package Reality

The user asked a very practical question: do these directions have official
code, or an implementation inside a serious forecasting package?

### 6.1 Verified today

| Direction | Official repo verified today? | Package-style route verified today? | Practical verdict |
|---|---:|---:|---|
| `SAMformer` | Yes | No package route verified | wrapper exists locally, generic smoke passes, and a narrow benchmark-clear job is now queued |
| `LightGTS` | Yes | No package route verified | benchmark-worthy custom addition |
| `OLinear` | Yes | No package route verified | blocked mainly by artifact generation, not by code absence |
| `UniTS` | Yes | No package route; script-based | integration possible but heavier than SAMformer |
| `ElasTST` | Yes, via ProbTS branch/toolkit | Yes, via ProbTS toolkit route | more practical than a from-scratch reimplementation |
| `TEMPO` | Yes | Yes, repo notes a `timeagi` pip route | foundation baseline candidate |
| `TabPFN-TS` | Yes | repo exists; standalone route verified | local wrappers exist; runtime import works, but practical use still needs gated terms acceptance or a local checkpoint |
| `Prophet` | Yes | Yes, pip package `prophet` | local wrapper exists, a user-local vendor install works, and a narrow benchmark-clear job is now queued |
| `CASA` | Yes | No package route verified | mechanism source, not first standalone entrant |
| `DistDF` | Yes | No package route verified | objective module, not first standalone entrant |

### 6.2 Not fully verified today

| Direction | Current status |
|---|---|
| `LiPFormer` | paper-level value is clear, but an official benchmark-ready code path was not verified in today’s pass |
| `TimeEmb` | paper / poster path verified, but an official reusable code repo was not verified in today’s pass |
| `TimePro` | local research notes already track a GitHub path, but today’s official verification was not completed end-to-end |

## 7. What is actually “higher value” than the current first wave?

Yes. A few directions are now at least as valuable as the older first-wave list,
and some are arguably **more valuable** than parts of it.

### Highest-value benchmark entrants right now

If the question is:

> what should enter the canonical benchmark next if we only care about maximum
> research value per unit engineering effort?

then the current best answer is:

1. `SAMformer`
2. `LightGTS`
3. `OLinear`
4. `ElasTST`
5. `UniTS`
6. `TabPFN-TS`
7. `TEMPO`
8. `Prophet`

### Why this order

- `SAMformer` is already partly done locally.
- `LightGTS` and `OLinear` are the strongest missing efficient challengers.
- `ElasTST` and `UniTS` are the most directly relevant to V740’s multi-horizon
  single-model ambition.
- `TabPFN-TS` gives a fast zero-shot line with very high information gain.
- `TEMPO` is a strong foundation-model comparator with decomposition/prompting
  flavor and multimodal relevance.
- `Prophet` is cheap and reviewer-friendly, even if it is not frontier-new.

## 8. Recommended Search Spaces

These are **proposed benchmark search spaces**, not already-run optimal settings.
Nothing below should be described as “optimal” until it is actually executed and
audited.

### 8.1 `SAMformer`

- input/context length: `{60, 90, 120}`
- hidden dim / model dim: `{64, 128}`
- lr: `{1e-4, 3e-4}`
- dropout: `{0.1, 0.2}`
- batch size: `{32, 64}`
- max epochs: `{3, 5}` for smoke and narrow benchmark

### 8.2 `LightGTS`

- context points: `{96, 192, 336, 528}`
- target points: `{our horizon}`
- patch length: `{16, 32, 48}`
- stride: `{8, 16, 48}`
- d_model: `{128, 256}`
- d_ff: `{256, 512}`
- e_layers: `{2, 3}`
- d_layers: `{1, 2}`
- lr: `{1e-4, 3e-4}`
- dropout / head_drop: `{0.1, 0.2}`

### 8.3 `OLinear`

- lookback / input length: `{60, 90, 120, 180}`
- CSL/ISL width: small vs default vs moderately widened
- Q-matrix artifact setting:
  - correlation window `{60, 90, 120}`
  - refresh granularity `{global, ablation-specific}`
- lr: `{1e-4, 3e-4, 1e-3}`
- weight decay: `{0, 1e-4}`

### 8.4 `ElasTST`

- training horizon set:
  - current Block 3 `{1, 7, 14, 30}`
  - extended research pack `{45, 60, 90, 180, 365}`
- context length: `{90, 120, 180, 365}`
- elastic horizon-conditioning size: `{small, default}`
- lr: `{1e-4, 3e-4}`
- dropout: `{0.1, 0.2}`

### 8.5 `UniTS`

- input length: `{96, 192, 336}`
- horizon curriculum:
  - current official
  - varied-horizon research pack
- task-conditioning width: `{small, default}`
- lr: `{1e-4, 3e-4}`
- batch size: `{16, 32}`
- freeze/unfreeze schedule:
  - `{fully frozen backbone, partially tuned, full finetune}`

### 8.6 `TabPFN-TS`

- lag set: `{[1,7,14,30], [1,7,14,30,60], [1,7,14,30,60,90]}`
- rolling summary windows: `{7, 14, 30}`
- exogenous feature packs:
  - `core_only`
  - `core_edgar`
  - `full`
- target transform:
  - none
  - `log1p` for heavy-tail funding
  - count transform for `investors_count`

### 8.7 `Prophet`

- seasonality mode: `{additive, multiplicative}`
- changepoint prior scale: `{0.01, 0.05, 0.1, 0.5}`
- seasonality prior scale: `{1, 5, 10}`
- yearly / weekly seasonality:
  - `{auto, forced-on, forced-off}`
- regressors:
  - none
  - EDGAR summary features
  - selected text summary features

## 9. What should be registered / integrated first

### Immediate benchmark-addition queue

1. `SAMformer`
2. `Prophet`
3. `TabPFN-TS`
4. `LightGTS`
5. `OLinear`
6. `ElasTST`
7. `UniTS`
8. `TEMPO`

### Immediate V740-mechanism queue

1. `DistDF`
2. `Selective Learning`
3. `CASA`
4. `TimeEmb`

## 10. What we can claim honestly today

As of 2026-03-27:

1. the benchmark already contains many strong modern families,
2. but it still does **not** contain every highest-value 2024-2026 direction,
3. and several of the most valuable missing directions now have verified
   official code paths,
4. so the next step should be:
   - integrate the highest-value standalone entrants into canonical comparison,
   - integrate mechanism papers into V740 rather than benchmarking them as
     separate “models,”
   - and only then claim a truly exhaustive 2026 frontier.
