# Block 3 SOTA Coverage Audit

> Last updated: 2026-03-27
> Purpose: answer a simple but critical question honestly:
> is the current Block 3 benchmark already broad enough to claim that it covers
> the latest relevant forecasting SOTA as of today?

## Short Answer

Not yet.

The current benchmark is already **substantial** and **clean enough to support
serious analysis**, but it is **not yet defensible** to claim that it covers
all of the most recent and most relevant 2024-2026 time-series forecasting
methods.

## 1. Current Internal Coverage

Based on the current canonical benchmark artifacts:

- raw materialized models: **137**
- non-retired real models: **116**
- active leaderboard models: **92**
- post-filter non-retired models with valid results in `all_results.csv`: **86**
- strict full clean comparable frontier: **55 models @ 160 shared cells**
- non-seed main-result comparable frontier: **58 models @ 112 shared cells**

This means the benchmark is already broad enough to support:

1. a meaningful internal frontier,
2. strong clean comparisons among established families,
3. a credible champion-mechanism analysis.

But it does **not** mean that every high-value 2024-2026 forecasting line has
already been integrated and benchmarked.

## 2. What Is Strongly Covered Already

The current fully landed clean frontier does already cover several of the most
important competitive families:

- `NBEATS`, `NHITS`, `GRU`, `TFT`, `DeepAR`
- `PatchTST`, `NBEATSx`, `KAN`, `TimesNet`, `Informer`, `Autoformer`, `FEDformer`, `iTransformer`, `TiDE`, `TimeMixer`
- `Chronos`, `ChronosBolt`, `Timer`, `TimesFM`
- strong classical baselines and tabular baselines

That is enough to say:

- the benchmark is **not shallow**,
- it is **not missing all mainstream strong baselines**,
- and the current top frontier is already nontrivial.

## 3. Why It Is Still Not “Complete SOTA Coverage”

As of today, several recent top-venue forecasting directions are still missing
from the canonical benchmark, or only exist as design references / local
prototypes rather than landed benchmark entrants.

### Verified recent missing or not-yet-benchmarked directions

1. **LightGTS**
   - lightweight general forecaster
   - accepted by **ICML 2025**
   - official source: `https://arxiv.org/abs/2506.06005`
   - current local status: missing from canonical benchmark

2. **OLinear**
   - orthogonally transformed linear forecasting model
   - official NeurIPS 2025 poster page:
     `https://neurips.cc/virtual/2025/poster/119234`
   - current local status: still missing from benchmark; artifact-generation path not finished

3. **SAMformer**
   - efficiency-oriented transformer recipe
   - paper: `https://arxiv.org/abs/2402.10198`
   - current local status: local wrapper exists, but not yet benchmark-cleared

4. **ElasTST**
   - varied-horizon forecasting design
   - official OpenReview PDF:
     `https://openreview.net/pdf/4747a9cde6909a55d4179490101ff0d5f36daf83.pdf`
   - current local status: missing from canonical benchmark

5. **UniTS**
   - unified multi-task time-series model
   - NeurIPS 2024:
     `https://papers.nips.cc/paper_files/paper/2024/file/fe248e22b241ae5a9adf11493c8c12bc-Paper-Conference.pdf`
   - current local status: design reference only, not benchmarked

6. **TimePerceiver**
   - generalized encoder-decoder forecasting framework
   - NeurIPS 2025 poster:
     `https://neurips.cc/virtual/2025/poster/118047`
   - current local status: missing

7. **Implicit Forecaster / Implicit Decoding**
   - forecasting-phase decoding improvement
   - NeurIPS 2025 poster:
     `https://nips.cc/virtual/2025/poster/116698`
   - current local status: missing

8. **TimeEmb**
   - lightweight static-dynamic disentanglement
   - NeurIPS 2025 poster:
     `https://neurips.cc/virtual/2025/poster/115701`
   - current local status: missing

9. **Selective Learning**
   - optimization strategy for noisy / non-generalizable timesteps
   - NeurIPS 2025 poster:
     `https://neurips.cc/virtual/2025/poster/116357`
   - current local status: not integrated as a benchmark method or training option

10. **DistDF**
    - joint-distribution objective for forecasting
    - ICLR 2026 OpenReview:
      `https://openreview.net/forum?id=VrdLwUmzBy`
    - current local status: objective reference only, not benchmarked as a separate line

11. **CASA**
    - efficient score-attention add-on with strong resource claims
    - IJCAI 2025:
      `https://www.ijcai.org/proceedings/2025/619`
    - current local status: missing

## 4. What This Means for a NeurIPS 2026 Oral Claim

### What can already be claimed honestly

We can already claim that the benchmark:

1. includes many strong classical, deep classical, transformer, foundation, and tabular baselines,
2. supports a meaningful and nontrivial clean frontier,
3. is strong enough to reveal stable champion structure,
4. is strong enough to motivate V740 as a serious research problem.

### What cannot yet be claimed honestly

We should **not** currently claim that:

1. all latest relevant 2024-2026 time-series forecasting SOTA has been integrated,
2. the current 55-model full clean frontier is the final exhaustive modern comparison pack,
3. the current benchmark alone is already sufficient for an unquestionable “all-frontier” oral-paper claim.

## 5. External Benchmark Families Repeatedly Used in Recent Papers

Across the recent papers above and the broader 2024-2026 literature pattern,
the recurring public forecasting datasets still center on:

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `Electricity (ECL)`
- `Traffic`
- `Weather`
- `Exchange`
- `Solar`
- `ILI`
- `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`

Evidence examples:

- `UniTS` explicitly spans dozens of forecasting tasks and datasets:
  NeurIPS 2024 paper, lines around Table 1 / datasets section.
- `DistDF` reports averaged forecasting results over horizons
  `96/192/336/720` on datasets such as `ECL`, `Weather`, `Traffic`, and `PEMS`.
- `CASA` reports evaluation on eight real-world long-term forecasting datasets.

## 6. Best Current Judgment

As of 2026-03-27:

- **55 models** is enough for a serious strict full clean leaderboard.
- **58 models** is enough for a serious non-seed main-result benchmark table.
- **86 non-retired models with post-filter results** is enough to show broad exploration.

But:

- it is **not yet enough** to say the benchmark already exhausts the latest
  relevant 2024-2026 forecasting frontier,
- and it is **not yet enough** to claim that the comparison pack is fully
  complete for a NeurIPS 2026 oral-level “final unified table”.

The correct next step is not to deny the current benchmark's value, but to be
precise:

1. use the current tables as a clean and substantial frontier,
2. continue integrating the highest-value missing recent models,
3. then extend the final paper benchmark with public-dataset generalization.
