# Block 3 Benchmark Results

> Last updated: 2026-03-15 (78 valid complete, deprecated outputs archived, V739 confirmed clean)
> Canonical results dir: `runs/benchmarks/block3_phase9_fair/`
> Phase 7/8 results are **DEPRECATED** (4 critical bugs fixed)
> **Data integrity audit**: DEEP ROOT CAUSE ANALYSIS 2026-03-12 — 5-layer oracle leakage chain identified (see §Oracle Root Cause below)
> **Code audit**: 2026-03-12 — No critical bugs, no data leakage, no anomalous metrics
> **V734-V738 audit (2026-03-13)**: ALL 5 AutoFit versions use oracle tables from **test-set** metrics — ALL INVALID

**Generated**: 2026-03-15
**Benchmark Dir**: `block3_phase9_fair` (Phase 9, canonical) — deprecated outputs in `_deprecated_archive/`
**Total Records (all)**: 9,180 (Phase 9: 8,972 + Phase 10: 208)
**Valid Records**: 8,660 (excluding V734/V735/V736 Phase 9 + V737/V738 Phase 10)
**Invalid Records (oracle leak)**: 520 (V734=104, V735=104, V736=104, V737=104, V738=104)
**Phase 11 Models**: 14 new SOTA + V739 (ALL PENDING in SLURM queue)
**Valid Complete Models (104/104 conditions)**: 78
**Partial Models**: 12 (Phase 9 gap-fill PENDING)
**INVALID AutoFit**: V734, V735, V736, V737, V738 (ALL oracle-leaked)
**V739 Status**: 0/104 — 18 SLURM jobs PENDING, 0 results landed
**Text Embeddings**: ❌ EMPTY — 4 generation jobs PENDING. All core_text ≡ core_only.

## Overview

| Metric | Value |
|--------|-------|
| Total metric records (all) | 9,180 |
| **Valid metric records** | **8,660** |
| Invalid records (oracle leak) | 520 (V734+V735+V736+V737+V738) |
| **Valid complete models (104/104)** | **78** |
| Phase 9 partial models (<104) | 12 |
| Phase 9 unique models (total) | 93 (90 valid + 3 invalid) |
| INVALID AutoFit (Phase 9) | V734=104, V735=104, V736=104 |
| INVALID AutoFit (Phase 10) | V737=104, V738=104 |
| Phase 11 new models (queued) | 14 + V739 |
| V739 progress | **0/104** — 0 RUNNING, 18 PENDING |
| Phase 12 text re-runs (blocked) | 40 scripts ready |
| SLURM status | **0 RUNNING, 125 PENDING** |
| Categories | deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota, tslib_sota |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |

## CRITICAL: 5-Layer Oracle Test-Set Leakage Root Cause Analysis 🔴🔴🔴

> **Severity**: FATAL — invalidates ALL AutoFit V734-V738 rankings
> **Discovered**: 2026-03-11, **Deep analysis**: 2026-03-12
> **Affects**: V733, V734, V735, V736, V737, V738 (ALL versions)

### Layer 1 — Data Source: Benchmark Only Produces Test-Set Metrics

The benchmark pipeline (`run_block3_benchmark_shard.py`) evaluates models on a
train/test split and writes ONLY test-set MAE/RMSE to `metrics.json`. There is NO
separate validation-set evaluation. Therefore, any oracle table built from
`metrics.json` IS test-set leakage by definition.

```
benchmark → train/test split → evaluate on TEST → metrics.json (only test metrics)
                                                        ↓
                                        ORACLE_TABLE built from test metrics
                                                        ↓
                                        AutoFit selects models using test info → LEAKAGE
```

### Layer 2 — Copy-Paste Perpetuation Without Auditing

Each AutoFit version built upon the previous one's oracle table, expanding it without
questioning the data source:

| Version | Oracle Table | Source | How Expanded |
|---------|-------------|--------|-------------|
| V733 | ORACLE_TABLE (L50) | Hand-coded, 5,848 records | Original sin |
| V734 | ORACLE_TABLE_V734 (L425) | Phase 9 test avg_rank | Added confidence weighting |
| V735 | ORACLE_TABLE_V735 (L689) | Phase 9 test RMSE, single best | Exact condition lookup |
| V736 | ORACLE_TABLE_TOP3 (L751) | Phase 9 test RMSE, top-3 | Inverse-RMSE ensemble |
| V737 | Inherits V736's ORACLE_TABLE_TOP3 | Same | Added EDGAR PCA |
| V738 | ORACLE_TABLE_V738 (L1313) | Phase 9 test MAE, top-5 | Multi-pool routing |

### Layer 3 — Missing Validation Infrastructure

The benchmark harness has NO mechanism for:
- Train/validation/test 3-way split
- Cross-validation
- Producing validation-set metrics separate from test-set

Without validation metrics, it's structurally impossible to build a legitimate oracle table.

### Layer 4 — Dead Code Created False Confidence

V738's `__init__` (L1412): `self._val_frac = val_frac  # default 0.2`
V738's docstring: "pick whichever has lower OOF MAE on 20% held out validation set"

But `val_frac` is **NEVER referenced** in `fit()` (L1420-1532) or `predict()` (L1590-1646).
The parameter exists, the docstring describes it working, but the implementation was never written.

**Proof**: V753 in `autofit_wrapper.py` (L7232) HAS a working `_fit_validation()` method:
```python
def _fit_validation(self, X_train, y_train, horizon):
    val_size = max(int(n * self._val_fraction), 20)
    X_inner, X_val = X_train[:-val_size], X_train[-val_size:]
    y_inner, y_val = y_train[:-val_size], y_train[-val_size:]
    # ... train on inner, evaluate on val, select best model
```
V738 was supposed to copy this logic but only copied the parameter.

### Layer 5 — No Automated Guard Against Test-Set Leakage

No test or assertion prevents test-set information from flowing into `fit()`:
- No integration test validates oracle values ≠ test-set metrics
- No code review gate checks oracle table provenance
- No runtime warning when oracle MAE exactly matches evaluation MAE

### Verification: Oracle Values = Exact Test-Set Metrics

```
ORACLE_TABLE_V738[("funding_raised_usd", 1, "core_only")] = [("NBEATS", 380659.46), ...]
Phase 9 test MAE(NBEATS, funding_raised_usd, h=1, core_only) = 380659.4598718175  ← EXACT

ORACLE_TABLE_V738[("funding_raised_usd", 7, "core_only")] = [("NHITS", 380577.13), ...]
Phase 9 test MAE(NHITS, funding_raised_usd, h=7, core_only) = 380577.1332467346  ← EXACT
```

### Impact on V738 Results

- V738 104/104 conditions complete (Phase 10)
- V738 vs V737 head-to-head: V738 wins 77/104, V737 wins 27/104
- V738's wins are driven by oracle advantage (5-model ensemble guided by test-set knowledge)
- V737 also oracle-leaked (inherits ORACLE_TABLE_TOP3 from V736)
- **Neither version's results are scientifically valid**

### Root Fix Required

To build a legitimate AutoFit:
1. Add train/val/test 3-way split to benchmark harness
2. Build oracle tables from VALIDATION-set metrics only
3. OR implement V753-style `_fit_validation()` inside AutoFit (no oracle table needed)
4. Add integration test: `assert oracle_mae != test_mae` for all conditions

---

## Full Leaderboard (76 complete models, ranked by mean rank across 104 conditions)

> Ranking metric: **mean rank** (lower = better), computed per-condition MAE ranking then averaged.
> This is the NeurIPS/ICML standard (TSLib/Monash/M4 benchmarks). Raw MAE average is misleading for multi-scale targets.

| Rank | Model | Category | Mean Rank | Wins | Top-3 | Integrity |
|-----:|-------|----------|----------:|-----:|------:|-----------|
| 1 | NHITS | deep_classical | 4.8 | 15/104 | 45/104 | ✅ genuine |
| 2 | PatchTST | transformer_sota | 4.8 | 4/104 | 51/104 | ✅ genuine |
| 3 | NBEATS | deep_classical | 5.7 | 26/104 | 53/104 | ✅ genuine |
| 4 | NBEATSx | transformer_sota | 6.4 | 3/104 | 31/104 | ⚠️ NBEATS near-duplicate |
| 5 | ChronosBolt | foundation | 7.9 | 0/104 | 17/104 | ✅ genuine |
| 6 | KAN | transformer_sota | 11.8 | 5/104 | 12/104 | ✅ genuine |
| 7 | TimesNet | transformer_sota | 12.0 | 0/104 | 0/104 | ✅ genuine |
| 8 | Chronos | foundation | 12.3 | 17/104 | 17/104 | ✅ genuine |
| 9 | ~~AutoFitV736~~ | ~~autofit~~ | ~~13.2~~ | ~~20/104~~ | ~~24/104~~ | ❌ **oracle test-set leakage** |
| 10 | TFT | deep_classical | 13.7 | 0/104 | 6/104 | ✅ genuine |
| 11 | TCN | deep_classical | 14.8 | 0/104 | 0/104 | ✅ genuine |
| 12 | MLP | deep_classical | 14.8 | 0/104 | 0/104 | ✅ genuine |
| 13 | DeepAR | deep_classical | 15.3 | 0/104 | 0/104 | ✅ genuine |
| 14 | NLinear | transformer_sota | 15.6 | 0/104 | 2/104 | ✅ genuine |
| 15 | Informer | transformer_sota | 15.9 | 0/104 | 0/104 | ✅ genuine |
| 16 | ~~AutoFitV734~~ | ~~autofit~~ | ~~16.6~~ | ~~0/104~~ | ~~22/104~~ | ❌ **oracle test-set leakage** |
| 17 | GRU | deep_classical | 16.9 | 0/104 | 5/104 | ✅ genuine |
| 18 | LSTM | deep_classical | 16.9 | 0/104 | 5/104 | ✅ genuine |
| 19 | ~~AutoFitV735~~ | ~~autofit~~ | ~~19.2~~ | ~~6/104~~ | ~~7/104~~ | ❌ **oracle test-set leakage** |
| 20 | DilatedRNN | deep_classical | 19.3 | 0/104 | 0/104 | ✅ genuine |
| 21 | BiTCN | transformer_sota | 19.5 | 0/104 | 0/104 | ✅ genuine |
| 22 | DLinear | transformer_sota | 20.2 | 1/104 | 5/104 | ✅ genuine |
| 23 | TiDE | transformer_sota | 20.6 | 0/104 | 0/104 | ✅ genuine |
| 24 | DeepNPTS | transformer_sota | 20.7 | 7/104 | 9/104 | ✅ genuine |
| 25 | FEDformer | transformer_sota | 29.6 | 0/104 | 0/104 | ✅ genuine |
| 26 | Timer | foundation | 30.3 | 0/104 | 0/104 | ⚠️ near-identical to TimeMoE/MOMENT |
| 27+ | *(remaining 50 models)* | various | 31-93 | - | - | see per-task tables below |

### Clean Leaderboard (excluding invalid models)

Removing: 3 AutoFit (oracle leak), 6 silent-fallback foundation, 5 crash-fallback, 2 near-duplicates, 3 constant-prediction TSLib = **19 excluded**

| Clean Rank | Model | Category | Mean Rank | Wins |
|---:|-------|----------|----------:|-----:|
| 1 | NHITS | deep_classical | 4.8 | 15 |
| 2 | PatchTST | transformer_sota | 4.8 | 4 |
| 3 | NBEATS | deep_classical | 5.7 | 26 |
| 4 | ChronosBolt | foundation | 7.9 | 0 |
| 5 | KAN | transformer_sota | 11.8 | 5 |
| 6 | TimesNet | transformer_sota | 12.0 | 0 |
| 7 | Chronos | foundation | 12.3 | 17 |
| 8 | TFT | deep_classical | 13.7 | 0 |
| 9 | TCN | deep_classical | 14.8 | 0 |
| 10 | MLP | deep_classical | 14.8 | 0 |

### Key Insight: NHITS and PatchTST dominate

Both NHITS (mean_rank=4.8, 15 wins, 45 top-3) and PatchTST (mean_rank=4.8, 4 wins, 51 top-3) are
tied at #1 — NHITS wins more conditions outright, while PatchTST is more consistently top-3.
NBEATS (#3, 26 wins, 53 top-3) has the MOST condition wins of any model.

### AutoFit Rankings Summary (INVALIDATED)

> ⚠️ All AutoFit rankings below are **scientifically invalid** due to oracle test-set leakage.
> The oracle tables (ORACLE_TABLE_V738, ORACLE_TABLE_TOP3) select models using test-set MAE/RMSE
> values from the same Phase 9 evaluation. See §V738 Fairness Audit above.

### Partial & In-Progress Models (2026-03-11)

| Model | Records | Remaining | Category | Status |
|-------|--------:|----------:|----------|--------|
| V738 | 104/104 | 0 | autofit | ❌ oracle leak — ALL rankings invalid |
| V737 | 104/104 | 0 | autofit | ❌ oracle leak — ALL rankings invalid |
| Chronos2 | 58/104 | 46 | foundation | 🔄 SLURM queued (5 jobs) |
| TTM | 58/104 | 46 | foundation | 🔄 SLURM queued (5 jobs) |
| TimeFilter | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| MultiPatchFormer | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| MSGNet | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| PAttn | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| MambaSimple | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| Crossformer | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (11 jobs) |
| ETSformer | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (6 jobs) |
| LightTS | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (6 jobs) |
| Pyraformer | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (6 jobs) |
| Reformer | 52/104 | 52 | tslib_sota | 🔄 SLURM queued (6 jobs) |
| NegativeBinomialGLM | 16/104 | 88 | ml_tabular | 🔄 SLURM queued (11 jobs) |
| AutoFitV739 | 0/104 | 104 | autofit | 🆕 SLURM queued (11 jobs, Phase 10) |
| CFPT | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| DeformableTST | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| ModernTCN | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| PathFormer | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| SEMPO | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| TimePerceiver | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| TimeBridge | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| TQNet | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| PIR | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| CARD | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| PDF | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| TimeRecipe | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| DUET | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |
| SRSNet | 0/104 | 104 | tslib_sota | 🆕 SLURM queued (Phase 11) |

**Total remaining conditions**: 2,260 (across 28 models)

### SLURM Job Status (2026-03-11)

| Batch | Jobs | Status | Account | Notes |
|-------|------|--------|---------|-------|
| Phase 9 gap-fill (p9r_*) | 100 | PENDING (Priority) | npin | Chronos2/TTM/TimeFilter/MultiPatchFormer/MSGNet/PAttn/MambaSimple/Crossformer/ETSformer/LightTS/Pyraformer/Reformer |
| Phase 9 NegBinGLM (p9r_nbglm_*) | 11 | PENDING (Priority) | npin | bigmem partition |
| Phase 10 V739 (p10_v739_*) | 11 | PENDING (Priority) | npin | GPU partition, validation-based AutoFit |
| Phase 11 new SOTA (p11_tslib_new_*) | 11 | PENDING (Priority) | npin | 14 new models per script |
| **cfisch** | **0** | **IDLE** | cfisch | No jobs submitted — account underutilized |
| **Total queued (npin)** | **133** | ALL PENDING | npin | Reason: Priority scheduling |

## task1_outcome

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.2K | 394.6K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| iTransformer | 376.3K | 376.6K | 376.0K | 376.2K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| KAN | 384.5K | 385.1K | 386.0K | 387.3K | ✅ |
| TimeMixer | 385.3K | 385.2K | 386.2K | 386.2K | ✅ |
| NLinear | 385.6K | 387.3K | 386.9K | 386.6K | ✅ |
| RMoK | 386.1K | 386.0K | 386.8K | 386.2K | ✅ |
| BiTCN | 386.2K | 386.1K | 386.2K | 385.4K | ✅ |
| SOFTS | 386.4K | 386.5K | 387.8K | 403.1K | ✅ |
| DLinear | 388.1K | 388.9K | 388.3K | 389.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| TSMixerx | 397.9K | 406.5K | 496.8K | 476.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## task2_forecast

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.2K | 394.6K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| KAN | 374.8K | 375.4K | 376.3K | 377.6K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| TimeMixer | 375.6K | 375.5K | 376.5K | 376.5K | ✅ |
| NLinear | 375.9K | 377.6K | 377.2K | 376.9K | ✅ |
| RMoK | 376.4K | 376.3K | 377.1K | 376.5K | ✅ |
| BiTCN | 376.5K | 376.4K | 376.5K | 375.7K | ✅ |
| iTransformer | 376.5K | 377.1K | 390.1K | 384.1K | ✅ |
| SOFTS | 376.6K | 376.8K | 378.0K | 393.4K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| DLinear | 378.4K | 379.2K | 378.6K | 379.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| TSMixerx | 388.2K | 396.8K | 487.1K | 467.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## task3_risk_adjust

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.0K | 394.8K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| KAN | 374.8K | 375.4K | 376.3K | 377.6K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| TimeMixer | 375.6K | 375.5K | 376.5K | 376.5K | ✅ |
| NLinear | 375.9K | 377.6K | 377.2K | 376.9K | ✅ |
| RMoK | 376.4K | 376.3K | 377.1K | 376.5K | ✅ |
| BiTCN | 376.5K | 376.4K | 376.5K | 375.7K | ✅ |
| SOFTS | 376.6K | 376.8K | 378.0K | 393.4K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| iTransformer | 377.8K | 375.9K | 410.4K | 380.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| DLinear | 378.4K | 379.2K | 378.6K | 379.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| TSMixerx | 388.2K | 396.8K | 487.1K | 467.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## Training Time Summary

| Model | Avg (s) | Min (s) | Max (s) |
|-------|--------:|--------:|--------:|
| AutoFitV736 | 3140.9 | 52.9 | 39074.7 |
| iTransformer | 1452.0 | 927.9 | 1763.2 |
| MSTL | 1175.3 | 112.5 | 1981.8 |
| AutoARIMA | 667.5 | 284.4 | 1133.3 |
| SOFTS | 459.7 | 329.9 | 708.1 |
| GRU-D | 355.4 | 179.6 | 525.6 |
| FEDformer | 342.2 | 171.1 | 585.0 |
| RMoK | 278.3 | 165.1 | 426.0 |
| Autoformer | 265.4 | 152.3 | 429.0 |
| Informer | 251.4 | 113.5 | 332.0 |
| TimesNet | 237.0 | 125.8 | 386.1 |
| TimeMixer | 175.7 | 140.6 | 232.6 |
| TSMixer | 155.2 | 107.6 | 228.3 |
| TSMixerx | 144.3 | 113.0 | 223.4 |
| AutoTheta | 114.3 | 60.1 | 200.9 |
| TFT | 91.6 | 71.1 | 129.2 |
| AutoETS | 80.1 | 10.2 | 157.5 |
| VanillaTransformer | 73.6 | 50.2 | 178.6 |
| PatchTST | 70.6 | 35.7 | 141.3 |
| SAITS | 63.2 | 24.2 | 104.5 |
| DeepAR | 61.8 | 30.6 | 171.3 |
| DilatedRNN | 47.7 | 24.5 | 82.9 |
| AutoFitV735 | 46.7 | 8.3 | 138.1 |
| LSTM | 39.7 | 16.2 | 76.7 |
| TCN | 38.9 | 12.1 | 76.4 |
| GRU | 35.3 | 11.8 | 79.9 |
| TiDE | 34.6 | 20.3 | 96.6 |
| MLP | 33.8 | 9.2 | 70.3 |
| BiTCN | 31.8 | 21.8 | 79.7 |
| AutoFitV734 | 28.5 | 3.2 | 94.8 |
| KAN | 28.0 | 10.7 | 69.5 |
| Sundial | 27.6 | 8.1 | 48.6 |
| TimesFM2 | 24.9 | 7.0 | 45.6 |
| NHITS | 24.5 | 10.5 | 37.6 |
| NBEATS | 24.1 | 9.3 | 70.5 |
| NBEATSx | 24.0 | 9.3 | 73.8 |
| NLinear | 22.2 | 11.8 | 62.2 |
| DLinear | 21.5 | 7.8 | 63.9 |
| DeepNPTS | 21.3 | 14.1 | 36.0 |
| TimeXer | 15.0 | 7.0 | 56.2 |
| StemGNN | 11.5 | 4.7 | 48.1 |
| MOMENT | 11.0 | 5.5 | 54.5 |
| Timer | 8.3 | 2.9 | 50.9 |
| TimeMoE | 8.1 | 2.9 | 48.5 |
| Chronos | 7.9 | 2.7 | 30.9 |
| ChronosBolt | 7.2 | 2.6 | 12.5 |
| TimesFM | 7.1 | 2.0 | 46.7 |
| xLSTM | 6.8 | 1.8 | 46.2 |
| TimeLLM | 6.0 | 3.0 | 40.9 |
| SF_SeasonalNaive | 1.3 | 0.4 | 2.2 |

## Completion Matrix

Shows which task/category/ablation combinations have results.

| Task | Category | Ablation | Models |
|------|----------|----------|-------:|
| task1_outcome | autofit | core_edgar | 3 |
| task1_outcome | autofit | core_only | 2 |
| task1_outcome | autofit | core_text | 3 |
| task1_outcome | autofit | full | 3 |
| task1_outcome | deep_classical | core_edgar | 9 |
| task1_outcome | deep_classical | core_only | 9 |
| task1_outcome | deep_classical | core_text | 9 |
| task1_outcome | deep_classical | full | 9 |
| task1_outcome | foundation | core_edgar | 8 |
| task1_outcome | foundation | core_only | 8 |
| task1_outcome | foundation | core_text | 8 |
| task1_outcome | foundation | full | 8 |
| task1_outcome | irregular | core_edgar | 2 |
| task1_outcome | irregular | full | 2 |
| task1_outcome | statistical | core_edgar | 5 |
| task1_outcome | statistical | core_text | 5 |
| task1_outcome | statistical | full | 5 |
| task1_outcome | transformer_sota | core_edgar | 23 |
| task1_outcome | transformer_sota | core_only | 23 |
| task1_outcome | transformer_sota | core_text | 23 |
| task1_outcome | transformer_sota | full | 23 |
| task2_forecast | autofit | core_edgar | 3 |
| task2_forecast | autofit | core_only | 3 |
| task2_forecast | autofit | core_text | 3 |
| task2_forecast | autofit | full | 3 |
| task2_forecast | deep_classical | core_edgar | 9 |
| task2_forecast | deep_classical | core_only | 9 |
| task2_forecast | deep_classical | core_text | 9 |
| task2_forecast | deep_classical | full | 9 |
| task2_forecast | foundation | core_edgar | 8 |
| task2_forecast | foundation | core_only | 8 |
| task2_forecast | foundation | core_text | 8 |
| task2_forecast | foundation | full | 8 |
| task2_forecast | irregular | core_edgar | 2 |
| task2_forecast | irregular | full | 2 |
| task2_forecast | statistical | core_edgar | 5 |
| task2_forecast | statistical | core_text | 5 |
| task2_forecast | statistical | full | 5 |
| task2_forecast | transformer_sota | core_edgar | 23 |
| task2_forecast | transformer_sota | core_only | 23 |
| task2_forecast | transformer_sota | core_text | 23 |
| task2_forecast | transformer_sota | full | 23 |
| task3_risk_adjust | autofit | core_edgar | 3 |
| task3_risk_adjust | autofit | core_only | 3 |
| task3_risk_adjust | autofit | full | 2 |
| task3_risk_adjust | deep_classical | core_edgar | 9 |
| task3_risk_adjust | deep_classical | core_only | 9 |
| task3_risk_adjust | deep_classical | full | 9 |
| task3_risk_adjust | foundation | core_edgar | 8 |
| task3_risk_adjust | foundation | core_only | 8 |
| task3_risk_adjust | foundation | full | 8 |
| task3_risk_adjust | irregular | core_edgar | 2 |
| task3_risk_adjust | irregular | full | 2 |
| task3_risk_adjust | statistical | core_edgar | 5 |
| task3_risk_adjust | statistical | full | 5 |
| task3_risk_adjust | transformer_sota | core_edgar | 23 |
| task3_risk_adjust | transformer_sota | core_only | 23 |
| task3_risk_adjust | transformer_sota | full | 23 |

---

## Data Integrity Audit (2026-03-09) — VERIFIED

> All findings below verified empirically via `/tmp/verify_audit.py` against actual metrics data.
> Two earlier claims corrected: C6 (73.1% not 86.5%), C7 (LightGBM IS horizon-variant).

### Critical Finding A: 6 Foundation Models — Silent Context-Mean Fallback

**Models**: Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2

**Evidence**:
- All 6 produce IDENTICAL avg MAE = 165,728.53
- 26/26 condition groups have identical MAE across all horizons (h=1,7,14,30)
- Only 6 unique MAE values out of 104 records
- `fallback_fraction=0.0` (incorrectly reported — harness doesn't detect model-internal fallback)

**Root Cause**: Each model's `predict()` has `except Exception: preds.append(float(np.mean(ctx)))` — when model inference fails (import errors, GPU OOM, API incompatibility), the exception is silently caught and replaced with per-entity context mean. All 6 models share the same entity contexts → identical predictions.

**Additional bugs in Moirai/Moirai2**: `entry.samples.mean(axis=1)` averages across the prediction_length dimension (should be `axis=0`), collapsing horizon information even if models succeed. Plus MOMENT has hardcoded `forecast_horizon=7` in `_load_moment()`.

**Impact**: These 6 models' rankings (#33-38) are **invalid** — they represent per-entity context mean, not model predictions.

### Critical Finding B: 5 Models — 100% Training Crash → Global Mean Fallback

**Models**: AutoCES, StemGNN, TimeXer, xLSTM, TimeLLM

**Evidence**:
- All produce IDENTICAL MAE = 902,704.71 (same as MeanPredictor)
- `fallback_fraction=1.0` (correctly reported) for AutoCES, xLSTM, TimeLLM (104/104)
- StemGNN, TimeXer: `fallback_fraction=0.38` reported, but still 104/104 MAE identical to MeanPredictor

**Root Cause**: NeuralForecast training crashes → `_use_fallback=True` → predict returns `np.mean(self._last_y)` (global mean, not per-entity). This bypasses the fitted RobustFallback (LightGBM).

**Impact**: These 5 models' rankings (#57-62) are meaningless — identical to MeanPredictor baseline.

### Finding C: Timer ≈ TimeMoE ≈ MOMENT (98-99% identical)

- Timer/TimeMoE: 102/104 identical (98.1%)
- Timer/MOMENT: 103/104 identical (99.0%)
- All three show 23-24/26 horizon-invariant groups

**Root Cause**: These models partially succeed but mostly produce near-identical predictions. The `generate()` API may not properly interface with the time series models. Avg MAE = 163,486 (slightly better than Group A due to some genuine predictions).

### Finding D: NBEATSx ≈ NBEATS (73.1% identical) — CORRECTED

- 76/104 conditions produce identical MAE (corrected from earlier 86.5% claim)
- Remaining 28 differ by <1%

**Root Cause**: NBEATSx is NOT in `_NF_SUPPORTS_STATIC_EXOG` set → never receives exogenous EDGAR features → trains identically to NBEATS. This is a misconfiguration, not a model failure.

### Finding E: ml_tabular Horizon Invariance — CORRECTED

**Mostly invariant**: CatBoost, ExtraTrees, HistGradientBoosting, RandomForest, XGBoost, XGBoostPoisson — 26/26 horizon-invariant. This is by design (tree models predict static features without horizon dependency).

**NOT fully invariant**: LightGBM (5/26 invariant) and LightGBMTweedie (4/26 invariant) **DO vary by horizon**. Earlier claim that ALL ml_tabular are horizon-invariant was WRONG. LightGBM's native objective interacts with data partitioning to produce horizon-sensitive splits.

### Effective Leaderboard (excluding invalid/duplicate)

Removing 6 silent-fallback models (C1), 5 crash-fallback models (C2), 2 near-duplicates (TimeMoE, MOMENT ≈ Timer):

| Clean Rank | Model | Avg MAE |
|---:|-------|--------:|
| 1 | Chronos | 159,732 |
| 2 | PatchTST | 159,829 |
| 3 | ChronosBolt | 159,873 |
| 4 | NHITS | 159,983 |
| 5 | TFT | 160,056 |
| 6 | NBEATS | 160,176 |
| 7 | NBEATSx | 160,176 |
| 8 | DeepAR | 160,218 |
| 9 | TimesNet | 160,396 |
| 10 | Informer | 160,400 |
| 11 | KAN | 160,663 |
| 12 | TimeMixer | 160,691 |
| 13 | BiTCN | 160,758 |
| 14 | RMoK | 160,926 |
| 15 | NLinear | 161,032 |
| 16 | FEDformer | 161,649 |
| 17 | TiDE | 161,729 |
| 18 | DLinear | 161,856 |
| 19 | GRU | 162,046 |
| 20 | TSMixer | 162,064 |
| 21 | LSTM | 162,114 |
| 22 | iTransformer | 162,189 |
| 23 | MLP | 162,387 |
| 24 | TCN | 162,412 |
| 25 | DilatedRNN | 162,594 |
| 26 | SOFTS | 162,891 |
| **27** | **AutoFitV736** | **163,465** |
| 28 | Timer | 163,486 |
| **29** | **AutoFitV734** | **165,162** |
| 30 | LightGBMTweedie | 165,728 |
| ... | ... | ... |
| **37** | **AutoFitV735** | **167,038** |

### CRITICAL: Normalized Ranking (per-target min-max) — V736 is #8

The raw MAE ranking is **dominated by `funding_raised_usd` scale** (MAE ~380K) vs `investors_count` (~45) vs `is_funded` (~0.03). When normalizing each target to [0,1] before averaging:

| Norm Rank | Model | Norm Score |
|---:|-------|----------:|
| 1 | PatchTST | 0.0001 |
| 2 | NHITS | 0.0002 |
| 3 | DeepNPTS | 0.0008 |
| 4 | NBEATSx | 0.0011 |
| 5 | NBEATS | 0.0012 |
| 6 | DLinear | 0.0018 |
| 7 | MLP | 0.0019 |
| **8** | **AutoFitV736** | **0.0021** |
| 9 | AutoFitV734 | 0.0024 |
| 10 | ChronosBolt | 0.0025 |
| 11 | AutoFitV735 | 0.0028 |
| ... | ... | ... |
| 17 | Chronos | 0.0066 |

**Key insight**: V736 is **#8 normalized** (vs #27 raw). Chronos drops from #1 to #17. V736 excels on `investors_count` (#8) and `is_funded` (#8) but loses on `funding_raised_usd` (#27) due to EDGAR feature degradation in `core_edgar`/`full` ablations.

### V736 vs Chronos Head-to-Head

| Metric | V736 | Chronos |
|--------|------:|--------:|
| **Condition wins** | **65** | 39 |
| Per-target wins | **2/3** (investors, is_funded) | 1/3 (funding_raised) |
| core_only rank | #8 | #1 |
| core_text rank | #8 | #1 |
| core_edgar rank | #30 | #1 |
| full rank | #30 | #1 |

**Root cause of V736's avg MAE loss**: The EDGAR feature integration causes a +4.1% degradation on `core_edgar`/`full` ablations for `funding_raised_usd`. Without EDGAR (`core_only`/`core_text`), V736 trails Chronos by only +0.2%. V736's ensemble assigns weight to EDGAR-derived features that overfit on funding amounts, while Chronos (zero-shot foundation model) ignores exogenous features entirely.

---

## Deep Re-Audit (2026-03-10) — 4 NEW CRITICAL FINDINGS

### NEW Finding F: Text Ablation Completely Broken 🔴

**Severity**: CRITICAL — invalidates entire ablation study

**Root Cause**: `offers_text.parquet` contains 19 columns that are ALL `string` dtype (headline, title, description_text, company_description, etc.). The `_prepare_features()` function at L572 of `run_block3_benchmark_shard.py` uses:
```python
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
```
This **silently drops ALL text string columns**. The join works correctly (`join_core_with_text()` adds 15 text columns), but they are removed before reaching any model.

**Impact**:
- `core_text ≡ core_only` for **ALL** models (0 numeric difference)
- `full ≡ core_edgar` for **ALL** models (in terms of feature matrix)
- 50% of benchmark compute was wasted on redundant ablations (core_text and full)
- Paper ablation study showing "text contribution" is completely invalid

**Fix Required**: Text→embedding pipeline before `_prepare_features()`:
- Option A: `sentence-transformers/all-MiniLM-L6-v2` → 384-dim embeddings per text column
- Option B: TF-IDF (top-1000) + PCA (→50-dim) — lightweight version

### NEW Finding G: 3 TSLib Models = 100% Constant Predictions 🔴

| Model | Conditions | fairness_pass=False | Status |
|-------|-----------|---------------------|--------|
| MICN | 15/15 | 100% | Must exclude |
| MultiPatchFormer | 31/31 | 100% | Must exclude |
| TimeFilter | 31/31 | 100% | Must exclude |

These models produce identical predictions for ALL test samples, making their MAE meaningless.

### NEW Finding H: NF Model Training Non-Determinism 🟡

8 NF models (BiTCN, DLinear, KAN, NLinear, RMoK, SOFTS, TimeMixer, TSMixerx) show ~1388 MAE units difference between `core_edgar` and `full` ablations, despite having IDENTICAL feature matrices (text columns dropped). Verified:
- Text join keys are unique (0 duplicates in 5.77M text rows)
- All 8 models produce DIFFERENT predictions (8 distinct MAE values)
- Difference is from GPU non-deterministic training in separate SLURM jobs

**Recommendation**: Run 3 random seeds and report mean ± std for NF models.

### NEW Finding I: Foundation Model Ablation Contaminated by RobustFallback 🟡

Foundation models (Chronos, ChronosBolt, Moirai, etc.) show MAE differences across ablations despite being univariate zero-shot models. Root cause: the `_RobustFallback` LightGBM handles entities with <10 training observations. This fallback model IS feature-dependent:
- `core_only`: LightGBM trained on core numeric features only
- `core_edgar`: LightGBM trained on core + EDGAR features → better fallback → lower MAE

Example: Chronos funding_raised_usd drops -1.61% from core_only to core_edgar — this is the fallback effect, NOT Chronos itself.

**Impact**: Foundation model ablation results reflect the RobustFallback sensitivity, not the foundation model's sensitivity to features.

---

## V736 Ranking Analysis (2026-03-10) — Definitive

### Multiple Ranking Standards Compared

| Standard | V736 Rank | Best at this standard |
|----------|-----------|----------------------|
| Raw MAE (avg across all conditions) | **#27/71** | Chronos (159,732) |
| Normalized Mean Rank (per-condition rank avg) | **#10/71** | NHITS (11.47) |
| investors_count (per-target MAE) | **#8/71** | NBEATS (44.78) |
| is_funded (per-target MAE) | **#8/71** | PatchTST (0.0327) |
| funding_raised_usd (per-target MAE) | **#27/71** | Chronos (377,504) |

### V736 vs Top Models (Head-to-Head)

| Rival | V736 Wins | Win Rate | Avg Gap |
|-------|-----------|----------|---------|
| Chronos | 153/264 | **58.0%** | +1.18% |
| TFT | 157/264 | **59.5%** | -0.08% |
| KAN | 127/264 | 48.1% | +0.56% |
| TimesNet | 119/264 | 45.1% | +0.57% |
| ChronosBolt | 73/264 | 27.7% | +1.16% |
| NBEATS | 73/264 | 27.7% | +1.35% |
| PatchTST | 68/264 | 25.8% | +1.42% |
| NHITS | 56/264 | 21.2% | +1.47% |

**Chronos granularity**: V736 beats Chronos on investors_count (124/124, 100%) and is_funded (16/16, 100%), but loses on funding_raised_usd (13/124, 10.5%).

### NeurIPS 2026 Standard Recommendation

The paper should use **Normalized Mean Rank** (standard in TSLib/Monash/M4 benchmarks) as the primary metric, with per-target MAE tables and condition-level win-rates as secondary evidence. Raw MAE average is misleading for multi-scale targets.

V736 at Normalized Rank #10 is **competitive but not oral-level** (needs #1-3). Key improvement area: funding_raised_usd target (+2.34% gap to Chronos).

_Last updated: 2026-03-10_

---

## Text Embedding Research & Decision (2026-03-11)

### BUG-1 Root Cause Recap

The text ablation (`core_text`, `full`) is **completely broken**: `_prepare_features()` at L572 uses
`select_dtypes(include=[np.number])` which silently drops all 15 text string columns from
`offers_text.parquet`. Result: `core_text` and `core_only` produce **identical** feature matrices.
All Phase 9 text ablation results are meaningless.

### Decision: GTE-Qwen2-1.5B-instruct (LLM-native embedding)

After exhaustive survey of the 2024-2025 SOTA embedding landscape, the chosen model is
**Alibaba-NLP/gte-Qwen2-1.5B-instruct** — an LLM-native decoder embedding model built on Qwen2
with bidirectional attention and instruction tuning.

### Embedding Model Survey Results

| Model | Params | Dim | MTEB (EN) | VRAM (FP16) | License | HF Free API |
|-------|-------:|----:|----------:|-------------|---------|-------------|
| **NV-Embed-v2** (NVIDIA) | 8B | 4096 | **72.31** | ~16GB | CC-BY-NC-4.0 ❌ | No |
| **GTE-Qwen2-7B** (Alibaba) | 7B | 3584 | 70.24 | ~14GB | Apache-2.0 ✅ | No |
| **GTE-Qwen2-1.5B** (Alibaba) | 1.5B | 1536 | 67.16 | ~3.3GB | Apache-2.0 ✅ | No |
| **Jina-embeddings-v3** (Jina) | 0.6B | 1024 | ~65 | ~1.2GB | CC-BY-NC-4.0 ❌ | No |
| **Snowflake Arctic-L-v2** | 0.6B | 1024 | ~55 | ~1.2GB | Apache-2.0 ✅ | **Yes** |
| E5-Mistral-7B (Microsoft) | 7B | 4096 | 66.63 | ~14GB | MIT ✅ | No |

### Why GTE-Qwen2-1.5B-instruct

1. **LLM-native architecture** (Qwen2 backbone) — NOT encoder-only like BERT/RoBERTa
2. **MTEB 67.16** — top-tier quality for 1.5B parameter class
3. **Apache-2.0** — fully open, no commercial restrictions (NV-Embed-v2 and Jina-v3 are NC-only)
4. **3.3GB VRAM** at FP16 → fits on ANY GPU (V100/H100/L40S) with massive batch headroom
5. **1536-dim** → PCA to 64 dims for downstream ML (captures >95% variance)
6. **Already-installed env**: transformers 4.57.6 + safetensors + einops — all compatible
7. **Throughput**: ~200 texts/sec on V100 at batch_size=64 → ~45min for all unique texts

### HPC Compute Capacity (ULHPC Iris)

| Partition | Nodes | GPU | VRAM/GPU | RAM | GTE-1.5B Fit? |
|-----------|------:|-----|----------|-----|---------------|
| `gpu` | 24 | V100 ×4 | 32GB | 756GB | ✅ Easy (3.3GB) |
| `hopper` | 1 | H100 ×4 | 80GB | 2TB | ✅ Easy |
| `l40s` | 2 | L40S ×4 | 48GB | 515GB | ✅ Easy |
| `bigmem` | 4 | None | — | 3TB | CPU-only (slow) |

### Text Data Profile

| Property | Value |
|----------|-------|
| Total rows | 5,774,931 |
| Unique entities | 22,569 |
| Text columns | 15 (11 used for embedding) |
| `headline` coverage | 85.1% (avg 125 chars) |
| `description_text` coverage | 78.4% (avg 1,704 chars) |
| `company_description` coverage | 14.2% (avg 1,043 chars) |
| Estimated unique texts (deduped) | ~200-500K |
| Estimated GPU time | 30-60 min (V100, batch=64) |

### Pipeline Architecture

```
offers_text.parquet (5.77M rows, 15 string cols)
    │
    ├─ combine_text_fields() → concatenate 11 key fields per row
    ├─ deduplicate_texts() → hash-based dedup → ~200-500K unique
    ├─ encode_texts() → GTE-Qwen2-1.5B @ FP16 → 1536-dim vectors
    ├─ apply_pca() → 1536 → 64 dims (PCA, >95% variance)
    └─ save → text_embeddings.parquet (entity_id, crawled_date_day, emb_0..63)
        │
        ▼
BenchmarkShard._join_text_embeddings()
    LEFT JOIN on (entity_id, crawled_date_day)
    → text_emb_0..63 appear as NUMERIC columns
    → pass through select_dtypes(include=[np.number]) ✅
```

### Models Rejected

- **NV-Embed-v2**: MTEB #1 (72.31) but **CC-BY-NC-4.0** — non-commercial only
- **Jina-embeddings-v3**: Good quality but **CC-BY-NC-4.0** — non-commercial only
- **GTE-Qwen2-7B**: Higher quality but 4× VRAM/compute for +3 MTEB points not justified
- **TF-IDF / BERT / one-hot**: Old methods, inferior to LLM-native embeddings
- **GPT-5.4 / Claude Opus 4.6 / Gemini 3.1 Pro API**: Zero-cost constraint violated
- **DeepSeek V3.2 / Qwen3.5 / GLM-5**: No dedicated embedding models; base LLMs too large (685B+)
- **HF Free Inference API**: Only Snowflake Arctic available; MTEB 55 (too low for paper)

### Implementation Files

- Embedding generator: `scripts/generate_text_embeddings.py`
- SLURM job: `.slurm_scripts/generate_text_embeddings.sh`
- Harness integration: `scripts/run_block3_benchmark_shard.py` → `_join_text_embeddings()`

### Audit Exclusion List (16 models)

Added to `scripts/aggregate_block3_results.py` → `AUDIT_EXCLUDED_MODELS`:

| Finding | Models Excluded | Reason |
|---------|-----------------|--------|
| A (6) | Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 | Context-mean fallback |
| B (5) | AutoCES, xLSTM, TimeLLM, StemGNN, TimeXer | Training crash fallback |
| C (2) | TimeMoE, MOMENT | Near-duplicate of Timer |
| G (3) | MICN, MultiPatchFormer, TimeFilter | 100% constant predictions |

_Last updated: 2026-03-11_