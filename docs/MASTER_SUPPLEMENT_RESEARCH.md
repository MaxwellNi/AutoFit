# MASTER SUPPLEMENT LIST — Model Research for Block 3 Phase 10 Expansion

**Date**: 2026-03-10 (updated)
**Purpose**: Comprehensive research on candidate models for benchmark expansion beyond the current 100-model Phase 9 registry.
**Scope**: P0 (must-add), P1 (strong missing baselines), P2 (LLM/multimodal), benchmarks

---

## ⚠️ CORRECTION LOG (2026-03-10)

Several entries below were previously marked "no public code" incorrectly. Corrections:
- **TimerXL**: ✅ IMPLEMENTED — shares `thuml/timer-base-84m` checkpoint. NOT "dropped as duplicate".
- **TiRex**: ✅ Available via TSLib `models/TiRex.py` AND `pip install uni2ts`. Should be RE-ENABLED.
- **Mamba**: Status is "env/dependency blocked" (needs `mamba-ssm` CUDA kernel), NOT "code-blocked". MambaSimple (pure PyTorch) is already running.
- **Kairos, SEMPO, TQNet, MMPD, interPDN, TimeMosaic, COSA, CFPT, PIR, SRSNet**: Marked "SKIP — no public code" — **PENDING VERIFICATION by user**. User claims several of these DO have public implementations. Awaiting user's Sections A-F correction document for exact URLs.

### Models to EXCLUDE from leaderboard (per deep audit 2026-03-10)
- **100% constant**: MICN, MultiPatchFormer, TimeFilter
- **Crashed→MeanPredictor**: AutoCES, StemGNN, TimeXer, xLSTM, TimeLLM
- **De-duplicate**: Timer≈TimeMoE (keep MOMENT only)

---

## Summary: Current Registry State

Already implemented in our codebase (`src/narrative/block3/models/`):
- **deep_classical (9)**: NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN
- **transformer_sota (23)**: PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer, TimeXer, TSMixerx, xLSTM, TimeLLM, DeepNPTS
- **foundation (15)**: Chronos, ChronosBolt, Chronos2, Moirai, MoiraiLarge, Moirai2, Timer, TimeMoE, MOMENT, LagLlama, TimesFM, Sundial, TTM, TimerXL, TimesFM2
- **tslib_sota (20)**: TimeFilter, WPMixer, MultiPatchFormer, TiRex, MSGNet, PAttn, MambaSimple, Mamba, Koopa, FreTS, Crossformer, MICN, SegRNN, ETSformer, NonstationaryTransformer, FiLM, SCINet, LightTS, Pyraformer, Reformer
- **statistical (15)**, **ml_tabular (15)**, **irregular (4)**, **autofit (21)**

Models in TSLib (`github.com/thuml/Time-Series-Library/models/`) as of 2026-03:
Autoformer, Chronos, Chronos2, Crossformer, DLinear, ETSformer, FEDformer, FiLM, FreTS, Informer, KANAD, Koopa, LightTS, MICN, MSGNet, Mamba, MambaSimple, Moirai, MultiPatchFormer, Nonstationary_Transformer, PAttn, PatchTST, Pyraformer, Reformer, SCINet, SegRNN, Sundial, TSMixer, TFT (TemporalFusionTransformer), TiDE, TiRex, TimeFilter, TimeMixer, TimeMoE, TimeXer, TimesFM, TimesNet, Transformer, WPMixer, iTransformer

---

## P0 — MUST ADD / MUST FIX

### 1. TimerXL (thuml)
| Field | Value |
|-------|-------|
| **Full Name** | Timer-XL: Long-Context Transformers for Unified Time Series Forecasting |
| **Paper** | ICLR 2025 |
| **arXiv** | 2410.04803 |
| **GitHub** | https://github.com/thuml/Timer-XL (125 stars) |
| **HuggingFace** | `thuml/timer-base-84m` (84M params) — **REAL checkpoint, shared with Timer** |
| **Package** | No pip — uses HF `transformers` with `trust_remote_code=True` |
| **In TSLib?** | Not directly (separate repo), but uses same checkpoint |
| **In NeuralForecast?** | No |
| **GPU?** | Yes |
| **Category** | foundation |
| **Status in our codebase** | ✅ Already implemented as `TimerXL` in `HFFoundationModelWrapper` |
| **Note** | Timer and Timer-XL share the `thuml/timer-base-84m` checkpoint. Timer-XL adds long-context (>2k) support. The HF checkpoint is REAL and functional. |

### 2. Chronos2 (Amazon)
| Field | Value |
|-------|-------|
| **Full Name** | Chronos-2: From Univariate to Universal Forecasting |
| **Paper** | arXiv 2025 (2510.15821) |
| **GitHub** | https://github.com/amazon-science/chronos-forecasting |
| **HuggingFace** | `amazon/chronos-2-*` (multiple sizes) |
| **Package** | `pip install chronos-forecasting` |
| **In TSLib?** | ✅ Yes — `models/Chronos2.py` |
| **In NeuralForecast?** | No |
| **GPU?** | Yes (recommended) |
| **Category** | foundation |
| **Status** | ✅ Already fixed in our codebase — `model_id = amazon/chronos-2` |

### 3. TTM / TinyTimeMixer (IBM Granite)
| Field | Value |
|-------|-------|
| **Full Name** | TinyTimeMixer: A Flexible Efficient Architecture for Accelerating Time Series Foundation Models |
| **Paper** | NeurIPS 2024 |
| **arXiv** | 2401.03955 |
| **GitHub** | https://github.com/ibm-granite/granite-tsfm (805 stars) |
| **HuggingFace** | `ibm-granite/granite-timeseries-ttm-r1`, `ibm-granite/granite-timeseries-ttm-r2` |
| **Package** | `pip install tsfm_public` (Python 3.10–3.13) |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Optional (<1M params, runs on CPU) |
| **Category** | foundation |
| **Status** | ✅ Already implemented in `HFFoundationModelWrapper` |

### 4. TimeMixer++ (TimeMixerPP)
| Field | Value |
|-------|-------|
| **Full Name** | TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis |
| **Paper** | arXiv 2024 (2410.16032) |
| **GitHub** | https://github.com/kwuking/TimeMixer (1.9k stars) |
| **Package** | No pip — git clone |
| **In TSLib?** | Partially — TSLib has `TimeMixer.py` (ICLR'24 original, NOT TimeMixer++) |
| **In NeuralForecast?** | ✅ TimeMixer (original) is in NeuralForecast |
| **GPU?** | Yes |
| **Category** | transformer_sota |
| **Feasibility** | MEDIUM — TimeMixer++ extends TimeMixer to 8 tasks but shares same backbone. Could add to TSLib wrapper or NF wrapper. Code in `kwuking/TimeMixer` repo. Separate from TSLib's `TimeMixer.py` |
| **Status** | Referenced in `configs/block3.yaml` line 179. Not yet implemented as distinct model. |
| **Recommendation** | Can wrap via TSLib adapter or standalone PyTorch. Architecture is MLP-based, so lightweight. Would need separate model file. |

### 5. TiRex
| Field | Value |
|-------|-------|
| **Full Name** | TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning |
| **Paper** | NeurIPS 2025 (arXiv 2505.23719) |
| **GitHub** | Part of Salesforce/uni2ts repo (https://github.com/SalesforceAIResearch/uni2ts) |
| **HuggingFace** | Available via uni2ts |
| **Package** | `pip install uni2ts` |
| **In TSLib?** | ✅ Yes — `models/TiRex.py` (zero-shot evaluation) |
| **In NeuralForecast?** | No |
| **GPU?** | Yes |
| **Category** | foundation |
| **Status** | In our tslib_models.py registry but EXCLUDED from Phase 9 benchmark (no compatible TSLib impl at the time). Now available in TSLib. |
| **Recommendation** | Re-enable via TSLib wrapper or use uni2ts directly for zero-shot. |

### 6. Kairos
| Field | Value |
|-------|-------|
| **Full Name** | Kairos: Building Cost-Efficient Machine Learning Time Series Pipelines (or "Kairos" by Salesforce) |
| **Paper** | Not a single landmark paper — "Kairos" appears in multiple contexts |
| **GitHub** | ❌ No public repo found at `SB-Kairos/Kairos` (404). May refer to internal/unpublished Salesforce work. |
| **Package** | None |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Unknown |
| **Category** | foundation (claimed) |
| **Feasibility** | ❌ NOT FEASIBLE — No public code or model weights found. Possible confusion with Salesforce Moirai ecosystem. |
| **Recommendation** | SKIP — no public implementation exists. |

### 7. TabPFN-TS
| Field | Value |
|-------|-------|
| **Full Name** | From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting |
| **Paper** | arXiv 2025 (2501.02945, v4 Jan 2026) |
| **GitHub** | https://github.com/PriorLabs/tabpfn-time-series |
| **Package** | `pip install tabpfn` (TabPFN-v2 backend) |
| **HuggingFace** | `Prior-Labs/TabPFN-v2-reg` |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Optional (11M params, CPU feasible) |
| **Category** | foundation (tabular-to-TS transfer) |
| **Feasibility** | HIGH — Treats forecasting as tabular regression. PriorLabs provides clear API. Wrappable via our ModelBase. Dependencies: `tabpfn`, `torch`. |
| **Recommendation** | ADD — Novel paradigm (tabular foundation → TS), competitive on GIFT-Eval. Small model, fast inference. |

### 8. Granite_TSFM (IBM)
| Field | Value |
|-------|-------|
| **Full Name** | Granite Time Series Foundation Model (umbrella for TTM, PatchTST, PatchTSMixer) |
| **Paper** | Multiple papers — TTM (NeurIPS'24), PatchTST (ICLR'23), PatchTSMixer |
| **GitHub** | https://github.com/ibm-granite/granite-tsfm (805 stars, v0.3.5) |
| **Package** | `pip install tsfm_public` |
| **In TSLib?** | No (PatchTST exists independently in TSLib) |
| **In NeuralForecast?** | PatchTST is in NeuralForecast |
| **GPU?** | Optional for TTM; Yes for larger models |
| **Category** | foundation |
| **Status** | ✅ TTM already implemented. Granite_TSFM is the umbrella repo, not a separate model. |
| **Recommendation** | No additional model needed — TTM covers the main Granite offering. |

### 9. SEMPO
| Field | Value |
|-------|-------|
| **Full Name** | Unknown — no paper found matching "SEMPO" + time series forecasting |
| **GitHub** | ❌ Not found |
| **Feasibility** | ❌ NOT FEASIBLE — Cannot identify this model. May be misspelled or unpublished. |
| **Recommendation** | SKIP — need clarification on what SEMPO refers to. |

### 10. TS-RAG (RAFT)
| Field | Value |
|-------|-------|
| **Full Name** | RAFT: Retrieval Augmented Time Series Forecasting |
| **Paper** | arXiv 2025 (2505.04163) |
| **Authors** | Sungwon Han, Seungeon Lee, Meeyoung Cha, Sercan O Arik, Jinsung Yoon (Google) |
| **GitHub** | Not yet publicly released (arXiv May 2025, preprint only) |
| **Package** | None |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Yes |
| **Category** | transformer_sota |
| **Feasibility** | LOW — No public code. Retrieval-augmented approach requires custom data pipeline. |
| **Recommendation** | DEFER until code release. |

### 11. Mamba4Cast
| Field | Value |
|-------|-------|
| **Full Name** | Mamba4Cast: Weather Forecasting with Mamba (or Mamba-based forecasting) |
| **Paper** | Various — Mamba for time series is primarily via MambaSimple (already in TSLib) |
| **GitHub** | The main Mamba TS work is `time-series-library/models/Mamba.py` and `MambaSimple.py` |
| **Package** | `mamba-ssm` (requires CUDA, Linux only) |
| **In TSLib?** | ✅ Both `Mamba.py` and `MambaSimple.py` are in TSLib |
| **In NeuralForecast?** | No |
| **GPU?** | Yes (CUDA kernel required for full Mamba; MambaSimple is pure PyTorch) |
| **Category** | tslib_sota |
| **Status** | ✅ Already in our registry as `Mamba` and `MambaSimple`. MambaSimple used in production (no mamba-ssm needed). |
| **Recommendation** | Already covered. |

### 12. TimeBridge
| Field | Value |
|-------|-------|
| **Full Name** | TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting |
| **Paper** | ICML 2025 (arXiv 2410.04442) |
| **GitHub** | https://github.com/hank0316/TimeBridge |
| **Package** | No pip — standalone PyTorch |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Yes |
| **Category** | transformer_sota |
| **Feasibility** | HIGH — Based on TSLib codebase. Drop-in model file. |
| **Recommendation** | ADD via TSLib wrapper. |

### 13. DUET
| Field | Value |
|-------|-------|
| **Full Name** | DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting |
| **Paper** | NeurIPS 2024 / ICML 2025 (arXiv 2412.10859) |
| **GitHub** | https://github.com/decisionintelligence/DUET |
| **Package** | No pip |
| **In TSLib?** | No |
| **In NeuralForecast?** | No |
| **GPU?** | Yes |
| **Category** | transformer_sota |
| **Feasibility** | MEDIUM — Standalone code, would need integration. |
| **Recommendation** | ADD if code is TSLib-compatible. |

### 14. TimePerceiver
| Field | Value |
|-------|-------|
| **Full Name** | TimePerceiver: Efficiently Processing Long Time Series (Perceiver-based) |
| **Paper** | arXiv ~2024 |
| **GitHub** | Search needed — limited info |
| **In TSLib?** | No |
| **GPU?** | Yes |
| **Feasibility** | LOW — Limited public code. |
| **Recommendation** | DEFER — not well-established yet. |

### 15. TimeRecipe
| Field | Value |
|-------|-------|
| **Full Name** | TimeRecipe: Self-Supervised Contrastive Recipe for Time Series |
| **Paper** | ICML 2025 |
| **GitHub** | Likely available (ICML camera-ready) |
| **In TSLib?** | No |
| **GPU?** | Yes |
| **Feasibility** | MEDIUM |
| **Recommendation** | DEFER — requires self-supervised pretraining step. |

### 16. MMPD
| Field | Value |
|-------|-------|
| **Full Name** | MMPD — Multi-scale Multi-resolution Patch-wise Decomposition |
| **Paper** | Recent top-conf (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | ❌ LOW — Cannot locate public implementation |
| **Recommendation** | SKIP until verifiable. |

### 17. interPDN
| Field | Value |
|-------|-------|
| **Full Name** | interPDN — Interpretable Patch Decomposition Network |
| **Paper** | Recent top-conf (unconfirmed venue) |
| **GitHub** | Not found publicly |
| **Feasibility** | LOW |
| **Recommendation** | SKIP until verifiable. |

### 18. TimeMosaic
| Field | Value |
|-------|-------|
| **Full Name** | TimeMosaic — mosaic-based time series model |
| **Paper** | Recent top-conf (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP until verifiable. |

### 19. CFPT
| Field | Value |
|-------|-------|
| **Full Name** | CFPT — Cross-Frequency Pre-Training for Time Series |
| **Paper** | Recent (unconfirmed venue) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP until verifiable. |

### 20. TQNet
| Field | Value |
|-------|-------|
| **Full Name** | TQNet — Temporal Quantile Network |
| **Paper** | Recent (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP — no public code. |

### 21. PIR
| Field | Value |
|-------|-------|
| **Full Name** | PIR — Patching with Instance-wise Routing |
| **Paper** | Recent (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP. |

### 22. SRSNet
| Field | Value |
|-------|-------|
| **Full Name** | SRSNet — Structured Recurrence for Sequence modeling |
| **Paper** | Recent (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP. |

### 23. COSA
| Field | Value |
|-------|-------|
| **Full Name** | COSA — possibly "Contrastive Seasonal-Aware" or similar |
| **Paper** | Recent (unconfirmed) |
| **GitHub** | Not found |
| **Feasibility** | LOW |
| **Recommendation** | SKIP — cannot verify. |

---

## P1 — STRONG MISSING BASELINES

### Comprehensive Table

| Model | Paper | Year/Venue | GitHub | In TSLib? | In NF? | GPU? | Feasibility | Category | Notes |
|-------|-------|-----------|--------|-----------|--------|------|-------------|----------|-------|
| **ModernTCN** | ModernTCN: A Modern Pure Convolution Structure | ICLR 2024 Spotlight | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) (409★) | ❌ No | ❌ No | Yes | HIGH — TSLib-style codebase, drop-in | transformer_sota | Pure convolution. Uses TSLib datasets directly. |
| **DeformableTST** | DeformableTST: Transformer with Deformable Attention for TS | arXiv 2024 | github.com/thefatherofall/DeformableTST | ❌ No | ❌ No | Yes | MEDIUM — standalone | transformer_sota | Deformable attention for TS |
| **CARD** | CARD: Channel Aligned Robust Blend Transformer | ICLR 2024 | github.com/wxie9/CARD | ❌ No | ❌ No | Yes | HIGH — TSLib-compatible | transformer_sota | Channel-aligned transformer |
| **CATS** | CATS: Context-Aware Time Series | ICML 2024 | github.com/dongbeank/CATS | ❌ No | ❌ No | Yes | HIGH — TSLib baseline | transformer_sota | Context-aware TS prediction |
| **S-Mamba** | S-Mamba: Structured State Space Mamba for TS | arXiv 2024 | github.com/wzhwzhwzh0921/S-Mamba | ❌ No | ❌ No | Yes | MEDIUM — needs mamba-ssm | transformer_sota | Structured Mamba for TS |
| **Bi-Mamba** | Bi-Mamba: Bidirectional Mamba for TS | arXiv 2024 | Limited availability | ❌ No | ❌ No | Yes | LOW — limited code | transformer_sota | Bidirectional SSM |
| **Mamba** | Mamba: Linear-Time Sequence Modeling | arXiv 2023 (2312.00752) | state-spaces/mamba | ✅ Yes | ❌ No | Yes (CUDA) | ✅ DONE | tslib_sota | Already in registry |
| **FITS** | FITS: Modeling Time Series with 10k Parameters | ICLR 2024 | github.com/VEWOXIC/FITS | ❌ No | ❌ No | Optional | HIGH — 10k params, lightweight | transformer_sota | Extremely compact model |
| **PDF** | PDF: Point-wise Decomposition Framework | ICLR 2024 | github.com/Levi-Ackman/PDF | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | Decomposition-based |
| **Pathformer** | Pathformer: Multi-scale Transformers with Adaptive Pathways | ICLR 2024 | github.com/decisionintelligence/pathformer | ❌ No | ❌ No | Yes | HIGH — clean code | transformer_sota | Multi-scale transformer |
| **Fredformer** | Fredformer: Frequency Debiased Transformer | arXiv 2024 | github.com/chenzRG/Fredformer | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | Frequency debiasing |
| **CycleNet** | CycleNet: Enhancing TS Forecasting through Modeling Periodic Patterns | NeurIPS 2024 | github.com/ACAT-SCUT/CycleNet | ❌ No | ❌ No | Yes | HIGH — clean TSLib-style | transformer_sota | Cyclic pattern modeling |
| **FilterTS** | FilterTS: Filter-Enhanced MLP for TS | arXiv 2024 | Limited | ❌ No | ❌ No | Yes | LOW | transformer_sota | Filtering MLP |
| **xPatch** | xPatch: Channel-independent Patch Transformer with Explainability | arXiv 2024 | github.com/zezhishao/xPatch | ❌ No | ❌ No | Yes | HIGH | transformer_sota | Explainable patching |
| **HCAN** | HCAN: Hierarchical Cross-Attention Network | arXiv 2024 | Limited | ❌ No | ❌ No | Yes | LOW | transformer_sota | Cross-attention hierarchy |
| **WPMixer** | WPMixer: Efficient Multi-Resolution Mixing | AAAI 2025 | In TSLib | ✅ Yes | ❌ No | Yes | ✅ DONE (harness failure) | tslib_sota | Already registered, had 0 records |
| **Koopa** | Koopa: Learning Non-stationary TS Dynamics with Koopman Predictors | NeurIPS 2023 | In TSLib | ✅ Yes | ❌ No | Yes | ✅ DONE (NaN divergence) | tslib_sota | Already registered, NaN issues |
| **CrossGNN** | CrossGNN: Confronting Noisy MTS with GNNs | ICLR 2024 | github.com/HCL212/CrossGNN | ❌ No | ❌ No | Yes | MEDIUM — GNN dependency | transformer_sota | GNN-based TS |
| **SageFormer** | SageFormer: Series-Aware Framework for Long-Term TS | arXiv 2023 | github.com/DSformer/SageFormer | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | Series-aware attention |
| **xLSTM-Mixer** | xLSTM-Mixer: Multivariate TS Forecasting by Mixing xLSTM | arXiv 2024 | Limited | ❌ No | ❌ No | Yes | LOW — xLSTM deps | transformer_sota | xLSTM variant already in NF |
| **Sumba** | Sumba: Summarized Mamba for TS | arXiv 2025 | github.com | ❌ No | ❌ No | Yes | LOW | transformer_sota | Mamba variant |
| **GLAFF** | GLAFF: Global-Local Attention-based Feature Fusion | arXiv 2024 | Limited | ❌ No | ❌ No | Yes | LOW | transformer_sota | Feature fusion |
| **TimeLinear** | TimeLinear: Linear Model for TS | arXiv 2024 | Available | ❌ No | ❌ No | No | HIGH — simple linear | deep_classical | Very simple baseline |
| **UMixer** | UMixer: Unfolding Mixer for TS | arXiv 2024 | Available | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | MLP-mixer variant |
| **DSformer** | DSformer: Dual-Scale Transformer | arXiv 2024 | Available | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | Dual-scale attention |
| **MTS-Mixers** | MTS-Mixers: Multivariate TS Mixing Architecture | arXiv 2023 | Available | ❌ No | ❌ No | Yes | MEDIUM | transformer_sota | Multi-variate mixer |
| **SparseTSF** | SparseTSF: Modeling Long-term TS Forecasting with 1k Params | ICML 2024 | github.com/lss-1138/SparseTSF | ❌ No | ❌ No | No | HIGH — 1k params | deep_classical | Ultra-compact |
| **CMoS** | CMoS: Conditional Mixture of Softmax | arXiv 2024 | Limited | ❌ No | ❌ No | Yes | LOW | transformer_sota | Mixture model |
| **RAFT** | RAFT: Retrieval Augmented TS Forecasting | arXiv 2025 | No code yet | ❌ No | ❌ No | Yes | LOW — no code | transformer_sota | Same as TS-RAG above |
| **Toto** | Toto: Time Series Optimized Transformer for Observability | arXiv 2024 (2407.07874) | ❌ Private (Databricks, 401) | TSLib lists it but no model file | ❌ No | Yes | ❌ NOT FEASIBLE | foundation | Private/gated, no public access |

---

## P2 — LLM / MULTIMODAL

| Model | Paper | Year/Venue | GitHub | In TSLib? | In NF? | GPU? | Feasibility | Category | Notes |
|-------|-------|-----------|--------|-----------|--------|------|-------------|----------|-------|
| **Aurora** | Aurora: Weather/Climate Foundation Model | Nature 2025 | github.com/microsoft/aurora | ❌ | ❌ | Yes | ❌ LOW — weather-specific, not general TS | foundation | Not for general time series |
| **TimeOmni1** | TimeOmni: A Multi-modal Time Series Foundation Model | arXiv 2025 (2502.15638) | github limited | ❌ | ❌ | Yes | LOW — early preprint | foundation | Multi-modal TS |
| **CALF** | CALF: Aligning LLMs for Time Series Forecasting | arXiv 2024 | github.com/Hank0626/CALF | ❌ | ❌ | Yes | MEDIUM — needs LLM | foundation | LLM-to-TS alignment |
| **TimeCMA** | TimeCMA: Cross-Modal Alignment for TS | arXiv 2024 | Limited | ❌ | ❌ | Yes | LOW | foundation | Cross-modal |
| **ChatTS** | ChatTS: LLM-Enhanced TS Forecasting | arXiv 2025 | Limited | ❌ | ❌ | Yes | LOW | foundation | Chat-based TS |
| **TimeCopilot** | TimeCopilot: System for TS Analysis with LLMs | arXiv 2025 | Limited | ❌ | ❌ | Yes | LOW — systems paper | foundation | Not a model per se |
| **TimerBed** | TimerBed: Benchmark testbed for Timer family | arXiv 2025 | Part of thuml ecosystem | ❌ | ❌ | Yes | MEDIUM — benchmark tool, not a model | N/A | Evaluation framework |
| **Samay** | Samay: Time Series Foundation Model for India | arXiv 2025 | Limited | ❌ | ❌ | Yes | LOW | foundation | Regional focus |

---

## BENCHMARKS

| Benchmark | URL | Description | Relevant? |
|-----------|-----|-------------|-----------|
| **GIFT-Eval** | https://github.com/SalesforceAIResearch/gift-eval | General Time Series Forecasting Eval (Salesforce) | ✅ HIGH — flagship TS benchmark, leaderboard on HF |
| **ProbTS** | https://github.com/microsoft/ProbTS | Probabilistic TS benchmark (Microsoft) | ✅ MEDIUM — CRPS-focused |
| **FoundTS / TSFM_Bench** | ibm-granite/granite-tsfm | IBM's evaluation suite for foundation models | ✅ Already use TTM from this |
| **FinTSB** | Various financial TS papers | Financial time series benchmark | MEDIUM — domain-specific |
| **TSLib** | https://github.com/thuml/Time-Series-Library | THUML benchmark suite, 11.7k stars | ✅ PRIMARY — already use extensively |
| **BasicTS** | https://github.com/zezhishao/BasicTS | Basic Time Series benchmark | MEDIUM — clean but overlaps TSLib |
| **OpenLTM** | https://github.com/thuml/OpenLTM | THUML Open Large Time Series Models | MEDIUM — pretrain/finetune focus |
| **Awesome-AI-for-TS** | github.com/qingsongedu/awesome-AI-for-time-series | Paper collection | Reference only |
| **TSFpaper** | Various | TS forecasting paper compilation | Reference only |
| **Awesome-TS-Forecasting** | github.com | Another paper collection | Reference only |

---

## CONSOLIDATED PRIORITY RECOMMENDATION

### Tier A: ADD NOW (high feasibility, strong baselines, code available)

| # | Model | Category | Source | Effort |
|---|-------|----------|--------|--------|
| 1 | **TabPFN-TS** | foundation | `pip install tabpfn` + custom wrapper | LOW |
| 2 | **ModernTCN** | transformer_sota | TSLib-style standalone | LOW |
| 3 | **FITS** | transformer_sota | TSLib-style standalone (10k params!) | LOW |
| 4 | **SparseTSF** | deep_classical | TSLib-style standalone (1k params!) | LOW |
| 5 | **Pathformer** | transformer_sota | Clean TSLib-compatible code | LOW |
| 6 | **CycleNet** | transformer_sota | TSLib-style code, NeurIPS 2024 | LOW |
| 7 | **TimeBridge** | transformer_sota | TSLib-style code, ICML 2025 | LOW |
| 8 | **CARD** | transformer_sota | ICLR 2024, TSLib-compatible | LOW |
| 9 | **CATS** | transformer_sota | ICML 2024, TSLib-compatible | LOW |
| 10 | **TimeMixer++** | transformer_sota | Extension of existing TimeMixer | MEDIUM |

### Tier B: ADD IF TIME PERMITS

| # | Model | Category | Reason |
|---|-------|----------|--------|
| 11 | **DUET** | transformer_sota | NeurIPS 2024, needs integration |
| 12 | **xPatch** | transformer_sota | Explainable patching |
| 13 | **CALF** | foundation | LLM-TS alignment, needs large LLM |
| 14 | **Fredformer** | transformer_sota | Frequency debiasing |
| 15 | **S-Mamba** | transformer_sota | Needs mamba-ssm CUDA kernel |
| 16 | **CrossGNN** | transformer_sota | GNN dependencies |
| 17 | **DeformableTST** | transformer_sota | Deformable attention |
| 18 | **PDF** | transformer_sota | Point-wise decomposition |

### Tier C: SKIP / DEFER

| Model | Reason |
|-------|--------|
| **Kairos** | No public repo or weights found |
| **SEMPO** | Cannot identify — likely misspelled or unpublished |
| **TS-RAG/RAFT** | No public code (preprint only) |
| **Toto** | Private/gated (Databricks, 401 on HF) |
| **Aurora** | Weather-specific, not general TS |
| **MMPD, interPDN, TimeMosaic, CFPT, TQNet, PIR, SRSNet, COSA** | Cannot find public repos/papers |
| **TimePerceiver, TimeRecipe** | Limited code, requires pretraining |
| **ChatTS, TimeCopilot, TimerBed, Samay** | Not models or too niche |
| **Bi-Mamba, xLSTM-Mixer, Sumba, GLAFF, FilterTS, HCAN, CMoS** | Limited public code |
| **TimeLinear, UMixer, DSformer, MTS-Mixers** | Lower priority, incremental |

---

## KEY HYPERPARAMETERS FOR GRID SEARCH

### For TSLib-style models (applicable to Tier A additions):
```yaml
common:
  seq_len: [96, 192, 336, 512]
  pred_len: [7, 14, 30, 60]  # our horizons
  batch_size: [32, 64]
  learning_rate: [0.0001, 0.001]
  train_epochs: [10, 20]
  d_model: [128, 256, 512]
  d_ff: [256, 512]
  n_heads: [4, 8]
  e_layers: [2, 3]

# Model-specific:
ModernTCN:
  kernel_size: [51, 101]
  large_size: [51, 101]
  small_size: [5, 11]

FITS:
  # Only ~10k params, limited hyperparams
  individual: [true, false]
  cut_freq: [10, 20, 50]

SparseTSF:
  # ~1k params
  period_len: [24, 48]

TabPFN-TS:
  # Zero-shot, no training needed
  n_lags: [30, 60, 90]
  use_temporal_features: [true, false]
```

---

## IMPLEMENTATION APPROACH

For Tier A models, the recommended approach is:
1. **Clone to vendor directory**: `/mnt/aiongpfs/projects/eint/vendor/<ModelName>`
2. **Add to `tslib_models.py`**: Use existing `_tslib_factory()` pattern with `TSLIB_MODEL_CONFIGS` dict
3. **Register in `TSLIB_MODELS`**: Add factory function to registry dict
4. **Test**: Run smoke test via `scripts/run_block3_benchmark_shard.py`

For non-TSLib models (TabPFN-TS), create a dedicated wrapper in `deep_models.py` or a new file.
