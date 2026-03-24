# Block 3 Champion Component Analysis

> Date: 2026-03-23 (update from 2026-03-22; Phase 15+H, benchmark 15,888 raw records, 92 active models)
> Scope: Research-grade dissection of 9 champion models across 160 evaluation cells
> Purpose: Inform V740+ AutoFit design — core mechanism selection, routing logic, feature interaction

> Previous version (2026-03-12): analyzed 8,660 records, 93 models, 104 conditions.
> Update 2026-03-18: full 6-ablation benchmark surface with real text embeddings and seed2 replication.
> Update 2026-03-18b: 14,418 records (+66 from running jobs). Key finding: 87% task redundancy.
> **Update 2026-03-23**: 15,888 raw records. Finding H remains active: 8 P15 models excluded (100% constant predictions). 24 total excluded models, 92 active leaderboard models. Champion distribution remains stable; the main change is that V740 should now be designed as a single-model synthesis system rather than a deterministic oracle router.
> **Update 2026-03-24**: design implication tightened further. `core` should remain a regular state stream, but `edgar` and `text` should no longer be treated as ordinary dense columns in V740. The right next step is a source-aware dual-clock event-memory path: `core` on a daily state grid, `edgar/text` as sparse event memories with recent-token and recency-bucket views.

## 0. Benchmark Evolution: From Seed-Replication to Full 6-Ablation Surface

### Phase 9 → Phase 12+15 Changes
| Metric | Phase 9 (2026-03-12) | Phase 12+15 (2026-03-18) | Phase 15+H (2026-03-23) | Change |
|---|---|---|---|---|
| Total records | 8,660 | 14,418 | 15,888 | +83.5% total |
| Raw models in benchmark | 93 (incl. partial) | 114 | 137 (116 real + 21 retired) | +47.3% |
| Excluded models | ~7 | ~17 | **24** (Finding A-H + structural) | +243% |
| Active (leaderboard) models | ~86 | ~97 | **92** | net −5 (more exclusions) |
| Complete models (@160) | 78 | 69 | **64** active + 13 excluded = 77 raw | stricter |
| Ablations | 4 (but core_text≡core_only, full≡core_edgar) | 6 (real text embeddings, real EDGAR, real seed2) | 6 (unchanged) |
| Effective conditions | 104 (== 52 truly independent) | 160 (all independent) | 160 (unchanged) |
| Text embeddings | absent (silent fallback) | real: 5.77M rows, 64 PCA dims, float32, 0 NaN | real (unchanged) |
| Seed2 replication | derived from text≡core rename | independent SLURM seed2 runs | independent (unchanged) |
| Text effect | none (0 champion changes) | real: core_text wins 227 vs core_only wins 1,077 | unchanged |
| P15 new models | n/a | 23 submitted | 15 valid + **8 excluded (Finding H)** |

### Current Ablation Structure
- **core_only**: temporal features only (seed1)
- **core_only_seed2**: temporal features only (seed2) — independent replication
- **core_text**: temporal + 64-dim PCA text embeddings from business descriptions
- **core_edgar**: temporal + EDGAR SEC filing features
- **core_edgar_seed2**: temporal + EDGAR (seed2) — independent replication
- **full**: temporal + text + EDGAR (all features combined)

### 17 Valid Conditions
- task1_outcome × 6 ablations = 6 conditions (72 records per model: 6 × 3 targets × 4 horizons)
- task2_forecast × 6 ablations = 6 conditions (48 per model)
- task3_risk_adjust × 5 ablations = 5 conditions (40 per model; task3 has no core_only_seed2)
- **Total: 17 conditions × 160 records = 160 evaluation cells per model**

## 1. Champion Distribution Summary (Updated 2026-03-23, 64 Active Complete Models @ 160 Records)

15,888 raw records across 137 models (116 real + 21 retired AutoFit). 24 models excluded (Finding A-H + structural). 92 active leaderboard models.
64 active models complete at 160/160 records (77 raw including 13 excluded@160).
160 evaluation cells = 3 tasks × {6 ablations for task1/task2, 5 for task3} × 3 targets × 4 horizons.
**Fair comparison uses only the 64 active complete models.**

> **Note (2026-03-23)**: Champion distribution is STABLE from 14,418→15,888 records. The newly landed records come from gap-fill jobs (especially seed2 / EDGAR) and partial Phase 15 completion. No champion changes observed. The 9-champion table below remains authoritative.

| Rank | Model | Wins/160 | Pct | Category | Mean Rank | Primary Domain |
|---:|---|---:|---:|---|---:|---|
| 1 | NBEATS | 65 | 40.6% | deep_classical | 5.01 | funding h=1, investors h≥7, cross-task |
| 2 | Chronos | 23 | 14.4% | foundation | 10.55 | funding h=14/30 (EDGAR-conditioned) |
| 3 | NHITS | 21 | 13.1% | deep_classical | 4.38 | funding h=7, is_funded (EDGAR) |
| 4 | KAN | 16 | 10.0% | transformer_sota | 10.46 | investors h=1 (all ablations) |
| 5 | DeepNPTS | 16 | 10.0% | deep_classical | 19.55 | is_funded (task1 ONLY, all ablations) |
| 6 | GRU | 11 | 6.9% | deep_classical | 13.22 | funding h=14 (core_only/core_text) |
| 7 | PatchTST | 4 | 2.5% | transformer_sota | 4.28 | is_funded h=1/14 (EDGAR) |
| 8 | NBEATSx | 3 | 1.9% | deep_classical | 5.81 | funding h=1 (seed2 ablation) |
| 9 | DLinear | 1 | 0.6% | transformer_sota | 18.57 | is_funded h=7 (seed2, task1 only) |

### Key Changes from Phase 9 (8,660 records → 14,352 records)
1. **NBEATS dominance increased**: 41→65 wins (+58%). Gains all 17 seed2 conditions it was favored in.
2. **Chronos gains**: 17→23 wins (+35%). Seed2 conditions confirm its h=14/30 advantage with EDGAR.
3. **DeepNPTS anomaly**: 8→16 wins but ALL wins are **task1_outcome ONLY**. Zero wins on task2/task3. This is a critical specialization — DeepNPTS is not a general champion but a task1-specific specialist.
4. **GRU strengthened**: 5→11 wins. Gains across all 3 tasks, confirming h=14 core_only niche.
5. **PatchTST**: Unchanged at 4 wins but has the **BEST mean rank** (4.28) — most consistent model across all cells.
6. **NBEATSx**: 3 wins, all on seed2 ablations (core_edgar_seed2) — confirms NBEATSx ≡ NBEATS within GPU noise.
7. **DLinear**: 1 win, now on core_edgar_seed2 instead of full — confirms marginal champion status.

### Mean Rank vs Win Count Discrepancy
The most striking finding: **PatchTST has the best mean rank (4.28) but only 4 wins (2.5%)**. This means PatchTST is the most consistently good model (rarely bad) but rarely the absolute best. Conversely, DeepNPTS has 16 wins but mean rank 19.55 — brilliant on its niche but terrible elsewhere. This is the classic generalist-vs-specialist tradeoff that V740 routing must handle.

## 2. Per-Model Core Component Analysis

### 2.1 NBEATS — 65 wins (Basis Expansion + Double Residual)

**Architecture**: Stack of blocks with structured basis functions.

**Config** (`PRODUCTION_CONFIGS["NBEATS"]`):
- `input_size=60` (2 months context)
- `stack_types=["trend", "seasonality"]` — dual basis decomposition
- `max_steps=1000`, `batch_size=128`, `lr=1e-3`
- `scaler_type="robust"` (median + IQR normalization)
- `num_lr_decays=3` — scheduled learning rate reduction

**Core Components Driving Wins**:

1. **Trend Basis (polynomial)**: Learns polynomial coefficients for level/slope extrapolation. For `funding_raised_usd` at h=1, this captures the dominant slow-moving dynamics of funding amounts. The polynomial trend acts as an adaptive smoother that extrapolates the most recent trajectory.

2. **Seasonality Basis (Fourier)**: Captures periodic patterns (day-of-week, monthly cycles) in financial data. For `investors_count` at h=7/14/30, weekly and bi-weekly investor activity patterns are captured by harmonic decomposition.

3. **Double Residual Architecture**: Each block produces both a *backcast* (reconstruction of input) and a *forecast*. The residual is passed to the next block. This hierarchical refinement mechanism provides effective denoising — critical for noisy financial series.

4. **Robust Scaler**: Median + IQR normalization handles heavy-tailed distributions (`funding_raised_usd` has kurtosis=125, skew=10.35). Without this, gradient magnitudes would be dominated by a few extreme values.

5. **LR Decay (num_lr_decays=3)**: Scheduled reduction allows coarse-to-fine optimization — finds the right trend/seasonality decomposition in early steps, then fine-tunes in later steps.

**Why it wins on its specific conditions**:
- `funding h=1`: Trend stack provides 1-step extrapolation of polynomial trajectory — the simplest and most reliable short-horizon forecast for continuous financial data.
- `investors h=7/14/30`: Fourier seasonality captures weekly investor activity cycles. At medium-to-long horizons, the structured inductive bias (trend+seasonality) outperforms flexible models that may overfit to noise.

---

### 2.2 Chronos — 23 wins (Pre-trained Tokenized Decoder) [Updated: was 17 in Phase 9]

**Architecture**: T5 language model adapted for time series via value tokenization.

**Config** (`FoundationModelWrapper._load_chronos`):
- Model: `amazon/chronos-t5-small`
- Context: last 128 time steps per entity
- Inference: batch_size=32, predict h=7 steps, take median of probabilistic output
- Zero-shot: NO fine-tuning on domain data

**Core Components Driving Wins**:

1. **Value Tokenization (quantile binning)**: Continuous values → discrete tokens via learned quantile bins. This provides natural regularization against extreme values and makes the model distribution-agnostic — critical for heavy-tailed `funding_raised_usd`.

2. **Pre-trained Decoder**: Trained on millions of diverse time series, the T5 decoder has learned universal temporal patterns (trends, periodicities, mean reversion, regime changes). For longer horizons, these pre-trained priors compensate for the limited domain-specific training signal.

3. **Zero-shot Inference**: No fine-tuning means no overfitting to the training panel. When test entities have different dynamics than training entities, zero-shot provides more robust generalization.

4. **Probabilistic Output → Median Aggregation**: The model generates multiple sample paths; taking the median provides a robust point estimate that's less sensitive to outlier scenarios.

5. **Entity-level Context** (`ctx[-128:]`): Each entity's full history (up to 128 days) is used independently, capturing entity-specific level and dynamics without cross-entity contamination.

**Why it wins on its specific conditions**:
- `funding h=14/30` (with EDGAR) and `funding h=30` (all ablations): At 2-4 week horizons, local pattern extrapolation becomes unreliable (too much noise accumulation). Chronos's pre-trained distributional priors from millions of series provide a strong regularizing signal that keeps multi-step forecasts anchored. The tokenization removes scale sensitivity.
- **Lost h=14/core_only to GRU**: Without EDGAR features, GRU's compact hidden state (25K params) provides better regularization than Chronos's 60M-param pre-trained model. But with EDGAR, the richer feature conditioning tips the balance back to Chronos.
- Doesn't win at h=1/7 because fine-tuned models (NBEATS, NHITS) capture domain-specific short-range patterns more precisely than a zero-shot model.

---

### 2.3 NHITS — 21 wins (Hierarchical Interpolation + Multi-Resolution)

**Architecture**: Multi-scale blocks with MaxPool downsampling and interpolation upsampling.

**Config** (`PRODUCTION_CONFIGS["NHITS"]`):
- `input_size=60`, `max_steps=1000`, `batch_size=128`, `lr=1e-3`
- `stack_types=["identity", "identity", "identity"]` — 3 flexible stacks
- `scaler_type="robust"`

**Core Components Driving Wins**:

1. **MaxPool Downsampling**: Each stack processes the input at a different temporal resolution via max-pooling. Stack 1 sees daily-level patterns, Stack 2 sees 3-day-level, Stack 3 sees weekly-level. This hierarchical decomposition captures multi-scale temporal structure.

2. **Interpolation Upsampling**: After processing at reduced resolution, predictions are upsampled via interpolation to the target resolution. This smoothing effect acts as implicit regularization.

3. **Identity Stacks** (vs NBEATS's trend/seasonality): More flexible than structured bases — can learn arbitrary temporal patterns. The 3×identity configuration gives NHITS maximum expressiveness per block.

4. **Multi-Resolution Processing**: The combination of different-scale stacks means NHITS can simultaneously model day-level micropatterning and week-level macrodynamics.

**Why it wins on its specific conditions**:
- `funding h=7`: The hierarchical multi-resolution architecture is optimally suited for exactly 1-week forecasts where both daily and weekly patterns matter. The MaxPool downsampling at ~7-day scale directly picks up weekly periodicity.
- `investors_count h=1` (core_edgar only): With EDGAR features, NHITS's flexible identity stacks can capture the interaction between filing events and investor activity.
- `is_funded` (various): Identity stacks can approximate step-function-like binary transitions better than NBEATS's smooth trend/seasonality basis.

---

### 2.4 KAN — 16 wins (Kolmogorov-Arnold Learnable Activations)

**Architecture**: Replaces fixed activation functions (ReLU) with learnable B-spline functions on edges.

**Config** (`PRODUCTION_CONFIGS["KAN"]`):
- `input_size=60`, `max_steps=1000`, `hidden_size=256`, `batch_size=64`, `lr=1e-3`
- `scaler_type="robust"`

**Core Components Driving Wins**:

1. **Learnable Edge Functions (B-splines)**: Instead of fixed `ReLU(Wx+b)`, KAN learns `φ(x)` where φ is a B-spline basis function on each edge. This allows arbitrarily shaped nonlinear transformations — crucial for capturing the nonlinear relationship between time series history and predictions.

2. **Kolmogorov-Arnold Representation Theorem**: Any continuous multivariate function can be expressed as compositions of univariate functions. KAN directly parameterizes this decomposition, providing a theoretically universal approximation framework.

3. **Larger Hidden Size (256)**: Compared to standard temporal models, KAN uses a wider hidden layer to accommodate the B-spline basis parameters on each edge.

4. **Smaller Batch Size (64)**: More gradient updates per epoch allows finer optimization of the complex B-spline parameters.

**Why it wins on its specific conditions**:
- `investors_count h=1` (core_only, core_text, full, and all tasks): At h=1, predicting the immediate next count value requires capturing complex nonlinear threshold effects. Investor counts have discontinuous jump patterns (0→50→0→200) that are poorly captured by polynomial (NBEATS) or interpolation (NHITS) bases. KAN's learnable B-spline edge functions can approximate these arbitrary step-like transitions.
- Only at h=1: The advantage of flexible activations diminishes at longer horizons where structured inductive biases (NBEATS's trend/seasonality) provide better regularization.
- Not with core_edgar at h=1: When EDGAR features are included, NHITS (with identity stacks) captures the entity-level context better.

---

### 2.5 DeepNPTS — 16 wins (Non-Parametric Distribution-Free TS)

**Architecture**: Learns attention-like weights over past observations for prediction, without assuming any distributional form.

**Config** (`PRODUCTION_CONFIGS["DeepNPTS"]`):
- `input_size=60`, `max_steps=1000`, `batch_size=64`, `lr=1e-3`
- `scaler_type="robust"`

**Core Components Driving Wins**:

1. **Non-Parametric Prediction**: Instead of learning a parametric mapping f(x)→ŷ, DeepNPTS learns a weighting function over historical observations: ŷ = Σᵢ wᵢ(context) · yᵢ. This is a weighted average of past values where weights are context-conditioned.

2. **Distribution-Free**: Makes NO assumption about the target distribution. For binary `is_funded` (Bernoulli), this is critical — Gaussian-based models impose a continuous distributional assumption on inherently discrete data.

3. **Context-Conditioned Attention Weights**: The learned weights adapt to the current temporal context, enabling the model to "attend" to the most relevant past observations. For binary data, this naturally produces probability-like outputs.

4. **Mixture Density Output**: The probabilistic variant models a mixture of historical values, providing well-calibrated predictive uncertainties.

**Why it wins on its specific conditions**:
- `is_funded` (core_only, core_text — all 4 horizons): On pure temporal signals without EDGAR, DeepNPTS outperforms because:
  1. Binary target requires distribution-free modeling — Gaussian-based models (NBEATS/NHITS) produce values outside [0,1]
  2. The non-parametric weighting over past binary values naturally produces probabilities
  3. With only temporal features, the context-conditioned attention focuses on recent binary state transitions
- Doesn't win when EDGAR features are available (core_edgar, full) because PatchTST's attention mechanism can better leverage the richer feature space.

---

### 2.6 PatchTST — 4 wins (Patching + Multi-Head Self-Attention)

**Architecture**: Divides input into overlapping patches and processes with Transformer encoder.

**Config** (`PRODUCTION_CONFIGS["PatchTST"]`):
- `input_size=64`, `max_steps=5000`, `hidden_size=128`
- `n_heads=16`, `patch_len=16`, `stride=8`
- `batch_size=64`, `lr=1e-4`, `scaler_type="robust"`

**Core Components Driving Wins**:

1. **Patching (patch_len=16, stride=8)**: Input is divided into overlapping patches of 16 days with 8-day stride. This creates 7 patches from 64-day input. Each patch captures ~2-week local structure while overlap ensures continuity.

2. **Multi-Head Self-Attention (16 heads)**: 16 attention heads capture diverse temporal relationships between patches. Some heads may focus on trend alignment, others on periodicity matching, others on anomaly detection.

3. **Channel Independence**: Each entity is processed independently — no cross-entity interference. Essential for heterogeneous financial entities.

4. **Extended Training Budget (max_steps=5000)**: PatchTST trains with a higher step budget than most models, allowing the Transformer to learn complex patch-level relationships without underfitting.

5. **Lower Learning Rate (1e-4)**: 10× slower than NBEATS (1e-3). Prevents catastrophic forgetting during the longer training and provides more stable optimization of attention weights.

**Why it wins on its specific conditions**:
- `is_funded` (core_edgar, full — h=1, h=14): PatchTST wins on binary targets specifically when EDGAR features are available:
  1. The 16 attention heads can capture long-range dependencies between EDGAR filing events and subsequent funding outcomes
  2. Patching creates local windows that capture pre/post-filing behavior patterns
  3. The extended training (3000 steps) allows learning from the enriched EDGAR signal
  4. Channel independence prevents EDGAR-rich entities from contaminating EDGAR-poor ones
- Doesn't win on core_only/core_text because without EDGAR, the simpler DeepNPTS is sufficient for binary temporal dynamics.

---

### 2.7 NBEATSx — 3 wins (Basis Expansion + Exogenous Support)

**Architecture**: NBEATS with additional exogenous feature input pathway.

**Config** (`PRODUCTION_CONFIGS["NBEATSx"]`):
- Same as NBEATS: `input_size=60`, `stack_types=["trend","seasonality"]`
- PLUS: accepts `stat_exog_list` (static entity-level covariates from EDGAR)

**Core Components Driving Wins**:

1. **All NBEATS Components** (trend basis, seasonality basis, double residual)
2. **Exogenous Feature Integration**: EDGAR entity-level features (CIK-matched, quarterly) are fed as static covariates, conditioning the basis expansion on entity context.
3. **Static Covariate Pathway**: Entity-level features influence the basis function coefficients, allowing the same trend/seasonality decomposition to adapt per entity.

**Why it wins on its specific conditions**:
- `funding h=1` (full ablation — all 3 tasks): NBEATSx wins EXACTLY when ALL features (core+text+EDGAR) are available at h=1. The exogenous pathway means EDGAR features directly condition trend/seasonality coefficients.
- Only at h=1: EDGAR-conditioning provides maximum value for immediate-step extrapolation. At longer horizons, Chronos's pre-trained priors dominate.
- Only with full ablation: Without EDGAR (core_only/core_text), there are no exogenous features to leverage, so NBEATS (without the x overhead) is better.

---

### 2.8 DLinear — 1 win (Decomposition + Linear Projection)

**Architecture**: Moving average decomposition + two separate linear layers.

**Config** (`PRODUCTION_CONFIGS["DLinear"]`):
- `input_size=60`, `max_steps=1000`, `batch_size=128`, `lr=1e-3`
- `moving_avg_window=25` (~monthly smoothing)
- `scaler_type="robust"`

**Core Components Driving Wins**:

1. **Moving Average Decomposition**: `trend = MovingAvg(x, 25)`, `seasonal = x - trend`. Separates slow-changing level from residual fluctuation. The 25-day window captures approximately monthly trends.

2. **Dual Linear Projection**: Separate linear layers for trend and seasonal components. Total parameters = 2 × input_size × horizon. Extremely lightweight.

3. **Simplicity Advantage**: For near-deterministic targets like `is_funded` (MAE≈0.032), a simple model avoids overfitting. The linear projection is the minimum-variance unbiased estimator when the true relationship is approximately linear.

**Why it wins on its specific condition**:
- `is_funded h=7` (full, task1 only): This single win occurs because:
  1. Binary `is_funded` at 1-week horizon with full features is a near-deterministic problem
  2. Decomposition separates the stable funding status (trend) from filing-event noise (seasonal)
  3. DLinear's extreme simplicity (linear projection) avoids overfitting that more complex models may suffer on this low-variance target
  4. The 25-day moving average captures the approximately monthly reporting cycle

---

### 2.9 GRU — 11 wins (Gated Recurrent Hidden State) [Updated: 2026-03-18, was 5 in Phase 9]

**Architecture**: Gated recurrent unit with update/reset gates controlling information flow.

**Config** (`PRODUCTION_CONFIGS["GRU"]`):
- `input_size=60`, `max_steps=1000`, `batch_size=128`, `lr=1e-3`
- `encoder_hidden_size=64`, `encoder_n_layers=2`
- `scaler_type="robust"`

**Core Components Driving Wins**:

1. **Update Gate (z_t)**: Controls how much of the previous hidden state carries forward. For slow-changing `funding_raised_usd`, the update gate learns a near-identity mapping (z≈1), preserving the most recent level estimate with minimal decay. This is functionally equivalent to an exponential smoothing with learned decay rate.

2. **Reset Gate (r_t)**: Controls how much prior state influences the candidate activation. For 2-week (h=14) forecasting, the reset gate can "forget" stale information from >14 days ago, focusing on the most recent trajectory.

3. **Compact Parameter Space**: With `hidden_size=64` and 2 layers, GRU has ~25K parameters — orders of magnitude fewer than Chronos (60M) or PatchTST (1.2M). At h=14 on core_only, this extreme parsimony prevents overfitting to the noise in ~22K entity time series.

4. **Hidden State as Sufficient Statistic**: Unlike NBEATS (which processes the full 60-day window in parallel), GRU's recurrent processing compresses the history into a fixed-dimensional hidden state. For `funding_raised_usd` at h=14, the relevant information (recent level + trend direction) fits well in 64 dimensions.

**Why it wins on its specific conditions**:
- `funding_raised_usd h=14` (core_only, core_text only, all 3 tasks × 2 ablations = 5 conditions + task3's core_only = 5 total):
  1. At h=14, NBEATS's polynomial trend begins to accumulate extrapolation error (quadratic/cubic basis diverges), while Chronos' pre-trained prior hasn't yet fully dominated
  2. GRU occupies the "sweet spot" between structured extrapolation (NBEATS) and distributional prior (Chronos) — it learns the specific recurrent dynamics of funding trajectories
  3. GRU beats LSTM (runner-up at MAE=380,796 vs GRU's 380,654) because the reset gate provides a simpler forgetting mechanism — fewer parameters, same expressive power for this data
  4. **Critical**: GRU wins ONLY without EDGAR features. With EDGAR (core_edgar/full), Chronos takes over at h=14. This suggests that on core_only, GRU's compact hidden state is a better regularizer than Chronos' broad pre-training, but EDGAR features shift the information balance toward Chronos' ability to condition on richer context

**Margin analysis**: GRU's 0.037% average margin to LSTM (#2) is very tight — among the tightest champion margins. This reflects that h=14 is a competitive transition zone between recurrent and foundation approaches.

---

## 3. Cross-Cutting Component Analysis

### 3.0 Structural Observations (2026-03-18, 160 evaluation cells)

**Observation 1: Task Redundancy is 87% (CRITICAL FINDING)**
Direct MAE comparison of overlapping cells (same target × horizon × ablation) across tasks:
- 120 overlapping cell comparisons → 104 IDENTICAL MAE, 16 different
- **87% of task-overlapping cells produce identical MAE** — truly unique evaluation cells ≈72 (not 160)
- "Different" cases are mostly floating-point precision (e.g., 380926.295 vs 380926.294) or minor run-to-run variability
- Task3 OOD evaluation currently uses same test set as task1/task2 (no actual OOD slicing)
- **V740 implication**: 160-cell surface is inflated; real signal lives in ~72 unique cells. Task dimension adds minimal information except for binary targets where DeepNPTS is task1-specific.

**Observation 1b: Task-Specific Champion Anomaly**
Despite 87% redundancy, DeepNPTS exhibits a genuine task anomaly:
- DeepNPTS wins 16 conditions — ALL on task1_outcome, zero on task2/task3
- For other champions (NBEATS, Chronos, NHITS, KAN), task invariance holds strongly
- **V740 implication**: any clean adaptive system must treat task as a first-class conditioning signal for binary targets, even if most continuous/count cells remain largely task-invariant.

**Observation 2: Text Embeddings are Mostly Harmful**
With real text embeddings (64-dim PCA from business descriptions):
- core_text wins only 227 out of 1,932 pairs against core_only (11.7%)
- core_only wins 1,077 pairs (55.7%), with 628 ties (32.5%)
- **Text embeddings HURT more than they help** for the majority of models
- This is a CRITICAL finding: the 64-dim PCA embeddings add noise that degrades time series forecasting
- Strongest text beneficiary: AutoARIMA (avg delta = -0.00%) — effectively zero benefit
- Most models show 0.00% text effect because they either: (a) don't use the extra columns, or (b) the PCA dims add noise that washes out
- **V740 should NOT include text embeddings in its feature set unless embedding quality improves dramatically**

**Observation 3: EDGAR Effect is Mixed (Not Uniformly Helpful)**
- core_edgar wins 670 / 1,932 pairs vs core_only's 924 wins (34.7% vs 47.8%)
- EDGAR features slightly degrade overall performance
- However, full (text+EDGAR) wins 737 vs core_edgar's 607 → text+EDGAR combo slightly better than EDGAR alone
- **Full vs core_only**: full wins 870/1,932 (45.0%), core_only wins 811 (42.0%), ties 251 — closest margin of any ablation pair
- **EDGAR benefits are target-specific**: helps binary (is_funded) and some funding conditions, hurts count (investors)

**Observation 3b: Category Win Distribution**
- deep_classical: 97 wins (60.6%) — NBEATS(65)+NHITS(21)+GRU(11) dominate
- transformer_sota: 40 wins (25.0%) — KAN(16)+PatchTST(4)+DLinear(1)+others
- foundation: 23 wins (14.4%) — Chronos(23) provides all foundation wins

**Observation 4: Seed Reproducibility is Excellent**
- 1,344 paired cells between core_only and core_only_seed2
- Avg |MAE delta|: 0.138%, Median: 0.000%
- Champion rankings are highly stable across seeds
- Deterministic models: exact 0.00% delta (AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive)
- Neural models: <0.1% for most champions
- Most seed-UNSTABLE: CSDI (avg=4.42%, max=36.8%), iTransformer (avg=2.12%), BRITS (avg=1.25%)
- None of the unstable models are champions — safe to exclude from V740

**Observation 5: Horizon-Dependent Architecture Selection**
Mean rank by horizon for top models:
| Model | h=1 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| NBEATS | 3.40 | 3.67 | 4.28 | **8.70** |
| NHITS | 3.58 | **2.08** | 8.38 | 3.48 |
| PatchTST | 4.20 | 4.22 | 3.75 | 4.92 |
| NBEATSx | 4.15 | 4.42 | 5.12 | 9.55 |
| ChronosBolt | 9.47 | 8.93 | 6.53 | **4.78** |

Key pattern: NBEATS degrades at h=30 (8.70), ChronosBolt improves (4.78). The crossover is at h≈14. PatchTST is horizon-invariant (3.75-4.92).

**Observation 6: Target-Specific Specialization**
Mean rank by target:
| Model | funding_raised | investors_count | is_funded |
|---|---:|---:|---:|
| NBEATS | 7.93 | **1.65** | 6.29 |
| NHITS | 6.34 | 2.84 | **3.17** |
| PatchTST | **3.34** | 5.84 | **2.50** |
| Chronos | **2.87** | 16.53 | 15.38 |
| DeepNPTS | 37.99 | 7.19 | **2.33** |

- Chronos excels on funding_raised (2.87) but terrible on investors (16.53) and is_funded (15.38)
- DeepNPTS: catastrophic on funding (37.99) but best on is_funded (2.33)
- PatchTST: most balanced — top-4 on all 3 targets

### 3.1 Horizon-Component Mapping (Updated 2026-03-12)

| Horizon | Dominant Component | Mechanism | Champion(s) | Avg Margin | Condition |
|---|---|---|---|---|---|
| h=1 | Basis expansion (polynomial) | 1-step polynomial extrapolation | NBEATS | 0.006% | heavy_tail targets |
| h=1 | Learnable activations (B-spline) | Flexible nonlinear approximation | KAN | ~0.01% | count targets only |
| h=1 | Non-parametric / Attention | Distribution-free / EDGAR-conditioned | DeepNPTS / PatchTST | ~0.3% | binary (ablation-dependent) |
| h=7 | Hierarchical interpolation | Multi-resolution weekly capture | NHITS | ~0.008% | heavy_tail targets |
| h=7 | Basis expansion | Weekly cycle Fourier capture | NBEATS | ~0.005% | count targets |
| h=7 | Non-parametric / Hierarchical | Distribution-free / interpolation | DeepNPTS / NHITS | ~0.3% | binary (ablation-dependent) |
| h=14 | Gated recurrence | Hidden state as sufficient statistic | GRU | 0.037% | heavy_tail/core_only |
| h=14 | Pre-trained tokenization prior | Distributional regulation + EDGAR | Chronos | 0.108% | heavy_tail/core_edgar |
| h=14 | Basis expansion | Multi-step count extrapolation | NBEATS | ~0.005% | count targets |
| h=30 | Pre-trained tokenization prior | Multi-step uncertainty anchoring | Chronos | ~0.12% | heavy_tail targets |
| h=30 | Basis expansion | Long-range count cycles | NBEATS | ~0.006% | count targets |

**Key insight**: Short horizons reward domain-specific inductive biases (basis expansion, learnable activations). Long horizons reward cross-domain pre-trained priors (Chronos). **h=14 is a critical transition horizon** where the champion depends on the ablation class: GRU wins on core_only (compact regularization), Chronos wins with EDGAR (richer conditioning). This transition is the most valuable signal for V740 AutoFit routing.

### 3.2 Target-Type-Component Mapping (Updated 2026-03-12)

| Target Type | Best Component | Mechanism | Champion(s) | EDGAR Effect |
|---|---|---|---|---|
| continuous + heavy-tail (`funding_raised_usd`) | Trend/seasonality basis + robust scaler | Polynomial extrapolation + IQR normalization | NBEATS (h≤7), GRU/Chronos (h=14), Chronos (h=30) | −1.6% MAE improvement |
| count-like (`investors_count`) | Learnable B-spline activations (h=1), basis expansion (h>1) | Arbitrary nonlinear threshold + Fourier cycles | KAN (h=1), NBEATS (h>1) | +0.15% MAE degradation |
| binary (`is_funded`) | Non-parametric weighting OR attention | Distribution-free probability estimation | DeepNPTS (w/o EDGAR), PatchTST/NHITS (w/ EDGAR) | −1.8% MAE improvement |

**Key insights**:
1. Target type determines which architectural component matters most
2. **EDGAR effect is target-dependent**: helps funding (−1.6%) and is_funded (−1.8%) but marginally hurts investors (+0.15%). The SEC filing features are informative for dollar amounts and binary status but add noise for count predictions.
3. Binary targets exhibit the strongest ablation sensitivity: champions switch between 4 different models across ablations

### 3.3 Ablation-Component Mapping (Updated 2026-03-18, Real Text Embeddings)

| Ablation Transition | Champion Shift | Conditions Affected | Quantitative Effect |
|---|---|---|---|
| core_only → core_text | **MOSTLY HARMFUL** | text wins 227 / 1,932 (11.7%) | 55.7% of pairs degraded by text |
| core_only → core_edgar | **MIXED** | edgar wins 670 / 1,932 (34.7%) | 47.8% of pairs degraded by EDGAR |
| core_edgar → full | **SLIGHT BENEFIT** | full wins 737 / 1,932 (38.1%) | Text+EDGAR > EDGAR alone |
| core_only → core_only_seed2 | **NO CHAMPION CHANGE** | 0 systematic changes | Avg |delta| = 0.12%, seed-stable |
| All ablation-invariant | NBEATS, KAN (most conditions) | ~75% of cells | Architecture > features for temporal models |

**Major revision from Phase 9:**
1. **Text is now a real signal, but negative**: The Phase 9 analysis noted core_text ≡ core_only (text was missing). Now with real 64-dim PCA text embeddings, most models perform WORSE with text. This is likely because:
   - PCA text embeddings capture company description similarity, not temporal predictive signal
   - 64 extra dimensions increase input noise for models with limited capacity
   - Statistical and foundation models naturally ignore them (correct behavior)
2. **EDGAR benefits are concentrated**: EDGAR helps primarily for binary (is_funded) predictions where SEC filing dates carry information about corporate activity. For continuous and count targets, EDGAR adds marginally useful or harmful information.
3. **Seed2 confirms robustness**: The independent seed2 runs (different GPU, different SLURM allocation) confirm that champion designations are not artifacts of specific random seed choices.

### 3.4 Finding H: Broken Phase 15 Models (2026-03-22)

**8 out of 23 new TSLib models produce 100% constant predictions** — a systematic failure unrelated to data or hyperparameters.

| Model | Constant Prediction Warnings | Raw Records | Fairness Pass | Excluded Under |
|---|---|---|---|---|
| CFPT | 130 | partial | 0% | Finding H |
| DeformableTST | 127 | partial | 0% | Finding H |
| MICN | 105 | partial | 0% | Finding G (already) + H |
| PathFormer | 92 | partial | 0% | Finding H |
| SparseTSF | 91 | partial | 0% | Finding H (note: different from SparseTSF paper; implementation bug) |
| SEMPO | 91 | partial | 0% | Finding H |
| TimePerceiver | 85 | partial | 0% | Finding H |
| TimeBridge | 85 | partial | 0% | Finding H |

**Root Causes Identified**:
1. **CUDA OOM → degraded mode**: DeformableTST fails with "CUDA out of memory" on both V100 (20.69MiB free, 318MiB requested) and L40S (202.12MiB free, 634MiB requested). Even on H100, dimension mismatch (expected 60, got 169) causes silent degradation to constant output.
2. **Architecture adaptation bugs**: TSLib models expect fixed input/output shapes. Our NeuralForecast wrapper adapts shapes, but some models (CFPT, SEMPO) produce shape-invariant constant outputs regardless of input.
3. **Forward-only models (encoder-only)**: Despite our `forward(x)` patches for 8 encoder-only models, PathFormer and SparseTSF still collapse to constant predictions — the patch doesn't address deeper architectural incompatibilities with our panel data format.

**Impact on Champion Analysis**:
- These 8 models were never competitive (all would rank last on any cell due to constant predictions)
- Excluding them REMOVES noise from the benchmark surface and makes fair comparisons cleaner
- Champion distribution (9 champions, 160-cell allocation) is completely UNAFFECTED by their removal
- Active model count drops from 99→92 but complete@160 active models remain at 64

**Lessons for V740**:
- **Fairness check is mandatory**: Any model with >50% constant-prediction warnings should be automatically flagged and excluded
- **TSLib integration is fragile**: 8/23 failure rate (35%) — TSLib adapters require per-model verification
- **V740 model pool (7 models) is safe**: All 7 pool models (NBEATS, NHITS, Chronos, KAN, DeepNPTS, PatchTST, GRU) have 0% constant-prediction rate and pass all fairness checks

## 4. 2026-03-23 Superseding Interpretation: Beyond Structural Oracle Routing

The 160-cell champion map is still extremely valuable, but it should no longer
be interpreted as the final V740 recipe. What it gives us is a clean map of
which **mechanisms** win where. What it does **not** justify anymore is ending
the project with a deterministic structural router.

The most important strategic shift on 2026-03-23 is:

> the champion table should now be treated as a mechanism atlas for a
> single-model V740 student, not as a hand-written deployment oracle.

### 4.1 What remains true from the champion map

Several empirical patterns remain stable and should directly constrain V740:

1. **Heavy-tail funding cells remain horizon-structured**
   - `h=1`: NBEATS-like basis decomposition remains strongest
   - `h=7`: NHITS-style multi-resolution interpolation remains strongest
   - `h=14`: temporal-only cells remain compact-state sensitive (GRU-like), while richer EDGAR-conditioned cells shift toward Chronos-like long-horizon priors
   - `h=30`: long-horizon prior strength remains the decisive factor

2. **Count forecasting remains split by horizon**
   - `h=1`: KAN-like nonlinear response remains uniquely useful
   - `h>=7`: NBEATS-style structured decomposition still dominates

3. **Binary forecasting remains regime-dependent**
   - pure temporal binary cells still favor DeepNPTS-like calibration behavior
   - EDGAR-enriched binary cells still favor PatchTST/NHITS-like context sensitivity

4. **Text embeddings remain harmful under the current pipeline**
   - the current PCA text path should not be treated as a core V740 strength
   - it is a candidate for redesign, not a component to preserve blindly

5. **Seed2 stability remains strong**
   - the champion map is not being driven by random initialization noise
   - this makes it safe to design V740 around persistent champion mechanisms

### 4.2 Why a deterministic oracle is no longer the right end-state

The old structural-router framing is now too narrow for four reasons.

1. **The current valid AutoFit line is already validation-based, not oracle-based**
   - V739 no longer proves that a hard-coded table is the right answer
   - it proves that clean adaptive selection is viable without test leakage

2. **A deterministic router does not satisfy the real project objective**
   - the target is no longer just ``pick the right specialist''
   - the target is ``approach the specialist frontier with one efficient model''

3. **Ambiguous cells remain genuinely ambiguous**
   - the h=14 funding regime is still a close race between compact temporal regularization and stronger long-horizon prior structure
   - a static table is a helpful summary, but too brittle to be the final intelligence layer

4. **Task redundancy does not eliminate the need for internal conditioning**
   - even if 87\% of overlapping cells are redundant at the MAE level,
     target semantics, calibration needs, and feature interactions still differ
   - V740 should compress this structure internally rather than flatten it away

### 4.3 What V740 should do instead

V740 should now be designed around the following principles:

1. **single training graph**
2. **single inference path**
3. **condition-aware internal modulation**
4. **offline champion distillation**
5. **no runtime ensemble**
6. **no test-derived routing**

Concretely, the champion set now looks more like a decomposition of useful
inductive biases than a menu of final models:

| Champion family | Mechanism V740 should absorb |
|---|---|
| NBEATS / NBEATSx | basis decomposition, residual refinement, cheap short-horizon structure |
| NHITS | multi-resolution interpolation and weekly-scale smoothing |
| Chronos / ChronosBolt | robust long-horizon priors and scale-insensitive extrapolation |
| KAN | highly nonlinear short-horizon count response |
| DeepNPTS | binary/event-style calibration behavior |
| GRU | compact temporal-state regularization |
| PatchTST | EDGAR-sensitive local context mixing |

### 4.4 New 2024-2026 Research Signals That Matter for V740

The deeper post-2024 literature search strengthens this ``single-model
mechanism synthesis'' direction rather than weakening it.

1. **SAMformer (ICML 2024 oral)**
   - strongest signal for a lightweight, efficient, attention-based forecaster
   - relevant because V740 must be closer to a normal single model than to a multi-model controller
   - now locally integrated as the first new efficiency-focused benchmark addition

2. **UniTS (NeurIPS 2024)**
   - supports the idea that one condition-aware model can span multiple tasks and settings
   - directly relevant to V740's task/target/horizon/ablation token design

3. **ElasTST (NeurIPS 2024)**
   - strengthens the case for one model across multiple horizons rather than horizon-specific submodels

4. **DIAN (IJCAI 2024) and TimeEmb (NeurIPS 2025)**
   - both reinforce the need for invariant/variant and static/dynamic disentanglement
   - this is highly aligned with our mixed panel, EDGAR, and entity-context structure

5. **LightGTS (ICML 2025), OLinear (NeurIPS 2025), LiPFormer (ICDE 2025)**
   - these are the most important missing efficiency-first benchmark additions
   - all three are closer to the V740 target than another heavyweight foundation model

6. **Time-o1, DistDF, QDF, Selective Learning, and Abstain-Mask-Retain-Core**
   - these papers collectively suggest that the next gains may come as much from the objective as from the backbone
   - V740 should therefore combine backbone synthesis with stronger heavy-tail and multistep training objectives

7. **CASA / DERITS / QuantileFormer**
   - not the first benchmark additions we should prioritize
   - but they add useful signals for efficient attention, non-stationarity handling, and calibrated uncertainty

### 4.5 Updated V740 Design Recommendations

Based on the current 160-cell map, the current 15,888-record benchmark state,
and the new 2024-2026 research layer:

1. **Stop treating the deterministic oracle as the V740 end-state**
   - keep it only as a historical compression of champion structure

2. **Preserve the seven truly important champion mechanisms**
   - NBEATS, NHITS, Chronos, KAN, DeepNPTS, GRU, PatchTST
   - NBEATSx remains useful mainly as an exogenous variant of NBEATS, not as a separate essential family

3. **Treat text as redesign territory, not as a current strength**
   - current PCA text embeddings hurt too often to justify a first-class V740 branch

4. **Make condition tokens a first-class part of the model**
   - task, target, horizon, and ablation should modulate one shared trunk

5. **Use target-specific heads**
   - continuous head for funding
   - count-aware head for investors
   - explicitly calibrated binary head for is_funded

6. **Add static-dynamic / invariant-variant factorization**
   - this is now one of the strongest research-backed improvements for our data type

7. **Upgrade the objective, not only the architecture**
   - Time-o1 / DistDF / QDF-style components should be considered part of V740, not optional afterthoughts

8. **Benchmark additions should prioritize efficiency-first missing models**
   - first real addition: SAMformer
   - then: OLinear preprocessing path
   - then: LightGTS and LiPFormer
   - then second-wave additions such as UniTS / CASA / DIAN-oriented baselines if needed

9. **PatchTST remains the right generalist sanity anchor**
   - best mean rank, few wins, very robust
   - useful as the benchmark's generalist reference even if V740 should not merely imitate it

10. **The final V740 target is not “oracle accuracy”**
    - it is champion-level behavior under a single efficient forward path

---

## 5. Evidence Summary (Updated 2026-03-23)

All quantitative observations are derived from:

### Primary Data Sources
- **15,888 raw benchmark records** across 137 models via `scripts/aggregate_block3_results.py`
- **92 active (leaderboard) models** after 24 exclusions (Finding A-H + structural)
- **64 active complete models** (@160/160 records each) used for fair comparison
- **160 evaluation cells** = 3 tasks × {6/6/5 ablations} × 3 targets × 4 horizons (but ~72 truly unique due to 87% task redundancy)
- **Canonical output**: `runs/benchmarks/block3_phase9_fair/`
- **Champion analysis computed from**: 14,418 records / 69 complete@160 (as of 2026-03-18); stable through 15,888 records (2026-03-23)

### Ablation Effect Statistics (from 1,932 paired cells each)
- **Text effect** (core_text vs core_only): core_text wins 227 (11.7%), core_only wins 1,077 (55.7%), ties 628 (32.5%)
- **EDGAR effect** (core_edgar vs core_only): core_edgar wins 670 (34.7%), core_only wins 924 (47.8%)
- **Full vs core_only** (text+EDGAR vs baseline): full wins 870 (45.0%), core_only wins 811 (42.0%), ties 251
- **Full vs core_edgar** (text+EDGAR vs EDGAR): full wins 737 (38.1%), core_edgar wins 607 (31.4%)
- **Seed reproducibility**: avg |delta| = 0.138%, median = 0.000%
- **Task redundancy**: 87% of overlapping cells produce identical MAE across tasks

### Architecture References
- `src/narrative/block3/models/deep_models.py` — NeuralForecast configs (NBEATS, NHITS, KAN, GRU, PatchTST, DeepNPTS, NBEATSx, DLinear)
- `src/narrative/block3/models/tslib_models.py` — TSLib models (23 new Phase 15 models)
- `src/narrative/block3/models/registry.py` — unified model registry
- `src/narrative/block3/models/nf_adaptive_champion.py` — AutoFit V739 implementation

### Key Numerical Constants (for reproducibility)
- **Mean rank leaders**: PatchTST (4.28), NHITS (4.38), NBEATS (5.01), NBEATSx (5.81), ChronosBolt (7.42)
- **Win count leaders**: NBEATS (65), Chronos (23), NHITS (21), KAN/DeepNPTS (16 each), GRU (11)
- **Horizon crossover**: NBEATS dominates h=1,7 (rank 3.40, 3.67); degrades h=30 (8.70). ChronosBolt: opposite pattern (9.47→4.78)
- **Target specialization**: Chronos rank 2.87 on funding, 16.53 on investors. DeepNPTS rank 2.33 on is_funded, 37.99 on funding.
- **DeepNPTS task anomaly**: 16/16 wins on task1_outcome; 0/0 on task2/task3

---

## 6. Compute Cost Analysis & V740 Efficiency Design (Updated 2026-03-23)

### 6.1 Phase 12+15 Compute Investment

**Current benchmark surface (2026-03-23)**: 92 active models × 160 conditions = 14,720 model-condition slots
- 64 active complete (@160): 10,240 records ✅
- 1 AutoFitV739 (@120): 5 af739 s2/e2 gap-fill RUNNING (very slow, ~3 conds per 2d)
- 4 models @113 (ETSformer/LightTS/Pyraformer/Reformer): covered by accel_v2, e2=0 critical gap
- 4 models @109 (Crossformer/MSGNet/MambaSimple/PAttn): covered by accel_v2
- 15 valid P15 models @~67: accel_v2 RUNNING (51 scripts across gpu/l40s/hopper)
- XGBoost @159: 1 missing (t1/full/is_funded — structural OOM, UNFIXABLE)
- XGBoostPoisson @157: 3 missing (structural OOM, UNFIXABLE)
- 24 excluded models: NegativeBinomialGLM (structural), 7 Finding A-G models, 8 Finding H models, 8 other

**Active compute (2026-03-23)**: 58 jobs (27 RUNNING + 31 PENDING)
- npin GPU: accel_v2, af739, gpu_cos2_t2
- npin L40S: l2_ac_* scripts (17 total)
- npin Hopper: h2_ac_* scripts (17 total)

**Estimated total compute**: ~5,500 GPU-hours invested across Phase 9-15
- Phase 12 text reruns: ~800 GPU-hours (48 scripts)
- Phase 15 new models: ~1,200 GPU-hours and counting (accel_v2: 51 optimized scripts)
- Gap-fill (seed2, EDGAR): ~600 GPU-hours

### 6.2 V740 Fast Iteration Plan

With the 160-cell champion analysis complete and stable (9 champions confirmed across 14,418→15,888 records), V740 iteration can focus on:

**7-model express benchmark** (drop NBEATSx, DLinear):
| Parameter | Full Benchmark | V740 Express | Savings |
|---|---|---|---|
| Models | 92 active | 7 (NBEATS, NHITS, Chronos, KAN, DeepNPTS, PatchTST, GRU) | 92% |
| Ablations | 6 | 2 (core_only + core_edgar) | 67% |
| Conditions | 17 | ~6 (2 ablations × 3 tasks, skip seed2) | 65% |
| Total cells | 160 | ~42 | **74% savings** |
| Estimated time | 2-3 weeks | 4-8 hours | 50×+ faster |

**When to expand**:
1. V740 single-model synthesis design validated on 7-model × 2-ablation express benchmark → if the champion map changes materially, re-open the mechanism set
2. Full 92-model benchmark only needed for final paper tables
3. 15 valid P15 models may reveal new champions → wait for completion before V740 finalization (Finding H removed 8 broken models, leaving 15 valid candidates)
4. accel_v2 scripts (51 jobs) cover remaining gap-fill — estimated completion 5-7 days

### 6.3 Unstable Models: Current Population

Models with high training variance from seed2 analysis:
- Deterministic models (exact 0.00% delta): StatsForecast, ML tabular, Foundation zero-shot
- Neural models with <0.1% delta: NBEATS, NHITS, GRU, PatchTST — champion-stable
- High-variance models: CSDI, iTransformer, BRITS — none are champions, safe to exclude from V740

---

## 7. Appendix: Full 160-Cell Champion Table

| # | Task | Target | Ablation | Horizon | Champion | MAE | Runner-Up | Runner-Up MAE | Gap% |
|---|---|---|---|---|---|---|---|---|---|
| See `runs/benchmarks/block3_phase9_fair/` consolidated data — 160 benchmark cells under the current 6-ablation Phase 12+15 surface |

*Full table omitted for brevity. Generate the current champion pivot from the canonical Phase 9 fair benchmark root rather than relying on older 104-condition historical exports.*
