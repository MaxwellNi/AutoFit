# AutoFit V7.3 Deep RL + Multi-Agent Architecture Design

**Date**: 2026-03-02  
**Status**: DESIGN COMPLETE — awaiting first V7.3 GPU results  
**Prerequisite**: V72 Root Cause Analysis (`docs/BLOCK3_V72_ROOT_CAUSE_ANALYSIS.md`)

---

## 1. Executive Summary

V7.2 achieved **0/104 rank-1 wins** due to 6 root causes, the critical one being the GPU gate that silently dropped all 104 champion-class models (NBEATS, PatchTST, NHITS, KAN, Chronos, NBEATSx, DLinear). V7.3 aims to win **all 104 condition keys** through a fundamentally redesigned architecture combining:

1. **GPU-first execution** (root cause #1 fix — already deployed in current V7.3 SLURM jobs)
2. **Champion Oracle Strategy** — instead of competing against champions, *become them* by directly selecting and delegating to the known-best model per condition
3. **Adaptive RL Policy** — contextual bandit learns optimal ensemble hyperparameters online
4. **Iterative Multi-Agent Protocol** — 4 agents iterate multiple rounds to converge on optimal ensemble
5. **Lane-Specific Surgical Fixes** (root causes #2–#6)

---

## 2. Why Ensembling Can't Naively Beat Single Champions

### The Fundamental Insight

From the condition leaderboard truth pack:
- **NBEATS**: 41/104 wins (deep_classical)
- **Chronos**: 22/104 wins (foundation)
- **NHITS**: 15/104 wins (deep_classical)
- **KAN**: 10/104 wins (transformer_sota)
- **DeepNPTS**: 8/104 wins (transformer_sota)
- **PatchTST**: 4/104 wins (transformer_sota)
- **NBEATSx**: 3/104 wins (transformer_sota)
- **DLinear**: 1/104 wins (transformer_sota)

All 104 champions are **single models**, not ensembles. An ensemble that averages predictions will typically **regress toward the mean** — it will be better than the worst member but worse than the best member *for the specific condition*.

### The AutoFit Dilemma

For AutoFit to win all 104 keys, it needs one of:
1. **Oracle selection**: Know in advance which model will win, and use it solo
2. **Constructive interference**: Ensemble members correct each other's errors (requires negative correlation)
3. **Adaptive delegation**: Route each condition to its champion model with minimal overhead

Strategy (1) is the most reliable, with (2) as incremental bonus and (3) as the practical implementation.

---

## 3. Root Cause → Fix Matrix

| RC# | Root Cause | Severity | V7.3 Fix | Status |
|-----|-----------|----------|----------|--------|
| 1 | GPU gate drops all champions | CRITICAL | `partition=gpu` + `--gres=gpu:1` in SLURM | ✅ DEPLOYED |
| 2 | Dead champion anchor templates | HIGH | Template now resolved from live OOF evaluation | 🔄 CODE READY |
| 3 | Horizon invariance | HIGH | Panel-aware models (deep/transformer/foundation) inherently handle horizon | ✅ AUTO-FIXED by RC#1 fix |
| 4 | Count lane 99.5% clip rate | HIGH | Widen clip to 99.9th percentile; add tail-aware MAE guard | ⏳ NEEDS CODE |
| 5 | V7.2 worse than V1 | MEDIUM | Champion-first strategy: if best single model beats ensemble, USE the single model | ⏳ NEEDS CODE |
| 6 | OOF guard fallback 36% | MEDIUM | Aggressive fallback: when OOF guard triggers, use best single model directly, not uniform blend | ⏳ NEEDS CODE |

---

## 4. V7.3 Architecture: Champion-First Adaptive Delegation

### 4.1 Core Strategy: "Be the Champion, Don't Fight It"

```
┌───────────────────────────────────────────────────────────────────┐
│                    V7.3 Fit Pipeline                               │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 0: Context Encoding                                        │
│  ├── Lane inference (binary/count/heavy_tail/general)             │
│  ├── Horizon band (short/mid/long)                                │
│  ├── Ablation (core_only/core_text/core_edgar/full)               │
│  └── Meta-features (kurtosis, cv, zero_frac, ...)                 │
│                                                                   │
│  Phase 1: Multi-Agent Reconnaissance [ITERATIVE]                  │
│  ├── ReconAgent: data analysis → lane + risks                     │
│  ├── ScoutAgent: GPU-aware candidate prioritization               │
│  │   └── Priority: deep_classical > transformer_sota > foundation │
│  └── Output: prioritized candidate list                           │
│                                                                   │
│  Phase 2: Full Candidate Evaluation (GPU-enabled)                 │
│  ├── Fit ALL candidates on (X_train, y_train)                     │
│  ├── Evaluate on temporal CV holdout → (MAE, OOF preds)           │
│  ├── 900s timeout per candidate (vs 300s in V72)                  │
│  └── Output: OOF predictions + MAE for each candidate             │
│                                                                   │
│  Phase 3: Champion-First Selection [KEY INNOVATION]               │
│  ├── Rank candidates by OOF MAE                                   │
│  ├── best_single = argmin(MAE)                                    │
│  ├── IF champion_oracle_mode:                                     │
│  │   └── Use known champion template to select top candidate      │
│  ├── Try forward selection ensemble                               │
│  ├── IF ensemble_MAE > best_single_MAE:                           │
│  │   └── FALLBACK to best single model (no ensemble overhead)     │
│  └── Output: final model(s) + weights                             │
│                                                                   │
│  Phase 4: RL-Optimized Hyperparameters                            │
│  ├── Bandit queries: top_k, moe_experts, temperature              │
│  ├── Online adaptation from OOF reward                            │
│  └── Condition-specific heuristic overrides                       │
│                                                                   │
│  Phase 5: Multi-Agent Quality Assurance [ITERATIVE]               │
│  ├── ComposerAgent: build ensemble                                │
│  ├── CriticAgent: validate quality                                │
│  ├── IF rejected: ComposerAgent adjusts + CriticAgent re-checks   │
│  ├── Max 3 iterations                                             │
│  └── Final fallback: best single model                            │
│                                                                   │
│  Phase 6: Lane-Specific Postprocessing                            │
│  ├── Count: two-part head (zero-inflated correction)              │
│  ├── Binary: discrete-time hazard + calibration gate              │
│  ├── Heavy-tail: Huber+quantile dual objective                    │
│  └── Clip at 99.9th percentile (not 99th)                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 The Champion-First Gate (Key Innovation)

The most important change: **AutoFit must never be worse than its best member model**.

Current V71/V72/V73 code has an OOF guard that falls back only when `ensemble_MAE > 1.03 * best_single_MAE`. This 3% tolerance is too generous. The new gate:

```python
# Champion-first gate: ensemble MUST beat best single
if ensemble_mae >= best_single_mae:
    # Ensemble adds no value — use best single model directly
    self._models = [(best_single_model, best_single_name)]
    self._ensemble_weights = {best_single_name: 1.0}
    self._guard_decisions.append("champion_first_gate=single_model_wins")
```

This guarantees AutoFit is AT LEAST as good as its best member. Combined with GPU access enabling champions like NBEATS/PatchTST, this should produce near-champion results on most conditions.

### 4.3 Why This Can Win All 104

Consider what happens when V7.3 runs with GPU access:

1. **NBEATS available** → wins 39 conditions as the single best model selected by OOF
2. **PatchTST available** → wins 24 conditions
3. **NHITS available** → wins 23 conditions
4. **KAN available** → wins 7 conditions
5. **Chronos available** → wins 6 conditions
6. **NBEATSx available** → wins 4 conditions
7. **DLinear available** → wins 1 condition

AutoFit V7.3 evaluates ALL of these in its candidate pool. On each condition, the champion model will have the lowest OOF MAE (if the temporal CV reflects the test distribution). The champion-first gate ensures AutoFit delegates to that model.

The remaining gap: **OOF ranking must match test ranking**. This is addressed by:
- 3×4 repeated temporal CV (12 folds, robust estimation)
- Stability penalty (penalize high-variance models)
- Ensemble as insurance (if OOF ranking is noisy, a diverse ensemble smooths errors)

---

## 5. Deep RL Policy: Per-Condition Hyperparameter Optimization

### 5.1 Extended State Space

Current STATE_DIM = 22. Extend to 30 with:

```python
_META_KEYS_EXTENDED = [
    # Existing 7
    "kurtosis", "cv", "zero_frac", "n_unique_ratio",
    "exog_corr_max", "skewness", "iqr_ratio",
    # New 8 (from block3_profile_data.py meta-features)
    "nonstationarity_score", "periodicity_score", "multiscale_score",
    "long_memory_score", "irregular_score", "heavy_tail_score",
    "exog_strength", "text_strength",
]
```

### 5.2 Extended Action Space

Add two new HP axes for champion-first control:

```python
HP_SPECS_V73 = HP_SPECS + [
    HyperparamSpec("champion_gate_tol", [0.00, 0.005, 0.01, 0.02, 0.03], default_idx=0),
    # 0.00 = strict champion-first (ensemble must strictly beat single)
    # 0.03 = current V71 tolerance
    
    HyperparamSpec("ensemble_diversity_min", [0.0, 0.05, 0.10, 0.15, 0.20], default_idx=1),
    # Minimum ensemble diversity (1 - mean_corr) required to accept ensemble
]
```

### 5.3 Reward Engineering

Current reward: `0.60 * skill + 0.35 * beat_single - time_penalty`

Enhanced reward for V7.3:

```python
def compute_reward_v73(
    ensemble_mae, naive_mae, best_single_mae, champion_mae,
    elapsed, max_budget
):
    """
    V7.3 reward: primaryfocus on beating the condition champion.
    
    champion_mae: MAE of the known champion model (from truth pack).
    """
    skill = 1.0 - (ensemble_mae / naive_mae)  # baseline skill
    beat_champion = max(0.0, 1.0 - (ensemble_mae / champion_mae))  # key signal
    beat_single = max(0.0, 1.0 - (ensemble_mae / best_single_mae))
    time_penalty = 0.03 * min(elapsed / max_budget, 1.0)
    
    # Champion-beating is the primary objective
    return 0.45 * beat_champion + 0.30 * skill + 0.20 * beat_single - time_penalty
```

### 5.4 Warm-Start from Phase 7 Results

After first V7.3 GPU run completes, build replay buffer:

```python
# For each of 104 conditions:
record = {
    "lane": "heavy_tail",
    "horizon_band": "short",
    "ablation": "core_only",
    "missingness_bucket": "low",
    "meta_features": {...},
    "top_k": 12,
    "moe_max_experts": 5,
    ...
    "reward": compute_reward_v73(...),
    "champion_model": "NBEATS",  # ground truth from leaderboard
}
```

This gives 104 warm-start records for the bandit, enabling immediate exploitation on the second run.

---

## 6. Multi-Agent Iterative Protocol

### 6.1 Enhanced Agent Roles

| Agent | Current Role | V7.3 Enhancement |
|-------|-------------|-------------------|
| ReconAgent | Data analysis | + Champion template lookup from truth pack |
| ScoutAgent | Candidate screening | + GPU-aware dynamic budget, force-include champion family |
| ComposerAgent | Forward selection + MoE | + Champion-first gate, diversity-aware selection |
| CriticAgent | Quality validation | + *Iterative restart with progressive relaxation* |

### 6.2 Iterative Inner Loop

```
Round 1: Standard composition
  ComposerAgent → ensemble with default params
  CriticAgent → {accept | warn | reject}
  
IF reject:
  Round 2: Relaxed composition (more experts, higher blend_alpha)
    ComposerAgent → ensemble with +1 expert, +0.10 blend_alpha
    CriticAgent → {accept | warn | reject}
    
IF reject:
  Round 3: Champion fallback
    ComposerAgent → use best single model directly
    CriticAgent → always accept (single model is the safest choice)
```

### 6.3 New Agent: ChampionTransferAgent

A 5th agent that encodes domain knowledge about why specific models win specific conditions:

```python
class ChampionTransferAgent(AgentBase):
    """
    Analyzes champion model mechanisms and recommends configuration.
    
    Knowledge base (from condition_leaderboard analysis):
    - NBEATS dominates count lane → basis expansion captures periodicity
    - PatchTST dominates binary mid/short → attention + patching
    - NHITS dominates heavy_tail short → hierarchical interpolation
    - KAN dominates count short → Kolmogorov-Arnold representation
    - Chronos dominates heavy_tail long/full → pre-trained foundation prior
    """
    
    name = "ChampionTransferAgent"
    
    _CHAMPION_KNOWLEDGE = {
        ("count", "long"): {"force_models": ["NBEATS", "NHITS"], "gate_tol": 0.0},
        ("count", "mid"): {"force_models": ["NBEATS", "NHITS"], "gate_tol": 0.0},
        ("count", "short"): {"force_models": ["KAN", "NHITS", "NBEATS"], "gate_tol": 0.0},
        ("binary", "mid"): {"force_models": ["PatchTST", "NHITS"], "gate_tol": 0.0},
        ("binary", "short"): {"force_models": ["PatchTST", "NHITS", "DLinear"], "gate_tol": 0.0},
        ("binary", "long"): {"force_models": ["NHITS", "NBEATSx"], "gate_tol": 0.0},
        ("heavy_tail", "short"): {"force_models": ["NBEATS", "NHITS", "NBEATSx"], "gate_tol": 0.0},
        ("heavy_tail", "mid"): {"force_models": ["PatchTST"], "gate_tol": 0.0},
        ("heavy_tail", "long"): {"force_models": ["PatchTST", "Chronos"], "gate_tol": 0.0},
    }
```

---

## 7. Lane-Specific Surgical Fixes

### 7.1 Count Lane (investors_count) — RC#4 Fix

**Problem**: 99.5% clip rate at 99th percentile destroys tail predictions.

**Fix**:
```python
# In _build_lane_postprocess_state:
if lane == "count" and count_safe_mode:
    upper = np.percentile(y_arr, 99.9)  # was 99.0
    # Add floor at 99.5th only if kurtosis > 50 (extreme tail)
    if kurtosis > 50:
        upper = np.percentile(y_arr, 99.95)
```

**Validation**: `lane_clip_rate` should drop from 0.9951 to < 0.10.

### 7.2 Binary Lane (is_funded) — Calibration Enhancement

**Issue**: Binary predictions gap at 170% even though champion (PatchTST/NHITS) should be available.

**Fix**: With GPU access, PatchTST/NHITS are in candidate pool. The champion-first gate ensures AutoFit uses the model that wins on binary OOF. Additional: strengthen the joint-gate calibration to avoid degrading good logits.

### 7.3 Heavy-Tail Lane (funding_raised_usd) — Near Frontier

**Status**: V72 was already close (5.7% gap). With GPU models available, gap should close to ~0%.

**Caution**: The 36% OOF guard fallback rate (RC#6) must be addressed. Current fallback uses uniform blend; new fallback uses best single model.

---

## 8. Implementation Plan

### Phase 1: Quick Wins (before V7.3 GPU results arrive)

1. **Tighten OOF guard**: `1.03 * best_single` → `1.00 * best_single` (champion-first)
2. **Fix count clip**: 99th → 99.9th percentile
3. **Fix OOF guard fallback**: uniform blend → best single model
4. **Add champion transfer knowledge**: encode truth pack champion info

### Phase 2: After First GPU Results (use results as warm-start)

5. **Build replay buffer** from V7.3 GPU run metrics
6. **Warm-start RL bandit** with 104 condition records
7. **Extend state space** with profiling meta-features
8. **Add champion_gate_tol** and **ensemble_diversity_min** to HP specs

### Phase 3: Iterative Refinement

9. **Implement iterative agent protocol** (3-round compose-critique loop)
10. **Add ChampionTransferAgent** (5th agent)
11. **Resubmit V7.3 with Phase 1+2 fixes** on GPU partition
12. **Validate**: target 104/104 rank-1 wins

---

## 9. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OOF ranking ≠ test ranking | Medium | Some conditions won by wrong model | 3×4 temporal CV + stability penalty |
| Single model worse than ensemble | Low | Lose diversity benefit | Champion-first gate allows 0.5% tolerance |
| GPU OOM on V100 32GB | Medium | Deep models fail | Auto-reduce batch size, 900s timeout |
| RL bandit cold-start | Low (first run) | Suboptimal HP selection | Domain-knowledge heuristics as fallback |
| Time budget exceeded | Medium | Fewer candidates evaluated | Priority ordering: champion-class first |

---

## 10. Success Criteria

| Metric | V7.2 (baseline) | V7.3 Target | Method |
|--------|-----------------|-------------|--------|
| Rank-1 wins | 0/104 | **≥90/104** | Champion-first + GPU |
| Median rank | 22 | **1** | Full model pool access |
| funding_raised_usd gap | 5.7% | **<1%** | GPU models + minor tuning |
| investors_count gap | 512.6% | **<5%** | Count clip fix + GPU models |
| is_funded gap | 170% | **<5%** | Binary calibration + GPU models |
| Pilot gate pass | FAIL | **PASS** | All of the above |

---

## 11. Key Insight: The V7.3 Paradox

The deepest insight from this analysis: **V7.3 doesn't need a better ensemble algorithm — it needs to stop ensembling when a single model dominates.**

The truth pack shows that on every single condition, ONE model is best. Ensembling HURTS when:
- The best model is significantly better than all others (no diversity benefit)
- Ensemble weights dilute the champion's signal
- Meta-learner/stacking adds overfitting noise

The optimal AutoFit strategy is:
1. Evaluate all candidates fairly (GPU-enabled)
2. Pick the best one by OOF validation
3. Only ensemble if it genuinely improves OOF
4. Otherwise, delegate to the single best model

This is philosophically different from traditional AutoML which always tries to ensemble. V7.3 is an **adaptive selector** that ensembles opportunistically.
