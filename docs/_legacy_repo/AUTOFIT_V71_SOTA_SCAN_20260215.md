# AutoFit V7.1 SOTA Scan and First Empirical Revision (2026-02-15)

Date: 2026-02-15
Scope: Time-series foundation models, tree/tabular SOTA, LLM-enhanced TS modeling, RL-style optimization ideas

## 1. Verified Sources (Primary)

### Time-Series Foundation / Transformer

- Chronos (Amazon): https://arxiv.org/abs/2403.07815
- Chronos-2 (Amazon): https://arxiv.org/abs/2510.15821
- Moirai2 (Salesforce): https://arxiv.org/abs/2410.10469
- Moirai-MoE: https://arxiv.org/abs/2502.20177
- Time-MoE: https://arxiv.org/abs/2409.16040
- TimesFM (decoder-only): https://arxiv.org/abs/2310.10688
- TimeOmni-1: https://arxiv.org/abs/2502.15638
- TiRex: https://arxiv.org/abs/2505.23719
- Retrieval-Augmented Forecasting: https://arxiv.org/abs/2505.04163
- Time-LLM: https://arxiv.org/abs/2310.01728

### Tree / Tabular

- XGBoost official release notes: https://xgboost.readthedocs.io/en/stable/changes/
- XGBoost 3.2 release page: https://xgboost.readthedocs.io/en/stable/changes/v3.2.0.html
- LightGBM releases: https://github.com/microsoft/LightGBM/releases
- CatBoost paper (ordered boosting): https://arxiv.org/abs/1706.09516
- TabPFN docs (PriorLabs): https://priorlabs.ai/docs/getting-started
- TabPFN on HuggingFace (tabpfn-v2-reg): https://huggingface.co/Prior-Labs/TabPFN-v2-reg

### RL / Policy-style optimization (TS-related)

- ReAugKD (RL for retrieval-augmented KD): https://arxiv.org/abs/2502.19634
- TimeR1 (reasoning TS benchmark): https://arxiv.org/abs/2506.09254

Notes:
- User-provided links `https://arxiv.org/abs/2506.10630` and `https://arxiv.org/abs/2508.07481` did not resolve at scan time.
- Keep RL integration constrained to train/validation feedback only (no test feedback loops).

## 2. Directly Actionable Components for AutoFit

### A. Lane-specialized experts (aligned with MoE trend)

- Keep lane routing (`binary`, `count`, `heavy_tail`, `general`).
- In count lane, preserve at least one count-specialist candidate (`XGBoostPoisson` / `LightGBMTweedie`) when near-tied on OOF MAE.
- Objective-specific meta learning remains enabled (`binary/logistic`, `count/poisson`, `heavy_tail/huber`).

### B. Retrieval-style regime descriptors (aligned with TS-RAG)

- Keep train-only prototype retrieval features.
- Do not use any test-target information.

### C. Count-target calibration (tree/foundation hybrid robustness)

- Apply deterministic lane-aware prediction postprocess for count/binary/heavy-tail.
- Ensure quick screen, repeated CV, and final prediction use the same postprocess policy.

## 3. First Empirical Revision Implemented (Round-1)

Code changes are now in place for AutoFit V7.1:

1. Lane-aware deterministic postprocess state from train targets only.
2. Unified postprocess in:
   - quick screen candidate evaluation
   - repeated temporal CV candidate evaluation
   - L2 blender validation path
   - final inference path
3. Count-lane anti-collapse rule:
   - force-keep one count-specialist model if within 2% OOF MAE degradation.

Expected impact:

- Improve `investors_count` robustness and reduce collapse to a single non-count model.
- Reduce prediction path mismatch between selection-time MAE and inference-time MAE.

## 4. Next Validation Plan

1. Run preflight gate (`preflight_block3_v71_gate.sh`).
2. Run Stage-A pilot with comparability filters on.
3. Compare V7.1 vs V7 at condition level:
   - GlobalNormalizedMAE
   - WinRate
   - investors_count gap median
4. Promote to full run only if pilot gates pass.

