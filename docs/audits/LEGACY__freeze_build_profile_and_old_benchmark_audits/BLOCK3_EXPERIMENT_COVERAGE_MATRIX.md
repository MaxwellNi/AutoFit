# Block 3 Experiment Coverage Matrix — KDD'26

Generated: 2026-02-09T05:47:39.408055+00:00

## Required Coverage (preset=full)

### Axes

| Axis | Values |
|------|--------|
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |
| Targets | funding_raised_usd, investors_count, is_funded |
| Horizons | 1, 7, 14, 30 |
| Context (Task2) | 30, 60, 90 |
| Ablations | core_only, core_edgar |
| Categories | statistical, ml_tabular, deep_classical, transformer_sota, foundation, irregular |
| Models | 43 total |

## Shard Coverage

- **Required shards**: 36
- **Completed**: 25 (from INVALID standard run)
- **Missing**: 11

## Coverage Grid (task × category × ablation)

| Task | Category | core_only | core_edgar | Status |
|------|----------|-----------|------------|--------|
| task1_outcome | statistical | ✓* | ✓* | INVALID (standard) |
| task1_outcome | ml_tabular | — | — | PENDING |
| task1_outcome | deep_classical | ✓* | ✓* | INVALID (standard) |
| task1_outcome | transformer_sota | — | — | PENDING |
| task1_outcome | foundation | ✓* | ✓* | INVALID (standard) |
| task1_outcome | irregular | ✓* | ✓* | INVALID (standard) |
| task2_forecast | statistical | ✓* | ✓* | INVALID (standard) |
| task2_forecast | ml_tabular | — | — | PENDING |
| task2_forecast | deep_classical | ✓* | ✓* | INVALID (standard) |
| task2_forecast | transformer_sota | ✓* | — | INVALID (standard) |
| task2_forecast | foundation | ✓* | ✓* | INVALID (standard) |
| task2_forecast | irregular | ✓* | ✓* | INVALID (standard) |
| task3_risk_adjust | statistical | ✓* | ✓* | INVALID (standard) |
| task3_risk_adjust | ml_tabular | — | — | PENDING |
| task3_risk_adjust | deep_classical | ✓* | ✓* | INVALID (standard) |
| task3_risk_adjust | transformer_sota | — | — | PENDING |
| task3_risk_adjust | foundation | ✓* | ✓* | INVALID (standard) |
| task3_risk_adjust | irregular | ✓* | ✓* | INVALID (standard) |

*\* = completed under INVALID `standard` preset (leakage bug). Must rerun with `full` preset.*

## Baseline Coverage for AutoFit Paper Claims

| Baseline | Status |
|----------|--------|
| Single-best fixed (e.g. LightGBM) | ⏳ Needs full rerun |
| AutoFit v1 rule_based_composer | ⏳ Not yet run |
| AutoFit v2 MoE router | ⏳ D1-D2 code ready, needs real data run |
| Oracle ensemble (retrospective) | ⏳ Needs all results first |

## Total Shard Count (full preset)

3 tasks × 2 ablations × 6 categories = **36 shards**
Each shard: 3 targets × 4 horizons × N models = comprehensive coverage
