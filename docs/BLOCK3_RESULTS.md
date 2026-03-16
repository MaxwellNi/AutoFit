# Block 3 Results (Current)

> Last verified: 2026-03-16
> Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`

This file reports only the current clean Phase 9 / V739 benchmark reality.
The previous large static result tables were archived to:
- `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md`

## Current Snapshot

The current benchmark surface is verified from:
- `docs/benchmarks/phase9_current_snapshot.md`
- `runs/benchmarks/block3_phase9_fair/all_results.csv`
- `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw metrics files | 132 | direct scan 2026-03-16 |
| raw records | 13407 | direct scan 2026-03-16 (Phase 12 near-complete) |
| raw models | 91 | direct scan 2026-03-16 |
| raw complete models (≥104) | 80 | direct scan 2026-03-16 |
| raw partial models | 11 | direct scan 2026-03-16 |

## Current AutoFit Reality

| Fact | Value | Evidence |
| --- | --- | --- |
| active AutoFit baseline | `AutoFitV739` | Root `AGENTS.md` |
| canonical landed conditions | `112/112` (ALL COMPLETE) | direct scan: 12 metrics.json under `*/autofit/*` |
| ablation breakdown | co=28, ce=28, ct=28, fu=28 | direct scan |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| mean rank score | 14.38 | lower is better |
| conditions won (champion) | 3/56 (1 per task) | best MAE in that condition |
| top-5 by mean rank | NHITS (#1, 4.21), PatchTST (#2, 4.36), NBEATS (#3, 4.77), NBEATSx (#4, 5.84), ChronosBolt (#5, 7.11) | 56 universal conditions |

## How to Interpret the Current Benchmark Surface

1. The canonical benchmark root is `runs/benchmarks/block3_phase9_fair/`.
2. The current physical Phase 9 result surface still reflects the seed-replication reinterpretation:
   - `core_only`
   - `core_only_seed2`
   - `core_edgar`
   - `core_edgar_seed2`
3. Text embedding artifacts fully functional. Phase 12 text reruns: 42/48 COMPLETED, 6 RUNNING (cfisch tslib only).
4. `core_text` now covers **91/91** models, `full` covers **91/91** models (NegBinGLM has partial records, structural failure).
5. Current champion model: **NBEATS** — 24/56 conditions won (43%), dominant across all 3 tasks.
6. Champion per task (top contributors): NBEATS(8), NHITS(5+), KAN(5), DeepNPTS(4), GRU(3), V739(3), PatchTST(2), Chronos(2+).

## Where to Inspect Actual Results

1. Current filtered leaderboard:
   - `runs/benchmarks/block3_phase9_fair/all_results.csv`
2. Current benchmark interpretation manifest:
   - `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`
3. Current fact snapshot:
   - `docs/benchmarks/phase9_current_snapshot.md`
4. Current model status summary:
   - `docs/BLOCK3_MODEL_STATUS.md`

## What Is No Longer Current

The following are preserved for history or reference only and must not be used as current benchmark truth:
- `docs/_legacy_repo/`
- `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/`
- Phase 7 / Phase 8 results
- V72 / early V73 benchmark narratives
- V734-V738 empirical outputs or design narratives
