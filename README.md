# AutoFit-TS

AutoFit-TS is an agentic temporal foundation framework for irregular, long-horizon
trajectories with heterogeneous exogenous signals. It automatically composes model
components to match dataset characteristics under a fixed compute budget, and it
produces auditable narrative indices through a concept bottleneck for faithful
explanations.

## Scope
- Data: irregular fundraising trajectories with static descriptors, multi-year
  snapshots, external filings, and narrative content
- Tasks: outcome prediction and trajectory forecasting with strict no-leakage
  evaluation
- Outputs: paper-ready tables, diagnostics, and structured explanation reports

## Core ideas
- Dataset-driven composition based on nonstationarity, long memory, multiscale
  periodicity, irregular sampling, and exogenous strength
- Multi-stream temporal backbone with exogenous conditioning (cross-attention,
  FiLM, bridge-token fusion)
- Concept bottleneck and additive concept models for auditable narrative indices
  and explanations

## Current status (recovery mode)
- Parquet-native pipeline, diagnostics, candidate composition, budget search, and
  report tables are implemented and tested
- EDGAR feature store, external dataset normalization, and multi-server sync
  templates are in place
- Concept bottleneck, NBI/NCI computation, and outcome models exist but need full
  tests
- Official B11 v2 grid is gated by strict-future validation; see `RUNBOOK.md` and
  `DECISION.md`

## Repo structure
- `src/narrative/` core pipeline, models, and explainability
- `scripts/` end-to-end runs, audits, and report generation
- `configs/` experiment and dataset configs
- `docs/` recovery notes and research pipeline status
- `runs/` local artifacts and audits (not tracked)

## Quick start (smoke)
1. Build a tiny `offers_core_smoke.parquet` (see `RUNBOOK.md`).
2. Run a minimal benchmark (`label_horizon=3`, `label_goal_min=50`,
   `limit_rows=5000`, `models=dlinear patchtst`).
3. Run Gates A-D and `scripts/summarize_horizon_audit.py`.

## Reproducibility and audits
- `docs/RESEARCH_PIPELINE_IMPLEMENTATION.md` tracks pipeline coverage and tests
- `docs/audits/` and `runs/` contain manifests, gates, and paper table outputs
- `QUICK_START.md` is the safety-first entry point
