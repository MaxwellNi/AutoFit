# Block3 Project Handoff (2026-02-21)

This handoff is intended for a new coding agent starting from zero context.
It summarizes the current Block3 state and points to the canonical evidence files.

## 1) Scope and Invariants

- Freeze pointer is fixed and read-only: `docs/audits/FULL_SCALE_POINTER.yaml`.
- Do not modify freeze assets under `runs/*_20260203_225620/`.
- Benchmark conclusions must use strict comparability only:
  - `fairness_pass == true`
  - `prediction_coverage_ratio >= 0.98`
- Canonical evidence doc:
  - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## 2) Current Execution Snapshot

- Snapshot time (UTC): `2026-02-21 22:19`
- Slurm queue:
  - `RUNNING=8`
  - `PENDING=77`
  - Main pending reasons: `QOSMaxJobsPerUserLimit` (batch), `QOSGrpNodeLimit` (gpu)
- Running jobs remaining time proxy (from current 8 running jobs):
  - min: `~26.6h`
  - avg: `~42.0h`
  - max: `~47.9h`

Evidence:
- `docs/benchmarks/block3_truth_pack/slurm_snapshot.json`
- `docs/benchmarks/block3_truth_pack/slurm_snapshot.md`

## 3) Where Results Stand

- Condition universe: `104` keys (`task x ablation x target x horizon`)
- Strict completion: `104/104` (for the current strict record pool)
- Latest truth-pack summary:
  - `raw_records=13871`
  - `strict_records=4489`
  - `legacy_unverified_records=9382`

Evidence:
- `docs/benchmarks/block3_truth_pack/truth_pack_summary.json`
- `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv`

## 4) AutoFit Family Status

- AutoFitV7: strict overlap exists but not complete in current strict pool (`97/104` keys observed).
- AutoFitV71: strict keys observed `104/104` (fully represented).
- AutoFitV72: strict keys observed `4/104` only (failure-pool heavy rerun scope).

Interpretation:
- V7.2 is not yet fully materialized at full benchmark scope.
- Gate-P/F remains blocked by insufficient V7.2 overlap coverage.

Evidence:
- `docs/benchmarks/block3_truth_pack/autofit_lineage.csv`
- `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json`
- `docs/benchmarks/block3_truth_pack/run_history_ledger.csv`

## 5) Known Risk Concentration

- Critical failure count remains `4` in taxonomy (historical catastrophic spikes in `investors_count` context).
- Current blockers are not freeze integrity issues; they are queue throughput and incomplete V7.2 coverage.

Evidence:
- `docs/benchmarks/block3_truth_pack/failure_taxonomy.csv`
- `docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json`

## 6) What To Run First (Next-Agent Checklist)

1. Refresh queue + truth pack + master doc.
2. Rebuild gate report and verify V7.2 overlap growth.
3. Keep fairness constraints unchanged.
4. Do not cancel unrelated pending jobs unless explicitly requested.

Commands:

```bash
cd /home/users/npin/repo_root
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python scripts/build_v72_pilot_gate_report.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python scripts/build_block3_truth_pack.py \
  --include-freeze-history \
  --bench-glob 'block3_20260203_225620*' \
  --capture-slurm \
  --slurm-since 2026-02-12 \
  --update-master-doc \
  --master-doc docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md
```

## 7) Canonical Document Map

- Master evidence and full tables:
  - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
- Lightweight status:
  - `docs/BLOCK3_MODEL_STATUS.md`
- Lightweight results:
  - `docs/BLOCK3_RESULTS.md`
- Truth-pack directory:
  - `docs/benchmarks/block3_truth_pack/`

## 8) Practical Handoff Note

If the next agent starts in another IDE session, read in this order:

1. `docs/PROJECT_HANDOFF_20260221.md`
2. `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
3. `docs/benchmarks/block3_truth_pack/truth_pack_summary.json`
4. `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json`

## 9) Local 4090 AutoFitV72 Completion Snapshot

- Completion date (UTC): `2026-02-21`
- Run root:
  `runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137/`
- Task completion:
  - `task1_outcome/core_only`: `12/12`
  - `task2_forecast/core_edgar`: `8/8`
  - Overall: `20/20`
- Local comparison package for Iris pull:
  - `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json`
  - `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.md`
  - `docs/benchmarks/block3_v72_local_4090/v72_metrics_autofitv72_20260221.csv`
