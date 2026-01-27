# Essential Files Checklist

This document defines the **minimum required files** for a safe, reproducible
workflow after disaster recovery. Use this as a gate before running any large
experiments.

## A) Must Exist (block if missing)

- `RUNBOOK.md`
- `DECISION.md`
- `scripts/run_full_benchmark.py`
- `scripts/audit_bench_configs.py`
- `scripts/check_label_leakage.py`
- `scripts/sanity_check_metrics.py`
- `scripts/audit_progress_alignment.py`
- `scripts/summarize_horizon_audit.py`
- `src/narrative/`
  - `models/tslib_runner.py`
  - `models/tslib_data.py`
  - `models/tslib_data_full.py`
  - `models/backbones/*.py`
  - `evaluation/report.py`
  - `evaluation/runner.py`
  - `evaluation/validation.py`
  - `data_preprocessing/*.py`
- `pyproject.toml`
- `env/tslib.yml`
- `QUICK_START.md`

## B) Recommended (audit / reproducibility / open-source hygiene)

- `docs/RESEARCH_PIPELINE_IMPLEMENTATION.md`
- `docs/transcripts/cursor_next_generation_ml_pipeline_impl.md`
- `scripts/verify_docs_against_runs.py`
- `scripts/audit_repo_consistency.py`
- `scripts/auto_backup.sh`
- `runs/backups/latest.txt`
- `configs/`
- `test/`

## C) Non-essential (keep for now, but review later)

- `scripts/run_official_benchmark.py`
- `scripts/run_official_models.py`
- `scripts/run_all_experiments.sh`
- `scripts/run_local_test.sh`
- `scripts/test_local_small_scale.py`
- `scripts/slurm/*.sbatch`
- `PROJECT_SUMMARY.md`
