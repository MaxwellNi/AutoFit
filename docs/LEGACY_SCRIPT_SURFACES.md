# Legacy Script Surfaces

> Last updated: 2026-03-14
> Purpose: identify scripts that still reference the retired V72 / early V73 truth-pack line or the archived pre-Phase-9 audit surface.

These scripts are preserved for history, compatibility, or postmortem analysis only.
They are not part of the current Phase 9 / AutoFitV739 execution surface.

## Legacy Truth-Pack and Handoff Scripts

- `scripts/assert_block3_execution_contract.py`
- `scripts/audit_block3_data_integrity.py`
- `scripts/audit_investors_count_stability.py`
- `scripts/build_block3_execution_status.py`
- `scripts/build_block3_full_sota_benchmark.py`
- `scripts/build_block3_truth_pack.py`
- `scripts/build_v72_cross_version_rootcause.py`
- `scripts/build_v72_missing_key_jobs.py`
- `scripts/build_v72_pilot_gate_report.py`
- `scripts/build_v72_policy_dataset.py`
- `scripts/build_v72_rank_table.py`
- `scripts/build_v73_handoff_pack.py`
- `scripts/estimate_block3_memory_requirements.py`
- `scripts/preflight_block3_v71_gate.sh`
- `scripts/pull_v73_handoff_from_iris.sh`
- `scripts/run_v72_4090_oneshot.sh`
- `scripts/run_v72_hyperparam_search.py`
- `scripts/soft_bump_v72_failure_pool_queue.sh`
- `scripts/submit_phase7_remaining.sh`
- `scripts/submit_v72_completion_controller.sh`
- `scripts/submit_v72_failure_pool_rerun_heavy.sh`
- `scripts/train_v72_offline_policy.py`
- `scripts/v72_coverage_guard.py`
- `scripts/watch_v72_fasttrack_release.sh`

## Legacy Audit / Freeze-Build Scripts

- `scripts/block3_benchmark_audit.py`
- `scripts/build_wide_freeze.sh`
- `scripts/run_b3_audit.sbatch`
- `scripts/run_wide_freeze_full.sh`

## Current Scripts To Prefer Instead

- `scripts/block3_verify_freeze.py`
- `scripts/build_phase9_current_snapshot.py`
- `scripts/phase12_prepare_text_rerun.py`
- `scripts/run_block3_benchmark_shard.py`
- `scripts/aggregate_block3_results.py`

## Current Truth Docs

- `docs/CURRENT_SOURCE_OF_TRUTH.md`
- `docs/PHASE9_V739_FACT_ALIGNMENT.md`
- `docs/V739_CURRENT_RUN_MONITOR.md`
- `docs/PHASE12_TEXT_RERUN_EXECUTION.md`

## Legacy Directory Naming Rule

- Any directory with prefix `LEGACY__` is archive-only, not current operational truth.
- Tombstone paths such as `docs/benchmarks/block3_truth_pack/README.md` exist only to redirect readers to the renamed legacy archive.
