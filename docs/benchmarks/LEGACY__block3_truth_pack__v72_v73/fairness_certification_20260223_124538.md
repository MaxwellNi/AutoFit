- generated_at_utc: **2026-02-23T12:45:38.254068+00:00**
- status: **NOT CERTIFIED**

| check | status | evidence_path | detail |
|---|---|---|---|
| temporal_split_embargo | PASS | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json | `{"embargo_non_negative": true, "train_before_val": true, "val_before_test": true}` |
| leakage_policy_coverage | PASS | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json | `{"all_targets_have_leak_group": true, "groups_non_empty": true}` |
| prediction_coverage_guard | PASS | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json | `{"fairness_pass_100": true, "global_normalized_mae_improvement_ge_8pct": false, "investors_count_gap_reduction_ge_50pct": false}` |
| catastrophic_spike_clear | NOT CERTIFIED | docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json | `{"catastrophic_spikes": 4}` |
| calibration_stability_distribution | PASS | docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json | `{"distribution_available": true, "ks_train_vs_test_lt_0_25": true, "psi_train_vs_test_lt_0_30": true}` |
| parameter_governance_no_test_feedback | PASS | docs/benchmarks/block3_truth_pack/hyperparam_search_ledger.csv | `{"best_config_exists": true, "compute_cost_exists": true, "hyperparam_search_ledger_exists": true}` |
