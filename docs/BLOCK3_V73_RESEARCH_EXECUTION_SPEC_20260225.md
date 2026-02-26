# Block3 V7.3 Research and Execution Spec (2026-02-25)

This document defines the handoff baseline for V7.3 research and execution on the finalized freeze.

## Current Benchmark Facts

<!-- BEGIN AUTO:V73_CURRENT_FACTS -->
| metric | value | evidence_path |
|---|---|---|
| generated_at_utc | 2026-02-26T00:08:32.072310+00:00 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| strict_conditions | 104/104 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| v72_coverage | 88/104 (0.8461538461538461) | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| v72_pilot_overall_pass | False | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| v72_global_improvement_pct | -0.37433548220324225 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| v72_investors_gap_reduction_pct | -2.9660362701370064 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| execution_v72_progress | [####################----] 88/104 (84.6%) | docs/benchmarks/block3_truth_pack/execution_status_latest.json |
| fairness_label | NOT CERTIFIED | docs/benchmarks/block3_truth_pack/fairness_certification_latest.json |

Condition keys are evaluation subtasks (`task × ablation × target × horizon`), not scheduler jobs.
<!-- END AUTO:V73_CURRENT_FACTS -->

## 104-key Task Universe (keys, not scheduler jobs)

<!-- BEGIN AUTO:V73_TASK_UNIVERSE -->
| task | keys | targets |
|---|---|---|
| task1_outcome | 48 | funding_raised_usd, investors_count, is_funded |
| task2_forecast | 32 | funding_raised_usd, investors_count |
| task3_risk_adjust | 24 | funding_raised_usd, investors_count |

Total condition keys: **104**.
<!-- END AUTO:V73_TASK_UNIVERSE -->

## Reuse Policy

<!-- BEGIN AUTO:V73_REUSE_POLICY -->
| metric | value | evidence_path |
|---|---|---|
| needs_rerun_true | 16 | docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv |
| needs_rerun_false | 88 | docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv |

| task | ablation | target | horizon | needs_rerun | reuse_from_run | reuse_reason |
|---|---|---|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 1 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | funding_raised_usd | 7 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | funding_raised_usd | 14 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | funding_raised_usd | 30 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | investors_count | 1 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | investors_count | 7 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | investors_count | 14 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | investors_count | 30 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | is_funded | 1 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | is_funded | 7 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | is_funded | 14 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_edgar | is_funded | 30 | false | block3_20260203_225620_phase7 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | funding_raised_usd | 1 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | funding_raised_usd | 7 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | funding_raised_usd | 14 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | funding_raised_usd | 30 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | investors_count | 1 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | investors_count | 7 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | investors_count | 14 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
| task1_outcome | core_only | investors_count | 30 | false | block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_materialized_reuse_enabled |
<!-- END AUTO:V73_REUSE_POLICY -->

## Champion Component Transfer Matrix

<!-- BEGIN AUTO:V73_CHAMPION_TRANSFER -->
| target_family | horizon_band | ablation | champion_models | key_components | transfer_priority | risk | verification_test |
|---|---|---|---|---|---|---|---|
| binary | long | core_edgar | NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | long | core_only | NBEATSx | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | long | core_text | NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | long | full | NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | mid | core_edgar | PatchTST | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | mid | core_only | PatchTST | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | mid | core_text | PatchTST | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | mid | full | PatchTST | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | short | core_edgar | PatchTST,NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | short | core_only | PatchTST,NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | short | core_text | PatchTST,NHITS | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| binary | short | full | PatchTST,DLinear | Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors. | high | Calibration gain may not translate to ranking; enforce joint MAE+calibration gate. | Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control. |
| count | long | core_edgar | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | long | core_only | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | long | core_text | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | long | full | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | mid | core_edgar | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | mid | core_only | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | mid | core_text | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | mid | full | NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | short | core_edgar | NHITS,NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | short | core_only | KAN,NHITS,NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | short | core_text | KAN,NHITS,NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| count | short | full | KAN,NBEATS | Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors. | critical | Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate. | investors_count median gap reduction vs V7 and catastrophic_spikes == 0. |
| heavy_tail | long | core_edgar | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | long | core_only | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | long | core_text | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | long | full | Chronos | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | mid | core_edgar | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | mid | core_only | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | mid | core_text | PatchTST | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | mid | full | Chronos | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | short | core_edgar | NBEATS,NHITS | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | short | core_only | NBEATS,NHITS | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | short | core_text | NBEATS,NHITS | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
| heavy_tail | short | full | NBEATSx,NHITS | Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95). | high | Tail-focused objectives can regress central tendency on short horizons. | Tail pinball and MAE stability across horizon bands. |
<!-- END AUTO:V73_CHAMPION_TRANSFER -->

## V7.3 Architecture (count / binary / heavy-tail)

1. Count lane uses two-part head and strict spike-safe guards.
2. Binary lane uses hazard head with OOF-only calibration selection.
3. Heavy-tail lane uses dual-objective robust losses and tail diagnostics.
4. Routing uses lane family + horizon band + ablation + missingness bucket.

## Offline RL Policy for Routing/HPO

<!-- BEGIN AUTO:V73_OFFLINE_RL_POLICY -->
```json
{
  "generated_at_utc": "2026-02-26T00:08:32.066343+00:00",
  "policy_name": "v73_offline_policy_v1",
  "policy_type": "contextual_bandit_with_safe_fallback",
  "state_schema": [
    "lane_family",
    "horizon_band",
    "ablation",
    "missingness_bucket",
    "nonstationarity_score",
    "periodicity_score",
    "heavy_tail_score",
    "exog_strength",
    "text_strength",
    "edgar_strength"
  ],
  "action_schema": [
    "template_id",
    "candidate_subset_id",
    "count_distribution_family",
    "binary_calibration_mode",
    "top_k"
  ],
  "reward": {
    "formula": "oof_improvement - compute_penalty - guard_penalty",
    "terms": {
      "oof_improvement": "delta of OOF MAE/logloss versus lane baseline",
      "compute_penalty": "normalized train_time and inference_time cost",
      "guard_penalty": "large penalty for fairness/coverage/guard violations"
    }
  },
  "constraints": {
    "selection_data_scope": "train_val_oof_only",
    "test_feedback_allowed": false,
    "fairness_guard_required": true,
    "coverage_threshold": 0.98,
    "spike_sentinel_required_for_count": true
  },
  "bootstrap_context": {
    "strict_completed_conditions": 104,
    "expected_conditions": 104,
    "v72_missing_keys": 16,
    "v72_coverage_ratio": 0.8461538461538461,
    "v72_pilot_overall_pass": false,
    "v72_overlap_keys": 88
  },
  "evidence_paths": {
    "truth_pack_summary": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
    "v72_pilot_gate_report": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
    "condition_leaderboard": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv"
  }
}
```
<!-- END AUTO:V73_OFFLINE_RL_POLICY -->

## Smoke / Pilot / Full Gates

1. Stage S: contract, freeze, fairness/coverage guards, lane telemetry checks.
2. Stage P: representative keys for all target families with OOF-only selection.
3. Stage F: full 104-key closure with strict-comparable reporting.

## Failure Handling (OOM / GPU reset / retry policy)

1. Never auto-cancel running jobs.
2. Retry failed keys with upgraded memory profile only when under-provisioned.
3. Persist retry provenance in queue action ledger.

## ETA and Queue Strategy

1. V72-first completion remains active until V7.2 closes strict coverage.
2. V7.3 runs use missing-key first and reuse-first submission policy.

## Reproducibility Checklist

1. Insider-only runtime (`python >= 3.11`).
2. Contract assertion before any submission.
3. Freeze pointer-only data access.
4. Train/val/OOF-only model selection and policy updates.

## 3090 Command Sequence (Pull-From-Iris, Dual-GPU)

```bash
cd /home/pni/project/repo_root

if [ -f /home/pni/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/pni/miniforge3/etc/profile.d/conda.sh
elif [ -f /home/pni/anaconda3/etc/profile.d/conda.sh ]; then
  source /home/pni/anaconda3/etc/profile.d/conda.sh
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate insider

# 1) Code baseline
git fetch origin
git checkout main
git pull --ff-only origin main

# 2) Non-GitHub handoff sync (delta)
bash scripts/pull_v73_handoff_from_iris.sh \
  --iris-login=npin@iris \
  --iris-repo=/home/users/npin/repo_root \
  --with-benchmarks=true

# 3) Runtime gate (must pass before runs)
bash scripts/check_3090_runtime_gate.sh --require-dual-gpu=true

# 4) Freeze and preflight
python3 scripts/block3_verify_freeze.py
bash scripts/preflight_block3_v71_gate.sh --v71-variant=g02 --skip-smoke --skip-audits

# 5) Refresh local evidence package
python3 scripts/build_v72_pilot_gate_report.py
python3 scripts/build_block3_truth_pack.py --include-freeze-history --capture-slurm --slurm-since 2026-02-12 --update-master-doc
python3 scripts/build_v73_handoff_pack.py --update-docs
```
