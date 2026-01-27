# RUNBOOK

> Reconstructed after workspace reset. Historical entries may be missing.

## Recovery note

- This file was recreated to unblock B11 v2 policy work.
- New entries will be appended below.

## Auto-backup setup (20260126_144350)

- Created `scripts/auto_backup.sh`
- Installed hourly cron job to run backup
- Initial snapshot: `runs/backups/20260126_144350/repo_backup.tgz`

## Strict-future policy upgrade (attempt, 20260126_144844)

Commands (h14 minimal 2-run with min deltas):

```
set -e; stamp=20260126_144844; for edgar in on off; do if [ "$edgar" = "on" ]; then edgar_arg="--edgar_features runs/edgar_feature_store/20260125_163720_smoke/edgar_features --use_edgar 1"; else edgar_arg="--use_edgar 0"; fi; exp="paper_min_matrix_edgar_${edgar}_v2_b11_lr500000_goal50_h14_fix2_sf_policy_v2"; PYTHONPATH=src python scripts/run_full_benchmark.py --offers_core runs/offers_core_v2_20260125_214521/offers_core.parquet $edgar_arg --limit_rows 500000 --exp_name "$exp" --plan paper_min --strict_matrix 1 --models dlinear patchtst itransformer timesnet --fusion_types none film --module_variants base nonstat multiscale --seeds 42 43 --sample_strategy random_entities --sample_seed 42 --split_seed 42 --label_goal_min 50 --label_horizon 14 --min_label_delta_days 1.0 --min_ratio_delta_rel 1e-4 |& tee -a runs/sanity_${stamp}/h14_policy_min_run.log; done
```

Outcome:

- Run failed with `RuntimeError: No samples available after filtering`.
- Drop counts (first run): `dropped_due_to_static_ratio=10190`, `dropped_due_to_min_delta_days=925`, `dropped_due_to_insufficient_future=39`.

## Recovery snapshot (20260127_021255)

- Snapshot dir: `runs/backups/repo_root_snapshot_20260127_021255/`
- Snapshot tar: `runs/backups/repo_root_snapshot_20260127_021255.tar.gz`
- Listing: `runs/backups/ls_repo_root_20260127_021255.txt`
- Checksums: `runs/backups/sha256_repo_root_20260127_021255.txt`

## Historical stamps (from transcript; outputs missing)

- `20260126_021733` — h1 fix2_sf full grid; outputs missing due to disaster.
- `20260126_084648` — h3 minimal + GateA fix; outputs missing due to disaster.
- `20260126_091417` — h3 full grid; outputs missing due to disaster.
- `20260126_093612` — h7 minimal; outputs missing due to disaster.
- `20260126_101559` — h7 full grid; outputs missing due to disaster.
- `20260126_105043` — h14 minimal; outputs missing due to disaster.
- `20260126_112613` — h14 full grid; outputs missing due to disaster.

## Smoke recovery run (20260127_023128)

- offers_core_smoke: `runs/offers_core_smoke_20260127_023118/offers_core_smoke.parquet`
- bench_dir: `runs/benchmarks/smoke_b11_recovery_h3_20260127_023128/`
- Gate A: `runs/sanity_20260127_023128/audit_summary.json`
- Gate B: `runs/sanity_20260127_023128_leakage_smoke/label_leakage_report.json`
- Gate C: `runs/sanity_20260127_023128/sanity_report.json`
- Gate D: `runs/sanity_20260127_023128/alignment_audit.json`
- Horizon compare: `runs/sanity_20260127_023128/horizon_compare.md`
- Manifest: `runs/sanity_20260127_023128/MANIFEST.json`

## Phase0 preflight (20260127_032955)

- Preflight compile log: `runs/sanity_20260127_032955/logs/preflight_compile.log`
- Preflight help log: `runs/sanity_20260127_032955/logs/preflight_help.log`
- Offers_core probe: `runs/sanity_20260127_032955/logs/offers_core_probe.log` (missing file)
- SHA256 scripts/configs: `runs/sanity_20260127_032955/logs/sha256_scripts_configs.txt`
- Phase0 manifest: `runs/sanity_20260127_032955/MANIFEST_phase0.json`

## Phase1 AB strict_future (20260127_032955)

- AB not run (missing offers_core): `runs/sanity_20260127_032955/RESULT_AB.txt`
- Bench list placeholder: `runs/sanity_20260127_032955/bench_dirs_ab.txt`

## Phase0-4 strict_future wiring (20260127_034904)

- Smoke offers_core selected: `runs/offers_core_smoke_20260127_023118/offers_core_smoke.parquet`
- Smoke validation log: `runs/sanity_20260127_034904/logs/offers_core_smoke_validate.log`
- Smoke AB bench_dirs: `runs/sanity_20260127_034904/bench_dirs_ab_smoke.txt`
- Smoke AB result: `runs/sanity_20260127_034904/RESULT_AB_SMOKE.md`
- Raw candidates log: `runs/sanity_20260127_034904/logs/offers_raw_candidates.txt`
- Built offers_core_v2: `runs/offers_core_v2_20260127_034904/offers_core.parquet`
- Build manifest: `runs/offers_core_v2_20260127_034904/MANIFEST.json`
- Real AB bench_dirs: `runs/sanity_20260127_034904/bench_dirs_ab_real.txt`
- Real AB result: `runs/sanity_20260127_034904/RESULT_AB_REAL.md` (sf1 failed: no samples)
- Gates (smoke/real): `runs/sanity_20260127_034904/audit_summary_ab_*.json`, `sanity_report_ab_*.json`, `alignment_audit_ab_*.json`
- Full manifest: `runs/sanity_20260127_034904/MANIFEST_full.json`

## Phase3 real AB unlock (20260127_045144)

- Feasibility rebuilt: `runs/sanity_20260127_045144/feasibility_rebuilt/feasibility_report.json`
- Recommended horizon: 45 (pct_delta_days_lt_min<=0.05 rule)
- AB script: `scripts/run_real_ab_minimal.sh`
- Bench dirs: `runs/sanity_20260127_045144/bench_dirs_ab_real.txt`
- Result: `runs/sanity_20260127_045144/RESULT_AB_REAL.md`
- Gates: `runs/sanity_20260127_045144/audit_summary_ab_real.json`, `sanity_report_ab_real.json`, `alignment_audit_ab_real.json`
- Manifest: `runs/sanity_20260127_045144/MANIFEST_full.json`

## Phase1 Gate B fix (20260127_090025)

- Preflight: `runs/sanity_20260127_090025/STATUS_PRE.txt`
- Gate B consistency note: `runs/sanity_20260127_090025/audit_gate_b_consistency.md`
- Leakage re-run (sf0): `runs/sanity_20260127_090025/leakage_ab_real_sf0/label_leakage_report.json`
- Leakage re-run (sf1): `runs/sanity_20260127_090025/leakage_ab_real_sf1/label_leakage_report.json`
- Manifest: `runs/sanity_20260127_090025/MANIFEST_phase1.json`
- Bundle: `runs/backups/repo_root_20260127_090025_phase1.bundle`

## Phase2 official horizon policy (20260127_091335)

- Feasibility policy report: `runs/sanity_20260127_091335/feasibility_policy/feasibility_report.json`
- Feasibility policy markdown: `runs/sanity_20260127_091335/feasibility_policy/feasibility_report.md`
- Manifest: `runs/sanity_20260127_091335/MANIFEST_phase2.json`

## Phase3 B11 v2 command templates (20260127_092307)

- Command template: `runs/sanity_20260127_092307/B11_V2_8RUN_COMMANDS.md`
- Next steps checklist: `runs/sanity_20260127_092307/NEXT_STEPS_CHECKLIST.md`
- Manifest: `runs/sanity_20260127_092307/MANIFEST_full.json`
- Bundle: `runs/backups/repo_root_20260127_092307_phase3.bundle`

## Phase4 minval edgar on/off (20260127_102903)

- Preflight: `runs/sanity_20260127_102903/STATUS_PRE.txt`
- Bench dirs: `runs/sanity_20260127_102903/bench_dirs_minval.txt`
- Gate B leakage (off): `runs/sanity_20260127_102903/leakage_b11_v2_minval_edgar_off_lr100000_goal50_h45_sf1_20260127_102903_20260127_102924/label_leakage_report.json`
- Gate B leakage (on): `runs/sanity_20260127_102903/leakage_b11_v2_minval_edgar_on_lr100000_goal50_h45_sf1_20260127_102903_20260127_103151/label_leakage_report.json`
- Minval audit summary: `runs/sanity_20260127_102903/MINVAL_AUDIT_SUMMARY.json`
- Manifest: `runs/sanity_20260127_102903/MANIFEST_full.json`
