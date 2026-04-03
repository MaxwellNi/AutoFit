# Phase 9 / V739 Current Snapshot

> Generated: 2026-04-03T13:11:50.677076+00:00
> Canonical benchmark: `runs/benchmarks/block3_phase9_fair`

## Verified Current Facts

| metric | value | evidence |
| --- | --- | --- |
| metrics_files | 137 | raw metrics scan |
| raw_records | 16299 | raw metrics scan |
| raw_models | 137 | raw metrics scan |
| raw_nonretired_models | 114 | raw metrics scan minus archived AutoFit-family lines |
| raw_retired_autofit_models | 23 | raw metrics scan (archived AutoFit-family lines) |
| raw_complete_models | 75 | raw metrics scan |
| raw_partial_models | 39 | raw metrics scan |
| filtered_records | 11976 | clean current-surface `all_results.csv` |
| filtered_models | 84 | clean current-surface `all_results.csv` |
| filtered_nonretired_models | 84 | clean current-surface `all_results.csv` |
| filtered_retired_autofit_models | 0 | archived AutoFit-family lines purged from current surface |
| filtered_complete_models | 55 | `all_results.csv` |
| filtered_partial_models | 29 | `all_results.csv` |
| v739_conditions_landed | 132 | raw metrics scan |
| v739_jobs_live | 0 | `squeue -u npin,cfisch` |
| v739_canonical_phase9_scripts | 13 | V739 script scan |
| v739_legacy_phase10_scripts | 32 | V739 script scan |
| text_embeddings_artifacts_complete | True | `runs/text_embeddings/` |

## Live Queue Snapshot

| metric | value |
| --- | --- |
| jobs_total | 34 |
| running | 8 |
| pending | 26 |
| npin_pending | 26 |
| cfisch_pending | 0 |
| v739_pending | 0 |
| v739_running | 0 |

### Pending Reasons

| reason | count |
| --- | --- |
| (Priority) | 25 |
| (Resources) | 1 |

## Text Embedding Artifacts

| field | value |
| --- | --- |
| directory_exists | True |
| artifacts_complete | True |
| parquet_path | runs/text_embeddings/text_embeddings.parquet |
| pca_model_path | runs/text_embeddings/pca_model.pkl |
| n_total_rows | 5774931 |
| n_unique_texts | 69697 |
| n_entities | 22569 |
| pca_dim | 64 |

## V739 Submission Surface

| metric | value |
| --- | --- |
| canonical_phase9_fair_scripts | 13 |
| legacy_phase10_scripts | 32 |

### V739 Script Families

#### phase10_fast_npin

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase10_fast/v739/v739f_t1_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t1_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739f_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739f_t2_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t2_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739f_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739f_t3_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t3_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r2_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739/v739r2_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739/v739r_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739r_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739r_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |

#### phase10_fast_cfisch

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t1_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t1_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_only |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t2_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t2_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_only |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_fu_hopper.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |

#### phase12_l40s_cfisch

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t1_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t1_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t2_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t2_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/full |

#### phase12_rerun_autofit

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t1_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t1_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/full |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t2_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t2_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/full |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t3_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t3_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/full |

## Partial Current-Surface Raw Models

| model_name | conditions | records |
| --- | --- | --- |
| XGBoost | 159 | 159 |
| XGBoostPoisson | 157 | 157 |
| AutoFitV739 | 132 | 132 |
| Chronos2 | 114 | 160 |
| TTM | 114 | 160 |
| Crossformer | 107 | 111 |
| MSGNet | 107 | 111 |
| MambaSimple | 107 | 111 |
| PAttn | 107 | 111 |
| MultiPatchFormer | 105 | 109 |
| TimeFilter | 105 | 109 |
| ETSformer | 94 | 114 |
| LightTS | 94 | 114 |
| Pyraformer | 94 | 114 |
| Reformer | 94 | 114 |
| CARD | 92 | 92 |
| DUET | 92 | 92 |
| FiLM | 92 | 92 |
| FilterTS | 92 | 92 |
| FreTS | 92 | 92 |
| Fredformer | 92 | 92 |
| ModernTCN | 92 | 92 |
| NonstationaryTransformer | 92 | 92 |
| PDF | 92 | 92 |
| PIR | 92 | 92 |
| SCINet | 92 | 92 |
| SRSNet | 92 | 92 |
| SegRNN | 92 | 92 |
| TimeRecipe | 92 | 92 |
| xPatch | 92 | 92 |
| CFPT | 69 | 69 |
| DeformableTST | 69 | 69 |
| MICN | 69 | 69 |
| PathFormer | 69 | 69 |
| SEMPO | 69 | 69 |
| SparseTSF | 69 | 69 |
| TimeBridge | 69 | 69 |
| TimePerceiver | 69 | 69 |
| NegativeBinomialGLM | 21 | 21 |
