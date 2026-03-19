# Phase 9 / V739 Current Snapshot

> Generated: 2026-03-19T11:16:19.311713+00:00
> Canonical benchmark: `runs/benchmarks/block3_phase9_fair`

## Verified Current Facts

| metric | value | evidence |
| --- | --- | --- |
| metrics_files | 132 | raw metrics scan |
| raw_records | 14681 | raw metrics scan |
| raw_models | 114 | raw metrics scan |
| raw_complete_models | 86 | raw metrics scan |
| raw_partial_models | 28 | raw metrics scan |
| filtered_records | 11132 | `all_results.csv` |
| filtered_models | 85 | `all_results.csv` |
| filtered_complete_models | 67 | `all_results.csv` |
| filtered_partial_models | 18 | `all_results.csv` |
| v739_conditions_landed | 112 | raw metrics scan |
| v739_jobs_live | 0 | `squeue -u npin,cfisch` |
| v739_canonical_phase9_scripts | 13 | V739 script scan |
| v739_legacy_phase10_scripts | 32 | V739 script scan |
| text_embeddings_artifacts_complete | True | `runs/text_embeddings/` |

## Live Queue Snapshot

| metric | value |
| --- | --- |
| jobs_total | 44 |
| running | 24 |
| pending | 20 |
| npin_pending | 20 |
| cfisch_pending | 0 |
| v739_pending | 0 |
| v739_running | 0 |

### Pending Reasons

| reason | count |
| --- | --- |
| (Priority) | 20 |

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

## Partial Raw Models

| model_name | conditions | records |
| --- | --- | --- |
| ETSformer | 92 | 112 |
| LightTS | 92 | 112 |
| Pyraformer | 92 | 112 |
| Reformer | 92 | 112 |
| DUET | 36 | 36 |
| DeformableTST | 36 | 36 |
| FilterTS | 36 | 36 |
| ModernTCN | 36 | 36 |
| PDF | 36 | 36 |
| PIR | 36 | 36 |
| PathFormer | 36 | 36 |
| SEMPO | 36 | 36 |
| SparseTSF | 36 | 36 |
| TimeRecipe | 36 | 36 |
| xPatch | 36 | 36 |
| CARD | 35 | 35 |
| CFPT | 35 | 35 |
| FiLM | 35 | 35 |
| FreTS | 35 | 35 |
| Fredformer | 35 | 35 |
| MICN | 35 | 35 |
| NonstationaryTransformer | 35 | 35 |
| SCINet | 35 | 35 |
| SRSNet | 35 | 35 |
| SegRNN | 35 | 35 |
| TimeBridge | 35 | 35 |
| TimePerceiver | 35 | 35 |
| NegativeBinomialGLM | 21 | 21 |
