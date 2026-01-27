# Research Pipeline Implementation (Recovery Status)

## Current status (20260127)

This repo is in disaster recovery mode after a workspace rollback and file loss.
All changes prioritize auditability, reproducibility, and rollback readiness.

### Recovery checkpoints completed

- Snapshot + checksum inventory created under `runs/backups/`.
- Essential scripts restored and de-duplicated.
- All `scripts/*.py` now pass `python <file> --help`.
- Smoke benchmark executed on a tiny `offers_core_smoke.parquet`.
- Gates A–D and horizon summarizer completed for the smoke run.
- MANIFEST.json generated for the smoke outputs.
- `strict_future` toggle added to benchmark + leakage gates (not yet validated on full data).

### Historical results (from transcript)

Outputs for the prior stamps are **missing** in this workspace due to disaster:
`20260126_021733`, `20260126_084648`, `20260126_091417`, `20260126_093612`,
`20260126_101559`, `20260126_105043`, `20260126_112613`.
They are listed in `RUNBOOK.md` as missing and should be re-run only after recovery.

### Next required steps (before any full grid)

1. Implement `strict_future` policy (drop samples without valid future label).
2. Re-run minimal h14 validation (2 runs, edgar on/off).
3. Only if validation passes, run the official B11 v2 8-run grid.
# 研究管道实现计划

## 研究目标对齐

### 核心目标
1. **NBI (Narrative Bias Index)** - 可解释的偏差维度评分
2. **NCI (Narrative Convergence Index)** - 模板化收敛程度
3. **GenAI 时代效应** - 测试 post-2023 对叙事和融资结果的影响

### 数据
- offers_snapshots parquet（619GB, panel over time）
- EDGAR parquet（187GB, filing+financial）

### 阶段 0：Parquet schema profiling ✅
**文件**：`scripts/profile_parquet_schemas.py` / `data_preprocessing/schema_profiler.py`

**功能**：
- 轮廓化 offers/EDGAR parquet schema
- 自动推断时间列与 join key 候选
- 输出 `runs/schema_profile/<timestamp>/schema_profile.json|csv`

**自测**：
- `PYTHONPATH=src pytest test/test_schema_profiler.py`

### 阶段 0.5：Entity key resolver ✅
**文件**：`data_preprocessing/key_resolver.py`

**功能**：
- entity_id 协议（cik → platform_name||offer_id → offer_id）
- offers ↔ EDGAR crosswalk
- 唯一性校验

**自测**：
- `PYTHONPATH=src pytest test/test_key_resolver.py`

### 阶段 0.7：Irregular timeline ✅
**文件**：`data_preprocessing/timeline.py`

**功能**：
- event_idx（不规则序列索引）
- time_delta_days / time_since_start_days
- cutoff mask 与 max_snapshots 截断

**自测**：
- `PYTHONPATH=src pytest test/test_timeline.py`

### 阶段 1：Parquet-native ingest ✅
**文件**：`pipeline.py` / `data_preprocessing/load_data.py`

**功能**：
- 使用 `load_offers_parquet` 读取 `data/raw/offers`（Hive partition）
- `allow_csv=false` 默认禁止 CSV
- 只读取必要列（id/时间/文本/json/outcomes）

**自测**：
- `PYTHONPATH=src pytest test/test_pipeline_smoke.py`

**运行示例**：
```bash
python -m narrative.pipeline \
  --config configs/experiment.yaml \
  --limit_rows 500 \
  --run_dir runs/dev_smoke_parquet
```

### 阶段 2：EDGAR feature store v2 ✅
**文件**：`data_preprocessing/edgar_feature_store.py` / `scripts/build_edgar_features.py`

**功能**：
- 从 EDGAR parquet 提取核心数值特征
- 生成 last/mean/ema 聚合（紧凑、可复现）
- 可选对 snapshots 进行 forward‑fill 对齐

**自测**：
- `PYTHONPATH=src pytest test/test_edgar_feature_store.py`

**运行示例**：
```bash
python scripts/build_edgar_features.py \
  --output_path data/processed/edgar_features.parquet
```

### 阶段 3：Irregular Patch Embed ✅
**文件**：`models/irregular_patch.py`

**功能**：
- 依据 `time_delta` 按时间跨度聚合 patch
- 输出 patch 表征与 mask（支持 padding）

**自测**：
- `PYTHONPATH=src pytest test/test_irregular_patch.py`

### 阶段 4：SSMEncoderV2 ✅
**文件**：`models/ssm_encoder_v2.py`

**功能**：
- chunked 深度可分卷积 + gated MLP
- causal / non‑causal 支持
- 输出 pooled 表征（或 sequence）

**自测**：
- `PYTHONPATH=src pytest test/test_ssm_encoder_v2.py`

### 阶段 5：Foundation model wrappers ✅
**文件**：`models/foundation_wrappers/*`

**功能**：
- 统一接口封装 Chronos / Lag‑Llama / Moirai
- 当前为 mock 实现（后续替换为官方包）

**自测**：
- `PYTHONPATH=src pytest test/test_foundation_wrappers.py`

### 阶段 6：Dataset diagnostics ✅
**文件**：`auto_fit/diagnose_dataset.py`

**功能**：
- 输出 `diagnostics.json` + `diagnostics_table.csv`
- 包含 irregularity / multiscale / nonstationarity / exogenous_strength 等指标

**自测**：
- `PYTHONPATH=src pytest test/test_diagnose_dataset.py`

### 阶段 7：Candidate composition ✅
**文件**：`auto_fit/compose_candidates.py`

**功能**：
- 扩展候选：PatchTST / iTransformer / TimeMixer++ / TimesNet / CATS
- 组合 IrregularPatch + SSMEncoderV2 + fusion + EDGAR on/off + explainability on/off

**自测**：
- `PYTHONPATH=src pytest test/test_compose_candidates.py`

### 阶段 8：Budget search ✅
**文件**：`auto_fit/budget_search.py`

**功能**：
- successive halving + early stopping
- 保存 checkpoints + best_config.yaml

**自测**：
- `PYTHONPATH=src pytest test/test_budget_search.py`

### 阶段 9：NBI supervision loaders ✅
**文件**：`nbi_nci/nbi_supervision.py` / `scripts/download_nbi_supervision.py`

**功能**：
- 加载 goemotions / financial_phrasebank / media_frames / persuasion_strategies / commitmentbank / bioscope
- 缺失数据集自动空表返回（graceful skip）

**自测**：
- `PYTHONPATH=src pytest test/test_nbi_supervision.py`

**下载示例**：
```bash
python scripts/download_nbi_supervision.py --output_dir data/raw/nbi_supervision
```

### 阶段 10：Dimension MoE router ✅
**文件**：`nbi_nci/dimension_moe.py` / `nbi_nci/nbi_computation.py`

**功能**：
- MoE gating（load-balance + sparsity regularization）
- 输出 `dim_scores` + `gating_diagnostics`

**自测**：
- `PYTHONPATH=src pytest test/test_dimension_moe.py`

### 阶段 11：Calibration ✅
**文件**：`nbi_nci/calibration.py`

**功能**：
- temperature scaling / isotonic 回归
- reliability metrics（ECE / Brier / NLL / accuracy）

**自测**：
- `PYTHONPATH=src pytest test/test_calibration.py`

### 阶段 12：Explainability faithfulness ✅
**文件**：`explainability/exporter.py`

**功能**：
- concept/time/exogenous attributions
- deletion / insertion / counterfactual faithfulness

**自测**：
- `PYTHONPATH=src pytest test/test_explainability_faithfulness.py`

### 阶段 13：Paper tables ✅
**文件**：`evaluation/report.py` / `scripts/run_full_benchmark.py`

**功能**：
- 输出 main / ablation / faithfulness / efficiency 四张表
- parquet‑native benchmark 结果自动汇总
 - 结果表路径：`runs/benchmark_matrix/<ts>/paper_tables/`

**自测**：
- `PYTHONPATH=src pytest test/test_report_tables.py`

**小样本跑法**：
```bash
python scripts/run_full_benchmark.py --offers_path data/raw/offers --limit_rows 5000 --max_runs 2
```

### 阶段 14：External datasets ✅
**文件**：`data_preprocessing/external_datasets.py` / `scripts/prepare_external_dataset.py`

**功能**：
- Kickstarter / Kiva / GoFundMe schema 统一
- 输出 parquet 到 `data/processed/external/`

**自测**：
- `PYTHONPATH=src pytest test/test_external_datasets.py`

**示例**：
```bash
python scripts/prepare_external_dataset.py --dataset kickstarter --input_path <raw.csv>
```

**跑基准示例**：
```bash
python scripts/run_full_benchmark.py --external_dataset kickstarter --external_path data/processed/external/kickstarter.parquet --max_runs 2
```

### 阶段 15：Multi-server sync ✅
**文件**：`scripts/sync_repo.sh` / `scripts/sync_results.sh` / `scripts/collect_results.py`

**功能**：
- 同步代码（排除 data/raw 与 runs）
- 同步 runs 到 `/work/projects/eint`
- 汇总结果到单目录

**自测**：
- `PYTHONPATH=src pytest test/test_collect_results.py`

### 阶段 16：Slurm templates ✅
**文件**：`scripts/slurm/*.sbatch`

**功能**：
- benchmark matrix / auto-fit search / final training
- 4×V100 资源配置模板

## 完整管道实现

### 阶段 1：文本段提取 ✅
**文件**：`preprocess_segments.py` (已存在)

**功能**：
- 从多字段提取文本段（description, reasons_to_invest, updates, comments 等）
- JSON 字段解析
- 时间索引（day_since_open）

**状态**：✓ 已实现

### 阶段 2：内容/风格解耦 ⚠️
**文件**：`nbi_nci/concept_bottleneck.py` (已创建)

**功能**：
- `StyleContentDisentangler`: 分离内容 c(x) 和风格 s(x)
- `ConceptBottleneck`: 预测概念向量 q(x)

**状态**：✓ 框架已创建，需要测试

### 阶段 3：NBI 计算 ⚠️
**文件**：`nbi_nci/nbi_computation.py` (已创建)

**功能**：
- `NBIHead`: 使用 KAN 计算每个偏差维度
- `NBIComputationModel`: 完整 NBI 模型

**偏差维度**：
1. tone_optimism
2. risk_disclosure
3. hype_vs_fundamentals
4. social_proof_fomo
5. numbers_vs_story
6. professionalization
7. esg_impact_focus
8. genai_likelihood

**状态**：✓ 框架已创建，需要测试

### 阶段 4：NCI 计算 ⚠️
**文件**：`nbi_nci/nci_computation.py` (已创建)

**功能**：
- `compute_nci_micro`: offer 级别收敛度
- `compute_nci_macro`: 群体级别多样性
- `NCIModel`: 完整 NCI 模型

**状态**：✓ 框架已创建，需要测试

### 阶段 5：结果预测 ⚠️
**文件**：`nbi_nci/outcome_model.py` (已创建)

**功能**：
- `OutcomePredictionModel`: 使用 NBI/NCI 预测融资结果
- 支持回归（funding_ratio_w）和分类（is_funded）
- 包含 post_llm × NBI 交互项

**状态**：✓ 框架已创建，需要测试

### 阶段 6：时序模型基准 ⚠️
**文件**：`models/tslib_*` (已创建)

**功能**：
- 36+ SOTA 时序模型
- 支持 snapshots 时序预测
- 早期成功预测

**状态**：✓ 框架已创建，需要完整测试

### 阶段 7：可解释性 ⚠️
**待实现**：
- TCAV: Concept activation vectors
- SHAP: Feature importance
- Integrated Gradients: Token attribution
- LIME: Local explanations
- KAN 函数可视化

**状态**：❌ 待实现

## 实现优先级

### P0（立即）
1. ✅ 修复环境问题（安装缺失包）
2. ✅ 测试小规模数据加载
3. ⚠️ 测试 NBI/NCI 计算（需要文本嵌入）

### P1（本周）
1. ⚠️ 完善文本嵌入生成
2. ⚠️ 测试 NBI/NCI 端到端
3. ⚠️ 测试结果预测模型

### P2（下周）
1. ⚠️ 实现可解释性工具
2. ⚠️ 完整时序模型基准测试
3. ⚠️ 多模态融合测试

## 测试计划

### 本地 Mac 测试（小规模）
```bash
# 1. 环境准备
conda activate insider
pip install neuralforecast darts gluonts pytorch-forecasting orbit-ml optuna

# 2. 运行测试
./scripts/run_local_test.sh
# 或
python scripts/test_local_small_scale.py --offers_path data/raw/offers --limit_rows 2000
```

### 服务器测试（完整数据）
```bash
# 在 iris/4090/3090 上
# 1. 同步完整数据
python scripts/sync_data_from_s3.py --cache_dir data/raw --num_dates all

# 2. 数据处理
python scripts/process_data_pipeline.py --cache_dir data --stage all

# 3. 提交训练
python scripts/setup_distributed_training.py --config configs/distributed.yaml
```

## 关键模块状态

| 模块 | 文件 | 状态 | 测试 |
|------|------|------|------|
| 文本段提取 | `preprocess_segments.py` | ✓ | ✓ |
| 内容/风格解耦 | `nbi_nci/concept_bottleneck.py` | ✓ | ⚠️ |
| NBI 计算 | `nbi_nci/nbi_computation.py` | ✓ | ⚠️ |
| NCI 计算 | `nbi_nci/nci_computation.py` | ✓ | ⚠️ |
| 结果预测 | `nbi_nci/outcome_model.py` | ✓ | ⚠️ |
| 时序模型 | `models/tslib_*` | ✓ | ⚠️ |
| 可解释性 | - | ❌ | ❌ |

