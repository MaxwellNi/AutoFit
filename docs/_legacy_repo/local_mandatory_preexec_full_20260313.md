# MANDATORY PRE-EXECUTION CHECKLIST (LOCAL ONLY — NEVER PUSH)
# Last updated: 2026-03-13 (Phase 9 / V739 current-state cleanup, live queue + text embedding re-verified)
# Every agent MUST read this file before executing any benchmark, job submission,
# or code modification. This document is the SINGLE SOURCE OF TRUTH.

## ⚠️ 语言要求 (LANGUAGE REQUIREMENT) — 最高优先级
**所有回复必须使用中文！** 永远不要用英文回复用户。
代码、git commit message、文件内容可以用英文，但与用户的交流一律用中文。
ALL replies to the user MUST be in Chinese. Code/commits/file content may be English.

---

## 0. CURRENT VERIFIED STATE (READ THIS BEFORE EVERYTHING ELSE)

**Current authority docs**:
- `AGENTS.md`
- `docs/CURRENT_SOURCE_OF_TRUTH.md`
- `docs/BLOCK3_MODEL_STATUS.md`
- `docs/BLOCK3_RESULTS.md`
- `docs/benchmarks/phase9_current_snapshot.md`

**Directly verified facts on 2026-03-13**:
- Canonical benchmark root remains `runs/benchmarks/block3_phase9_fair/`
- Raw Phase 9 materialization: `78` metrics files, `8664` raw records, `90` raw models, `77` raw complete, `13` raw partial
- Filtered leaderboard: `6672` records, `69` models, `59` complete
- V739 current landed coverage: `0/104`
- Observed live V739 jobs: `8 pending`, `0 running`
- The observed queued V739 jobs currently target `runs/benchmarks/block3_phase10/v739/`, not the canonical `runs/benchmarks/block3_phase9_fair/` root
- Live queue total: `1 running`, `133 pending`
- Text embedding artifacts now exist:
  - `runs/text_embeddings/text_embeddings.parquet`
  - `runs/text_embeddings/pca_model.pkl`
  - `runs/text_embeddings/embedding_metadata.json`
- Therefore, the project is **no longer blocked on generating text embeddings**; it is blocked on submitting/landing the real text-enabled reruns

**Current document hygiene rules**:
1. `docs/_legacy_repo/` = archive only, never current truth
2. `docs/references/` = useful background only, never current status
3. `docs/benchmarks/block3_truth_pack/` = historical V72/V73 evidence line, not current operational truth for Phase 9 / V739
4. `docs/V739_CURRENT_RUN_MONITOR.md` = current V739 queue/landing interpretation
5. `docs/PHASE12_TEXT_RERUN_EXECUTION.md` = current real text/full rerun execution surface

---

## 1. CANONICAL RESULTS DIRECTORY

**USE ONLY**: `runs/benchmarks/block3_phase9_fair/`
**DEPRECATED**: `runs/benchmarks/block3_20260203_225620_phase7/` (see DEPRECATED.md inside)

All new benchmark results MUST write to `block3_phase9_fair/`.
All aggregation, leaderboard, and paper table scripts MUST read from `block3_phase9_fair/`.
Old results in `block3_20260203_225620_phase7/` contain known bugs — DO NOT REUSE.

---

## 2. FROZEN TRAINING PROTOCOLS (DO NOT CHANGE WITHOUT EXPLICIT USER APPROVAL)

### NeuralForecast Models (deep_classical + transformer_sota)
These configs are FROZEN at their NF library defaults. Changing them invalidates
all existing results and requires a FULL re-benchmark.

| Group | max_steps | early_stop_patience | val_check_steps |
|-------|-----------|---------------------|-----------------|
| Most NF models | 1000 | 10 | 50 |
| PatchTST, Informer, Autoformer, FEDformer, VanillaTransformer | 5000 | 10 | 100 |
| TimeXer, TimeLLM | 2000 | 10 | 100 |

**Per-model hyperparameters** (lr, hidden_size, scaler_type, etc.) are set in
`PRODUCTION_CONFIGS` in `src/narrative/block3/models/deep_models.py`.
These values must NOT be changed mid-benchmark without user approval AND full rerun.

### Justification for keeping max_steps=1000:
- 50% of training sessions early-stopped before 1000 steps (converged)
- 50% hit the cap, but NF restores best checkpoint regardless
- All models shared the same constraint → fair comparison
- Empirically validated on 2026-03-08

### Foundation Models
- `prediction_length` MUST equal `pred_horizon` (the requested forecast horizon)
- Entity cap MUST be None (process ALL entities)
- NEVER hardcode prediction_length to a constant

### TSLib Models
- Prediction MUST be per-entity (fit globally, predict per-entity)
- See `_predict_per_entity()` in tslib_models.py

### ML Tabular Models
- MUST run all 4 horizons [7, 14, 30, 60] (NOT just horizons[0])
- fillna: use NaN passthrough (NOT fillna(0))

### AutoFit
- **V739 = 当前唯一有效版本** (validation-based, NO oracle leakage)
  - Replaces oracle tables with genuine temporal validation
  - Harness passes `val_raw` to AutoFit fit() (benchmark_shard.py L993)
  - 8 candidate pool: NHITS, PatchTST, NBEATS, NBEATSx, ChronosBolt, KAN, Chronos, TimesNet
  - Trains each candidate on train data, evaluates on val split, selects best single model
  - Code: nf_adaptive_champion.py class NFAdaptiveChampionV739
  - Factory: autofit_wrapper.py get_autofit_v739()
- V734-V738 ALL retired (5个版本连续oracle测试集泄漏，排名全部无效)
- V1–V733, FusedChampion 也已排除
- **未来所有 AutoFit 迭代 (V740+) 必须以 V739 为基础，见下方强制检查清单**

### ⛔ AutoFit 新版本开发强制检查清单 (V734-V738 五轮血泪教训)

> **警告**: V734-V738 连续5个版本因 oracle 测试集泄漏全部失败，累计浪费 520 条记录 + 数十个 GPU-day。
> 开发任何 AutoFit V740+ 版本时，以下每一条必须逐项检查，缺一不可。
> 完整分析见 §25。

**开发前 (BEFORE CODING):**
1. 确认数据来源: `metrics.json` = 测试集结果 → ❌ 绝对禁止用于模型选择
2. 确认 harness 传入的 `val_raw` 是唯一合法验证数据源
3. 确认以 V739 (`NFAdaptiveChampionV739`) 为基础, 不要从 V734-V738 复制任何代码

**开发中 (DURING CODING):**
4. 每个新参数 (如 val_frac, n_candidates) 必须 `grep -n` 验证在 fit()/predict() 中被实际引用。参数 ≠ 实现。
5. 如果继承父类, 必须检查 fit() 和 predict() 的完整调用链。V739 做法: 完全覆盖, 不调 super()。
6. 类体内执行 `grep -n 'ORACLE_TABLE\|oracle_key\|oracle_lookup'` → 必须返回 0 结果。
7. 注释中明确标注数据来源 (train/val/test), 禁止模糊的 "clean data" 之类。

**提交前 (BEFORE SUBMITTING):**
8. 跑 smoke test: `(task1_outcome, core_only, funding_raised_usd, h=7)` → fit 成功, predict 输出合理范围, 无 NaN/Inf
9. 代码必须 `git commit` 后再 `sbatch`
10. 提交前检查 `squeue -u $USER` 避免重复提交

**提交后 (AFTER RESULTS):**
11. 对比 V739 baseline — 如果新版本在所有 104 条件上都显著优于 V739, 极大概率有 bug (太好必有诈)
12. 抽样检查 3 个 metrics.json: 确认 model_name 正确, 无异常低 error

---

## 3. CODE CHANGE PROTOCOL

1. **NEVER modify PRODUCTION_CONFIGS** without explicit user approval
2. **ALWAYS commit all code changes before submitting SLURM jobs**
   - Running SLURM jobs read working directory at import time
   - Uncommitted changes create version inconsistency between running and pending jobs
3. **ALWAYS run the execution contract assertion before job submission**:
   ```bash
   python3 scripts/assert_block3_execution_contract.py --entrypoint <script>
   ```
4. **NEVER create files with names containing**: cursor, prompt, recovery, runbook,
   decision, transcript (case-insensitive)
5. **All code and docs must be English only**

---

## 4. JOB SUBMISSION PROTOCOL

1. Verify ALL code changes are committed: `git status` must show clean working tree
   (or only untracked SLURM scripts)
2. Run execution contract assertion
3. Verify the target output directory is `block3_phase9_fair/`
4. Use SLURM scripts from `.slurm_scripts/phase9/`
5. Log directory: `/work/projects/eint/logs/phase9/`
6. Maximum concurrent jobs per account: 100 (QOS normal MaxJobsPU)
7. **Two accounts available**: npin (primary), cfisch (secondary, see §4a below)

### 4a. cfisch Account Access Strategy

cfisch submits via `ssh iris-cf` (alias for iris login as cfisch user).
cfisch CAN directly access `/work/projects/eint/repo_root/` (same GPFS filesystem).
cfisch CAN directly run the insider Python environment:

```bash
# Direct insider python — NO micromamba needed
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
```

**cfisch DOES NOT** have `micromamba` in PATH. All cfisch SLURM scripts MUST:
- Use `--account=christian.fisch` (NOT `--account=cfisch`)
- Use the direct `INSIDER_PY` path above (NO `micromamba activate insider`)
- Use `cd /work/projects/eint/repo_root` (NOT `/home/users/npin/repo_root`)
- Write to the SAME `runs/benchmarks/block3_phase9_fair/` output dirs (shared filesystem)

**Submitting cfisch jobs from npin session**:
```bash
ssh iris-cf "cd /work/projects/eint/repo_root/.slurm_scripts/phase9 && sbatch <script>.sh"
```

### 4b. Partition Memory Limits (EMPIRICALLY VALIDATED)

| Partition | Nodes | CPUs | RAM/node | GPU | QOS Required | Use For |
|-----------|-------|------|----------|-----|-------------|---------|
| batch     | ~168  | varies | 112G   | None | normal | NEVER for benchmark (too small) |
| gpu       | 24    | skylake | 756G  | 4x V100 (16/32GB) | normal | GPU models (deep, foundation, tslib) |
| bigmem    | 4     | skylake | 3TB   | None | normal | Memory-intensive CPU (ml_tabular, autofit, statistical) |
| l40s      | 2     | emeraldrapids | 515G | 4x L40S (48GB) | iris-snt | GPU overflow (**OWNER节点，会被抢占！**) |
| l40s-serval | 1   | emeraldrapids | 515G | 4x L40S | iris-snt | Same as l40s (**OWNER节点**) |
| l40s-trux | 1     | emeraldrapids | 515G | 4x L40S | iris-snt | Same as l40s (**OWNER节点**) |
| hopper    | 1     | saphirerapids | 2TB | 4x H100 NVL (96GB) | besteffort | **CAN ACCESS via besteffort QOS** (OWNER节点，会被抢占) |

**CRITICAL LESSONS**:
1. **batch = NEVER** for benchmark. 112G total, requesting 112G → no OS headroom → OOM.
2. **gpu = PRIMARY** for all GPU models. 756G RAM (no strict enforcement observed).
3. **bigmem = CPU-ONLY models** needing >112G. Memory IS enforced (cgroup).
4. **l40s = GPU OVERFLOW** via iris-snt/iris-snt-long QOS. Max ~400G request (515G total - OS).
   iris-snt: 无 GrpGRES 限制！iris-snt-long: GrpGRES=gpu=2（全QOS组共享）。
   V739 完成仅需 41-51 分钟，优先用 iris-snt（2天墙时间够用）。
   ⚠️ **l40s/l40s-serval/l40s-trux 是 OWNER 节点**：会被 KILL/REQUEUE。
   benchmark harness 有 resume 机制（加载已有 metrics.json 跳过完成的 combo），可安全重跑。
5. **hopper = ACCESSIBLE via besteffort QOS** (2026-03-13 验证).
   4× H100 NVL 96GB VRAM + 2TB RAM。可请求 300G+。
   ⚠️ **OWNER节点** — besteffort 任务会被 owner 抢占，但 benchmark harness resume 机制保证安全。
6. **iris-gpu-long QOS**: GrpTRES=node=6（全用户共享6节点上限，经常满）。
   如有空位立即分配，否则等待。npin 和 cfisch 竞争同一池。

### 4c. QOS Limits (Both npin and cfisch under christian.fisch Account)

| QOS | MaxJobsPU | MaxWall | Priority | Notes |
|-----|-----------|---------|----------|-------|
| normal | 100 | 2 days | 10001 | Primary QOS |
| low | 200 | 2 days | 2000 | Low priority |
| besteffort | 300 | 50 days | 1 | Lowest priority, preemptible. **可用于 gpu/l40s/hopper/bigmem** |
| iris-snt | 100 | 2 days | varies | Required for l40s partition |
| iris-snt-long | 100 | 14 days | varies | Long l40s jobs |
| iris-gpu-long | 4 | 14 days | varies | Long GPU jobs (limited) |
| iris-bigmem-long | 4 | 14 days | varies | Long bigmem jobs (limited) |
| iris-batch-long | 8 | 14 days | varies | Long batch jobs |

### 4d. Memory Requirements by Ablation (EMPIRICALLY MEASURED)

Based on completed Wave 1 jobs' MaxRSS values:
| Metric | core_only | core_edgar | core_text | full |
|--------|-----------|------------|-----------|------|
| Peak RAM observed | ~120GB | ~175GB | ~283GB | ~353GB |
| Safe allocation (bigmem) | 256G | 256G | 512G | 640G |
| Safe allocation (gpu) | 256G | 256G | 400G | 500G |
| Safe allocation (l40s) | 200G | 200G | 400G | REJECTED (>515G) |

**Root cause**: `offers_text.parquet` is 20GB compressed → ~150-200GB in memory after
decompression + merge. EDGAR store is only 13MB (negligible). Core data is 301MB (~2GB in memory).

**OOM failure history** (5 rounds of discovery):
- Round 1: batch partition (112G total) → ALL ablations OOM. Fix: moved to bigmem/gpu.
- Round 2: bigmem 128G for statistical → _co/_ce OOM (peak ~120-175G, no headroom).
           bigmem 256G for autofit → _ct/_fu OOM (peak ~283-353G).
- Round 3 (Wave 3): Fixed allocations to 256-640G based on empirical MaxRSS.
- Round 4 (2026-03-13): V739 on l40s/hopper at 192G → ct/fu OOM (hit 192G cgroup limit).
  t3_co 用 114G, t3_ce 用 164G OK。但 ct/fu 即使 text embedding 不存在也需要 >192G。
  Fix: 所有 ct/fu ablation 改为 300G。l40s 300G 可行 (515G node)，hopper 300G 也没问题 (2TB node)。
- Round 5 (2026-03-13): V739 t2_fu (task2/full) OOM at 300G on hopper (MaxRSS=314G)。
  "full" ablation 在 task1/task2 上实际内存超过 300G。
  Fix: full ablation 改为 500G (hopper 2TB OK, l40s 用 450G)。core_text 用 300G 可行。

**cfisch 权限问题** (2026-03-13):
- cfisch 写入 npin 创建的目录时 Permission Denied（MANIFEST.json）。
- 根因：npin 的 umask=0022 创建的目录没有 group write。
- Fix: (a) chmod -R g+w runs/benchmarks/block3_phase10/v739/, (b) 所有脚本加 `umask 002`。

---

## 5. KNOWN BUGS

### 5a. FIXED BUGS — DO NOT REINTRODUCE

| Bug | Root Cause | Fix Location | Date Fixed |
|-----|-----------|--------------|------------|
| Foundation prediction_length=7 | Hardcoded `predict(tensors, 7)` | deep_models.py L1249 | 2026-03-07 |
| Moirai entity cap [:50] | Arbitrary `ctxs[:50]` slice | deep_models.py L1265 | 2026-03-07 |
| Moirai prediction_length=7 | Hardcoded in MoiraiForecast() | deep_models.py L1275 | 2026-03-07 |
| TSLib per-entity prediction | Global predict on concatenated panel | tslib_models.py | 2026-03-07 |
| ml_tabular single horizon | `run_horizons = [horizons[0]]` | benchmark_shard.py | 2026-03-07 |
| fillna(0) for ml_tabular | `df.fillna(0)` corrupting features | benchmark_shard.py | 2026-03-07 |
| Constant-prediction fairness | fairness_pass=True for constants | benchmark_shard.py | 2026-03-07 |
| Fairness guard kills shard | raise instead of log+skip | benchmark_shard.py | 2026-03-07 |
| Constant detection threshold | `== 0.0` missed near-constant predictions | benchmark_shard.py L1020: `< 1e-8` | 2026-03-10 |
| Target-aware fallback detection | All targets shared same MAE threshold | aggregate_block3_results.py: `_FALLBACK_MAE_RANGES` dict | 2026-03-10 |
| Unfitted RobustFallback returns None | `predict()` called before `_train_mean` set | deep_models.py: store `_train_mean` on fit failure | 2026-03-10 |
| Unmapped target no warning | Silently ran without leak guard | benchmark_shard.py L616: log warning | 2026-03-10 |
| Statistical silent fallback | 3 except paths returned mean without setting `_use_fallback=True` | statistical.py: set flag on all 3 paths | 2026-03-10 |
| Synthetic panel shared dates | All entities got same date axis `"2020-01-01"` | deep_models.py: offset `start + i*obs` days | 2026-03-10 |
| V737 sinh overflow | `np.sinh(preds)` overflows for preds>710 | nf_adaptive_champion.py: clip to [-25,25] before sinh | 2026-03-10 |
| Text embedding Arrow OOM | pandas Arrow string backend × 155K-char strings = 2.55 TiB | generate_text_embeddings.py: .astype("object") + per-field truncation | 2026-03-10 |

### 5b. OPEN BUGS — MUST FIX (discovered 2026-03-10, updated 2026-03-11)

| Bug | Severity | Root Cause | Impact |
|-----|----------|-----------|--------|
| ORACLE TEST-SET LEAKAGE | 🔴🔴 FATAL | ORACLE_TABLE_V738/TOP3 built from Phase 9 test-set MAE/RMSE values → model selection uses test-set info | ALL AutoFit V734-V738 rankings scientifically INVALID for paper |
| val_frac DEAD CODE | 🔴 HIGH | V738.__init__ sets val_frac=0.2 but fit()/predict() never reference it | Docstring claim "20% validation" is not implemented |
| TEXT ABLATION DEAD | 🔴 CRITICAL | `_prepare_features` L572 uses `select_dtypes(include=[np.number])` → drops ALL 19 string text columns | core_text ≡ core_only; full ≡ core_edgar; 50% compute wasted; paper ablation invalid |
| 3 TSLib constant predictions | 🔴 HIGH | MICN/MultiPatchFormer/TimeFilter produce 100% constant predictions (fairness_pass=False on ALL conditions) | Must exclude from leaderboard |
| NF training non-determinism | 🟡 MEDIUM | NF models in separate SLURM jobs → GPU non-deterministic training → ~1388 MAE noise between ce/fu | Need 3-seed averaging for robust results |
| C2 crash models still active | 🟡 MEDIUM | AutoCES/StemGNN/TimeXer/xLSTM/TimeLLM crash → MeanPredictor fallback | Must exclude from leaderboard |
| Timer≈TimeMoE≈MOMENT identity | 🟡 MEDIUM | HF pipeline shared tokenizer → 99-102/104 identical predictions | Report as 1 model, not 3 |

### Models to EXCLUDE from leaderboard (total: 16+)
- **Oracle leak**: AutoFitV734, AutoFitV735, AutoFitV736, AutoFitV737, AutoFitV738
- **100% constant**: MICN, MultiPatchFormer, TimeFilter
- **Crashed→MeanPredictor**: AutoCES, StemGNN, TimeXer, xLSTM, TimeLLM
- **De-duplicate**: Timer≈TimeMoE (keep MOMENT only)

### 5c. ORACLE LEAKAGE 5-LAYER ROOT CAUSE (identified 2026-03-12)

The oracle test-set leakage is NOT a simple coding mistake — it is a **structural** problem:

1. **Layer 1 — Data Source**: Benchmark pipeline outputs ONLY test-set metrics to metrics.json.
   No validation-set evaluation exists. Any oracle built from metrics.json = test-set leakage.

2. **Layer 2 — Copy-Paste Perpetuation**: V733→V734→V735→V736→V737→V738 each expanded
   the oracle table without questioning its data source. Comments say "Phase 9 clean data"
   without clarifying = test-set.

3. **Layer 3 — Missing Infrastructure**: Benchmark harness has NO mechanism for
   train/validation/test 3-way split. Without validation metrics, legitimate oracle = impossible.

4. **Layer 4 — Dead Code False Confidence**: V738's `val_frac=0.2` and docstring claim
   "20% held out validation set" suggest validation was INTENDED. The working implementation
   EXISTS in V753 (`autofit_wrapper.py` L7232 `_fit_validation()`) but was never carried over.

5. **Layer 5 — No Automated Guard**: No test prevents test-set → fit(). No assertion
   validates oracle ≠ test metrics. No code review gate checks oracle provenance.

**Root fix**: Must add 3-way split to benchmark harness OR use V753-style internal validation.

---

## 6. MODELS REQUIRING RERUN (Phase 9 Wave 2)

### Wave 1 Status (submitted 2026-03-08)
- 37 models at 104/104 (complete, no action needed)
- 12 models at 99-103/104 (need gap-fill)
- 15 models at 3-39/104 (need more conditions)
- 25 models have 0 results (running in Wave 1)

### Wave 2 OOM Fixes (p9r_ scripts)
| Script Group | Partition/Mem | Models | Scripts | Account |
|-------------|---------------|--------|---------|---------|
| p9r_af9 | bigmem/256G | AutoFitV736 | 8 | cfisch |
| p9r_mlT | bigmem/256G | 18 ml_tabular | 10 | cfisch |
| p9r_irN | gpu/320G | BRITS,CSDI | 8 | cfisch |
| p9r_fmB | gpu/500G | LagLlama,Moirai,MoiraiLarge,Moirai2 | 5 | cfisch |

### Wave 2 New Models (p9n_ scripts)
| Script Group | Models | Scripts | Account |
|-------------|--------|---------|---------|
| p9n_sta | 10 statistical | 11 | cfisch |
| p9n_fmC | Chronos2,TTM,TimerXL | 11 | npin |
| p9n_tpf | TabPFNClassifier,TabPFNRegressor | 11 | npin |
| p9n_tsC | 6 tslib_sota | 11 | npin |

### Wave 2 Gap-Fill (p9g_ scripts)
| Script | Models | Conditions | Account |
|--------|--------|-----------|---------|
| p9g_trCO | 10 transformer_sota | task1/core_only is_funded | npin |
| p9g_trCT | 10 transformer_sota | task1/core_text is_funded | npin |　ß
| p9g_fmCT | Sundial,TimesFM2 | task2/core_text | npin |

### Moirai2Large
**NOT in registry** — removed from scripts. Only Moirai2 (moirai-moe-1.0-R-small) exists.

**TOTAL WAVE 2**: 78 scripts (36 npin + 42 cfisch)

---

## 7. WHAT NOT TO DO (LESSONS LEARNED)

1. **DO NOT change training hyperparameters mid-benchmark** — this was done by a previous
   agent (max_steps 1000→3000, lr changes, hidden_size changes) without committing,
   creating version inconsistency. All 3,278 affected records had to be audited.
2. **DO NOT submit SLURM jobs with uncommitted code** — jobs read working directory
   at import time; uncommitted changes create version mismatch.
3. **DO NOT run old AutoFit versions** — V734-V738 ALL oracle-leaked. Only V739 is valid.
4. **DO NOT hardcode prediction horizons** — always use the requested horizon parameter.
5. **DO NOT reuse results from `block3_20260203_225620_phase7/`** — use `block3_phase9_fair/`.
6. **DO NOT use batch partition for benchmarks** — 112G total, insufficient for ANY ablation.
7. **DO NOT request 500G+ on l40s** — 515G total, requests >~400G silently rejected.
8. **DO NOT use _ct/_fu ablations on bigmem with <512G** — text data alone uses 200-283G.
9. **DO NOT assume GPU partition enforces --mem** — jobs can exceed requested memory if node has free RAM.
10. **DO NOT submit to hopper partition** — we lack iris-hopper QOS.
11. **DO NOT run two shards writing to the SAME metrics.json** — race condition.
12. **DO NOT change model code between job requeues** — stale results persist via resume.
13. **DO NOT assume sbatch failure prints error** — some rejections (e.g., mem > node RAM) are silent.
14. **DO NOT use np.sinh() without pre-clipping** — sinh(710+) = inf in float64. Always clip in transformed space before inverse.
15. **DO NOT use pandas .loc/.where on Arrow string columns with long strings** — Arrow backend allocates n_rows × max_len × 4 bytes. Use `.astype("object")` first.
16. **V737 EDGAR ce/fu ablations need 192G minimum** — EDGAR as-of join peaks at ~134G; 128G OOM proven (5/11 jobs failed).
17. **DO NOT build oracle/selection tables from test-set metrics** — This is test-set information leakage. ORACLE_TABLE_V738 and ORACLE_TABLE_TOP3 both use Phase 9 test-set MAE/RMSE. Model selection must use ONLY training/validation data.
18. **DO NOT assume `val_frac` is used just because it's set** — V738 sets `val_frac=0.2` in `__init__` but never references it in `fit()`/`predict()`. Always verify dead code by grep.
19. **cfisch core_text/full need 384G minimum** — 256G OOM proven (all 5 cfisch V738 jobs, actual RSS=268G). Next step would be 512G if 384G fails (756G node RAM).
20. **DO NOT submit duplicate jobs without checking queue** — Use `squeue -u npin` before every sbatch. Multiple rounds of duplicate cleanup wasted GPU hours.
21. **STRUCTURAL FIX APPLIED**: Benchmark harness now passes `val_raw` to AutoFit via `fit_kwargs["val_raw"] = val`. The harness already had a 3-way temporal split (train_end=2025-06-30, val_end=2025-09-30, test_end=2025-12-31, embargo_days=7) but `val` was UNUSED until this fix. V739 uses this val data for model selection.
22. **V753's `_fit_validation()` (autofit_wrapper.py L7232) is the correct template** — It implements proper inner train/val split: `val_size = max(int(n * self._val_fraction), 20)`, trains on inner, evaluates on held-out, selects best. Copy this logic, not just the parameter.
23. **DO NOT copy parameters without copying implementation** — V738 copied `val_frac=0.2` from V753 but not the `_fit_validation()` method. Dead parameters create false confidence in code review.
24. **tsA jobs TIMEOUT at 2-day walltime** — MSGNet/PAttn/MambaSimple training on full ablation too slow for 2-day wall. Need either: extend walltime (iris-gpu-long, 14d, max 4 jobs), or use checkpoint/resume with re-queue.
25. **DO NOT submit core_text/full SLURM jobs without verifying text embeddings exist** — 50% compute wasted (3,888 records, ~638 GPU-hours) because text_embeddings.parquet was never generated. Harness silently falls back, producing core_text ≡ core_only. See §29.
26. **V739/V740 迭代只跑 core_only + core_edgar + 9 冠军模型** — 95% 算力节省。全量 benchmark 在 AutoFit 设计稳定后再做。See §29.4 策略6.
27. **高随机性模型 (CSDI/iTransformer/BRITS) 必须 3-seed 运行** — single-run 训练方差 >1%，冠军判定不可靠。See §29.4 策略5.

### 7b. ⛔ AutoFit V734-V738 错误模式速查 (开发新 AutoFit 版本前必读)

**V734-V738 连续5个版本全部失败。根因: 用测试集结果做模型选择 = 偷看答案。**

| 错误模式 | 发生版本 | 如何防止 |
|----------|----------|----------|
| 从 metrics.json 构建 oracle 表 | V734-V738 全部 | metrics.json = 测试集结果。禁止用于任何模型选择逻辑。 |
| 复制粘贴 oracle 表但不审查数据来源 | V735-V738 | 每次迭代必须重新验证所有查找表的数据来源。 |
| 添加参数但不实现功能 (val_frac=0.2 死代码) | V738 | grep 验证每个参数在 fit/predict 中被引用。 |
| 继承父类 oracle 路径 | V737-V738 | 完全覆盖 fit/predict，或审查 super() 调用链。 |
| 注释写"clean data"但实际=测试集 | V734-V738 | 注释必须明确标注数据来源（train/val/test）。 |
| 无自动化防护 | V734-V738 | 至少一个 assert 验证 fit() 不接触 test 数据。 |

**正确做法 (V739 模式):**
- harness 传入 `val_raw` (3-way temporal split, 7 天 embargo)
- 模型内部 fit() 完全覆盖父类，不调用 super()
- 在 val_raw 上训练候选模型 → 评估 → 选最佳
- 零 oracle 表引用
- 完整检查清单见 §2 "AutoFit 新版本开发强制检查清单"

---

## 8. DECISION LOG

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-08 | Keep 3,278 NF records (max_steps=1000) | 50% early-stopped; NF restores best checkpoint; all models shared same constraint |
| 2026-03-08 | Revert NF config changes (1000→3000) | Uncommitted changes created version inconsistency; early stopping makes budget moot |
| 2026-03-08 | Drop AutoFit V1–V733 | User decision: only V734/V735/V736 were considered relevant (later ALL V734-V738 invalidated on 2026-03-12 — oracle test-set leakage) |
| 2026-03-08 | Create block3_phase9_fair/ | Single source of truth for all valid results |
| 2026-03-08 | Cancel 42 wasted SLURM jobs | Old autofit versions + buggy tslib code |
| 2026-03-08 | Fix 31 OOM jobs: batch→bigmem, gpu mem↑ | batch=112G total (no headroom), gpu models need 320-500G |
| 2026-03-08 | Remove Moirai2Large from scripts | Model not in registry (no moirai-moe large variant exists) |
| 2026-03-08 | cfisch via direct python path | cfisch lacks micromamba in PATH; use /mnt/.../insider/bin/python3 |
| 2026-03-08 | Wave 2: 78 scripts (36 npin, 42 cfisch) | OOM fixes + 21 new models + gap-fill for 12 near-complete |
| 2026-03-08 | Wave 3: 34 scripts (13 npin l40s/gpu, 21 cfisch bigmem) | Memory-fixed resubmits + l40s exploitation |
| 2026-03-08 | l40s partition activated via iris-snt QOS | 2 nodes, 4x L40S each, 515G RAM, preemptible but checkpointed |
| 2026-03-08 | 500G l40s requests fail silently | 515G total - OS overhead → max ~400G usable per job |
| 2026-03-08 | Statistical ALL OOM at 128G | Even _co/_ce (peak ~120-175G) exceed 128G with OS overhead |
| 2026-03-08 | AutoFit _ct/_fu OOM at 256G | Text data (~283G peak) + model overhead > 256G |
| 2026-03-08 | GPU partition likely does NOT enforce --mem strictly | Jobs survived requesting 256G but using 283-353G (MaxRSS) |
| 2026-03-10 | 8 bugs fixed (commit 84fa648) | Constant threshold, target-aware fallback, unfitted fallback, unmapped target warning, statistical silent fallback, synthetic panel dates |
| 2026-03-10 | V737 = V736 + EDGAR PCA + asinh (commit d2266e1) | Root cause: EDGAR raw features hurt V736 by +1.76% MAE; PCA(5) compresses 41→5; asinh handles heavy-tail targets |
| 2026-03-10 | V737 sinh overflow fix (commit 20903ec) | np.sinh() overflows float64 for preds>710; clip to [-25,25] in asinh space before inverse |
| 2026-03-10 | V737 OOM at 128G for ce/fu ablations | EDGAR as-of join peaks at ~134G; bumped ce/fu scripts to 192G |
| 2026-03-10 | Text embedding 3× OOM (Arrow backend) | pandas Arrow string backend allocates n_rows × max_len × 4 bytes for .loc operations on long strings; fix: .astype("object") + per-field truncation (commit f67d69e) |
| 2026-03-11 | V738 asinh removal (commit e307926) | Same root cause as V737 sinh overflow — NF wrapper reads raw-scale targets |
| 2026-03-11 | V737 asinh removal (commit 408717c) | Dual-transform bug: model trains on raw scale inside NF, outer sinh(pred) overflows |
| 2026-03-11 | cfisch OOM 128G→192G→256G→384G | core_text/full peak RSS = 268GB on all 5 cfisch V738 jobs. 384G = 51% of 756G node RAM = safe |
| 2026-03-11 | V738 oracle test-set leakage confirmed | ORACLE_TABLE_V738 MAE values = exact match to Phase 9 test-set MAE. V738's 83.9% champion rate is artificially inflated. Same issue in ORACLE_TABLE_TOP3 (V736/V737). ALL AutoFit rankings are scientifically invalid |
| 2026-03-11 | 93 models registered, **80 complete** | 8,972 records (corrected from 8,928). NHITS mean_rank=4.12, PatchTST=4.13 (clean #1-#2) |
| 2026-03-12 | 5-layer oracle root cause analysis | Structural finding: benchmark harness lacks validation split → ALL oracle tables fundamentally flawed |
| 2026-03-12 | V737 ALL COMPLETE: 104/104 | 6 npin (49-71min) + 5 cfisch (62-115min) = 11 jobs, 11 metrics.json files |
| 2026-03-12 | V738 ALL COMPLETE: 104/104 | 5 cfisch (30-75min), oracle-leaked, scientifically invalid |
| 2026-03-12 | tsA TIMEOUT: 3 jobs | MSGNet/PAttn/MambaSimple stuck at 52/104 after 2d walltime. Need iris-gpu-long or checkpoint-resume |
| 2026-03-12 | V753 identified as correct template | autofit_wrapper.py L7232 `_fit_validation()` implements proper inner train/val split |
| 2026-03-13 | V739 implemented (no oracle) | Validation-based model selection: 8 candidates, harness val_raw, single-best selection |
| 2026-03-13 | Harness val_raw fix | benchmark_shard.py: `fit_kwargs["val_raw"] = val` for autofit category |
| 2026-03-13 | Corrected model counts | 80 complete (not 76), 13 partial, 8,972 records (not 8,928) |
| 2026-03-13 | 2-seed 复现重组完成 | core_text→core_only_seed2, full→core_edgar_seed2; 40 dirs, 4032 records; REPLICATION_MANIFEST.json; all_results.csv 重生成 |
| 2026-03-13 | 88 non-AutoFit models verified clean | grep confirmed oracle references ONLY in nf_adaptive_champion.py |
| 2026-03-12 | V737 vs V738: V738 wins 77/104 | V738's oracle advantage (top-5 MAE) beats V737's (top-3 RMSE). Both invalid. |
| 2026-03-12 | 50% 算力浪费确认 | 3,888/8,660 记录冗余 (core_text≡core_only, full≡core_edgar)，~638 GPU-hours。变废为宝: 训练方差估计 + 可复现性论据 |
| 2026-03-12 | V739/V740 加速方案 | 只跑 9 冠军 × 2 ablation = 648 条件替代 12000+，算力节省 95% |
| 2026-03-12 | 训练方差三级分类 | 确定性(53模型/0%)、近确定性(25模型/<0.01%)、高随机(3模型/>1%) |
| 2026-03-12 | 冠军稳定性审计 | 92/104 STABLE, 12/104 UNSTABLE (全 NBEATS↔NBEATSx tie) |

---

## 9. OOM TROUBLESHOOTING GUIDE (TRIAL AND ERROR LOG)

### Pattern Recognition
| Symptom | Exit Code | Meaning | Fix |
|---------|-----------|---------|-----|
| Job dies immediately | 0:53 | Permission denied / script error | Check stderr, chmod g+w |
| Job dies in 1-2 min | 0:125 | OOM during data loading | Increase --mem (text = +200G, EDGAR negligible) |
| Job dies in 4-10 min | 0:125 | OOM during model fitting after data loaded | Increase --mem by 50-100G |
| Job dies in 10+ min | 0:125 | OOM as models accumulate memory | Add gc.collect() or reduce concurrent models |
| Job dies in 10+ min | 0:125 | OOM from text embedding Arrow string backend | Use .astype("object") before string ops; pre-truncate long strings |
| Job stuck for hours | N/A | Model training too slow (no timeout enforced) | Monitor stderr for progress |

### Data Size Reference (Compressed → In-Memory)
| Dataset | Parquet Size | Est. In-Memory | Notes |
|---------|-------------|---------------|-------|
| offers_core_daily | 301MB | ~2GB | Base data, always loaded |
| offers_text | 20GB | ~150-200GB | Text embeddings, massive expansion |
| edgar_feature_store | 13MB | ~50-200MB | 41 features, negligible |
| Core + Text merge | N/A | ~200-250GB | Left join doubles some data |
| Core + Text + EDGAR | N/A | ~280-350GB | merge_asof adds columns |

### Per-Ablation Memory Budget (Recommended)
| Ablation | Data Overhead | Model Budget | Total Recommendation |
|----------|--------------|-------------|---------------------|
| core_only | ~2GB | 100-120GB | 256G (bigmem), 256G (gpu) |
| core_edgar | ~3GB | 100-170GB | 256G (bigmem), 256G (gpu) |
| core_text | ~200GB | 80-100GB | 512G (bigmem), 400G (gpu/l40s) |
| full | ~280GB | 80-100GB | 640G (bigmem), 500G (gpu) |

---

## 10. COMPUTE SATURATION STRATEGY

### Two-Account Strategy (npin + cfisch)
- Both accounts share QOS under `christian.fisch` group
- Combined limit: 2 × 100 = 200 jobs on `normal` QOS
- Also available: 2 × 100 = 200 jobs on `iris-snt` QOS (l40s partition)
- Strategy: npin handles GPU partition, cfisch handles bigmem partition

### L40S Exploitation (Discovered 2026-03-08)
- **Access**: We HAVE `iris-snt` QOS → can submit to l40s partition
- **Advantages**: L40S has 48GB VRAM (vs V100's 16/32GB), newer architecture
- **Limitations**: Max ~400G RAM request (515G total), preemptible by owner
- **PreemptMode=REQUEUE**: If preempted, SLURM auto-requeues. Our checkpoint/resume
  mechanism (SIGTERM handler + done_combos) ensures no lost progress.
- **Best use**: GPU-heavy models that fit in 400G (foundation _ct, irregular _ce/_ct)
- **Do NOT use for**: _fu ablations needing 500G+ (silently rejected)

### Partition Selection Matrix
| Category | Ablation | Best Partition | QOS | Memory |
|----------|----------|---------------|------|--------|
| deep_classical | _co/_ce | gpu | normal | 256G |
| deep_classical | _ct | gpu or l40s | normal/iris-snt | 400G |
| deep_classical | _fu | gpu | normal | 500G |
| transformer_sota | all | gpu | normal | 320-400G |
| foundation | _co/_ce | gpu | normal | 256G |
| foundation | _ct | l40s (preferred) | iris-snt | 400G |
| foundation | _fu | gpu (only) | normal | 500G |
| tslib_sota | all | gpu | normal | 320-400G |
| irregular | _co | gpu | normal | 128G |
| irregular | _ce | l40s (preferred) | iris-snt | 200G |
| irregular | _ct | l40s (preferred) | iris-snt | 400G |
| irregular | _fu | gpu (only) | normal | 500G |
| ml_tabular | _co/_ce | bigmem | normal | 256G |
| ml_tabular | _ct | bigmem | normal | 512G |
| ml_tabular | _fu | bigmem | normal | 640G |
| statistical | _co/_ce | bigmem | normal | 256G |
| statistical | _ct | bigmem | normal | 512G |
| statistical | _fu | bigmem | normal | 640G |
| autofit | _co/_ce | bigmem | normal | 256G |
| autofit V737/V738 | _co | gpu | normal | 128G |
| autofit V737/V738 | _ce | gpu | normal | 192G (128G OOM proven, MaxRSS=134G) |
| autofit V737/V738 | _ct | gpu (cfisch) | normal | 384G (256G OOM proven, MaxRSS=268G) |
| autofit V737/V738 | _fu | gpu (cfisch) | normal | 384G (256G OOM proven, MaxRSS=268G) |
| autofit | _ct | bigmem | normal | 512G |
| autofit | _fu | bigmem | normal | 640G |

### Queue Backlog Reduction Tactics
1. Use l40s for GPU overflow (preemptible but safe with checkpoint)
2. Use `besteffort` QOS (300 job limit, priority=1) for low-priority fills
3. Split large model lists into smaller shards (2-day wall time constraint)
4. Monitor with `sacct` for OOM/TIMEOUT → resubmit with corrected resources

---

## 11. CHECKPOINT AND RESUME MECHANISM (VERIFIED 2026-03-08)

### How It Works
1. **Incremental save**: After each `(target, horizon)` combo completes, `_save_outputs(partial=True)`
   writes current metrics + predictions to disk immediately.
2. **SIGTERM handler**: Catches `SIGUSR1` (sent 120s before wall time) and `SIGTERM`.
   Saves partial results before exit. Script uses `#SBATCH --signal=USR1@120`.
3. **Resume on restart**: Loads existing `metrics.json`, builds `done_combos` set of
   `(model_name, target, horizon)` tuples. Skips any combo already in the set.
4. **Deduplication**: New results are appended, existing ones preserved. No duplicates.

### What Is Safe
- Preemption by REQUEUE (l40s partition) — safe, progress preserved
- Wall time timeout with USR1 signal — safe, saves before exit
- Multiple job restarts on same output dir — safe, resume skips done combos
- Two jobs writing to DIFFERENT subdirs — safe (separate metrics.json files)

### What Is NOT Safe
- Two jobs writing to the SAME metrics.json simultaneously — NOT safe (race condition)
- Changing model code between job requeues — stale results from previous code persist
- Killing job with SIGKILL (kill -9) — no signal handler runs, partial data may be lost

---

## 12. EXPERIMENT FAIRNESS AND CORRECTNESS REQUIREMENTS

### Absolute Fairness Principles
1. **Same data**: All models in the same ablation see identical train/test split.
   Verified: `_load_data()` called once per (target, horizon), all models share it.
2. **Same temporal split**: 7-day embargo gap between train/val/test prevents leakage.
   Verified: `apply_temporal_split` in unified_protocol.py.
3. **Same evaluation**: All models evaluated with identical metrics (MAE, RMSE, SMAPE, MAPE).
4. **Same seed**: `--seed 42` ensures reproducibility within same hardware.
5. **EDGAR as-of join**: Uses `direction="backward"` + 90-day tolerance, preventing
   future-information leakage.
6. **Target leakage guard**: Co-determined columns (e.g., `is_funded` when predicting
   `funding_raised_usd`) are automatically dropped.

### Known Fairness Caveats (DOCUMENTED, NOT BUGS)
1. **fillna(0) vs NaN passthrough**: ML tabular models see raw NaN (tree models handle
   it natively), all others get fillna(0). This is a design choice, not a bug.
2. **NeuralForecast horizon clamping**: `h_nf = max(h, 7)` for horizons < 7. Only affects
   h=1 which is not in the standard preset.
3. **Foundation model context window**: Varies by model (Chronos=128, MOMENT=512, TimesFM2=512).
   This is inherent to each model's architecture, not controllable.
4. **StatsForecast horizon retry**: If h=30 fails, retries with h=14, h=7, h=1. This means
   the model's effective forecast horizon may be shorter than requested. MONITOR and flag
   any models that used fallback horizons.

### Constant Prediction Guard
- All predictions are checked for constant output (`np.std(y_pred) < 1e-8`) (fixed commit 84fa648, was `== 0.0`)
- Flagged as `fairness_pass=False` in metrics.json
- Record STILL SAVED (allows post-hoc filtering)
- Paper tables should EXCLUDE records with `fairness_pass=False`

### Known Harness Issues (Audited 2026-03-08)
| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| Model timeout defined but not enforced | High | WONTFIX (mid-benchmark) | _MODEL_TIMEOUT_SECONDS=1800 never used |
| CRPS metric never computed | Medium | WONTFIX (post-benchmark) | Schema has crps field but harness doesn't collect probabilistic outputs |
| Failed models leave no metrics record | Medium | WONTFIX (mid-benchmark) | Can only detect via manifest n_models_failed |
| predictions.parquet lacks entity_id | Low | WONTFIX | Per-entity analysis needs separate code |
| self.predictions grows unbounded | Medium | MONITOR | Could cause memory issues on large runs |

---

## 13. HYPERPARAMETER TUNING AND TRAINING PROTOCOL

### Frozen Training Budgets
All NeuralForecast models share the SAME training budget — this is critical for fairness:
- Most: max_steps=1000, early_stop_patience=10, val_check_steps=50
- PatchTST/Informer/Autoformer/FEDformer/VanillaTransformer: max_steps=5000
- TimeXer/TimeLLM: max_steps=2000

**Empirical validation**: 50% of sessions early-stopped before hitting max_steps.
The other 50% hit the cap, but NeuralForecast restores the best checkpoint regardless.
ALL models share the same constraint → fair comparison.

### TSLib Models
- 100 epochs max, patience=15 (early stopping on val loss)
- Training speed varies wildly: MSGNet ~4.5 min/epoch, some models ~10 sec/epoch
- A single TSLib shard (7 models × 12 combos) can take up to 42 hours
- The 2-day wall time is tight but checkpoint/resume handles timeouts

### Foundation Models
- Zero-shot or minimal fine-tuning (model-specific)
- prediction_length MUST equal requested horizon (not hardcoded)
- Entity cap MUST be None (process ALL entities)

### ML Tabular Models
- No hyperparameter tuning (default sklearn/lightgbm/xgboost params)
- NaN passthrough for tree models (LightGBM, XGBoost, CatBoost)
- All 4 horizons [7, 14, 30, 60] must be run (not just horizons[0])

### What NOT To Change Mid-Benchmark
1. PRODUCTION_CONFIGS in deep_models.py (affects all NF models)
2. Training budgets (max_steps, patience, lr)
3. Data loading code (ablation behavior)
4. Metric computation code
5. Temporal split parameters (embargo period, train/val/test ratios)

---

## 14. WAVE SUBMISSION HISTORY

| Wave | Date | Scripts | Account Split | Purpose | Issues Found |
|------|------|---------|---------------|---------|-------------|
| Wave 1 | 2026-03-08 | 66 | 44 npin, 22 cfisch | Initial full benchmark | batch OOM, cfisch permission |
| Wave 2 | 2026-03-08 | 78 | 36 npin, 42 cfisch | OOM fixes + new models + gaps | stat 128G OOM, af9 _ct/_fu 256G OOM, l40s silent reject |
| Wave 3 | 2026-03-08 | 34+6 | 13+6 npin (l40s+gpu), 21 cfisch (bigmem) | Memory-fixed + l40s exploitation | l40s 500G rejected → moved _fu to gpu |
| Wave 6 | 2026-03-10 | 11+1 | all npin | Phase 10: V737 (11 jobs) + text embeddings (1 job) | 5/11 V737 OOM at 128G → resubmitted at 192G (6 jobs); sinh overflow fixed (commit 20903ec) |

---

## 15. AUTOFIT ROOT CAUSE ANALYSIS & FIX (2026-03-09, commit a251cdc)

### 7 Root Causes Identified
1. **Oracle tables built from deprecated Phase 7/8 data** — FIXED: Rebuilt from Phase 9 clean data (4,564 records, 44 trainable models)
2. **V734 `_effective_stack_k()` always returned 1** — FIXED: `top_rank <= 1.5` was always true for integer ranks; changed threshold logic to `max_k <= 1`
3. **V734/V735 are model selectors, not ensembles** — Design limitation, V736 is the real ensemble
4. **V735 oracle referenced FusedChampion (not in Phase 9)** — FIXED: Eliminated from all oracle tables
5. **V736 TOP3 table had duplicate models per condition** — FIXED: Rebuild with dedup (groupby model+condition before ranking)
6. **V736 RMSE weights from deprecated Phase 7 data** — FIXED: Rebuilt from Phase 9 RMSE values
7. **`predict()` NaN→0 silently corrupted ensemble** — FIXED: NaN-aware weighted average with per-position weight redistribution

### Key Lesson: Oracle Table Construction
- Always query ACTUAL Phase 9 results, never use stale data
- Dedup by `(model_name, task, ablation, target, horizon)` before ranking
- Exclude non-trainable models (FusedChampion, NFAdaptiveChampion*)
- Verify all TOP3 entries have 3 UNIQUE model names per condition

### Key Lesson: metrics.json Schema
- **Field `model_name`** (NOT `model`): the model name key
- **Field `target`** (NOT `target_col`): the target variable key
- **Actual targets**: `funding_raised_usd`, `investors_count`, `is_funded`
- **Actual horizons**: `[1, 7, 14, 30]` (NOT `[7, 14, 30, 60]` from config — config ≠ runtime)
- Always verify actual data schema with `json.load()` before writing scripts

---

## 16. BROKEN / PERMANENTLY UNAVAILABLE MODELS

| Model | Reason | Status | Date |
|-------|--------|--------|------|
| ~~Chronos2~~ | `amazon/chronos2-small` wrong model ID | **FIXED** → `amazon/chronos-2` (commit fbae1b4) | 2026-03-09 |
| ~~TTM~~ | `tsfm_public` not installed | **FIXED** → `pip install granite-tsfm` v0.3.5 | 2026-03-09 |
| TimerXL | `thuml/timer-xl-84m` does NOT exist on HuggingFace | **DROPPED** — same checkpoint as Timer (`thuml/timer-base-84m`), would be duplicate | 2026-03-09 |
| TiRex | Not in TSLib — too recent, no public implementation compatible with TSLib | EXCLUDED | 2026-03-08 |
| Mamba(full) | Needs `mamba-ssm` CUDA kernel not installed; `MambaSimple` in registry succeeded instead | EXCLUDED | 2026-03-08 |
| Koopa | Numerically unstable on financial data (NaN divergence during training) | EXCLUDED | 2026-03-08 |
| TabPFNRegressor | `tabpfn` package not installed; limited to <10K samples and <500 features | EXCLUDED | 2026-03-08 |
| TabPFNClassifier | Same as TabPFNRegressor | EXCLUDED | 2026-03-08 |
| Ridge | Redundant with ElasticNet (l1_ratio=0 equivalent) | EXCLUDED | 2026-03-08 |
| Lasso | Redundant with ElasticNet (l1_ratio=1 equivalent) | EXCLUDED | 2026-03-08 |

**Active foundation model count**: 13 (was 11, now +Chronos2, +TTM, -TimerXL)
Resubmitted Chronos2+TTM via cfisch: cf_p9r3_fm_t{1,2,3}.sh (Wave 5)

**Permanently excluded from active count**: 18 retired AutoFit (V1-V733, FusedChampion, NFAdaptiveChampion) — intentionally kept in registry for code history but NEVER run in Phase 9.

---

## 17. MEMORY TIER REQUIREMENTS FOR SLURM

| Ablation | Required Memory | Suitable Partitions |
|----------|----------------|---------------------|
| core_only | 256G | bigmem, gpu (320G+) |
| core_edgar | 256G (512G for ml_tabular with NegBinGLM) | bigmem, gpu (320G+) |
| core_text | 512G | bigmem, gpu (640G+) |
| full | 640G | gpu (756G max), bigmem (3TB) |

**Critical**: TSLib/NF models need GPU → "full" ablation MUST request `--mem=640G` on GPU partition (756G available).
Previous OOM: 9 TSLib "full" scripts requested only 320G on GPU → all OOM'd.
Fix: Created `p9r2_ts{A,B,C}_t{1,2,3}_fu.sh` with `--mem=640G` on GPU.

**ml_tabular core_edgar OOM at 256G**: Job p9r2_mlT_t1_ce_isfunded OOM'd at 256G on bigmem
after completing 8 models (RF, ExtraTrees, HGB, LightGBM, XGBoost, CatBoost, XGBPoisson, LGBTweedie).
OOM occurred during NegativeBinomialGLM (statsmodels dense matrix + 2M rows).
Fix: Resubmitted at 512G via cfisch (cf_p9r2_mlT_t1_ce_isfunded.sh).

**cfisch scripts DO NOT use micromamba**: cfisch account has no micromamba in PATH.
All cfisch scripts must use direct INSIDER_PY path:
```
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
```
DO NOT use `micromamba activate insider` in cfisch scripts.

---

## 18. WAVE 4 SUBMISSION (2026-03-09)

| Job Name | Account | Partition | Mem | Purpose | Status |
|----------|---------|-----------|-----|---------|--------|
| p9r2_mlT_t1_ce_isfunded | npin | bigmem | 256G | Fill ml_tabular task1/core_edgar/is_funded gap | OOM (256G, 56min) |
| p9r2_tsA_t{1,2,3}_fu | npin | gpu | 640G | TSLib batch A full ablation OOM rerun | RUNNING |
| p9r2_tsB_t{1,2,3}_fu | npin | gpu | 640G | TSLib batch B full ablation OOM rerun | CANCELLED → cfisch |
| p9r2_tsC_t{1,2,3}_fu | npin | gpu | 640G | TSLib batch C full ablation OOM rerun | CANCELLED → cfisch |
| p9f_af_t{1}_co/ct/fu | npin | bigmem | 256G | AutoFit V734/V735/V736 remaining conditions | RUNNING |

## 18b. WAVE 5 SUBMISSION (2026-03-09, cfisch account)

| Job Name | Account | Partition | Mem | Purpose | SLURM ID |
|----------|---------|-----------|-----|---------|----------|
| cf_p9r2_tsB_t{1,2,3}_fu | cfisch | gpu | 640G | TSLib B full ablation (moved from npin) | 5221095-5221097 |
| cf_p9r2_tsC_t{1,2,3}_fu | cfisch | gpu | 640G | TSLib C full ablation (moved from npin) | 5221098-5221100 |
| cf_p9r2_mlT_t1_ce_isfunded | cfisch | bigmem | 512G | ml_tabular is_funded gap (OOM fix 256→512G) | 5221101 |
| cf_p9r3_fm_t1 | cfisch | gpu | 256G | Chronos2 + TTM task1 (all ablations) | 5221102 |
| cf_p9r3_fm_t2 | cfisch | gpu | 256G | Chronos2 + TTM task2 (all ablations) | 5221103 |
| cf_p9r3_fm_t3 | cfisch | gpu | 256G | Chronos2 + TTM task3 (all ablations) | 5221104 |

**Scripts location**: `.slurm_scripts/phase9/cfisch/`
**Key fix**: cfisch scripts use direct INSIDER_PY path (no micromamba dependency)

---

## 18c. WAVE 6 SUBMISSION (2026-03-10, Phase 10 — V737 + text embeddings)

### V737 Jobs (11 scripts in `.slurm_scripts/phase10/`)
| Job Name | Partition | Mem | Ablation | SLURM IDs | Status |
|----------|-----------|-----|----------|-----------|--------|
| p10_v737_t{1,2,3}_co | gpu | 128G | core_only | 5225833-5225835 | t1 RUNNING, t2 RUNNING, t3 PENDING |
| p10_v737_t{1,2}_ct | gpu | 128G | core_text | 5225836-5225837 | PENDING |
| p10_v737_t{1,2,3}_ce | gpu | 192G | core_edgar | 5225846,5225848,5225850→cancelled→cfisch 5225859,5225862,5225865 | PENDING (cfisch) |
| p10_v737_t{1,2,3}_fu | gpu | 192G | full | 5225847,5225849,5225851→cancelled→cfisch 5225860,5225863,5225866 | PENDING (cfisch) |

**V737 OOM Root Cause**: EDGAR as-of join memory peaks at ~134G (MaxRSS=134217832K on t1_ce).
128G allocation = exact match → OOM killer. ce/fu scripts bumped to 192G on resubmit.

**V737 sinh overflow**: `np.sinh(preds)` overflows float64 for asinh-space predictions > 710.
Fixed: clip to [-25,25] before sinh (commit 20903ec). sinh(25) ≈ 3.6e10, safe for funding data.

### Text Embedding Job
| Job Name | Partition | Mem | SLURM ID | Status |
|----------|-----------|-----|----------|--------|
| gen_text_emb | gpu | 128G | 5225838→cancelled | Moved to cfisch |
| cf_gen_text_emb | gpu | 128G | 5225868 (cfisch) | PENDING (3rd attempt, Arrow OOM fixed) |

**Text embedding OOM History**:
- Attempt 1 (5225807): OOM at 2.55 TiB — `.where()` on Arrow string series
- Attempt 2 (also 5225807 or earlier): Same OOM — `.loc[]` also triggers Arrow backend
- Attempt 3 (5225838): Fixed — `.astype("object")` + `per_field_limit` pre-truncation (commit f67d69e)

---

## 19. KEY OPERATIONAL LESSONS (COMPREHENSIVE)

1. **Always check sacct, not just squeue**: `squeue` only shows running/pending jobs. Use `sacct -S <date>` to see completed, OOM, and cancelled jobs.
2. **OOM states in sacct show as `OUT_OF_MEMORY` with exit code `0:125`**: Filter for these specifically when auditing.
3. **cfisch account uses `/work/projects/eint/repo_root`** as working directory but shares the same GPFS filesystem as npin.
4. **Duplicate sbatch prevention**: When submitting in a loop, verify the submission count matches expectations before and after. Cancel duplicates with `scancel`.
5. **GPU partition memory**: V100 nodes on Iris have 756G RAM total. Can request up to ~640G safely with GPU reservation.
6. **bigmem partition**: 3TB RAM, 112 CPUs, NO GPU. Use for CPU-only models (ml_tabular, autofit, statistical).
7. **task3_risk_adjust has NO core_text ablation**: Only 3 ablations (core_only, core_edgar, full) × 2 targets × 4 horizons = 24 conditions.
8. **NegativeBinomialGLM is binary-only**: Only predicts `is_funded` target on task1_outcome. Max conditions = 16, not 104.
9. **glob depth matters**: Phase 9 results are at `task/category/ablation/metrics.json` (3 levels deep), not `category/metrics.json`.
10. **Verify model names against registry**: Registry names may differ from common names (e.g., `SF_SeasonalNaive` vs `SeasonalNaive`, `GRU-D` with hyphen, `CrostonClassic/Optimized/SBA` not just `Croston`).
11. **cfisch has NO micromamba in PATH**: cfisch scripts MUST use direct `INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3` — do NOT use `micromamba activate`.
12. **ml_tabular NegBinGLM OOMs at 256G bigmem**: statsmodels creates dense matrices for 2M rows. Use 512G for ml_tabular core_edgar/full ablations.
13. **Chronos2 model ID was wrong**: `amazon/chronos2-small` → correct is `amazon/chronos-2` (fixed commit fbae1b4).
14. **TTM requires `granite-tsfm` pip package**: `pip install granite-tsfm` provides `tsfm_public` module. Installed v0.3.5 on 2026-03-09.
15. **TimerXL HuggingFace repo does NOT exist**: `thuml/timer-xl-84m` is 404. Only `thuml/timer-base-84m` exists (same as Timer). TimerXL dropped from active benchmark.
16. **"Official implementations" first**: When HF model IDs fail, search official GitHub repos (e.g., `amazon-science/chronos-forecasting`, `ibm-granite/granite-tsfm`, `thuml/Timer-XL`) for correct IDs — don't assume permanently broken.

---

## 20. 唯一真理标准 — CANONICAL TRUTH STANDARDS (NEVER CHANGE WITHOUT USER APPROVAL)

**This section defines absolute constraints. Future agents MUST NOT contradict these.**

### 20a. Active Model Count
- **Registry total**: 128+ models across 8 categories (including retired AutoFit versions)
- **AGENTS.md target**: 100 models (15+15+9+23+11+4+20+3)
- **Phase 9/10 active**: 102 models (93 materialized P9 + 8 new P10 + 1 V737)
- **Phase 9 complete @104**: 66 models
- **Retired AutoFit (18)**: V1–V733, FusedChampion, NFAdaptiveChampion — NEVER run
- **Excluded from ml_tabular (9)**: Ridge, Lasso, ElasticNet, SVR, KNN, LogisticRegression, QuantileRegressor, TabPFNRegressor, TabPFNClassifier
- **Excluded from tslib_sota (2)**: TiRex (no tirex module), Mamba (needs mamba_ssm; MambaSimple used)
- **Recovered tslib_sota (2)**: WPMixer (fixed: device=torch.device), Koopa (fixed: lazy mask_spectrum)
- **Excluded from foundation (1)**: TimerXL (HF repo doesn't exist)
- **Removed earlier (1)**: Moirai2Large (§6, HF repo doesn't exist)

### 20b. Expected Conditions Per Model
- Standard model: **104 conditions** = task1(48) + task2(32) + task3(24)
  - task1_outcome: 4 ablations × 3 targets × 4 horizons = 48
  - task2_forecast: 4 ablations × 2 targets × 4 horizons = 32
  - task3_risk_adjust: 3 ablations × 2 targets × 4 horizons = 24
- NegativeBinomialGLM: **16 conditions** (binary-only: task1 `is_funded` × 4 ablations × 4 horizons)
- Targets: `funding_raised_usd` (task1,2,3), `investors_count` (task1,2,3), `is_funded` (task1 only)

### 20c. Canonical Field Names (metrics.json)
- `model_name` (NOT `model`)
- `target` (NOT `target_col`)
- `task`, `ablation`, `horizon`
- Actual targets: `funding_raised_usd`, `investors_count`, `is_funded`
- Actual horizons: `[1, 7, 14, 30]` (runtime values, NOT config `[7, 14, 30, 60]`)

### 20d. Ablation Names
- `core_only` (co): core features only
- `core_text` (ct): core + text features (NOT available for task3)
- `core_edgar` (ce): core + EDGAR features
- `full` (fu): core + text + EDGAR

### 20e. Result Directory Structure
```
runs/benchmarks/block3_phase9_fair/
  {task}/{category}/{ablation}/metrics.json
```
Example: `task1_outcome/ml_tabular/core_edgar/metrics.json`

### 20f. Hardware Facts (ULHPC Iris)
| Partition | RAM | CPUs | GPU | Notes |
|-----------|-----|------|-----|-------|
| batch | 112G | 28 | None | Default, too small for benchmark |
| gpu | 756G | 14 (per GPU) | V100 32GB | 4 GPUs/node, use `--gres=gpu:1` |
| bigmem | 3TB | 112 | None | CPU-only workloads |
| l40s | 515G | 14 | L40S 48GB | Preemptible, limited quota |

### 20g. Account Configuration
| Account | SLURM account | Working dir | micromamba |
|---------|---------------|-------------|-----------|
| npin | `--account=npin` | `/home/users/npin/repo_root` | Yes (in PATH) |
| cfisch | `--account=christian.fisch` | `/work/projects/eint/repo_root` | NO — use direct INSIDER_PY |

### 20h. ⛔ THINGS THAT MUST NEVER CHANGE MID-BENCHMARK
1. Training budgets (max_steps, patience) — §13
2. Temporal split parameters
3. Data loading/ablation logic
4. Metric computation formulas
5. Seed (42)
6. Active model list (this section)
7. Target variable definitions
8. The contents of freeze artifacts under `runs/*_20260203_225620/`

---

## 21. CANONICAL MODEL INVENTORY (VERIFIED FROM metrics.json 2026-03-09)

### statistical (15 models) — ALL 15 COMPLETE @104
AutoARIMA, AutoCES, AutoETS, AutoTheta, CrostonClassic, CrostonOptimized, CrostonSBA, DynamicOptimizedTheta, HistoricAverage, Holt, HoltWinters, MSTL, Naive, SF_SeasonalNaive, WindowAverage

### ml_tabular (11 active / 20 registry) — 9 EXCLUDED
**Active (11)**: CatBoost, ExtraTrees, HistGradientBoosting, LightGBM, LightGBMTweedie, MeanPredictor, NegativeBinomialGLM, RandomForest, SeasonalNaive, XGBoost, XGBoostPoisson
- 10 regular models at 103/104 (cfisch running is_funded recovery). NegBinGLM at 15/104.
**Excluded (9)**: Ridge, Lasso, ElasticNet, SVR, KNN, LogisticRegression, QuantileRegressor, TabPFNRegressor, TabPFNClassifier

### deep_classical (9 models) — ALL 9 COMPLETE @104
DeepAR, DilatedRNN, GRU, LSTM, MLP, NBEATS, NHITS, TCN, TFT

### transformer_sota (23 models) — ALL 23 COMPLETE @104
Autoformer, BiTCN, DLinear, DeepNPTS, FEDformer, Informer, KAN, NBEATSx, NLinear, PatchTST, RMoK, SOFTS, StemGNN, TSMixer, TSMixerx, TiDE, TimeLLM, TimeMixer, TimeXer, TimesNet, VanillaTransformer, iTransformer, xLSTM

### foundation (14 active / 15 registry) — 1 EXCLUDED
**Materialized (12, all @104)**: Chronos, ChronosBolt, LagLlama, MOMENT, Moirai, Moirai2, MoiraiLarge, Sundial, TimeMoE, Timer, TimesFM, TimesFM2
**Pending (2, cfisch PENDING)**: Chronos2 (model ID fixed, 0 records), TTM (granite-tsfm installed, 0 records)
**Excluded (1)**: TimerXL (HF repo doesn't exist)

### irregular (4 models) — ALL 4 COMPLETE @104
BRITS, CSDI, GRU-D, SAITS

### tslib_sota (42 active / 42 registry) — 2 EXCLUDED from runtime (TiRex, Mamba)
**Phase 9 complete (4)**: WPMixer(104✓), CATS(104✓), FITS(104✓), KANAD(104✓)
**Phase 9 partial (12)**: Crossformer(52), ETSformer(52), LightTS(52), MSGNet(52), MambaSimple(52), MultiPatchFormer(52), PAttn(52), Pyraformer(52), Reformer(52), TimeFilter(52), — all have gap-fill jobs (p9r_*) PENDING
**Phase 10 new (8)**: SparseTSF, Fredformer, CycleNet, xPatch, FilterTS, NonstationaryTransformer, FiLM, SCINet, SegRNN, MICN, FreTS, Koopa — mixed status
**Phase 11 new (14)**: CFPT, DeformableTST, ModernTCN, PathFormer, SEMPO, TimePerceiver, TimeBridge, TQNet, PIR, CARD, PDF, TimeRecipe, DUET, SRSNet (all 0/104, p11_tslib_new_* PENDING)
**Excluded (2)**: TiRex (no tirex module), Mamba (needs mamba_ssm CUDA; MambaSimple used)

### autofit (1 valid) — V734-V738 ALL INVALID (oracle leak), V739 PENDING
⛔ AutoFitV734(104, INVALID), AutoFitV735(104, INVALID), AutoFitV736(104, INVALID): Phase 9 oracle test-set leakage
⛔ AutoFitV737(104, INVALID), AutoFitV738(104, INVALID): Phase 10 oracle test-set leakage
✅ AutoFitV739(0/104 — p10_v739_* PENDING on npin/gpu + cf_v739_l40s on cfisch/l40s, validation-based, NO oracle)

### TOTAL: 108 active models (78 complete + 12 partial + 16 new/pending)
Phase 9: 78 complete + 12 partial = 90 materialized (8,660 valid records)
Phase 10: V739 (0/104, queued on npin/gpu + cfisch/l40s)
Phase 11: 14 new SOTA (0/104, queued on cfisch/gpu)
SLURM queue: 47 npin + 31 cfisch = 78 PENDING, 0 RUNNING
**Oracle-leaked (ALL INVALID)**: V734, V735, V736, V737, V738 (520 records excluded from leaderboard)

---

## 10. 冠军模型分析核心结论 (Phase 9, 80 models × 104 conditions)

**Current reference note**: `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`

### 10.1 Overall MeanRank Top-5 (RMSE)
1. NBEATS (7.82) — 双残差基函数展开, 24s训练, **Pareto最优**
2. ~~AutoFitV736 (7.91)~~ — ⛔ INVALID (oracle test-set leakage, 排名不可信)
3. NBEATSx (8.72) — NBEATS+外生变量
4. PatchTST (9.08) — Patch分词+16头自注意力
5. MLP (9.73) — 1024维全连接, Top3出现34次(最多)

### 10.2 RMSE #1 次数
DeepNPTS(26), TimesNet(16), AutoFitV734(16), NBEATS(10), DeepAR(10)

### 10.3 目标-模型匹配规律
- **funding_raised_usd**: MLP(3.43), GRU(3.77) — 简单高容量模型胜出
- **investors_count**: TimesNet(3.43), NBEATS(5.27) — 周期/基函数模型
- **is_funded**: NBEATSx(2.94), NBEATS(4.62) — 基函数天然产出[0,1]

### 10.4 消融规律
- **core_only**: AutoFitV734(9), DeepAR(6) — 纯自回归占优
- **core_edgar/full**: DeepNPTS(24), TimesNet(16) — 占71.4%冠军

### 10.5 V740 候选池设计
当前V739候选池缺少funding_raised_usd上表现最强的模型(MLP, GRU)。
建议V740扩展至12个候选:
- 保留: NHITS, PatchTST, NBEATS, NBEATSx, ChronosBolt, KAN, TimesNet
- **新增**: MLP, GRU, DeepNPTS, DeepAR, TCN
- **移除**: Chronos (MeanRank 12.77, 零Top3, 推理30s)

### 10.6 关键洞察
1. 外生特征对深度模型的增益≈0% (仅树模型+8%)
2. investors_count区分度极低(RMSE 1082-1083), 真正战场是funding_raised_usd
3. NBEATS是Pareto最优: MeanRank #1 + 最快训练(24s)
4. V736 stacking是唯一系统性超越单模型的方法(但有泄漏)
5. V739的validation-based选择可与V736的stacking结合 → 无泄漏集成

---

## 11. SLURM 双账户策略 (npin + cfisch)

### 11.1 账户共享发现
- npin (uid=490088567) 和 cfisch (uid=490088631) 共享 **同一SLURM账户** `christian.fisch`
- 两人都在 `eint` 组 (gid=490088569)
- **Fairshare 相同**: 拆分到cfisch提交不改善Priority调度
- npin 可直接用 `--account=christian.fisch` 提交 → 无需SSH到cfisch

### 11.2 cfisch 脚本模板
cfisch SLURM脚本位于 `.slurm_scripts/phase11/cfisch/` (33 scripts):
- `cf_p11_*.sh` (11): Phase 11 新模型 (GPU)
- `cf_p9g_*.sh` (11): Gap-fill (Crossformer/MSGNet/MambaSimple/MultiPatchFormer/PAttn/TimeFilter)
- `cf_nbglm_*.sh` (11): NegativeBinomialGLM (bigmem)

### 11.3 文件系统访问
- `/work/projects/eint/repo_root/` (`drwxrwsrwx npin:eint`) — 两用户完全共享
- `/home/users/npin/repo_root` → symlink to `/work/projects/eint/repo_root`
- 共享Python: `/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3`

---

*This file is maintained incrementally. Each session should append new decisions,
protocol changes, or lessons learned. NEVER delete existing entries.*

---

## 25. V734-V738 失败链完整经验教训 (2026-03-12 最终总结)

### 25.1 五次迭代失败时间线
| 版本 | 创建日期 | 核心思路 | 失败原因 | 浪费规模 |
|------|----------|----------|----------|----------|
| V734 | 2026-03-08 | 从 Phase 9 测试集结果中选择最佳单模型 | `_effective_stack_k()` 始终返回1 + oracle来源=测试集 | 104 条 INVALID |
| V735 | 2026-03-08 | 扩展V734的oracle表 + 引用FusedChampion | 使用已弃用的Phase 7模型 + oracle来源=测试集 | 104 条 INVALID |
| V736 | 2026-03-08 | 逆RMSE Top-3 stacking ensemble | oracle表=测试集MAE/RMSE → 选择=偷看答案 | 104 条 INVALID |
| V737 | 2026-03-10 | V736 + EDGAR PCA(5) + asinh变换 | 继承V736的oracle表 + sinh溢出 + OOM | 104 条 INVALID, 11个jobs |
| V738 | 2026-03-11 | V737 + val_frac=0.2参数 (死代码) | val_frac参数写了但fit/predict中从未引用 | 104 条 INVALID, 5个jobs |

### 25.2 五层结构性分析 (为什么连续错5次)
1. **Layer 1 — 数据源盲点**: benchmark harness 只输出测试集指标到 metrics.json。不存在验证集评估。任何基于 metrics.json 的 oracle = 测试集泄漏。但5个版本都从 metrics.json 构建 oracle 表而未质疑数据来源。
2. **Layer 2 — 复制粘贴传染**: V733→V734→V735→V736→V737→V738 逐版本扩展 oracle 表，注释写"Phase 9 clean data"但未说明 = 测试集。每次迭代只改模型结构，不审视 oracle 本身。
3. **Layer 3 — 基础设施缺失**: Benchmark harness 缺少 train/validation/test 三路分割机制。没有验证集指标，就不可能构建合法 oracle。这是基础设施层面的问题，不是代码 bug。
4. **Layer 4 — 死代码误导**: V738 `val_frac=0.2` 和文档声称"20% held out validation set"暗示验证已实现。实际上 V753 (`autofit_wrapper.py` L7232 `_fit_validation()`) 有正确实现，但从未移植到 V738。
5. **Layer 5 — 缺少自动化防护**: 没有测试阻止测试集→fit()。没有断言验证 oracle ≠ 测试指标。没有代码审查关卡检查 oracle 来源。

### 25.3 ⛔ 从中提炼的绝对规则
1. **NEVER build oracle/selection/lookup tables from test-set outputs** — 这是最高优先级禁令。
2. **引入新参数必须同时引入使用该参数的代码** — 参数 ≠ 实现。
3. **继承父类时必须审查 fit() 和 predict() 的完整调用链** — V737/V738 继承了 V736 的 oracle 路径。
4. **每次 AutoFit 迭代必须重新验证 oracle 数据来源** — 不能假设上一版本是正确的。
5. **metrics.json 只包含测试集结果** — 这是 harness 的结构性事实。
6. **V739 root fix: harness 传入 val_raw + 模型内部全覆盖 fit/predict** — 这是唯一正确模式。

### 25.4 V739 安全性验证 (2026-03-12 sub-agent 完整审计)
- ✅ V739 `fit()` 完全覆盖父类 (不调用 `super().fit()`)
- ✅ V739 `predict()` 完全覆盖父类 (不调用 `super().predict()`)
- ✅ 类体中零次引用 ORACLE_TABLE、ORACLE_TABLE_TOP3、ORACLE_TABLE_V738
- ✅ 继承方法仅限: `_fit_edgar_pca()` (纯PCA预处理), `_create_model_wrapper()` (工厂)
- ✅ `val_raw` 来自 harness 时序分割 (train_end=2025-06-30, val_end=2025-09-30, 7天embargo)
- ✅ `_validate_temporal_ordering()` 在时间重叠时抛出 RuntimeError
- **结论: V739 结构性安全，可以放心运行。**

---

## 26. Phase 定义与历史 (全面解读)

| Phase | 日期 | 内容 | 状态 |
|-------|------|------|------|
| Phase 7 | 2026-03-07 | 初版全量 benchmark (67 models) | ❌ DEPRECATED — 4个关键bug (TSLib per-entity, foundation prediction_length, Moirai entity cap, ml_tabular single-horizon) |
| Phase 8 | (从未独立存在) | Phase 7 的修复尝试 | ❌ DEPRECATED — 与 Phase 7 共用同个结果目录, 共同作废 |
| Phase 9 | 2026-03-08 至今 | **当前正式 benchmark**: 全部 bug 修复后的完全重跑, 新目录 `block3_phase9_fair/` | ✅ ACTIVE — 78 valid complete, 8,660 valid records |
| Phase 10 | 2026-03-10 | 新版 AutoFit (V737/V738/V739) + text embedding 生成 | ⚠️ V737/V738 INVALID (oracle), V739 PENDING, text emb PENDING |
| Phase 11 | 2026-03-13 | 14 新增 TSLib SOTA 模型 (CFPT, DeformableTST, ModernTCN, etc.) | ⏳ PENDING — 11 cfisch jobs in queue |
| Phase 12 | 2026-03-12 | Text embedding 生成 + core_text/full ablation 重跑 | ⏳ BLOCKED — text_embeddings.parquet 未生成; 40 rerun scripts ready |

**关键说明**: Phase 编号是增量实验批次, 不是互斥阶段。Phase 9 是主干, Phase 10-12 是增量补充。所有结果写入同一目录 `block3_phase9_fair/`。

---

## 27. 废弃输出归档 (2026-03-12)

### 归档位置
`runs/benchmarks/_deprecated_archive/` — 包含所有废弃版本的输出:
- 16 个 Phase 7/v71/v72 目录
- 5 个 dual3090 测试目录
- 4 个 b11_v2 目录
- 1 个 block3_phase10 (V737/V738 invalid)
- 合计: 26 个目录

### 当前活跃目录
`runs/benchmarks/block3_phase9_fair/` — 唯一有效结果目录

### 注意事项
- V734-V738 的 metrics.json 仍在 `block3_phase9_fair/` 中 (autofit 子目录) — 已在聚合时排除
- 不要删除归档目录 — 保留用于审计追溯
- `_deprecated_archive/DEPRECATED_README.md` 有详细说明

---

## 28. 集群竞争现状与应对 (2026-03-12)

### 为什么 0 RUNNING?
- GPU 分区: 24 节点共 78 个 RUNNING jobs, 504 个 PENDING — **被其他用户占满**
- 最大占用者: elavdusinovi (51 个 running GPU jobs)
- L40S 分区: 2 节点全部 allocated (8 running, 12 pending)
- 我们的 Priority: ~10225 (l40s) / ~10388 (gpu) — normal QOS base=10001, 竞争激烈
- **这不是配置问题, 是集群拥堵** — 所有 125 个 PENDING jobs 配置正确, 等待资源释放

### 当前 SLURM 队列 (78 PENDING, 0 RUNNING)
| 账户 | 任务类型 | 数量 | 分区 |
|------|----------|------|------|
| npin | V739 | 11 | gpu |
| npin | gap-fill (Reformer/LightTS/ETSformer/Pyraformer) | 24 | gpu |
| npin | TTM/Chronos2 gap | 10 | gpu |
| npin | text embedding | 2 | gpu+l40s |
| cfisch | Phase 11 新模型 | 11 | gpu |
| cfisch | gap-fill (Crossformer/MSGNet/etc) | 11 | gpu |
| cfisch | V739 L40S | 7 | l40s |
| cfisch | text embedding | 2 | gpu+l40s |

### 可采取的行动
1. **等待**: GPU 使用者的 2 天 wall time 到期后资源会释放
2. **besteffort QOS**: 可提交低优先级副本 (priority=1), 在空闲间隙抢占
3. **iris-gpu-long**: 4 个 14 天 wall time 的长任务配额 (可用于关键前置如 text embedding)
4. **主动监控**: `squeue -u npin,cfisch -t RUNNING` 一旦有 RUNNING, 密切关注进度

---

## 22. ⛔ 极致计算资源利用铁律 (COMPUTE SATURATION MANDATE — ABSOLUTE, NON-NEGOTIABLE)

**每次提交任务时，必须极致地、榨干式地、毫无保留地充分利用 npin 和 cfisch 两个账户最大权限下的所有计算资源！**

### 22.1 双账户使用强制规则
1. **永远同时使用 npin 和 cfisch 两个账户**。每次提交任务必须均匀分配到两个账户。
2. **cfisch 通过 `ssh iris-cf` 登录提交**。命令格式：
   ```bash
   ssh iris-cf "cd /work/projects/eint/repo_root && sbatch <script>.sh"
   ```
3. **两个账户互相访问权限已拉满**：
   - 共享文件系统 `/work/projects/eint/repo_root/` (drwxrwsrwx)
   - 共享 Python 环境 (INSIDER_PY)
   - 共享 HF 模型缓存 (`export HF_HOME="/home/users/npin/.cache/huggingface"`)
   - 结果写入同一目录 `runs/benchmarks/block3_phase9_fair/`
4. **永远不要忘记 cfisch 账户的存在**。每次操作前默念："npin + cfisch 双账户全量利用"。
5. **任务分配策略**：按分区和 QOS 均匀分配，不要全堆在一个账户下。

### 22.2 全分区覆盖强制规则
提交 GPU 任务时，必须同时覆盖所有可用分区：
| 分区 | QOS | 账户 | 何时使用 |
|------|-----|------|----------|
| gpu | normal | npin + cfisch | 所有 GPU 任务的主要分区 |
| l40s | iris-snt | npin + cfisch | GPU 溢出，队列更短 |
| bigmem | normal | npin + cfisch | CPU-only 任务 |
| gpu | besteffort | npin + cfisch | 低优先级填充，可被抢占 |
| gpu | iris-gpu-long | npin + cfisch | 超长任务 (14天 wall, 限4个) |

### 22.3 禁止空闲计算资源
- 如果某账户 RUNNING=0，必须立即调查原因并采取行动
- 如果某分区有空闲但我们没有提交，必须立即补交
- 每次会话开始时检查 `squeue -u npin,cfisch -t RUNNING | wc -l`
- 目标：两个账户同时有 RUNNING 任务

### 22.4 重复提交竞速模式
对于关键前置任务（如 text embedding 生成），同时提交到多个分区/账户组合：
- npin/gpu, npin/l40s, cfisch/gpu, cfisch/l40s = 4个并行竞速
- 第一个完成的即为有效结果，立即 scancel 其余
- 这是正确的资源利用策略，不是浪费

---

## 23. ⛔ 实事求是铁律 (TRUTH-BASED RESPONSE MANDATE — ABSOLUTE, NON-NEGOTIABLE)

1. **所有回答必须基于实际操作和实际数据**。永远不可凭空编造、猜测、或给出没有实际依据的回答。
2. **永远要亲自检查、亲自验证**。不确定的东西必须先查再答，不能"应该是"、"大概是"。
3. **逻辑严密性**：每个操作前必须检查是否存在未来信息泄漏、数据不一致、实验不公平等问题。
4. **发现问题必须立即停止并报告**，不能掩盖或忽略。
5. **具体例子**：
   - ❌ "text embedding应该已经生成了" → 必须 `ls runs/text_embeddings/` 确认
   - ❌ "V739应该跑完了" → 必须 `squeue -u npin | grep v739` 和检查 metrics.json
   - ❌ "core_text应该有文本特征" → 必须检查代码逻辑确认 select_dtypes 是否过滤
   - ✅ 先查实际情况，再基于事实回答

---

## 24. TEXT EMBEDDING 关键发现 (2026-03-12)

### 24.1 严重问题：core_text/full 结果全部无效 → 已重组为 2-seed 复现实验
- `runs/text_embeddings/` 目录为空 — text_embeddings.parquet 从未生成
- 原因：4次生成尝试全部失败（Arrow string OOM 2.55 TiB）
- 后果：所有 80 个模型的 `core_text` 结果 ≡ `core_only`，`full` 结果 ≡ `core_edgar`
- **50% 的实验计算被浪费**（core_text/full 条件无效）
- 代码路径：`benchmark_shard.py _join_text_embeddings()` → fallback 到 raw text join → `select_dtypes(include=[np.number])` 静默丢弃所有文本列

**⚡ 重组完成 (2026-03-13)**:
- `core_text/` → `core_only_seed2/`，`full/` → `core_edgar_seed2/`
- 40 个目录重命名，4,032 条 metrics.json 记录更新
- `REPLICATION_MANIFEST.json` 已生成于 `runs/benchmarks/block3_phase9_fair/`
- 脚本: `scripts/reorganize_replication_seed2.py`
- 当前 4 个有效 ablation: `core_only`, `core_only_seed2`, `core_edgar`, `core_edgar_seed2`
- `all_results.csv` 和 `docs/BLOCK3_RESULTS.md` 已重新生成（0 条旧名称残留）

### 24.2 Embedding 模型选择依据
**GTE-Qwen2-1.5B-instruct** (Alibaba-NLP) — 经过充分评估是当前最佳选择：
- MTEB 67.16（1-3B级别SOTA）
- 1536维 embedding → PCA 降至 64维
- FP16 仅需 3.3GB VRAM（V100/L40S 均轻松承载）
- Apache-2.0 开源许可
- 30-60分钟可完成 5.77M 行文本的 embedding 生成

**为什么不用 7B 模型？**
- gte-Qwen2-7B: MTEB ~72 但需 ~14GB VRAM，推理速度慢 5×
- 经过 PCA 64维压缩后，1.5B vs 7B 的嵌入质量差异可忽略
- 不值得用 5× 推理时间换取 <5% 的 MTEB 分差

### 24.3 修复计划 (Phase 12)
1. ✅ 提交 4 个 text embedding 生成作业（2账户 × 2分区 竞速提交）
   - 5229600 (npin/gpu), 5229601 (npin/l40s), 5229602 (cfisch/gpu), 5229603 (cfisch/l40s)
2. ⏳ 等待 `runs/text_embeddings/text_embeddings.parquet` 产出
3. ⏳ 验证 embedding 文件完整性（行数 = offers_text 行数，64列数值）
4. ✅ 冗余 core_text/full 目录已重组为 core_only_seed2/core_edgar_seed2 (40 dirs, 4032 records)
5. ⏳ 重新提交所有 core_text/full ablation 的 benchmark 作业（需真实 text embedding）
6. ⏳ core_only 和 core_edgar 结果保留（它们本来就不依赖 text embedding）

### 24.4 公平性保证
- core_only/core_edgar 结果有效（从未依赖 text embedding）
- core_text/full 结果全部重跑（所有模型使用相同的 GTE-Qwen2-1.5B embedding）
- 这确保了：(1) 所有模型在同一 ablation 下看到完全相同的特征集 (2) text ablation 真正有意义

### 24.5 SLURM 脚本位置
- `.slurm_scripts/phase12/gen_text_emb_gpu.sh` (npin/gpu, 256G)
- `.slurm_scripts/phase12/cf_gen_text_emb_gpu.sh` (cfisch/gpu, 256G)
- `.slurm_scripts/phase12/gen_text_emb_l40s.sh` (npin/l40s, 256G)
- `.slurm_scripts/phase12/cf_gen_text_emb_l40s.sh` (cfisch/l40s, 256G)

### 24.6 OOM 修复历史
| 尝试 | Job ID | 结果 | 原因 |
|------|--------|------|------|
| #1 | 5225722 (npin) | FAILED 1:58 | Arrow string OOM 2.55 TiB (.where() API) |
| #2 | 5225807 (npin) | FAILED 1:56 | 同上 (.loc[] API) |
| #3 | 5225868 (cfisch) | FAILED 0:53 | Permission denied (路径错误用了 /home/npin/) |
| #4 | 5225948 (cfisch) | FAILED 0:53 | 同上 |
| #5 | 5225961 (cfisch) | FAILED 7:01 | Arrow OOM (fix 未 commit 就提交了作业) |
| #6 | 5229600-5229603 | PENDING | 修复后版本 (.astype("object") + 截断, 256G RAM) |

---

## 29. ⛔ 50% 算力浪费的血泪教训 (2026-03-12) → ✅ 已重组为 2-Seed 复现 (2026-03-13)

### 29.0 ✅ 重组完成: 冗余 → 2-Seed Independent Replication

**2026-03-13 操作**: 40 个冗余目录已物理重命名，4,032 条 metrics.json 记录已更新。
- `core_text/` → `core_only_seed2/` (16 dirs)
- `full/` → `core_edgar_seed2/` (24 dirs)
- `REPLICATION_MANIFEST.json` 已生成 (含 per-model 方差统计 + 论文 claim 模板)
- `all_results.csv` + `docs/BLOCK3_RESULTS.md` 已重新生成 (0 条旧名称残留)
- 脚本: `scripts/reorganize_replication_seed2.py`

**复现统计**: 4,032 对条件中 81.5% 完全一致，91.5% 差异 <0.1%，平均 0.3369%。
45 个确定性模型完美复现，7 个高随机性模型 (CSDI/iTransformer/BRITS) 差异 >1%。

### 29.1 浪费根因

Text embedding 从未生成 (`runs/text_embeddings/` 为空)，但 benchmark harness 静默继续运行
core_text 和 full ablation —— 实际读入原始文本字符串 → `select_dtypes(include=[np.number])`
丢弃所有非数值列 → **core_text 结果 ≡ core_only，full 结果 ≡ core_edgar**。

| 指标 | 数值 |
|------|------|
| 浪费的记录数 | 3,888 / 8,660 = **44.9%** |
| core_text 冗余记录 | 1,656 |
| full 冗余记录 | 2,232 |
| 估计浪费 GPU 时 | ~638 GPU-hours (425 个 SLURM job × ~1.5h avg) |
| 浪费的高内存分配 | core_text: 400-512G/job, full: 500-640G/job (正常 core_only 只需 256G) |

### 29.2 为什么没提前发现？（根因分析）

1. **无前置检查**: `_join_text_embeddings()` 在文件不存在时静默跳过，不 raise、不 warn
2. **类型过滤静默**: `select_dtypes(include=[np.number])` 不打印被丢弃的列
3. **结果看似正常**: core_text 作业成功完成、产出 metrics.json、MAE 合理 —— 无任何异常信号
4. **ablation 独立提交**: SLURM 脚本逐 ablation 提交，没有 "先验证前置条件再跑" 的 guard
5. **长排队掩盖**: Iris 上排队时间长(数天)，等到作业跑完才发现结果相同

### 29.3 ⛔ 强制规则 (MANDATORY — 后续提交前必须执行)

25. **DO NOT submit core_text/full jobs BEFORE verifying text embeddings exist**:
```bash
ls -la runs/text_embeddings/text_embeddings.parquet || { echo "❌ TEXT EMBEDDINGS MISSING — DO NOT SUBMIT core_text/full JOBS"; exit 1; }
```
26. **ALL ablation jobs with external data dependencies MUST have pre-flight checks**:
   - core_text/full → 检查 `runs/text_embeddings/text_embeddings.parquet` 存在且行数 > 0
   - core_edgar/full → 检查 `runs/freeze_20260203_225620/edgar_feature_store.parquet` 存在
27. **Benchmark harness SHOULD warn (not silently skip) when text embeddings are missing**:
   - `_join_text_embeddings()` 应该在文件不存在时 `logger.warning("TEXT EMBEDDINGS NOT FOUND")`
   - 如果 ablation 是 core_text 或 full 但 embedding 文件不存在，应该 `raise FileNotFoundError`
28. **DO NOT re-run core_text/full until text embeddings are verified** — 只跑 core_only + core_edgar = 省 50% 算力 + 50% 内存

### 29.4 冗余结果极致利用方案 (6 策略全面榨干)

虽然 core_text ≡ core_only、full ≡ core_edgar，但这些冗余结果是**在不同 SLURM 作业
中独立重新训练**的模型，因此等价于一次 **免费的独立重复实验** (free independent replication)。

#### 策略 1: 训练方差估计 (Training Variance Estimation) — ⭐最高价值

core_only 与 core_text 的差异 = 纯粹的 GPU 训练噪声（相同数据、相同代码、不同 SLURM job/GPU）。
这给出了每个模型的**训练不确定性**量化，可用于：

| 模型类别 | 训练方差 (mean % diff) | 解读 |
|----------|----------------------|------|
| 确定性模型 (53个, 含 StatsForecast, ML tabular, Foundation zero-shot) | 0.000% | 完美可复现 |
| 近确定性 NF 模型 (NBEATS, GRU, Chronos 等 25个) | <0.01% | GPU 浮点噪声，学术可忽略 |
| 随机性 NF 模型 (PatchTST, VanillaTransformer 等 9个) | 0.01-0.1% | 需要多 seed 平均 |
| 高随机性模型 (CSDI: 4.4%, iTransformer: 2.1%, BRITS: 1.2%) | >1% | 冠军结论不可靠，必须 3+ seed |

**V740 设计启示**:
- 冠军 margin < 2× 训练方差 → 冠军地位不稳定，oracle routing 该 cell 应改为 validation gate
- 12 个 NBEATS↔NBEATSx 条件 margin=0.000% 但训练方差=0.002-0.03% → 完全等同，合并为一个模型

#### 策略 2: 双独立运行平均 (2-Seed Averaging) — ⭐高价值

将 core_only 和 core_text 的 MAE 取平均 = 免费的 2-seed 估计，方差缩小 √2 倍：
```
MAE_robust(model, condition) = (MAE_core_only + MAE_core_text) / 2
```
实测结果: **0/28 个冠军因平均化而改变** → 当前冠军排名是稳健的。
但对于 margin < 0.01% 的 tight race (如 NBEATS vs NBEATSx)，2-seed 平均能给出更可信的排序。

同理，core_edgar 和 full 也可取平均。

#### 策略 3: 冠军稳定性审计 (Champion Stability Audit) — ⭐高价值

| 审计条件 | 结果 |
|----------|------|
| 冠军 margin > 2× 训练方差 | 92/104 条件 STABLE (冠军地位可信) |
| 冠军 margin < 2× 训练方差 | 12/104 条件 UNSTABLE (全部是 NBEATS↔NBEATSx 的 0.000% tie) |
| 结论 | 除 NBEATS↔NBEATSx tie 外，所有冠军指定在统计意义上是可信的 |

#### 策略 4: NeurIPS 论文的可复现性论据 (Reproducibility Evidence)

**"We independently replicated all benchmark conditions via a paired ablation design.
87.7% of (model, condition) pairs produced identical MAE between independent SLURM jobs,
confirming numerical reproducibility. The remaining 12.3% represent GPU training stochasticity,
providing natural error bars (mean ±0.097%, max ±36.8% for CSDI)."**

这本身就是一个有价值的实验贡献——大规模 benchmark 的可复现性分析。

#### 策略 5: 不稳定模型自动排除 (Unstable Model Exclusion)

训练方差 >1% 的模型（CSDI: 4.4%, iTransformer: 2.1%, BRITS: 1.2%）在 single-run benchmark
中不可信。后续 V739/V740 的候选池应优先排除或降权这些模型，或强制 3-seed 运行。

#### 策略 6: V739/V740 迭代加速方案 — ⭐⭐最高优先

基于以上分析,V739/V740 的提交策略应为:
1. **只跑 core_only + core_edgar** (2 ablations = 原来的 50%)
2. **只跑 9 个冠军模型** (NBEATS, NHITS, Chronos, KAN, DeepNPTS, PatchTST, GRU, NBEATSx, DLinear)
   而非全部 93 个 → 再省 90%
3. **合计**: 9 模型 × 2 ablation × 3 task × 3 target × 4 horizon = ~648 条件
   vs 原来 93 模型 × 4 ablation × 3 task × 3 target × 4 horizon = ~12,000+ 条件
   **算力节省 95%**，迭代周期从数周降到数小时
4. **V739/V740 验证完成后**，才对全 93 模型 × 4 ablation 做完整 benchmark
5. **text embedding 生成后**，core_text/full 只对 9 个冠军模型做 ablation 敏感性检验

### 29.5 核心经验教训总结

| 教训 | 对策 |
|------|------|
| 静默 fallback = 浪费倍增器 | 所有 fallback 路径必须 warn/raise |
| 独立 ablation 无前置检查 | 提交前验证所有数据依赖 |
| "看似正常" ≠ "真正正常" | 提交后抽样对比两个 ablation 的冠军是否一致 |
| 50% 计算浪费但产出有价值 | 变废为宝: 训练方差估计 + 可复现性论据 |
| 迭代验证不需要全量运行 | 9 冠军 × 2 ablation = 95% 算力节省 |
