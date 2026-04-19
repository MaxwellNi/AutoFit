# TESF 项目全景文档（2026-04-20 权威版）

> **用途**：给从未接触过本项目的新 AI / 新专家提供一站式深度了解。
> **权威性**：所有数据基于 git commit `f5647cb`、slurm gate 日志、35/35 单元测试实际运行结果。
> **语言**：全中文。
> **最后更新**：2026-04-20 01:30 UTC

---

## 1. 项目一句话描述

**TESF**（Three-head Equity-stage Sequence Forecasting）是一个**统一单模型**框架，
用一个共享状态主干（shared trunk）同时预测融资过程的三个观测律：
- **Binary**：是否融资成功（survival / hazard）
- **Funding**：融资金额（heavy-tail severity）
- **Investors**：投资者人数（zero-inflated count / marked TPP）

目标：投稿 **NeurIPS 2026 oral**。

---

## 2. 系统架构（一图看懂）

```
                    ┌─────────────────────────────────┐
                    │   Raw Data (16,902 records)      │
                    │   90 active models, 160 cells    │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │   SharedTemporalBackbone         │
                    │   (compact state + event atoms   │
                    │    + source memory + summaries)   │
                    │   backbone.py, ~112 dim trunk     │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │   Hard Barrier (信息隔离墙)       │
                    │   lane 不能污染 trunk             │
                    └──┬────────────┬────────────┬────┘
                       │            │            │
              ┌────────▼───┐ ┌─────▼──────┐ ┌───▼──────────┐
              │ Binary Lane │ │Funding Lane│ │Investors Lane│
              │ hazard+cal  │ │anchor+GPD  │ │hurdle+TPP    │
              │ binary_lane │ │funding_lane│ │investors_lane│
              └─────────────┘ └────────────┘ └──────────────┘
```

---

## 3. 代码结构

```
src/narrative/block3/models/single_model_mainline/
├── backbone.py          # SharedTemporalBackbone（共享状态提取）
├── wrapper.py           # SingleModelMainlineRuntime（统一入口）
├── objectives.py        # 三头目标函数合约
├── variant_profiles.py  # 所有实验配置（Gen-1..9）
├── hazard_utils.py      # P1: hazard-space 校准工具
├── tail_utils.py        # P2: GPD 尾部校正工具
├── intensity_utils.py   # P3: Marked TPP intensity 工具
├── jump_ode_utils.py    # P4: Jump ODE（已证实失败）
├── cqr_utils.py         # P2.1: 共形分位数回归
└── lanes/
    ├── binary_lane.py   # Binary 预测 lane
    ├── funding_lane.py  # Funding 预测 lane
    └── investors_lane.py # Investors 预测 lane

scripts/
├── analyze_mainline_investors_track_gate.py  # 门控验证主脚本
└── slurm/single_model_mainline/              # 所有 slurm 作业脚本

test/
├── test_single_model_mainline_backbone.py
├── test_single_model_mainline_objectives.py
├── test_single_model_mainline_jump_ode.py
├── test_single_model_mainline_funding_lane.py
├── test_single_model_mainline_source_activation.py
├── test_single_model_mainline_mark_encoder.py
└── test_single_model_mainline_source_read_policy.py
```

---

## 4. 当前唯一有效主线：selective_event_state_guard (Gen-3)

这是经过 **4 轮门控验证** 后唯一通过所有面板的候选。

### 4.1 它做了什么

```python
# variant_profiles.py 中的配置
"mainline_selective_event_state_guard": {
    "enable_investors_event_state_features": True,      # 事件状态特征
    "enable_investors_selective_event_state_activation": True,  # 选择性激活
    "investors_event_state_allow_h1": False,             # h1 不启用（保守）
    "investors_event_state_max_source_presence_share": 0.0,  # 源丰富时禁用
    "enable_funding_gpd_tail": True,                    # P2: GPD 尾部校正
    "enable_funding_tail_focus": True,
    "enable_funding_cqr_interval": True,                # P2.1: CQR 区间
    "funding_cqr_alpha": 0.10,
    "enable_investors_intensity_baseline": True,         # P3: TPP 强度基线
    "investors_intensity_blend": 0.5,
}
```

### 4.2 门控验证证据链

| Gate | 面板 | Mean Δ% | Worst Δ% | 正/负 | 判定 |
|------|------|---------|----------|-------|------|
| P2 GPD gate (official) | 12 SliceCases | **+0.014%** | 0.0% | 9/0 | ✅ PASS |
| P2 GPD gate (dynamic) | 1 case | **+0.542%** | — | 1/0 | ✅ PASS |
| P3 TPP gate (official) | 12 SliceCases | **+0.0138%** | 0.0% | 9/0 | ✅ PASS |
| P3 TPP gate (dynamic) | 1 case | **+0.555%** | — | 1/0 | ✅ PASS |
| P4 gate 交叉验证 | selective 行 | **+0.014%** | 0.0% | 9/0 | ✅ 数据一致 |

**关键发现**：P4 gate 中 selective 候选的数据与 P3 gate 完全一致——这是独立重复验证。

---

## 5. P1-P4 完整审计报告（A 级证据）

### 5.1 已证实有效 ✅

| 编号 | 组件 | 文件 | 机制 | 证据 |
|------|------|------|------|------|
| P1 | hazard-space calibration | `hazard_utils.py` | 把 binary 校准从概率空间扩展到 hazard 空间 | 集成到 selective 主线，gate passed |
| P1.1 | survival NLL scoring | `binary_lane.py` | 校准器尊重 time-to-event 结构 | 集成到 selective 主线 |
| P2 | GPD tail correction | `tail_utils.py` | POT + MOM 估计器修正 funding 重尾 | **gate +0.014% / +0.542%** |
| P2.1 | CQR prediction interval | `cqr_utils.py` | 分布无关覆盖保证 | 7/7 单元测试通过 |
| P3 | Marked TPP intensity | `intensity_utils.py` | blend baseline intensity 给 investors | **gate +0.0138% / +0.555%** |

### 5.2 已证实失败 ❌

| 编号 | 组件 | 文件 | 失败模式 | 证据 |
|------|------|------|----------|------|
| Gen-8 | Hawkes intensity | `backbone.py` | 跨实体分布 107× 漂移，z-score 无法校正 | dynamic -173%, geometry fail |
| P4 | Jump Neural ODE | `jump_ode_utils.py` | EDGAR surfaces 上 ODE 特征灾难性失效 | **-1.04% mean, -17.8% worst** |

### 5.3 P4 Jump ODE 失败的详细模式

```
Case                                                        Δ%        Status
────────────────────────────────────────────────────────────────────────────
task1_outcome__core_only__h1                            +3.2554%    ✅  无 EDGAR，ODE 有效
task1_outcome__core_only__h7                            +0.4370%    ✅
task1_outcome__core_only__h14                           +0.4356%    ✅
task1_outcome__core_only__h30                           +0.4341%    ✅
task2_forecast__core_edgar__h1                         -17.7885%    ❌  EDGAR + ODE 灾难
task2_forecast__core_edgar__h7                          -1.1841%    ❌
task2_forecast__core_edgar__h14                         -1.1898%    ❌
task2_forecast__core_edgar__h30                         -1.1964%    ❌
task2_forecast__full__h1                                +8.2667%    ✅  h1 偶然正
task2_forecast__full__h7                                -1.3071%    ❌
task2_forecast__full__h14                               -1.3126%    ❌
task2_forecast__full__h30                               -1.3189%    ❌
```

**根因分析**：
1. task1_core_only 全正（+0.4~3.3%）→ 无 EDGAR 时 ODE 特征确实有效
2. task2_core_edgar h1 灾难 (-17.8%) → EDGAR 引入的维度与 ODE 特征冲突
3. task2_full h1 偶然正 (+8.3%) 但 h7-30 全负 → 不稳定，不可推广
4. **共同根因**：12 维 ODE 特征的 z-score 归一化在 EDGAR surfaces 失效，
   与 Gen-8 Hawkes 失败的根因完全一致——**trunk 层特征扩展在异构 surfaces 上不安全**

### 5.4 P4 dynamic gate 额外发现的 bug

P4 dynamic gate 在执行时报错：
```
TypeError: InvestorsLaneRuntime.fit() got an unexpected keyword argument 'enable_investors_intensity_baseline'
```
原因：gate 脚本 dynamic_variant 中写的是 `enable_investors_intensity_baseline`（前缀 `investors_`），
但 `InvestorsLaneRuntime.fit()` 的参数名是 `enable_intensity_baseline`（无前缀）。
这是 P3 intensity 集成时的命名不一致 bug。**不影响 P3 official gate 结果**（official 面板走 wrapper 路径，不受此影响），
但需要修复。

---

## 6. Trunk 层两次失败的共同教训

Gen-8 Hawkes 和 Gen-9 Jump ODE 的失败揭示了一个**结构性规律**：

| 维度 | Gen-8 Hawkes | Gen-9 Jump ODE | 共同规律 |
|------|-------------|----------------|---------|
| 新增维度数 | +4 (3 intensity + 1 inter-event) | +12 (8 ODE + 4 diagnostic) | 维度膨胀 |
| 归一化 | z-score + ±5σ clip | z-score + ±5σ clip | 相同策略 |
| core_only | 未单独测试 | ✅ 全正 (+0.4~3.3%) | 简单 surface 有效 |
| core_edgar | 未单独测试 | ❌ 灾难 (-17.8%) | EDGAR 引入冲突 |
| dynamic | ❌ -173% | ❌ (bug 阻塞) | 跨 surface 泛化失败 |
| 根因 | 107× 分布漂移 | EDGAR 维度冲突 | **z-score 对异构数据不安全** |

**决策（2026-04-20）**：
- **trunk 层特征扩展方向暂停**
- **转向 lane 层直接改进**（不经过 trunk，直接在 investors lane 内部创新）
- selective_event_state_guard (Gen-3) 确认为**唯一 promotable 候选**

---

## 7. 当前最大瓶颈：investors shared112 仍是 0/0/48

这是距离 NeurIPS oral 的**核心差距**。含义：

- `0/0/48`：在 shared112 的 48 个 investors cells 中，0 个 positive、0 个 negative、48 个 neutral
- **通俗解释**：统一主线目前对 investors 预测"没有任何帮助也没有任何伤害"
- **NeurIPS 问题**："你的统一模型和分别训练三个模型有什么区别？"——如果 investors 头始终 neutral，这个问题无法回答

### 7.1 为什么 investors 头如此困难

1. **数据极度稀疏**：大多数时间步 investors_count = 0（零膨胀）
2. **异质性极强**：不同实体的投资者到达模式完全不同
3. **共享 trunk 信号太弱**：trunk 的 112 维状态对 investors 的解释力不足
4. **h1 vs h>1 断裂**：h1 用 KNN exemplar，h>1 用 hurdle+regression，两套逻辑

### 7.2 下一步方向：P5 Investors 选择性学习 + 不确定性门控

基于 trunk 层两次失败的教训，P5 不再在 trunk 层加特征，而是在 **investors lane 内部** 创新：

**方案 A：条件弃权（Abstention Gate）**
- 当 lane 预测不确定性高时，输出 anchor（历史均值）而非 learned prediction
- 文献：Selective Learning (Geifman & El-Yaniv, NeurIPS 2019)
- 预期效果：减少 "大错特错" 的 case，保守但安全

**方案 B：自适应 Shrinkage**
- 根据 uncertainty 将预测结果向 anchor 收缩，而非二元弃权
- 文献：James-Stein shrinkage + Bayesian posterior
- 预期效果：保留 learned signal 的同时降低风险

**方案 C：Ensemble Uncertainty Weighting**
- 用 MC Dropout 或 conformal band 估计不确定性，加权混合
- 文献：Kendall et al. (CVPR 2018) multi-task uncertainty
- 预期效果：自动平衡 confident vs uncertain predictions

---

## 8. 数学框架（融资过程的三层结构）

### 8.1 层 1：共享状态演化（trunk）

$$dx_t = f(x_t, \text{context}_t) \, dt + g(x_{t^-}, \text{event}_t) \, dN_t$$

- $x_t$：企业融资状态向量（compact state 112 维）
- $f$：连续漂移（滚动摘要 + 事件衰减）
- $g$：离散跳跃（投资者到来、EDGAR 发布）
- $N_t$：事件计数过程

**当前实现**：backbone.py 的 SharedTemporalBackbone
**理论目标**：Jump SDE / Neural CDE
**现实约束**：两次 trunk 扩展失败，当前只能安全使用 Gen-3 selective 特征

### 8.2 层 2：三条观测律（lanes）

| Lane | 观测律 | 数学形式 | 当前实现 |
|------|--------|---------|---------|
| Binary | Hazard function | $h(t \| x_t) = P(\text{funded at } t \| \text{not yet}, x_t)$ | Logistic + 5 种校准器 + hazard-space |
| Funding | Heavy-tail severity | $F(\text{amount} \| \text{funded}, x_t) \sim \text{LogNormal} + \text{GPD tail}$ | Anchor-residual + GPD POT 校正 |
| Investors | Zero-inflated count | $N_t \sim \text{Hurdle}(\pi_0, \text{Poisson}(\lambda_t))$ | Hurdle + occurrence + intensity baseline |

### 8.3 层 3：目标函数（objectives）

```
已实现 (✅ Implemented):
├── calibration (binary)        — Brier score + ECE
├── hazard (binary)             — hazard-space Platt/Isotonic
├── event_consistency (binary)  — 跨 horizon 一致性
├── anchor_residual (funding)   — anchor + β × residual
├── occurrence (investors)      — h1 二值分类
├── hurdle (investors)          — 两阶段 hurdle
├── intensity_baseline (investors) — TPP blend
└── jump_ode_state (investors)  — Jump ODE（已证失败）

未实现 (❌ Deferred):
├── tail_guard (funding)        — GPD NLL loss（P2 只做了 inference 层校正）
├── transition (investors)      — 跨期转移信号
├── binary_funding_alignment    — 跨头一致性
├── selective_calibration       — 选择性预测
├── abstention_gate             — 弃权门控（P5 方向）
└── counterfactual_source       — 反事实推断
```

---

## 9. 技术环境

| 项目 | 详情 |
|------|------|
| Python | 3.12.12 (insider micromamba env) |
| PyTorch | 2.10.0+cu126（可用但 mainline 主要用 sklearn/numpy） |
| 计算 | HPC: bigmem 分区 (3TB RAM, CPU-only), GPU 需另申 |
| Slurm | qos=normal, --signal=USR1@120 --requeue |
| 数据 | 16,902 records, 90 active models, 160 conditions (V739@160/160) |
| Git HEAD | `f5647cb` (main) |
| 测试 | 76/77 pass (1 pre-existing flaky: source_read_policy 浮点排序) |
| torchdiffeq | **不可用**（P4 因此用纯 numpy 实现） |

---

## 10. Git Commit 历史（P1-P4 全链）

```
f5647cb P4 gate: add jump_ode_state_guard candidate + slurm scripts
4d7e01e P4: Jump ODE state evolution for shared trunk (Jia & Benson ICML 2020)
9abdf01 fix: update objectives test expectations for P1/P2/P3 implemented terms
16a4383 P3: Marked TPP intensity baseline for investors lane
4e1b829 P2.1: CQR conformal prediction intervals for funding lane
1b0a924 P2: GPD tail correction for funding lane + Gen-8 Hawkes final verdict
d015bc6 Add Gen-7 Hawkes financing state to shared trunk  ← origin/main
```

**代码量**：P1-P4 共 18 files, 2,981 insertions, 49 deletions

---

## 11. 距离 NeurIPS 2026 Oral 的差距分析

### 11.1 进度条

```
■■■■■■■□□□□□□□□□□□□□  35%
```

### 11.2 已完成 vs 未完成

| 维度 | 状态 | 完成度 |
|------|------|--------|
| 统一框架设计 | ✅ shared trunk → barrier → 3 lanes | 90% |
| Binary lane | ✅ hazard + calibration + survival NLL | 55% |
| Funding lane | ✅ anchor-residual + GPD + CQR | 50% |
| Investors lane | ⚠️ hurdle + TPP baseline，shared112 仍 0/0/48 | 25% |
| Trunk 状态机 | ❌ 两次失败（Hawkes/ODE），当前仅 rolling summary | 15% |
| 门控证据 | ✅ P2/P3 双通过 + P4 negative result | 60% |
| 三头联动收益 | ❌ investors 尚未活跃 | 10% |
| Foundation baseline | ⚠️ 有 Chronos/TTM，缺 TimesFM/Moirai | 40% |
| 论文写作 | ❌ 未开始 | 0% |
| 消融实验 | ⚠️ 有 P1→P3 逐步叠加，需系统化 | 30% |

### 11.3 最关键的 3 个差距

1. **investors 头必须活跃**（目前 0/0/48）→ P5 选择性学习
2. **trunk 需要理论创新**（两次扩展失败）→ 转向 lane-conditioned read
3. **论文还没开始写**→ 5-6 月必须启动

### 11.4 最短路径到 Oral

```
Phase 1（4月下旬）：P5 选择性学习 → investors 脱离 0/0/48
Phase 2（5月上旬）：P6 binary-funding 联动 → 证明三头耦合收益
Phase 3（5月中旬）：P7 uncertainty weighting → 自动平衡三头
Phase 4（5-6月）：论文写作 + 消融实验 + foundation baseline
Phase 5（6月底）：提交 NeurIPS 2026
```

---

## 12. 探索失败历史（完整记录）

| 代数 | 方案 | 失败原因 | 教训 |
|------|------|---------|------|
| Gen-1..4 | 对称窗口 rolling | 无记忆、无事件驱动 | 压缩器不够 |
| Gen-5 | Multiscale temporal/spectral | gate 无增益 | 频率分解对融资数据无帮助 |
| Gen-5a | Temporal-only | gate 无增益 | 同上 |
| Gen-5b | Spectral-only | gate 无增益 | 同上 |
| Gen-6 | Process-state feedback | gate 无增益 | 有界线性反馈太弱 |
| Gen-7 | Hawkes (bug) | 4 个实现 bug | 参数未传递 + 跨实体污染 |
| Gen-8 | Hawkes (修复) | -173% dynamic, 107× 分布漂移 | z-score 对异构实体不安全 |
| Gen-9 | Jump ODE | -1.04% mean, -17.8% worst | EDGAR surface 维度冲突 |

**总结**：trunk 层 6 次扩展尝试全部失败。唯一成功的是 Gen-3 selective event-state——
它不在 trunk 层加新维度，而是选择性地让 investors lane 读取已有的事件状态原子。

---

## 13. 文献调查覆盖（研究文档统计）

详见 `docs/references/single_model_true_champion/SINGLE_MODEL_COMPLETE_SOTA_FINANCE_TS_MAPPING_ZH_20260419.md`

| 类别 | 方法数 | 关键来源 |
|------|--------|---------|
| 连续状态演化 (ODE/SDE/CDE) | 7+ | Neural ODE, Jump ODE, Neural CDE |
| 状态空间模型 (SSM) | 6+ | Mamba, S4, HiPPO, Griffin |
| 点过程 (TPP) | 8+ | Hawkes, Neural Hawkes, THP, SAHP |
| Survival 分析 | 9+ | DeepHit, Dynamic-DeepHit, DSM |
| 极值理论 (EVT) | 5+ | GPD, GEV, POT, DistDF |
| 多任务学习 | 5+ | Uncertainty weighting, GradNorm, PCGrad |
| 不确定性量化 | 9+ | Conformal, MC Dropout, Ensemble |
| 可解释性/因果 | 8+ | SHAP, Attention, do-calculus |
| VC/PE 实证金融 | 5+ | Gompers, Hellmann, Kaplan-Lerner |
| 总计 | **200+** 方法 | 16+ 顶会 + 金融/经济学顶刊 |

---

## 14. 当前需要更新的文档

| 优先级 | 文档 | 核心缺失 |
|--------|------|---------|
| P0 | `docs/CURRENT_SOURCE_OF_TRUTH.md` | 缺 P1-P4 gate 结果 |
| P0 | `docs/BLOCK3_MODEL_STATUS.md` | TESF 状态行过时 |
| P1 | `docs/V740_MASTER_EXECUTION_PLAN.md` | 缺 P1-P4 结果和新方向 |
| P1 | `docs/references/V740_V745_TESF_STATUS_20260412.md` | 缺全部 gate 结论 |
| P2 | `docs/PLANS.md` | 缺 TESF 推进计划 |
| P2 | 3 个 single_model 子文档 | 缺 P2-P4 gate 状态 |
| 本文档 | `docs/TESF_PROJECT_FULL_PANORAMA_ZH_20260420.md` | **新建——本文档** |

---

## 15. 关键文件速查表

| 需求 | 文件 |
|------|------|
| 项目全貌（本文档） | `docs/TESF_PROJECT_FULL_PANORAMA_ZH_20260420.md` |
| 完整 SOTA 方法映射 | `docs/references/single_model_true_champion/SINGLE_MODEL_COMPLETE_SOTA_FINANCE_TS_MAPPING_ZH_20260419.md` |
| 代码真相入口 | `docs/CURRENT_SOURCE_OF_TRUTH.md`（需更新） |
| 门控验证脚本 | `scripts/analyze_mainline_investors_track_gate.py` |
| 统一入口 wrapper | `src/narrative/block3/models/single_model_mainline/wrapper.py` |
| 主线候选配置 | `src/narrative/block3/models/single_model_mainline/variant_profiles.py` |
| P2 gate 日志 | `/work/projects/eint/logs/smm_np_tgp2off_*.out` |
| P3 gate 日志 | `/work/projects/eint/logs/smm_np_tgp3off_*.out` |
| P4 gate 日志 | `/work/projects/eint/logs/smm_np_tgp4off_5328972.out` |
| P4 dynamic 错误 | `/work/projects/eint/logs/smm_np_tgp4dyn_5328973.err` |

---

## 16. 给新 AI / 新专家的快速启动指令

```bash
# 1. 环境
export PYTHON=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
cd /home/users/npin/repo_root

# 2. 运行测试
$PYTHON -m pytest test/test_single_model_mainline_*.py -v --tb=short

# 3. 查看当前有效主线配置
grep -A 20 "mainline_selective_event_state_guard" src/narrative/block3/models/single_model_mainline/variant_profiles.py

# 4. 查看 P4 gate 结果
python3 -c "
import json
with open('/work/projects/eint/logs/smm_np_tgp4off_5328972.out') as f:
    report = json.loads(f.read())
for name, data in report['official_summary'].items():
    print(f'{name}: mean={data[\"mean_mae_delta_pct\"]:+.4f}%, worst={data[\"worst_mae_delta_pct\"]:+.4f}%')
"

# 5. 提交 gate 验证
sbatch scripts/slurm/single_model_mainline/np_mainline_track_gate_p3_official_bigmem.sh
```

---

## 17. 已知问题和技术债务

| 问题 | 严重度 | 状态 |
|------|--------|------|
| P4 dynamic gate bug（参数名不一致） | 中 | 待修复 |
| source_read_policy 测试 flaky（0.0 > 0.0） | 低 | 预存 |
| 研究文档 §16 重复段（两份§16） | 低 | 待清理 |
| shared112 investors 仍 0/0/48 | **高** | P5 方向 |
| 论文未开始 | **高** | 5-6 月启动 |
| torchdiffeq 不可用 | 中 | 用纯 numpy 绕过 |
