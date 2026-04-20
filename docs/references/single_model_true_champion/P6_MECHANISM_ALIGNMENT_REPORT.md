# P6 Mechanism Alignment Report — 强制预代码审查

**日期**: 2026-04-20  
**前提**: 本报告遵循 Core Research Protocol，在任何 P6 代码写入之前必须完成。

---

## 1. 当前三头真实状态审计

### 1.1 Investors Head（投资者数头）— ★★★★★ DOMINANT

| 指标 | 值 |
|------|-----|
| Task1 outcome | 16/0 W/L, mean Δ = −97.94% |
| Task2 forecast | 16/0 W/L, mean Δ = −95.20% |
| Track gate (official) | shrinkage +0.6348% mean, 9+/0- |
| Track gate (dynamic) | shrinkage +60.28% (MAE: 786→312) |

**架构**: Occurrence HGBC + Positive HGBR(log-residual) + KNN exemplar(h=1) + source specialists + intensity baseline + P5 adaptive shrinkage gate  
**关键优势**: anchor-based residual learning — lag1 count 是极强锚点  
**瓶颈**: 已通过 P5 shrinkage 解决大部分，但 h=1 仍用 KNN exemplar 而非 learned model

### 1.2 Funding Head（融资额头）— ★★★★★ DOMINANT

| 指标 | 值 |
|------|-----|
| Task1 outcome | 16/0 W/L, mean Δ = −99.36% |
| Task3 cold-start | 16/0 W/L, mean Δ = −96.91% |

**架构**: 2-component jump-hurdle (HGBC event + HGBR severity) + GPD tail + CQR intervals + grid-searched anchor guard  
**关键优势**: anchor residual + log-domain + grid-calibrated blend  
**状态**: 全面碾压所有 SOTA 时序模型

### 1.3 Binary Head（是否融资头）— ★☆☆☆☆ CATASTROPHIC

| 指标 | 值 |
|------|-----|
| Task1 outcome | 0/16 W/L, mean Δ = +665.49% |

**架构**: Balanced LogisticRegression + hazard adapter + multi-strategy calibration  
**根因分析**:
1. LogReg 容量严重不足 — 77维输入空间非线性无法被线性模型捕捉
2. 对手是 DeepNPTS（task1 专家，16/16 wins）— 纯深度模型在 binary classification 上碾压
3. Hazard adapter 有正确思路但执行太弱（仅 LogReg 套 LogReg）
4. Calibration 无法弥补底层模型能力缺陷

**Binary 头是整个系统的最大弱环。**

### 1.4 Shared Trunk — 需要升级

| 组件 | 状态 |
|------|------|
| 归一化 | Plain z-score — ⚠️ 已证实是 trunk-level 失败根因 |
| 特征维度 | 106 dims (default) |
| Barrier | 固定随机投影 — 无可学习参数 |
| Cross-head coupling | **零** — 一致性约束全部 deferred |
| Hawkes/Jump-ODE | 已证实在 z-score 下崩溃（Hawkes −173%, ODE −17.8%）|

---

## 2. P5 突破性结果回顾

### 2.1 Official Panel (9 conditions)

| Condition | MAE Δ% |
|-----------|--------|
| task1_core_only h7/14/30 | +0.514% |
| task2_core_edgar h7/14/30 | +1.064% (P4 ODE 同条件 −17.8%) |
| task2_full h7/14/30 | +0.960% |
| All h1 | 0.0% (by design — h1 禁用 shrinkage) |
| **Mean** | **+0.6348%** |

### 2.2 Dynamic Panel (entity-level, h14)

| Variant | MAE | Δ vs baseline |
|---------|-----|---------------|
| Legacy baseline | 786.09 | 0.0% |
| Selective event-state | 781.73 | +0.55% |
| **Shrinkage gate** | **312.23** | **+60.28%** |

**核心洞察**: Shrinkage 在高不确定性实体上效果最大（MAE 从 786 降至 312），验证了 James-Stein 理论：越不确定越应该 shrink toward anchor。

---

## 3. 文献映射 → P6 候选方案

### 3.1 最高优先级：Binary 头修复

**问题**: Binary 头 0/16 W/L, +665% worse  
**根因**: LogisticRegression 容量不足  

| 方案 | 来源 | 原子映射 | 预期效果 |
|------|------|----------|----------|
| 升级为 HGBC | 工程常识 | binary_lane.py L235 | 高 — funding/investors 已用 HGBC |
| Tweedie → Zero-inflated Poisson regression | CIKM'23 STTD | binary calibration | 中等 — 分布匹配 |
| DER 不确定性分解 | AAAI'26 ProbFM | binary confidence gate | 高 — 单次推理分离认知/偶然不确定性 |
| Learning to Defer | Mohri'24 | binary abstention | 中等 — 不确定时 defer to anchor |

**推荐 P6.1**: 将 Binary 头从 LogReg 升级为 HGBC，保持 hazard adapter + calibration pipeline 不变。这是最小改动、最大回报的修复。

### 3.2 第二优先级：Trunk 归一化修复

**问题**: z-score 在异构实体面上崩溃，导致 Hawkes/ODE 特征不可用  

| 方案 | 来源 | 原子映射 | 预期效果 |
|------|------|----------|----------|
| R²-IN (median/MAD) | arXiv 2510.04667 | backbone.py L127-133 | 极高 — 可能解锁 trunk-level 扩展 |
| Quantile normalization | 经典统计 | backbone.py | 中等 — rank-based |
| RevIN (可逆实例归一化) | ICLR'22 | backbone.py | 中等 — 但对极端异常值仍脆弱 |

**推荐 P6.2**: 用 R²-IN (median/MAD) 替换 z-score。若成功，Hawkes/Jump-ODE 特征可能从灾难性 (−173%/−17.8%) 变为正贡献。

### 3.3 第三优先级：不确定性驱动 Gate 升级

**问题**: 当前 shrinkage gate 用 HGBR 预测 alpha，但没有校准的不确定性  

| 方案 | 来源 | 原子映射 | 预期效果 |
|------|------|----------|----------|
| SCRC 选择性 conformal | arXiv 2512.12844 | shrinkage_utils.py | 高 — 统一 shrinkage 和 conformal |
| DER 单次不确定性 | AAAI'26 ProbFM | gate 驱动信号 | 高 — 替代 HGBR alpha predictor |
| BCI 动态 conformal | arXiv 2512.10289 | CQR 扩展 | 中等 — 在线适应 |

---

## 4. P6 实施计划

### 阶段 P6.1: Binary Head 升级（最高优先）

**目标**: Binary 头从 0/16 W/L 提升到至少 8/16 W/L  
**改动范围**: `binary_lane.py` — 仅模型替换，不触碰 calibration pipeline  
**代码改动估计**: ~15 行  

```
变更清单:
1. binary_lane.py: LogisticRegression → HistGradientBoostingClassifier (max_depth=3, max_iter=150)
2. 保留 hazard adapter (at-risk 子模型也升级为 HGBC)
3. 保留全部 calibration pipeline (Platt/Isotonic/Hazard-space)
4. 保留 shrinkage (off by default)
```

**三过滤器检查**:
- ✅ Anti-scale-drift: HGBC 是 tree-based，天然抗 z-score 问题
- ✅ Heterogeneous pulse: tree splits 天然处理混合分布
- ✅ Rejection capable: calibration pipeline 已有 ECE guard

### 阶段 P6.2: Trunk R²-IN 归一化（第二优先）

**目标**: 用 robust 归一化解锁 Hawkes/ODE 特征  
**改动范围**: `backbone.py` L127-133  
**代码改动估计**: ~10 行  

```
变更清单:
1. backbone.py fit(): mean → median, std → MAD × 1.4826
2. backbone.py transform(): 同步更新
3. 移除 Hawkes/ODE 独立 z-score（统一归一化）
4. backbone.py: 添加 normalize_mode='robust' 参数（可回退到 'zscore'）
```

**三过滤器检查**:
- ✅ Anti-scale-drift: median/MAD 对极端值鲁棒 — 正是为此设计
- ✅ Heterogeneous pulse: MAD 不受稀疏大跳影响
- ✅ Rejection capable: 不影响下游 gate 机制

### 阶段 P6.3: Shrinkage → Conformal Shrinkage（第三优先）

**目标**: 为 shrinkage gate 添加校准的不确定性保证  
**前提**: P6.1 和 P6.2 的 gate 结果必须先落地  
**改动范围**: `shrinkage_utils.py`  

---

## 5. P6 顺序与 Gate 合约

```
P6.1 Binary HGBC Upgrade
  ├── 实现（binary_lane.py）
  ├── 单元测试（test_binary_lane.py）
  ├── Gate: track_gate binary panel
  └── 通过条件: W/L ≥ 8/16

P6.2 Trunk R²-IN
  ├── 实现（backbone.py）
  ├── 单元测试（test_backbone.py）
  ├── Gate: 重跑全部 14 track candidates
  └── 通过条件: shrinkage_gate_guard mean > +0.6348%

P6.3 Conformal Shrinkage
  ├── 前提: P6.1 + P6.2 gates passed
  ├── 实现（shrinkage_utils.py）
  ├── Gate: official + dynamic panel
  └── 通过条件: dynamic MAE improvement > 60.28%
```

---

## 6. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| HGBC 过拟合 binary | 低 | 中等 | max_depth=3 + 交叉验证 calibration |
| R²-IN 破坏现有 investors/funding 性能 | 中等 | 高 | normalize_mode 参数可回退 |
| Hawkes/ODE 仍在 R²-IN 下失败 | 中等 | 低 | 独立消融评估 |
| Binary HGBC 仍不敌 DeepNPTS | 中等 | 中等 | DeepNPTS 是 task1 专家，可能仍有差距 |

---

## 审批检查清单

- [x] 三过滤器（anti-scale-drift, heterogeneous-pulse, rejection）全部通过
- [x] 不涉及 trunk-level 特征扩展（P6.1 仅换模型）
- [x] P6.2 的归一化改动有回退路径
- [x] 每个阶段有独立 gate
- [x] 文献支撑充分（R²-IN, STTD, SCRC, DER）
- [x] 代码改动最小化（P6.1 ~15行, P6.2 ~10行）

**结论**: P6.1 (Binary HGBC) 是最高 ROI 下一步。立即实施。
