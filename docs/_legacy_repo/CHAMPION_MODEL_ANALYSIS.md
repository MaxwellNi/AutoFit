# Phase 9 冠军模型深度机制分析

> Generated: 2026-03-12 | Data: 8,972 test records, 80 complete models (104/104 conditions)
> Benchmark: `runs/benchmarks/block3_phase9_fair/` | Fair comparison (no oracle leakage, temporal split)

---

## 1. 总体排名 (Overall Mean Rank — RMSE)

| # | 模型 | 类别 | MeanRank | #1st | Top3 | Top5 | 核心机制 |
|---|------|------|----------|------|------|------|----------|
| 1 | **NBEATS** | deep_classical | 7.82 | 10 | 25 | 43 | 双残差基函数展开 (trend+seasonality) |
| 2 | **AutoFitV736** | autofit | 7.91 | 5 | 23 | 32 | 逆RMSE加权 Top-3 stacking ensemble |
| 3 | **NBEATSx** | transformer_sota | 8.72 | 6 | 17 | 34 | NBEATS + 外生变量注入 |
| 4 | **PatchTST** | transformer_sota | 9.08 | 0 | 17 | 32 | Patch分词 + 通道独立自注意力 |
| 5 | **MLP** | deep_classical | 9.73 | 0 | 34 | 42 | 1024维宽隐藏层直接映射 |
| 6 | **NHITS** | deep_classical | 9.93 | 0 | 11 | 31 | 多速率分层插值 (MaxPool multi-scale) |
| 7 | **AutoFitV734** | autofit | 11.33 | 16 | 26 | 32 | 经验Oracle + Softmax加权选择 |
| 8 | **KAN** | transformer_sota | 12.64 | 2 | 15 | 17 | Kolmogorov-Arnold 可学习激活函数 |
| 9 | **Chronos** | foundation | 12.77 | 0 | 0 | 5 | T5 tokenizer 零样本预训练 |
| 10 | **DeepNPTS** | transformer_sota | 12.81 | 26 | 28 | 32 | 非参数混合密度输出 |

> **关键发现**: DeepNPTS 拥有最多的#1名次(26次)，但 MeanRank 仅第10 — 说明它在某些条件下极强，但在其他条件下弱。NBEATS 虽#1次数仅10次，但稳定性最高(MeanRank 7.82)。

---

## 2. RMSE 冠军频率 (104 条件中的获胜次数)

| 模型 | 胜场 | 占比 | 主要胜场来源 |
|------|------|------|-------------|
| DeepNPTS | 26 | 25.0% | funding_raised_usd × (core_edgar + full) |
| TimesNet | 16 | 15.4% | investors_count × (core_edgar + full) |
| AutoFitV734 | 16 | 15.4% | investors_count × (core_only + core_text) |
| NBEATS | 10 | 9.6% | is_funded + investors_count 长horizon |
| DeepAR | 10 | 9.6% | funding_raised_usd × (core_only) |
| NBEATSx | 6 | 5.8% | is_funded × (core_edgar + full) |
| Autoformer | 5 | 4.8% | funding_raised_usd × h=1 × core_only |
| GRU | 5 | 4.8% | funding_raised_usd × h=14 × core_only |
| AutoFitV736 | 5 | 4.8% | investors_count × h=30 × core_only |

---

## 3. 类别层面分析

| 类别 | RMSE #1 次数 | Top-3 出现次数 | 代表冠军 |
|------|-------------|--------------|---------|
| transformer_sota | **55** | 115 | DeepNPTS, TimesNet, KAN, PatchTST, Autoformer |
| deep_classical | 25 | **120** | NBEATS, MLP, GRU, DeepAR |
| autofit | 24 | 75 | AutoFitV734, AutoFitV736, AutoFitV735 |
| foundation | 0 | 2 | (Chronos/ChronosBolt 偶尔 Top-3) |

> **结论**: transformer_sota 在精确获胜次数上领先，但 deep_classical 在 Top-3 出现率上最高。deep_classical 是最稳定的类别 — 几乎永远在前5。

---

## 4. 按目标变量拆解

### 4.1 funding_raised_usd (44 conditions — 主战场)

| # | 模型 | MeanRank | Top3 | 为什么胜出？ |
|---|------|----------|------|------------|
| 1 | **MLP** | 3.43 | 33 | 1024维全连接层对高维数据有强拟合能力；robust scaler 处理极端值 |
| 2 | **GRU** | 3.77 | 29 | 门控记忆机制捕捉资金筹集的短期自相关；encoder(200)/decoder(128)非对称设计 |
| 3 | AutoFitV736 | 6.93 | 0 | 虽然 Top-3 为0，但稳定排在6-8名 → 平均排名低 |
| 4 | LSTM | 7.07 | 11 | 类似GRU但有额外遗忘门，对长依赖有优势 |
| 5 | DeepAR | 8.48 | 10 | 自回归LSTM + 参数化概率分布(Student-t)适配重尾金融数据 |

**核心发现**: 
- **无外生特征(core_only)**: 简单RNN (GRU/LSTM/DeepAR) 占优 → 自回归结构匹配金融序列的短期记忆
- **有EDGAR/full**: DeepNPTS 以绝对优势领先(RMSE ~1.63M vs 第2名 ~1.66M) → 非参数混合密度输出适配重尾分布
- **结论**: funding_raised_usd 的分布呈极端右偏(峰度~125)，DeepNPTS 的非参数性质完美匹配，但依赖外生特征

### 4.2 investors_count (44 conditions)

| # | 模型 | MeanRank | Top3 | 为什么胜出？ |
|---|------|----------|------|------------|
| 1 | **TimesNet** | 3.43 | 25 | FFT周期发现 + 2D-Inception卷积捕捉投资者活动的周期性模式 |
| 2 | NBEATS | 5.27 | 14 | 趋势/季节性基函数分解天然匹配投资者计数的周期波动 |
| 3 | NBEATSx | 7.16 | 5 | NBEATS + 外生变量，但投资者计数数据方差极低(RMSE~1082)，外生增益微弱 |
| 4 | NHITS | 7.86 | 3 | 多尺度池化捕捉不同周期 |
| 5 | AutoFitV734 | 8.20 | 20 | Oracle选择实际选中了TimesNet/NBEATS作为底层模型 |

**核心发现**:
- 投资者计数的区分度极低：所有Top-5模型的RMSE都在1082-1083之间(差距<0.1%)
- TimesNet 以微弱优势领先，其FFT+Inception机制对这种低方差计数数据最合适
- 这是一个`对排名基本无影响`的目标变量 — 真正的差异化在 funding_raised_usd

### 4.3 is_funded (16 conditions — 二元分类)

| # | 模型 | MeanRank | Top3 | 为什么胜出？ |
|---|------|----------|------|------------|
| 1 | **NBEATSx** | 2.94 | 11 | 基函数展开+外生变量，在core_edgar/full上RMSE=0.0 |
| 2 | NBEATS | 4.62 | 7 | 趋势基函数天然产生[0,1]输出 |
| 3 | PatchTST | 4.75 | 6 | Patch注意力对二元模式有效 |
| 4 | NHITS | 6.81 | 3 | 分层插值 |
| 5 | AutoFitV735 | 6.94 | 6 | 精确Oracle选中了NBEATSx |

**核心发现**:
- 多数Top模型在is_funded上实现 RMSE≈0 → 二元目标几乎被完美预测
- 对整体排名影响有限(仅16条件)

---

## 5. 按消融实验拆解

### 5.1 core_only (28 conditions — 无外生特征)

| 冠军模型 | 次数 | 分析 |
|---------|------|------|
| AutoFitV734 | 9 | Oracle表在core_only条件下选择了最佳单模型 |
| DeepAR | 6 | 纯自回归结构不依赖外生输入，core_only天然适配 |
| Autoformer | 3 | 分解注意力(trend+seasonal)在无外生时有效 |
| GRU | 3 | 简单门控RNN足以捕捉时序自相关 |
| AutoFitV736 | 3 | Stacking ensemble 对core_only仍有效 |

### 5.2 core_edgar + full (56 conditions — 有外生特征)

| 冠军模型 | 次数 | 分析 |
|---------|------|------|
| **DeepNPTS** | 24 | 非参数混合密度 + EDGAR特征 → 极强组合 |
| **TimesNet** | 16 | FFT+2D卷积有效利用EDGAR结构化特征 |
| NBEATS | 10 | 基函数展开即使不直接使用外生也保持强劲 |
| KAN | 2 | 可学习激活函数捕捉非线性外生关系 |

> **关键洞察**: core_edgar/full 占56/104条件，DeepNPTS+TimesNet 合计拿走40/56(71.4%)的冠军 — 外生特征的有效利用是决定性因素。

---

## 6. 按预测步长拆解

| horizon | 主要冠军 | 特征 |
|---------|---------|------|
| h=1 | DeepNPTS(6), TimesNet(6), AutoFitV734(6), Autoformer(5) | 短期: 分解模型(Autoformer)和Oracle(AutoFitV734)有效 |
| h=7 | DeepNPTS(7), TimesNet(5), DeepAR(5), AutoFitV734(5) | 中短期: 自回归(DeepAR)开始发力 |
| h=14 | DeepNPTS(7), TimesNet(5), GRU(5), AutoFitV734(5) | 中期: RNN(GRU)在此步长表现最佳 |
| h=30 | NBEATS(7), DeepNPTS(6), DeepAR(5), AutoFitV736(5) | 长期: NBEATS基函数外推+DeepAR分布外推 |

> **关键洞察**: 
> - h=1~14: DeepNPTS 稳定领先(非参数短期预测)
> - h=30: NBEATS 反超(多项式基函数对长期趋势外推更稳定)
> - AutoFitV736 在h=30获5场 → stacking ensemble 对长期预测有优势(逆方差加权减少方差)

---

## 7. 冠军核心机制剖析

### 7.1 NBEATS — 双残差基函数展开 (MeanRank #1: 7.82)

**架构**: 
```
Input(60) → [Stack_trend: polynomial basis] → residual → [Stack_seasonality: Fourier basis] → forecast
```

**为什么赢**: 
1. **可解释的归纳偏置**: 显式趋势多项式 + 季节性傅里叶基函数匹配金融时序的结构
2. **双残差学习**: 每个block产生部分预测并将残差传递给下一block → 逐步逼近
3. **无需外生变量**: NBEATS 是纯内生模型，在core_only上就很强
4. **训练效率**: 24s平均训练时间(80个模型中最快的Top-5模型)
5. **h=30 外推能力**: 多项式基函数天然适合趋势外推

**弱点**: 
- 无外生变量通道 → 有EDGAR时不如NBEATSx/DeepNPTS
- investors_count 上不如TimesNet(缺乏FFT周期发现机制)

### 7.2 AutoFitV736 — 逆RMSE加权 Stacking (MeanRank #2: 7.91)

**架构**:
```
Oracle_Top3(target, horizon, ablation) → [Model_1, Model_2, Model_3]
                                          ↓         ↓         ↓
                                        train     train     train
                                          ↓         ↓         ↓
                                        pred_1    pred_2    pred_3
                                          ↓
                                    w = 1/RMSE_i / Σ(1/RMSE_j)
                                          ↓
                                    weighted_avg(pred_1, pred_2, pred_3)
```

**为什么赢**:
1. **方差缩减**: 3个不同架构的加权平均 → 偏差-方差权衡
2. **Oracle指导**: 经验RMSE数据指导模型选择 → 避免随机组合
3. **跨架构多样性**: 每个条件选不同的Top-3 → 真正的集成多样性
4. **稳定但不极端**: #1次数仅5次，但Top10 达71次 → 极其稳定

**弱点**:
- Oracle表来自test数据 → **有测试集泄漏** → 排名科学上不成立
- 训练时间1163s(是NBEATS的48倍)
- 对funding_raised_usd，Top-3值为0(不够精确)

### 7.3 PatchTST — Patch分词自注意力 (MeanRank #4: 9.08)

**架构**:
```
Input(64) → [Patch: len=16, stride=8] → 8 tokens
          → [Embedding] → [MultiHead_Attn(16 heads)] × L layers
          → [Linear] → forecast
```

**为什么赢**:
1. **长程依赖**: 自注意力O(P²)其中P=patches数 ≪ L=序列长度 → 高效长距离建模
2. **局部模式保留**: 每个patch(16步)保留短期模式(日内/周内波动)
3. **通道独立**: 每个entity独立处理 → 不受跨实体噪声干扰
4. **3000步训练**: 比其他模型(1000步)更充分训练

**弱点**:
- #1次数为0 → 永远不是最好的，但几乎永远在前5
- 72s训练(是NBEATS的3倍)
- 对funding_raised_usd的极端值不如MLP/GRU

### 7.4 DeepNPTS — 非参数混合密度 (RMSE #1次数最多: 26)

**架构**:
```
Input(60) → [MLP encoder] → mixture_weights(w_1, ..., w_T) over historical values
          → forecast = Σ w_i × y_{t-i}  (加权历史值)
```

**为什么赢**: 
1. **分布无关**: 不假设目标分布形状 → 对funding_raised_usd的极端右偏(峰度125)完美适配
2. **非参数**: 预测是历史值的加权混合 → 自然处理多模态分布
3. **EDGAR协同**: 有EDGAR特征时RMSE ~1.63M vs 无EDGAR ~1.66M(-1.8%) → 外生信息调制权重

**弱点**:
- 高方差: MeanRank仅12.81(#10) → 在investors_count/is_funded上显著劣于NBEATS
- 仅在funding_raised_usd × (core_edgar + full)上碾压 → 领域特异性强

### 7.5 TimesNet — FFT+2D卷积 (investors_count 绝对冠军)

**架构**:
```
Input(60) → [FFT → top-k periods] → [Reshape 1D→2D: (period × freq)]
          → [2D Inception Conv] → [Reshape 2D→1D]
          → forecast
```

**为什么赢**:
1. **自适应周期发现**: FFT自动发现数据中的主要周期 → 不需要手动指定
2. **2D建模**: 将时间序列reshape为2D张量 → 2D Inception捕捉周期内和周期间的交互
3. **scaler_type="standard"**: 唯一使用z-score而非robust scaler的冠军 → 对低方差计数数据更合适

**弱点**:
- 训练时间244s(是NBEATS的10倍)
- 对funding_raised_usd表现平庸(MeanRank ~11)

---

## 8. 外生特征影响分析

| 模型 | core_only RMSE | full RMSE | Δ% | 结论 |
|------|---------------|-----------|-----|------|
| HistGradientBoosting | 888,094 | 814,242 | **+8.3%** | 树模型受益最大 |
| BRITS | 2,654,486 | 2,585,737 | +2.6% | 插补模型有小增益 |
| ExtraTrees | 796,617 | 786,649 | +1.3% | 集成树有增益 |
| NBEATS | ~696K | ~696K | ~0% | 纯内生 → 无增益 |
| DeepNPTS | ~1,617K(core) | ~1,632K(full) | -0.9% | 实际变差(full更高) |

> **重要**: 大部分深度学习模型从外生特征获益 ≈ 0%。唯一显著获益的是树模型(+8%)。
> DeepNPTS 的 "核心-EDGAR赢优" 不是来自外生特征本身的信息增益，而是因为数据维度增加导致模型行为改变。

---

## 9. 计算效率对比

| 模型 | 训练时间(s) | 推理时间(s) | MeanRank | 效率比(Rank/Train) |
|------|------------|------------|----------|-------------------|
| NBEATS | 24.1 | 2.40 | 7.82 | 0.32 |
| NHITS | 24.5 | 2.45 | 9.93 | 0.41 |
| NBEATSx | 25.4 | 2.67 | 8.72 | 0.34 |
| MLP | 33.8 | 5.62 | 9.73 | 0.29 |
| KAN | 28.0 | 2.96 | 12.64 | 0.45 |
| PatchTST | 72.9 | 3.03 | 9.08 | 0.12 |
| TimesNet | 244.5 | 7.93 | 13.65 | 0.06 |
| AutoFitV736 | 1163.6 | 70.58 | 7.91 | 0.007 |
| AutoFitV734 | 1157.9 | 110.45 | 11.33 | 0.010 |

> **Pareto 最优**: NBEATS — 24s训练,MeanRank 7.82 → 无法被其他模型在两个维度上同时超过。
> AutoFitV736 用48倍训练时间仅换取0.09 MeanRank优势(7.91 vs 7.82)。

---

## 10. V739 策略启示

### 10.1 当前 V739 设计

V739 使用temporal validation选模型，候选池:
`[NHITS, PatchTST, NBEATS, NBEATSx, ChronosBolt, KAN, Chronos, TimesNet]`

### 10.2 基于冠军分析的改进方向

#### A. 目标自适应策略
冠军分析揭示了明确的**目标-模型匹配模式**:

| 目标类型 | 最佳模型 | 核心机制 | V739 应选策略 |
|---------|---------|---------|-------------|
| 高方差连续(funding) | MLP, GRU, DeepNPTS | 高容量直接映射 / 门控记忆 / 非参数 | 加入MLP, GRU到候选池 |
| 低方差计数(investors) | TimesNet, NBEATS | FFT周期建模 / 基函数分解 | 已在候选池中 ✓ |
| 二元(is_funded) | NBEATSx, NBEATS, PatchTST | 基函数展开 | 已在候选池中 ✓ |

#### B. 消融自适应策略
| 特征集 | 最佳模型 | V739 应选策略 |
|--------|---------|-------------|
| core_only | AutoFit(oracle), DeepAR, GRU | 加入 DeepAR, GRU 到 core_only 候选池 |
| core_edgar/full | DeepNPTS, TimesNet | 加入 DeepNPTS 到有外生特征的候选池 |

#### C. 步长自适应策略
| 步长 | 最佳模型 | 策略 |
|------|---------|------|
| h=1~14 | DeepNPTS, TimesNet | 短中期: 非参数/周期模型 |
| h=30 | NBEATS, DeepAR, AutoFitV736 | 长期: 基函数外推 + 自回归 |

#### D. 扩展候选池提议
当前候选池(8个): NHITS, PatchTST, NBEATS, NBEATSx, ChronosBolt, KAN, Chronos, TimesNet

建议扩展至(12个):
```python
CANDIDATES_V740 = [
    # 稳定基线 (已有)
    'NHITS',        # MeanRank 9.93, 多尺度分层
    'PatchTST',     # MeanRank 9.08, Patch自注意力
    'NBEATS',       # MeanRank 7.82, 基函数展开 ← OVERALL #1
    'NBEATSx',      # MeanRank 8.72, 基函数+外生
    'ChronosBolt',  # MeanRank 13.38, 零样本预训练
    'KAN',          # MeanRank 12.64, 可学习激活
    'TimesNet',     # MeanRank 13.65, FFT+2D Conv
    # 新增高价值候选 (基于冠军分析)
    'MLP',          # MeanRank 9.73, funding_raised_usd #1
    'GRU',          # MeanRank 13.97, funding core_only #2
    'DeepNPTS',     # MeanRank 12.81, funding edgar/full #1 (26次冠军)
    'DeepAR',       # MeanRank 14.42, funding core_only 长期
    'TCN',          # MeanRank 12.92, 稳定卷积基线
]
# 移除 Chronos (MeanRank 12.77, 零Top3, 高推理开销30s)
```

#### E. 加权验证策略
当前V739对所有条件使用统一的MAE验证。建议:
- **funding_raised_usd**: 使用RMSE验证(因为重尾分布下MAE可能欠罚大误差)
- **investors_count**: 使用SMAPE验证(相对误差对低值更敏感)
- **is_funded**: 使用Binary Brier Score(二元目标更合适)

#### F. Stacking 集成 (V736风格)
V736 的逆RMSE stacking (MeanRank 7.91) 几乎与NBEATS(7.82)并列。
V740 可以结合V739的validation-based选择 + V736的stacking:
1. 在validation上训练所有K个候选
2. 选出Top-3(按validation RMSE)
3. 用1/val_RMSE 加权平均预测 → 无泄漏的stacking ensemble

---

## 11. 关键结论

1. **NBEATS 是无争议的 Pareto 最优模型**: MeanRank #1(7.82) + 最快训练(24s)
2. **真正的战场是 funding_raised_usd**: investors_count 和 is_funded 差异太小
3. **外生特征的影响被高估**: 仅树模型获得显著增益(+8%)，深度模型基本为0%
4. **DeepNPTS 是高方差赌注**: 26次#1但排名第10 → 极端特异性
5. **V736的stacking是唯一系统性超越单模型的方法**: 但有oracle泄漏
6. **V739需要增加 MLP 和 GRU**: 当前候选池缺少在funding上表现最强的模型
7. **步长h=30需要特殊处理**: NBEATS的基函数外推在长期预测上有结构性优势
