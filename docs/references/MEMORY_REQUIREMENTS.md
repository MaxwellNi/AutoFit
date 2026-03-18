# Block 3 Benchmark — Memory Requirements Reference

> **Last updated:** 2026-03-18
> **Source:** `sacct` actual OOM failures + completed job MaxRSS profiles
> **Purpose:** 防止 OOM 导致的计算资源浪费，为未来任务提交提供内存需求参考

---

## 1. 快速参考表 (Recommended Memory per Job Type)

| 任务类型 | 模型数 | Ablation | 推荐内存 | 推荐分区 | 实际峰值(观测) |
|----------|--------|----------|---------|----------|---------------|
| 单类别 co/ce (core_only / core_edgar) | 5-22 models | co/ce | **200G** | gpu / l40s | 114-164G |
| 单类别 ct (core_text) | 5-22 models | ct | **256G** | gpu | 125-166G |
| 单类别 fu (full) | 5-22 models | fu | **320G** | gpu / bigmem | 172-217G |
| AutoFit V739 co/ce | 1 (8 candidates) | co/ce | **200G** | l40s / gpu | 114-164G |
| AutoFit V739 ct | 1 (8 candidates) | ct | **300G** | l40s / gpu | 126-285G |
| AutoFit V739 fu | 1 (8 candidates) | fu | **384G** | gpu | 337G |
| ml_tabular 全部10模型 一次性 | 10 models | fu | ❌ 不可行 | — | >640G (必须拆分) |
| ml_tabular 单模型逐个跑 | 1 model | fu | **200G** | gpu | ~170G (estimated) |
| Phase 15 新模型 (23 TSLib) | 23 models | any | **256G** | gpu | TBD (running) |
| ALL33 加速脚本 gap-fill | varies | any | **256G** | gpu | TBD |
| NegativeBinomialGLM | 1 model | any | ❌ 结构性失败 | — | >640G (excluded) |
| TSLib 全批次 (>40 models) | 40+ | any | ❌ 不可行 | — | TIMEOUT@2d |

---

## 2. 实际 OOM 失败记录 (Complete Failure Log from sacct)

### 2.1 AutoFit V737 (retired — oracle leakage, 已废弃)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| cf_v737_t1_ct | 128G | gpu | OOM in 1m25s |
| cf_v737_t1_fu | 192G | gpu | OOM in 2m43s |
| cf_v737_t2_ct | 128G | gpu | OOM in 1m23s |
| cf_v737_t2_fu | 192G | gpu | OOM in 3m05s |
| cf_v737_t3_fu | 192G | gpu | OOM in 2m52s |

**教训:** AutoFit ct 需要 >128G，fu 需要 >192G

### 2.2 AutoFit V738 (retired — oracle leakage, 已废弃)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| cf_v738_t1_ct | 128G → 256G | gpu | 两次都 OOM |
| cf_v738_t1_fu | 192G → 256G | gpu | 两次都 OOM |
| cf_v738_t2_ct | 128G → 256G | gpu | 两次都 OOM |
| cf_v738_t2_fu | 192G → 256G | gpu | 两次都 OOM |
| cf_v738_t3_fu | 192G → 256G | gpu | 两次都 OOM |

**教训:** V738 AutoFit 内存需求极大，ct >256G，fu >256G。成功跑通需 384G。

### 2.3 AutoFit V739 (current baseline)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| v739f_t1_ct | 192G | hopper | OOM in 2m |
| v739f_t2_ct | 192G | hopper | OOM in 2m |
| v739f_t3_fu | 192G | l40s | OOM in 1m36s |
| v739f_t2_fu | 192G | hopper | OOM in 2m24s |
| v739r_t1_fu | 300G | l40s | OOM in 8m |
| v739r_t2_fu | 300G | hopper | OOM in 9m24s |

**成功案例:**
- v739 co: 114G @ 200G l40s ✅
- v739 ce: 163-164G @ 200G l40s ✅
- v739 ct: 125-126G @ 300G l40s ✅
- v739 fu: 337G @ 384G gpu ✅ (cf_v738 用 384G 后成功)

**教训:** V739 fu 实际用 337G → 需要 384G+ 分区

### 2.4 Foundation Models (early phase)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| fc_t*_co | 64G | l40s | OOM (<1m) |
| fc_t*_ce | 64G → 128G | l40s | 64G OOM，128G 也 OOM |
| fc_t*_ct | 64G → 128G → 192G | l40s → gpu | 三级递增才到 192G gpu 也 OOM |
| fc_t*_fu | 64G → 128G → 192G | l40s → gpu | 三级递增也 OOM |
| p9_fmB_t*_ct/fu | 256G | gpu | OOM in 5-7m |

**教训:** Foundation models (14 models batch) co 需 ≥200G, ce 需 ≥200G, ct 需 320G, fu 需 320G

### 2.5 ml_tabular (10 models — 全部一次性)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| ml_t1_fu_fix2 | 320G | gpu | OOM after 3h |
| ml_t1_fu_bm | 640G | bigmem | OOM after 2.5h |
| p12_ml_t_t1_fu_fix | 640G | bigmem | OOM after 3.5h |

**教训:** ml_tabular 10 models 全部加载 fu ablation = **不可行**，即使 640G bigmem 也不够。

**解决方案:** 拆分为逐模型执行 (`ml_t1_fu_split.sh`)，每个模型 200G 足够。

### 2.6 ModernTCN
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| mtcn_t2_ce | 128G | gpu | OOM after 2h |
| mtcn_t3_ce | 128G | gpu | OOM after 2h |
| mtcn_t3_co | 128G | gpu | TIMEOUT@12h |

**教训:** ModernTCN 单模型内存大（12.5M参数），需要 200G+。训练也慢（~30min/epoch），需要长时限。

### 2.7 NegativeBinomialGLM
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| p9r_nbglm_t1_ct | 512G | bigmem | 完成但用了 365G |
| p9r_nbglm_t1_fu | — | bigmem | 推断 >640G |

**教训:** NegativeBinomialGLM 结构性 OOM，已被排除。单模型 ct 就用 365G，fu 不可行。

### 2.8 TSLib 全批次 (>40 models all-at-once)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| p9_tsA_t*_{co,ce,ct} | 320G | gpu | 全部 TIMEOUT@2d |
| p9_tsB_t*_{co,ce,ct} | 320G | gpu | 全部 TIMEOUT@2d |
| p9r2_tsA_t*_fu | 640G | gpu | 全部 TIMEOUT@2d |

**教训:** 40+ TSLib models 在 2 天时限内跑不完。需要拆分为更小的批次 (10-15 models per job) 或 per-model。

### 2.9 Legacy V72/V73 (historical, 已废弃)
| Job | Req Mem | Partition | 失败原因 |
|-----|---------|-----------|---------|
| p7r_v72_t1_ct | 378G | gpu | OOM after 8h |
| p7r_v72_t2_ct | 378G | gpu | OOM after 8.7h |
| p7r_v73_t1_ct | 378G | gpu | OOM after 10h |
| p7r_v73_t2_ct | 378G | gpu | OOM after 8.5h |

---

## 3. 成功任务的实际内存使用 (Verified Peaks)

### 3.1 Per-Ablation Memory Pattern (from Phase 12 completed jobs)
| Ablation | 类别 | 实际使用 | 分配内存 | 利用率 |
|----------|------|---------|---------|-------|
| co (core_only) | statistical(14) | ~114G | 200G l40s | 57% |
| co (core_only) | foundation(14) | ~114G | 200G l40s | 57% |
| ce (core_edgar) | statistical(14) | ~164G | 200G l40s | 82% |
| ce (core_edgar) | foundation(14) | ~164G | 200G l40s | 82% |
| ct (core_text) | statistical(14) | 164-166G | 320G bigmem | 52% |
| ct (core_text) | deep_classical(9) | 128G | 320G gpu | 40% |
| ct (core_text) | foundation(14) | 134-135G | 320G gpu | 42% |
| ct (core_text) | irregular(4) | 125-127G | 320G gpu | 40% |
| fu (full) | statistical(14) | 216-217G | 320G bigmem | 68% |
| fu (full) | deep_classical(9) | 174-175G | 320G gpu | 55% |
| fu (full) | foundation(14) | 177-181G | 320G gpu | 56% |
| fu (full) | irregular(4) | 172-173G | 320G gpu | 54% |

### 3.2 Key Memory Driver Analysis
内存主要来自三个来源:
1. **数据加载** (~100-110G): freeze 数据集 + pandas wide 表
2. **特征工程** (+20-50G): EDGAR columns (+20G), text PCA embeddings (+30G)
3. **模型实例化** (+10-100G): 取决于模型参数量和 batch size

**内存公式 (经验):**
- `co/s2`: ~110G baseline
- `ce/e2`: ~110G + 20G(EDGAR) + 30G(overhead) ≈ 160G
- `ct`: ~110G + 30G(text embeddings) + overhead ≈ 130-170G
- `fu`: ~110G + 20G(EDGAR) + 30G(text) + 50G(overhead) ≈ 180-220G

---

## 4. 集群分区参考

| 分区 | 节点数 | 每节点内存 | GPU | 最佳用途 |
|------|--------|-----------|-----|---------|
| gpu | 24 | ~320G | V100-16GB | 主要计算分区，co/ce/ct 任务 |
| l40s | 2 | ~300G | L40S | co/ce tasks (较快GPU) |
| hopper | 1 | ~200G+ | H100 | 快速单模型训练 |
| bigmem | 4 | 640G | 无GPU | statistical models (CPU only), fu 超大任务 |

---

## 5. 决策规则 (Job Submission Checklist)

### 提交前必须确认:
1. **模型数量**: 超过 20 个模型的批次 → 考虑拆分
2. **Ablation 类型**: fu > ct > ce > co (内存递增)
3. **超大模型**: ModernTCN, NegBinGLM → 单独提交或排除
4. **ml_tabular fu**: **必须**逐模型拆分，禁止全批次
5. **TSLib 全批次**: **禁止** >30 models 一次性提交，必须分批

### 推荐提交策略:
```
# 安全提交: 单类别 (5-22 models)
co/ce → --mem=200G -p gpu (或 l40s)
ct    → --mem=256G -p gpu
fu    → --mem=320G -p gpu (或 bigmem)

# AutoFit V739
co/ce → --mem=200G -p l40s
ct    → --mem=300G -p l40s (或 320G gpu)
fu    → --mem=384G -p gpu

# ml_tabular 拆分
单模型 → --mem=200G -p gpu

# Phase 15 新模型 (23个)
any   → --mem=256G -p gpu

# 如果 OOM → 升级路径:
200G → 256G → 320G → 384G → 640G(bigmem)
```

---

## 6. OOM 应急处理流程

1. **收到 OOM 通知** → 检查 `sacct -j JOBID --format=MaxRSS,ReqMem`
2. **记录到本文档** (Section 2)
3. **判断升级可行性**:
   - 如果 MaxRSS ≈ ReqMem → 增加 50% 内存重提
   - 如果 MaxRSS << ReqMem → 可能是瞬时峰值，尝试 batch size 调小
   - 如果已用 640G 还 OOM → **拆分**或**排除**
4. **更新本文档**的推荐内存值

---

## 附录: 完整 OOM 清单 (chronological)

```
# AutoFit V737 (retired)
OOM  cf_v737_t1_ct   req=128G  gpu   00:01:25
OOM  cf_v737_t1_fu   req=192G  gpu   00:02:43
OOM  cf_v737_t2_ct   req=128G  gpu   00:01:23
OOM  cf_v737_t2_fu   req=192G  gpu   00:03:05
OOM  cf_v737_t3_fu   req=192G  gpu   00:02:52

# AutoFit V738 (retired)
OOM  cf_v738_t1_ct   req=128G  gpu   00:01:22
OOM  cf_v738_t1_ct   req=256G  gpu   00:06:18
OOM  cf_v738_t1_fu   req=192G  gpu   00:02:53
OOM  cf_v738_t1_fu   req=256G  gpu   00:05:46
OOM  cf_v738_t2_ct   req=128G  gpu   00:01:24
OOM  cf_v738_t2_ct   req=256G  gpu   00:04:55
OOM  cf_v738_t2_fu   req=192G  gpu   00:02:54
OOM  cf_v738_t2_fu   req=256G  gpu   00:05:39
OOM  cf_v738_t3_fu   req=192G  gpu   00:02:51
OOM  cf_v738_t3_fu   req=256G  gpu   00:05:04

# AutoFit V739 (current)
OOM  v739f_t1_ct     req=192G  hopper  00:02:02 (t2 same)
OOM  v739f_t2_fu     req=192G  hopper  00:02:24
OOM  v739f_t3_fu     req=192G  l40s    00:01:36
OOM  v739r_t1_fu     req=300G  l40s    00:08:05
OOM  v739r_t2_fu     req=300G  hopper  00:09:24

# Foundation Models (fc = foundation combined)
OOM  fc_t*_co        req=64G   l40s    <1m
OOM  fc_t*_ce        req=64-128G  l40s  <3m
OOM  fc_t*_ct        req=64-192G  l40s/gpu  <3m
OOM  fc_t*_fu        req=64-192G  l40s/gpu  <3m
OOM  p9_fmB_t*_ct/fu req=256G  gpu    ~6m

# ml_tabular (all-at-once)
OOM  ml_t1_fu_fix2   req=320G  gpu     02:57:58
OOM  ml_t1_fu_bm     req=640G  bigmem  02:36:06
OOM  p12_ml_t_t1_fu  req=640G  bigmem  03:30:06

# ModernTCN
OOM  mtcn_t2_ce      req=128G  gpu     02:04:17
OOM  mtcn_t3_ce      req=128G  gpu     02:04:49
TMO  mtcn_t3_co      req=128G  gpu     12:00:21

# NegativeBinomialGLM
# 365G used for ct (512G allocated) — fu未尝试

# TSLib 全批次 (TIMEOUT, 不是OOM但同样浪费)
TMO  p9_tsA_t*       req=320G  gpu     2-00:00 (全部)
TMO  p9_tsB_t*       req=320G  gpu     2-00:00 (全部)
TMO  p9r2_tsA_t*_fu  req=640G  gpu     2-00:00 (全部)

# Legacy V72/V73
OOM  p7r_v72_t1/t2_ct  req=378G  gpu   ~8-9h
OOM  p7r_v73_t1/t2_ct  req=378G  gpu   ~9-10h
```
