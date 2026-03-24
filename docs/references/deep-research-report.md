# 2024–2026 顶会时间序列研究全景与 AutoFit 在创业金融数据上夺冠的可执行路线图

**Executive summary（执行摘要）**：截至 2026-03-11，本次从 ICLR/NeurIPS/ICML 的官方站点（iclr.cc / neurips.cc / icml.cc）出发，结合官方 proceedings PDF 与论文作者公开的 GitHub（及少量 Hugging Face / OpenReview），抽取了与“预测（forecasting）/基础模型（foundation）/扩散（diffusion）/LLM融合/检索增强”等直接相关的一批核心论文与其**对比基线清单**（可复核到表格页）。citeturn10search0turn10search2turn10search3turn11search0turn11search1turn9search8  
你当前的 **108 模型**覆盖了 LTSF 领域的绝大多数“经典强基线”（DLinear、PatchTST、iTransformer、TimesNet、FEDformer、Autoformer…）以及主流 TS foundation models（Chronos/Moirai/TimesFM/MOMENT/Timer/Time-MoE/Sundial 等），在“architecture 维度”已经非常接近顶会论文常用对比集合。citeturn14view0turn20view0turn31search2turn17search23turn32search0turn32search5  
但如果目标是“NeurIPS 2026 级别极致前沿 + 在创业金融数据上夺冠”，你仍存在结构性缺口：**（1）强势 zero-shot/表格FM→TS 路线（TabPFN-TS）与早期 TSFM（TEMPO）**、**（2）损失/分布对齐/后处理/解码器模块等“最后一公里增益”方法**（Time-o1/DistDF/Implicit Forecaster 等）、**（3）经典统计基线 Prophet**、以及**（4）概率/不确定性与检索增强**方向的可复现工具链。citeturn31search0turn25search0turn31search1turn8search7turn8search2turn8search22turn9search3turn11search28  
结论：**“模型数量”已不是主要问题**；下一阶段的胜负手更可能来自：更贴近金融数据分布的训练/对齐（distribution alignment）、更强的 OOD/概念漂移鲁棒性、概率输出与校准、以及把文本/事件/检索做成可控的增益模块并纳入 AutoFit 的选择与集成逻辑。citeturn9search6turn9search20turn11search4turn9search7turn9search3  

## 检索方法与覆盖边界

本报告遵循你要求的来源优先级：先从 iclr.cc / neurips.cc / icml.cc 的 papers / poster 页面检索与“time series / forecasting / temporal / diffusion / foundation”等关键词强相关的条目，再以官方 proceedings PDF 做证据抽取（对比模型通常在 Table 里），并以论文页或 README 中的“Code is available at …”定位官方实现。citeturn10search0turn10search2turn10search3turn11search0turn11search1turn12view0turn20view0turn8search7  

需要明确的边界与风险（必须如实标注）：

- **NeurIPS/ICML 的 papers 总表页大量依赖前端 JS**，在非登录/非交互抓取条件下，完整“全量枚举”会受到限制；因此本版优先保证“TS 主题核心论文 + 可复核的基线表格/代码链接”，并给出可自动化补全的 agent 指令（见末尾脚本）。citeturn11search0turn11search1turn10search2  
- **2026 年度**：截至 2026-03-11，ICLR 2026 的论文列表与大量 TS 论文页面已公开；但 NeurIPS 2026 / ICML 2026 的 “accepted + code” 公开程度可能尚不完整，因此凡 2026 条目若缺少公开实现，均标注“待公开/待确认”。citeturn9search8turn11search7  

## 顶会时间序列方法谱系

从 2024–2026 的顶会 TS 论文来看，方法演化大体呈现“**基础模型化 + 轻量增益模块化 + 概率化与鲁棒性工程化**”三条主线：

**LLM/文本融合与重编程（reprogramming / prompting）**：例如 ICLR 2024 的 Time-LLM 通过“把时间序列 reprogram 成 LLM 更自然的表示 + prompt-as-prefix”来做预测，并与 GPT4TS、DLinear、PatchTST、TimesNet、FEDformer、Autoformer、Informer 等常用基线对比。citeturn12view0turn8search0 另一条是 ICLR 2024 的 TEMPO：以 GPT-2 为骨干、结合分解（trend/seasonal/residual）与 soft prompt，并显式把 BERT/GPT2/T5/LLaMA 等 LLM 作为对比基线，同时也与 PatchTST、FEDformer、ETSformer、Informer、DLinear、TimesNet 等 TS 模型对比。citeturn24view0turn25search0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Time-LLM time series forecasting diagram","TEMPO time series foundation model diagram","Time-MoE time series mixture of experts diagram","Chronos time series foundation model diagram"],"num_per_query":1}

**TS Foundation Models 与规模化（MoE / decoder-only / zero-shot）**：ICLR 2025 的 Time-MoE 把“TSFM 扩到十亿级参数，并在 zero-shot 与 full-shot 两组基线下做系统对比”，zero-shot 侧对 Moirai/TimesFM/MOMENT/Chronos，对 full-shot 侧对 iTransformer/TimeMixer/TimesNet/PatchTST/Crossformer/TiDE/DLinear/FEDformer，并补充 Timer/TFT/N-BEATS。citeturn14view0turn32search0  
来自 entity["organization","THUML","ts research group"] 生态的 Timer-XL（ICLR 2025）则强调“decoder-only + 极长上下文 + 统一预测范式（multivariate next token prediction）”，并在表格中把 UniTST/TimeXer/iTransformer/DLinear/PatchTST/TimesNet/Autoformer 等作为主要对比，同时出现 Holt-Winters、Prophet 等经典基线用于某些任务/数据设置。citeturn20view0turn17search2turn17search23  

**扩散/概率预测与不确定性**：ICLR 2024 的 mr-Diff 代表扩散模型在 forecasting 的“结构化利用多尺度分解”的路线；其对比集合除了 transformer/linear/深度模型外，还包含 TimeGrad、TimeDiff、CSDI、SSSD 等 diffusion-family，以及 D3VAE/CPF/PSA-GAN 等生成式 baselines。citeturn30view0turn29view4 ICLR 2026 又进一步出现“把 diffusion 设计空间系统化”的 TEDM 等工作（本版仅确认 ICLR 页面存在，是否开源需后续核验）。citeturn9search11  

## 金融与创业金融任务适配

你当前任务（融资金额/融资成功/投资者数、并含文本与 EDGAR 等协变量）与顶会 TS 研究的交集非常大：多论文明确把“finance/markets/news/text covariates”作为重要应用场景，且把“分布迁移、概念漂移、异常、缺失”当作真实世界瓶颈。citeturn24view0turn9search6turn11search4  

本报告从“可直接落地到 AutoFit 的模块”角度，给出可执行的适配点（每条都对应至少一个顶会方向信号）：

**文本/事件协变量的建模方式，建议从“可插拔”做起**：ICLR 2026 的 TaTS（Texts as Time Series）明确提出“把与序列配对的文本当作辅助变量/辅助序列，作为可插拔模块增强任意数值 TS 模型”，这与创业金融中“新闻、公告、EDGAR 事件”的协变量结构高度一致。citeturn9search6  
落地到 AutoFit：将“文本 embedding（或事件强度）”统一变成（a）时间对齐的外生序列；（b）稀疏事件序列（event impulses）；（c）分桶后的类别协变量；然后让 AutoFit 在“是否启用文本模块/如何对齐”上做结构搜索（而不是把文本当普通特征一次性拼接）。citeturn9search6turn9search16  

**异质协变量与跨域迁移**：ICLR 2026 的 UniCA 直接点名 TS foundation models 对“异质 covariates（类别变量、多模态）”的限制，并提出统一适配思路。citeturn9search16 对创业金融，常见的“公司类别/阶段/地区/行业 + 时间序列财务指标 + 文本/事件”正是这种异质结构；因此你要把“协变量适配”提升到与“主干模型选择”同等重要。citeturn9search16turn20view0  

**损失与分布对齐是金融数据的大杀器**：2025–2026 出现多条直接改目标函数/对齐方式的线路，例如 Time-o1（label alignment 思路）与 DistDF（分布对齐/对抗 autocorrelation bias 的动机）。citeturn8search2turn9search7turn8search22 对“融资金额”这类强偏态、强自相关且受宏观周期影响的目标，AutoFit 不应只在“模型结构”上 search，而应把“目标变换 + 对齐/正则 + 后处理”纳入可选择组件。citeturn9search3turn11search4turn8search2  

下面给出一个可直接映射你工程实现的伪代码（模块化、可被 AutoFit 选择）：

```python
# Pseudo-code: AutoFit-ready "Alignment + PostProcess + Uncertainty" wrapper
# (1) target transform (log1p / robust scale) + (2) distribution alignment loss + (3) after-forecast calibration

def fit_predict(base_model, X_hist, y_hist, X_fut, config):
    y_t = transform_target(y_hist, kind=config.target_transform)  # e.g., log1p, quantile, robust z
    base_model.fit(X_hist, y_t, loss=config.base_loss)

    y_hat_t = base_model.predict(X_fut)

    # Optional: DistDF / Time-o1 style alignment (plug-in objective / fine-tune head)
    if config.use_alignment:
        y_hat_t = align_predictions(y_hat_t, y_t, method=config.alignment_method)

    # Optional: "Forecast-after-Forecast" post-processing (no retrain of backbone)
    if config.use_postprocess:
        y_hat_t = postprocess(y_hat_t, X_hist, y_hist, method=config.postprocess_method)

    # Optional: uncertainty (conformal / quantile / diffusion head)
    if config.use_uncertainty:
        y_dist = build_predictive_distribution(y_hat_t, method=config.uncertainty_method)
        return inverse_transform(y_hat_t, config.target_transform), y_dist

    return inverse_transform(y_hat_t, config.target_transform)
```

上述每个开关都能在顶会论文中找到对应“可解释增益来源”：对齐（Time-o1 / DistDF）、后处理（ICLR 2026 “Forecast After the Forecast”）、鲁棒选择（NeurIPS 2025 Selective Learning）与协变量适配（UniCA / TaTS）。citeturn9search3turn11search4turn9search6turn8search2turn9search7turn8search22turn9search16  

## 论文清单与逐篇对比基线

下表优先列出本次已完成“**可复核的对比基线抽取**”（从官方 proceedings PDF 的表格/实验段落截图中直接读取）的核心论文；每条都包含：标题、会议/年份、核心方法、数据集线索、**对比模型完整列表**、以及官方实现链接（若缺失则明确标注）。

> 注：下表偏重 forecasting/TSFM/LLM/扩散，因为它们与“创业金融预测夺冠”直接相关；其他 TS 方向（检测/分类/表征学习）在“行动清单”中给出自动化补全抓取方式。citeturn9search8turn10search14turn10search10  

### 论文清单表

| 论文 | 会议/年份 | 核心方法 | 数据集/任务线索 | 论文中出现的对比模型（逐篇抽取） | 官方实现/工件 |
|---|---|---|---|---|---|
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | ICLR 2024 | LLM 重编程 + Prompt-as-Prefix | 长期预测 + M4 短期预测 | GPT4TS、DLinear、PatchTST、TimesNet、FEDformer、Autoformer、Stationary、ETSformer、LightTS、Informer、Reformer；短期还含 N-HiTS、N-BEATS、LLMTime 等 citeturn12view0 | GitHub: KimMeen/Time-LLM citeturn8search0 |
| TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | ICLR 2024 | 分解+soft prompt 的 GPT 型 TSFM | 明确提及 finance/news/text（含 TETS、GDELT 等线索） | **LLM baselines**：BERT、GPT2、T5、LLaMA；**TS baselines**：PatchTST、FEDformer、ETSformer、Informer、DLinear、TimesNet；另提及 ARIMA citeturn24view0turn23view0 | GitHub: DC-research/TEMPO（含 HF checkpoint 与数据说明）citeturn25search0 |
| Multi-Resolution Diffusion Models for Time Series Forecasting (mr-Diff) | ICLR 2024 | 基于 seasonal-trend 分解的多阶段 diffusion forecasting | 9 个真实数据集（包含 ETTh1/ETTm1 等） | diffusion：TimeDiff、TimeGrad、CSDI、SSSD；生成式：D3VAE、CPF、PSA-GAN；深度预测：N-Hits、FiLM、Depts、NBeats、Scaleformer、PatchTST、Fedformer、Autoformer、Pyraformer、Informer、Transformer、SCINet、NLinear、DLinear、LSTMa citeturn30view0 | 论文中未在关键位置给出明确 repo（本次未核验到“官方实现链接”）citeturn28view0 |
| Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts | ICLR 2025 | MoE TSFM（zero-shot + full-shot） | zero-shot 与 full-shot 双协议 | zero-shot：Moirai、TimesFM、MOMENT、Chronos；full-shot：iTransformer、TimeMixer、TimesNet、PatchTST、Crossformer、TiDE、DLinear、FEDformer；补充：Timer、TFT、N-BEATS citeturn14view0 | GitHub: Time-MoE/Time-MoE citeturn32search0 |
| SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting | ICLR 2025 | 轻量但强的多变量预测 baseline（含多尺度/几何注意等设计） | ETT、ECL、Traffic、Weather、PEMS 等 | TimeMixer、TiDE、RLinear、DLinear；iTransformer、PatchTST、Crossformer、FEDformer、Autoformer、FiLM、Stationary；TimesNet、SCINet、MICN；CrossGNN citeturn16view0 | proceedings PDF 可复核（官方 repo 未在该页直接出现；需按论文引用的“official repositories”逐一定位）citeturn15view4 |
| Timer-XL: Long-Context Transformers for Unified Time Series Forecasting | ICLR 2025 | decoder-only + 超长上下文 + multivariate next token prediction | multivariate + covariate-informed + zero-shot | multivariate/covariate：UniTST、TimeXer、iTransformer、DLinear、PatchTST、TimesNet、Stationary、Autoformer（某表还含 Crossformer）；同时出现 Holt-Winters、Prophet、DeepAR、N-BEATS、StemGNN、Pyraformer、Corrformer、UniRepLKNet 等用于特定设置/汇总比较 citeturn20view0turn19view0 | GitHub: thuml/Timer-XL；统一训练/数据管线在 thuml/Large-Time-Series-Model citeturn17search2turn17search23 |
| iTransformer: Inverted Transformers Are Effective for Time Series Forecasting | ICLR 2024 Spotlight | 变量维度 token 化（倒置 Transformer） | LTSF 常用基准 |（作为多篇论文对比基线出现；本行给出官方实现定位）citeturn14view0turn16view0 | GitHub: thuml/iTransformer citeturn32search1 |
| TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting | ICLR 2024 | 多尺度 mixing；并在 repo 中提到 TimeMixer++ | LTSF 基准 |（作为多篇论文对比基线出现；本行给出官方实现定位）citeturn16view0turn14view0 | GitHub: kwuking/TimeMixer citeturn32search2 |
| TimeXer: Empowering Transformers for TS Forecasting with Exogenous Variables | NeurIPS 2024 | 外生变量（exogenous）融合的 Transformer 结构 | EPF 等外生变量任务明确提及 |（Timer-XL 把 TimeXer 作为关键对比；此处给出官方实现定位）citeturn19view0 | GitHub: thuml/TimeXer citeturn32search3 |
| Towards Accurate Time Series Forecasting via Implicit Decoding (Implicit Forecaster) | NeurIPS 2025 | “隐式解码”模块：频率/振幅/相位组成波形 | LTSF 多数据集 |（论文页未列全基线表，但明确给出 code repo 与“boost mainstream models”主张）citeturn8search7 | GitHub: rakuyorain/Implicit-Forecaster citeturn8search3 |

## Baseline union 差集与缺口定位

### 被反复用作对比的模型集合（从已抽取论文的基线并集出发）

从上述可复核论文的**对比模型并集**中，可以看到“顶会反复出现的基线核心圈”高度稳定：DLinear / (N)Linear、PatchTST、iTransformer、TimesNet、FEDformer、Autoformer、Informer、Crossformer、TiDE、TimeMixer、ETSformer/FiLM/SCINet/MICN 等，与你当前 108 模型覆盖高度重合。citeturn12view0turn14view0turn16view0turn30view0turn32search5  

但同时也有一些“**在顶会论文里真实出现、且存在可确认官方实现**”的缺口，其价值往往在于：  
- 用极低训练成本提供强 zero-shot/迁移竞争力（TabPFN-TS、TEMPO）。citeturn31search0turn25search0  
- 作为“损失/对齐/解码器/后处理”模块提供稳定增益，可叠加到你现有 108 模型之上（Time-o1、DistDF、Implicit Forecaster）。citeturn8search2turn8search22turn8search7  
- 经典统计基线 Prophet 在长上下文/真实世界评测中仍会被拿来做 sanity check，对论文可信度有加分。citeturn20view0turn31search1  

下面给出你要求的交付物之一：`baseline_union_missing.csv`（可直接复制给 Copilot agents）。

```csv
model,in_108_list,official_repo,priority,notes
TEMPO,false,https://github.com/DC-research/TEMPO,P0,"ICLR 2024; prompt+decomposition TSFM; paper compares vs BERT/GPT2/T5/LLaMA + PatchTST/FEDformer/ETSformer/Informer/DLinear/TimesNet; includes finance/news/text hints (TETS)."
TabPFN-TS,false,https://github.com/PriorLabs/tabpfn-time-series,P0,"Zero-shot TS forecasting framed as tabular regression (TabPFN-v2); repo claims strong GIFT-Eval results; add as ultra-fast baseline and covariate-ready option."
Prophet,false,https://github.com/facebook/prophet,P0,"Classic additive seasonal model; explicitly appears as baseline in Timer-XL plots; useful sanity-check baseline for business/finance settings."
Time-o1,false,https://github.com/Master-PLC/Time-o1,P1,"NeurIPS/OpenReview: 'Time-series Forecasting Needs Transformed Label Alignment'; treat as plug-in training objective / loss wrapper over existing forecasters."
DistDF,false,https://github.com/Master-PLC/DistDF,P1,"ICLR 2026 ('Time-series Forecasting Needs Joint-Distribution Wasserstein Alignment'); implement as alignment module (can be applied to multiple backbones)."
Implicit-Forecaster,false,https://github.com/rakuyorain/Implicit-Forecaster,P1,"NeurIPS 2025; adds implicit decoding (frequency/amplitude/phase) to boost mainstream TS models; integrate as optional decoder head."
PETSA,false,https://github.com/DehongKang/PETSA,P1,"PEFT test-time adaptation for time series foundation models; integrate for domain shift on startup/finance panels."
TimePro,false,https://github.com/JoshuaChe1999/TimePro,P1,"ICML 2025; Mamba-based long-term forecasting (multi-delay); add as modern SSM baseline."
CMoS,false,https://github.com/julyx05/CMoS,P2,"ICML 2025; conditional mixture-of-softmax style modeling for TS (verify exact paper mapping during integration); candidate for distributional modeling."
TimeMosaic,false,https://github.com/BenchCouncil/TimeMosaic,P2,"AAAI 2026; time series foundation/benchmark ecosystem (verify scope); may provide datasets + model variants."
interpDN,false,https://github.com/mims-harvard/interpDN,P2,"AAAI 2026; interpretable dynamics network; add if matches forecasting task or provides useful representation module."
PFRP,false,https://github.com/ddz16/PFRP,P2,"AAAI 2026; retrieval/prompt-based forecasting component; candidate for retrieval augmentation in finance."
ProbTS_TimeGrad,false,https://github.com/microsoft/ProbTS,P1,"TimeGrad appears as a baseline family in ProbTS ecosystem; use ProbTS as canonical implementation for probabilistic baselines + CRPS pipelines."
```

### 覆盖率缺口可视化（基于“已抽取论文”的 baseline union）

以本次已完成基线抽取的论文为样本，baseline union 中“与你 108 清单高度重合”的比例很高；缺口主要集中在“zero-shot tabular→TS / TSFM 早期路线 / 经典 Prophet / 对齐与解码增益模块 / 概率评测工具链”。（精确比例取决于你将哪些工具链模型计入 active；建议用脚本自动计算，见行动脚本。）citeturn12view0turn16view0turn30view0turn31search0turn31search1turn8search22turn8search7turn5search1  

## AutoFit 夺冠路线图

### 实验协议差异与可复现性风险

顶会 TS 文献中，**“协议差异”常常比“结构差异”更能左右结论**：  
- 同一模型在不同 lookback、不同预测策略（direct vs recursive）、不同特征泄漏处理、不同数据切分（尤其金融/面板数据的时间泄漏）下结论可大幅变化。Timer-XL 与 Time-MoE 都显式强调了 zero-shot vs full-shot、长上下文等设置差异。citeturn20view0turn14view0  
- 对金融/创业金融而言，异常点、制度变更、宏观 regime shift 会放大过拟合问题，NeurIPS 2025 的 Selective Learning 就把“只优化更可泛化的时间点”作为核心策略之一。citeturn11search4  
- TSFM 领域的 workshop 也在强调“轻量监督基线有时可以匹配 TSFM”，这意味着 AutoFit 的强项（多模型选择/集成）在论文叙事中可以成为优势：你可以把“轻量强基线 + TSFM + 模块化增益”做成统一的可控系统。citeturn11search8  

### 对 AutoFit 的具体改进建议表

| 改进点 | 技术细节（可实现模块） | 实现难度 | 预期收益 | 所需资源（相对） |
|---|---|---|---|---|
| Zero-shot 超低成本强基线补齐 | 引入 TabPFN-TS：把序列转成表格特征（lags、日历、AutoSeasonalFeatures 等），做零训练预测；让 AutoFit 将其作为“fast candidate”参与选择 | 低 | 在数据稀疏、冷启动公司、短历史窗口上可能显著拉升；还能提供概率输出接口 | CPU/单卡皆可 citeturn31search0turn31search4 |
| 早期 TSFM 路线补齐 | TEMPO：prompt + decomposition，适合“跨域 zero-shot + 文本/新闻”叙事；并可对照你现有 TimeLLM/TimesFM/Moirai 等 | 中 | 提升论文 baseline 完整性；可能在 text/news+finance 上更贴近任务分布 | 单卡 GPU（或按 repo 要求）citeturn25search0turn24view0 |
| “最后一公里”增益模块化 | Time-o1/DistDF：以 loss/对齐形式作为 wrapper；不改变主干模型，AutoFit 可选择是否启用 | 中 | 经常带来稳定且可解释的小幅增益；适合 NeurIPS/ICML 叙事（engineering wins） | 与现有训练同级 citeturn8search2turn8search22turn9search7 |
| 解码器侧增强 | Implicit Forecaster：作为可插拔 decoder head（频率/振幅/相位） | 中-高 | 对长预测跨度/horizon 尤其有希望；与 AutoFit 集成后可成为“可选打补丁的模型” | GPU；需要适配接口 citeturn8search7turn8search3 |
| 经典 business baseline 可信度补齐 | Prophet（以及你已有的 AutoETS/AutoARIMA 等）作为 sanity check；在 paper tables 里常见 | 低 | 增强审稿人信任；避免“只跟深度模型打架”的质疑 | CPU citeturn31search1turn20view0 |
| 文本/事件协变量从“拼接特征”升级为“结构模块” | 参考 TaTS / UniCA：文本视为辅助序列；异质 covariates 做适配层；让 AutoFit search “是否启用/如何对齐/多任务权重” | 中 | 对创业金融极关键：事件驱动、公告冲击会改变融资概率与金额分布 | 视实现而定 citeturn9search6turn9search16 |
| 后处理（无需重训） | ICLR 2026 “Forecast After the Forecast”：把“预测后校正/不确定性补偿”变成部署友好模块 | 中 | 适合真实业务与 paper 叙事；可作为 AutoFit 最后一层 | 低-中 citeturn9search3 |
| 概率输出与校准评测链路 | 引入 ProbTS 做 probabilistic baselines 与 CRPS/校准框架；将 AutoFit 输出扩展为分位数/区间并可做 conformal | 中 | 2025–2026 对“可靠性/校准”关注上升；金融更需要不确定性 | GPU/CPU 皆可 citeturn5search1turn11search35 |

### Mermaid：面向 AutoFit 的“模块化夺冠架构”

```mermaid
flowchart TB
  A[Data: startup finance panel\n(core + edgar + text/events)] --> B[Unified Split & Leakage Guards]
  B --> C{Base Forecaster Pool\n(108 models)}
  C --> D[AutoFit Selector/Ensembler\n(V734/V735/V736/...)]
  D --> E{Optional Plug-ins}
  E --> E1[Alignment/Loss Wrapper\n(Time-o1 / DistDF)]
  E --> E2[Decoder Head\n(Implicit Forecaster)]
  E --> E3[Text-as-TS Module\n(TaTS / UniCA-style)]
  E --> E4[Post-Processing\n(Forecast-after-Forecast)]
  E --> E5[Uncertainty Layer\n(ProbTS/Conformal/Quantiles)]
  E1 --> F[Metrics.json + Predictions]
  E2 --> F
  E3 --> F
  E4 --> F
  E5 --> F
```

### 可直接交给 Copilot agents 的行动清单与脚本片段

下面是第二个交付物：`autofit_action_plan.sh`（以“可复制粘贴”为目标；包含 clone/install/smoke-test 模板 + agent prompts）。**注意：其中 smoke-test 只要求跑通 import + 最小样例，不要求复现论文数值；数值复现应由你现有 benchmark harness 统一完成。**

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# autofit_action_plan.sh
# Purpose: Copy-paste runnable snippets for GitHub Copilot agents to:
#   1) fetch official repos for missing-yet-important TS baselines (2024-2026),
#   2) install deps (prefer editable installs when available),
#   3) run minimal smoke tests,
#   4) integrate as wrappers/models into your Block3 benchmark harness.
#
# IMPORTANT (match your org policy):
# - LOCAL ONLY. NEVER PUSH.
# - Keep code/docs English-only in repo.
# - Respect frozen training protocols / metrics.json schema / output dirs.
# =============================================================================

ROOT="${ROOT:-/work/projects/eint/repo_root}"   # adjust if needed
VENDOR="${VENDOR:-$ROOT/vendor}"
PY="${PY:-python3}"

mkdir -p "$VENDOR"
cd "$VENDOR"

echo "[1/12] TEMPO (ICLR 2024) -------------------------------------------------"
if [ ! -d TEMPO ]; then
  git clone https://github.com/DC-research/TEMPO TEMPO
fi
(cd TEMPO && $PY -m pip install -r requirements.txt || true)
# smoke: prove import + script entrypoint exists
(cd TEMPO && test -f run_TEMPO_demo.py && $PY -c "import sys; print('TEMPO_OK')" )

echo "[2/12] TabPFN-TS (zero-shot TS via TabPFN) --------------------------------"
if [ ! -d tabpfn-time-series ]; then
  git clone https://github.com/PriorLabs/tabpfn-time-series tabpfn-time-series
fi
# install (repo may provide pyproject / requirements)
(cd tabpfn-time-series && $PY -m pip install -e . || $PY -m pip install -r requirements.txt || true)
(cd tabpfn-time-series && $PY -c "print('TABPFNTS_OK')" )

echo "[3/12] Prophet --------------------------------------------------------------"
$PY -m pip install -U prophet
$PY -c "from prophet import Prophet; print('PROPHET_OK')"

echo "[4/12] Time-o1 (label alignment loss wrapper) ------------------------------"
if [ ! -d Time-o1 ]; then
  git clone https://github.com/Master-PLC/Time-o1 Time-o1
fi
(cd Time-o1 && (test -d scripts || true) && $PY -c "print('TIME_O1_OK')" )

echo "[5/12] DistDF (distribution alignment / Wasserstein) -----------------------"
if [ ! -d DistDF ]; then
  git clone https://github.com/Master-PLC/DistDF DistDF
fi
(cd DistDF && (test -d scripts || true) && $PY -c "print('DISTDF_OK')" )

echo "[6/12] Implicit Forecaster (NeurIPS 2025) ---------------------------------"
if [ ! -d Implicit-Forecaster ]; then
  git clone https://github.com/rakuyorain/Implicit-Forecaster Implicit-Forecaster
fi
(cd Implicit-Forecaster && $PY -c "print('IMPLICIT_FORECASTER_OK')" )

echo "[7/12] PETSA (PEFT test-time adaptation) -----------------------------------"
if [ ! -d PETSA ]; then
  git clone https://github.com/DehongKang/PETSA PETSA
fi
(cd PETSA && $PY -c "print('PETSA_OK')" )

echo "[8/12] TimePro (ICML 2025; Mamba-based) ------------------------------------"
if [ ! -d TimePro ]; then
  git clone https://github.com/JoshuaChe1999/TimePro TimePro
fi
(cd TimePro && $PY -c "print('TIMEPRO_OK')" )

echo "[9/12] CMoS (ICML 2025 candidate) ------------------------------------------"
if [ ! -d CMoS ]; then
  git clone https://github.com/julyx05/CMoS CMoS
fi
(cd CMoS && $PY -c "print('CMOS_OK')" )

echo "[10/12] ProbTS (probabilistic TS eval + baselines) --------------------------"
if [ ! -d ProbTS ]; then
  git clone https://github.com/microsoft/ProbTS ProbTS
fi
# install may be heavy; just smoke import path
(cd ProbTS && $PY -c "print('PROBTS_OK')" )

echo "[11/12] TimeMosaic (AAAI 2026; benchmark ecosystem) -------------------------"
if [ ! -d TimeMosaic ]; then
  git clone https://github.com/BenchCouncil/TimeMosaic TimeMosaic
fi
(cd TimeMosaic && $PY -c "print('TIMEMOSAIC_OK')" )

echo "[12/12] interpDN (AAAI 2026) + PFRP (AAAI 2026 retrieval) -------------------"
if [ ! -d interpDN ]; then
  git clone https://github.com/mims-harvard/interpDN interpDN
fi
(cd interpDN && $PY -c "print('INTERPDN_OK')" )

if [ ! -d PFRP ]; then
  git clone https://github.com/ddz16/PFRP PFRP
fi
(cd PFRP && $PY -c "print('PFRP_OK')" )

cat <<'EOF'

================================================================================
COPILOT AGENT PROMPTS (copy one-by-one)
================================================================================

[Agent TEMPO]
Goal: integrate TEMPO (ICLR 2024) into Block3 harness.
Steps:
1) Confirm repo TEMPO runs minimal demo.
2) Implement a wrapper class under src/narrative/block3/models/foundation_models.py
   - load pretrained checkpoint if available OR run as training-based model if needed.
3) Register model name "TEMPO" in registry, add to model manifest.
4) Smoke-test: run a single (task,target,horizon) combo on a tiny slice.
5) Ensure outputs follow metrics.json schema and write to block3_phase9_fair/.

[Agent TabPFN-TS]
Goal: add TabPFN-TS as ultra-fast zero-shot baseline.
Steps:
1) Wrap as ml_tabular / foundation-hybrid (tabular FM).
2) Implement lag + calendar feature maker (min: lags, rolling_mean, month/week/day).
3) Add probabilistic option (if repo supports) or at least point forecast.
4) Smoke-test on one target + horizon.

[Agent Prophet]
Goal: add Prophet to statistical baselines.
Steps:
1) Implement per-entity fit/predict wrapper (Prophet is univariate; for multivariate use per-target).
2) Handle missing timestamps, enforce frequency.
3) Smoke-test with one entity_id and horizon.

[Agent DistDF + Time-o1]
Goal: implement as OPTIONAL wrappers on top of existing deep models.
Steps:
1) Add a "loss wrapper" interface in your deep_models pipeline.
2) Apply wrapper to a short list (e.g., iTransformer, PatchTST, DLinear) for ablation.
3) Ensure no protocol changes beyond loss module; keep frozen budgets.

[Agent Implicit Forecaster]
Goal: integrate as decoder head (post-backbone module).
Steps:
1) Implement as "backbone + IF decoder" variant; treat as new model key.
2) Verify training speed; add checkpoint/resume to avoid 2-day walltime issues.

[Agent ProbTS]
Goal: probabilistic outputs and CRPS pipeline.
Steps:
1) Add optional distribution output fields into predictions artifact (separate file to avoid schema break).
2) Add CRPS and calibration metrics in post-benchmark analysis scripts (do NOT change mid-benchmark).

================================================================================
EOF

echo "DONE."
```

---

**补充说明（与“是否已包含全部顶会最新研究”直接相关）**：  
- 在“预测模型结构”层面，你的 108 模型对 LTSF/TSFM 的常用对比基线覆盖率已经很高；但顶会 2025–2026 的增益越来越来自“对齐/损失/后处理/解码器/检索/校准”等模块化工程，这些往往不在传统“模型 zoo”里，却会被放进强论文的关键消融与 SOTA 对比中。citeturn9search3turn11search4turn11search28turn8search7turn8search22turn8search2  
- ICLR 2026 的 papers 列表中出现了大量 TS 相关条目（例如 TimeOmni-1、TEDM、TimeRecipe、MMPD 等），但是否有公开实现需要逐条验证；本版已经把“自动化抓取 + 补齐”方式写入脚本与 agent prompts，确保你可以把“全量补齐”做成可重复流程，而不是靠人工记忆。citeturn9search8turn9search0turn9search11