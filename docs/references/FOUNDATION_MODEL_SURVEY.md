# Time Series Foundation Model Survey (2024–2026)

Last updated: 2026-02-11

## Summary Table

| Model | Org | GitHub Stars | Pip Install | Python 3.12 | HuggingFace | Last Active | License |
|-------|-----|-------------|-------------|-------------|-------------|-------------|---------|
| **TimesFM 2.5** | Google Research | 7.7k | `pip install timesfm` | ≥3.10 (torch) | `google/timesfm-2.5-200m-pytorch` | Jan 2026 | Apache-2.0 |
| **Sundial** | THUML (Tsinghua) | 187 | No (HF `transformers`) | Yes (3.10+) | `thuml/sundial-base-128m` | Jul 2025 | Apache-2.0 |
| **Toto** | Databricks | N/A | No public release | Unknown | Gated/private (401) | Unknown | Unknown |
| **Time-MoE** | Tsinghua/ICLR'25 | 905 | No (git clone) | 3.10+ | `Maple728/TimeMoE-50M`, `TimeMoE-200M` | Dec 2025 | Apache-2.0 |
| **Xihe** | — | — | — | — | — | — | Not found |
| **YingLong** | Alibaba | 9 | No (HF `transformers`) | Yes | `qcw2333/YingLong_300m` (+3 sizes) | Sep 2025 | CC-BY-4.0 |
| **Timer-XL** | THUML (Tsinghua) | 125 | No (HF `transformers`) | Yes | `thuml/timer-base-84m` | Jul 2025 | Apache-2.0 |
| **Timer** | THUML (Tsinghua) | 932 | No (HF `transformers`) | Yes | `thuml/timer-base-84m` | Jul 2025 | MIT |
| **Lag-Llama** | ServiceNow/MS | 1.5k | `pip install lag-llama` (from source) | Likely yes | `time-series-foundation-models/Lag-Llama` | Jun 2025 | Apache-2.0 |
| **TTM** | IBM Research | 780 | `pip install tsfm_public` | 3.10–3.13 | `ibm-granite/granite-timeseries-ttm-r1` | Feb 2026 | Apache-2.0 |
| **MOMENT** | CMU AutonLab | 703 | `pip install momentfm` | 3.11 (testing 3.13) | `AutonLab/MOMENT-1-large` | Feb 2026 | MIT |
| **UniTS** | Harvard MIMS | 614 | No (git clone) | Yes | Weights via GitHub releases | Feb 2024 | MIT |

---

## 1. TimesFM (Google Research)

- **Paper**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) (ICML 2024)
- **GitHub**: https://github.com/google-research/timesfm (7.7k stars)
- **PyPI**: `pip install timesfm` (v1.3.0 for v1/v2; v2.5 requires editable install)
- **Latest version**: TimesFM 2.5 (200M params, 16k context, quantile head)
- **Python**: Torch ≥3.10, PAX requires 3.10
- **Docker**: No official image
- **Last commit**: ~2 weeks ago (Jan 2026)

### Installation (v2.5 — recommended)
```bash
git clone https://github.com/google-research/timesfm.git && cd timesfm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch]"
# For covariates: pip install -e ".[xreg]"
```

### Installation (v2.0 via PyPI)
```bash
pip install "timesfm[torch]"
```

### Inference API (v2.5)
```python
import torch, numpy as np, timesfm

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024, max_horizon=256,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[np.sin(np.linspace(0, 20, 100)), np.linspace(0, 1, 67)],
)
# point_forecast.shape = (2, 12)
# quantile_forecast.shape = (2, 12, 10)
```

### Inference API (v2.0 via PyPI)
```python
import timesfm
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                     horizon_len=128, num_layers=50,
                                     use_positional_embedding=False, context_len=2048),
    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)
import numpy as np
point, quantile = tfm.forecast([np.sin(np.linspace(0, 20, 100))], freq=[0])
```

---

## 2. Sundial (THUML, Tsinghua University)

- **Paper**: [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816) (ICML 2025 Oral, Top 1%)
- **GitHub**: https://github.com/thuml/Sundial (187 stars)
- **PyPI**: **No** — uses HuggingFace `transformers` directly
- **Model**: 128M params, pre-trained on 1 trillion time points
- **Python**: 3.10+ recommended
- **GIFT-Eval**: 1st MASE on the leaderboard
- **Last commit**: ~5 months ago

### Installation
```bash
pip install transformers==4.40.1 torch
```

### Inference API
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)

batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length)

forecast_length = 96
num_samples = 20
output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)
# output.shape = (1, num_samples, forecast_length) — use for mean/quantile estimation
```

**Key advantage**: Generative model — outputs multiple samples for probabilistic forecasting (confidence intervals, quantiles) from a single forward pass.

---

## 3. Toto (Databricks)

- **Status**: **NOT PUBLICLY AVAILABLE** as of Feb 2026
- All HuggingFace model pages return 401 (gated/private)
- No public GitHub repository found
- No public blog post or documentation accessible
- Databricks HF org shows 0 public models
- **Conclusion**: Cannot be used outside Databricks platform currently

---

## 4. Time-MoE (ICLR 2025 Spotlight)

- **Paper**: [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://arxiv.org/abs/2409.16040)
- **GitHub**: https://github.com/Time-MoE/Time-MoE (905 stars)
- **PyPI**: **No** — git clone + requirements.txt
- **Models**: 50M (base) and 200M (large), trained on Time-300B dataset (300B+ time points)
- **Context**: Up to 4096
- **Python**: 3.10+
- **Requires**: `transformers==4.40.1`
- **Last commit**: ~2 months ago

### Installation
```bash
git clone https://github.com/Time-MoE/Time-MoE.git && cd Time-MoE
pip install -r requirements.txt
# Optional: pip install flash-attn==2.6.3
```

### Inference API
```python
import torch
from transformers import AutoModelForCausalLM

seqs = torch.randn(2, 512)  # [batch_size, context_length]
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',  # or 'Maple728/TimeMoE-200M'
    device_map="cuda",
    trust_remote_code=True,
)

# Normalize
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# Forecast
prediction_length = 96
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]
predictions = normed_predictions * std + mean
```

---

## 5. Xihe

- **Status**: **NOT FOUND**
- No GitHub repositories matching "xihe time series" (0 results)
- No HuggingFace models matching "xihe time series" (0 results)
- May be confused with a weather/climate model or not yet released
- **Conclusion**: Does not appear to exist as a public time series forecasting model

---

## 6. YingLong (Alibaba)

- **Paper**: [arXiv:2506.11029](https://huggingface.co/papers/2506.11029) (May 2025)
- **GitHub**: https://github.com/wxie9/YingLong (9 stars)
- **PyPI**: **No** — uses HuggingFace `transformers`
- **Models**: 4 sizes: 6M, 50M, 110M, 300M (pre-trained on 78B time points)
- **Python**: 3.10+
- **License**: CC-BY-4.0
- **Last commit**: ~9 months ago

### Installation
```bash
pip install xformers transformers torch
# Optional for faster inference:
pip install flash-attn --no-build-isolation
```

### Inference API
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'qcw2333/YingLong_300m',  # or YingLong_6m, _50m, _110m
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

seqs = torch.randn(1, 2880).bfloat16().cuda()
output = model.generate(seqs, future_token=96)
# output.shape = (1, 96)
```

---

## 7. Timer-XL / Timer (THUML, Tsinghua University)

Timer and Timer-XL are from the same group and share a unified codebase.

### Timer (ICML 2024)
- **Paper**: [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368)
- **GitHub**: https://github.com/thuml/Large-Time-Series-Model (932 stars)

### Timer-XL (ICLR 2025)
- **Paper**: [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://arxiv.org/abs/2410.04803)
- **GitHub**: https://github.com/thuml/Timer-XL (125 stars)
- **HuggingFace**: `thuml/timer-base-84m` (84M params)
- **PyPI**: **No** — uses HuggingFace `transformers`
- **License**: Apache-2.0 / MIT

### Installation
```bash
pip install transformers torch
```

### Inference API (Timer / Timer-XL zero-shot)
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

seqs = torch.randn(1, 2880)
# Normalize
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

prediction_length = 96
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_pred = output[:, -prediction_length:]
predictions = normed_pred * std + mean
```

---

## 8. Lag-Llama (ServiceNow / Morgan Stanley)

- **Paper**: [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)
- **GitHub**: https://github.com/time-series-foundation-models/lag-llama (1.5k stars)
- **PyPI**: **No** (pip install from source git)
- **HuggingFace**: `time-series-foundation-models/Lag-Llama`
- **Python**: 3.10+
- **License**: Apache-2.0
- **Last commit**: ~8 months ago (kv_cache fix)
- **Built on**: GluonTS framework

### Installation
```bash
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
# Or clone manually:
git clone https://github.com/time-series-foundation-models/lag-llama.git && cd lag-llama
pip install -r requirements.txt
```

### Inference API
```python
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="time-series-foundation-models/Lag-Llama",
                             filename="lag-llama.ckpt")

# Create estimator for zero-shot
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=24,
    context_length=32,
    input_size=1,
    use_rope_scaling=True,
)
predictor = estimator.create_lightning_module().to("cuda")
# See Colab demos for full pipeline
```

**Note**: More complex setup than other models due to GluonTS dependency. Best used via the provided Colab notebooks.

---

## 9. TTM — Tiny Time Mixers (IBM Research / Granite)

- **Paper**: [Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting](https://arxiv.org/abs/2401.03955) (NeurIPS 2024)
- **GitHub**: https://github.com/ibm-granite/granite-tsfm (780 stars)
- **PyPI**: `pip install tsfm_public`
- **HuggingFace**: `ibm-granite/granite-timeseries-ttm-r1` (R1) and `ibm-granite/granite-timeseries-ttm-r2` (R2)
- **Python**: 3.10, 3.11, 3.12, **3.13** — widest compatibility
- **Model size**: <1M parameters — runs on CPU!
- **License**: Apache-2.0
- **Downloads**: 2.28M/month (most popular by downloads)
- **Last commit**: 2 days ago (very active)

### Installation
```bash
pip install "tsfm_public[notebooks]"
# Or from source:
git clone https://github.com/ibm-granite/granite-tsfm.git && cd granite-tsfm
pip install ".[notebooks]"
```

### Inference API
```python
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from transformers import Trainer

# Load model
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r1",
    revision="main"  # 512-96 variant
)

# Zero-shot evaluation
zeroshot_trainer = Trainer(model=model, args=zeroshot_forecast_args)
zeroshot_output = zeroshot_trainer.evaluate(dset_test)

# Fine-tuning (freeze backbone)
for param in model.backbone.parameters():
    param.requires_grad = False

finetune_trainer = Trainer(
    model=model, args=finetune_args,
    train_dataset=dset_train, eval_dataset=dset_val,
)
finetune_trainer.train()
```

**Variants**: `512-96` (main), `1024-96` (1024-96-v1). R2 models trained on 700M samples.

---

## 10. MOMENT (CMU AutonLab)

- **Paper**: [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885) (ICML 2024)
- **GitHub**: https://github.com/moment-timeseries-foundation-model/moment (703 stars)
- **PyPI**: `pip install momentfm` (v0.1.4)
- **HuggingFace**: `AutonLab/MOMENT-1-large`, `MOMENT-1-base`, `MOMENT-1-small`
- **Python**: 3.11 recommended (testing up to 3.13)
- **License**: MIT
- **Last commit**: yesterday

### Installation
```bash
pip install momentfm
# Or from source:
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

### Inference API
```python
from momentfm import MOMENTPipeline

# Forecasting
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "forecasting", "forecast_horizon": 96},
)
model.init()

# Classification
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "classification", "n_channels": 1, "num_class": 2},
)
model.init()

# Embedding / Representation Learning
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "embedding"},
)
model.init()

# Anomaly Detection / Imputation
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"},
)
model.init()
```

**Key advantage**: Multi-task (forecasting + classification + anomaly detection + imputation + embedding) from one model.

---

## 11. UniTS (Harvard MIMS)

- **Paper**: [UniTS: Building a Unified Time Series Model](https://arxiv.org/abs/2403.00131) (NeurIPS 2024)
- **GitHub**: https://github.com/mims-harvard/UniTS (614 stars)
- **PyPI**: **No** — git clone + requirements.txt
- **Weights**: GitHub releases (ckpt tag)
- **Python**: 3.10+ (PyTorch 2.0+)
- **License**: MIT
- **Last commit**: ~2 years ago (no longer actively maintained?)

### Installation
```bash
git clone https://github.com/mims-harvard/UniTS.git && cd UniTS
pip install -r requirements.txt
bash download_data_all.sh
```

### Usage
```bash
# Supervised learning
bash ./scripts/supervised_learning/UniTS_supervised.sh
# Pretrain + Prompt learning
bash ./scripts/pretrain_prompt_learning/UniTS_pretrain_x128.sh
# Zero-shot on new data
bash ./scripts/zero_shot/UniTS_zeroshot_newdata.sh
```

**Note**: Script-based rather than API-based. No simple `from_pretrained` pattern. Best for research benchmarking rather than production pipelines.

---

## Recommendations for Block 3 Integration

### Tier 1 — Easy integration (pip install, clean API)
| Model | Install | Why |
|-------|---------|-----|
| **TimesFM 2.5** | `pip install timesfm[torch]` or editable | Best point forecaster, quantile head, Google-backed |
| **TTM** | `pip install tsfm_public` | <1M params, CPU-friendly, Python 3.10–3.13, IBM-backed |
| **MOMENT** | `pip install momentfm` | Multi-task (forecast + classify + embed), MIT license |

### Tier 2 — Moderate integration (HuggingFace `transformers` API)
| Model | Install | Why |
|-------|---------|-----|
| **Sundial** | `pip install transformers==4.40.1` | ICML 2025 Oral, 1st on GIFT-Eval, probabilistic |
| **Timer/Timer-XL** | `pip install transformers` | ICML 2024 / ICLR 2025, same API pattern |
| **Time-MoE** | `pip install transformers==4.40.1` | ICLR 2025 Spotlight, MoE architecture, 2.4B params |
| **YingLong** | `pip install transformers` | Alibaba, 4 model sizes, CC-BY-4.0 |

### Tier 3 — Complex integration (custom setup needed)
| Model | Why complex |
|-------|-------------|
| **Lag-Llama** | GluonTS dependency, manual checkpoint download |
| **UniTS** | Script-based, no from_pretrained API |

### Not available
| Model | Status |
|-------|--------|
| **Toto** | Private/gated on Databricks |
| **Xihe** | Does not exist as a public time series model |
| **TimesXL** (unrelated) | Confused with Timer-XL above |

---

## Common Installation Pattern (HuggingFace models)

Most THUML/Alibaba models follow the same pattern:
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '<repo_id>',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

seqs = torch.randn(batch, context_len).bfloat16().cuda()
output = model.generate(seqs, max_new_tokens=horizon)
predictions = output[:, -horizon:]
```

This applies to: **Sundial**, **Timer/Timer-XL**, **Time-MoE**, **YingLong**.

---

## Docker Notes

No models in this survey provide official Docker images. For containerized deployment:
1. Use `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` as base
2. Install `transformers`, `safetensors`, `huggingface_hub`
3. Pre-download weights: `huggingface-cli download <repo_id>`
