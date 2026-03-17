#!/usr/bin/env python3
"""
TSLib Model Wrapper — Integration of thuml/Time-Series-Library models.

Wraps standalone PyTorch models from TSLib (https://github.com/thuml/Time-Series-Library)
into our Block 3 ModelBase interface. This enables direct comparison of 14 additional
SOTA models from NeurIPS, ICML, ICLR, AAAI (2022-2025) that are NOT available in
NeuralForecast.

Architecture:
  1. TSLibModelWrapper(ModelBase): handles data conversion, training loop, inference
  2. _TSLibConfigs: namespace object mimicking TSLib's argparse configs
  3. Entity-panel training: TSLib models are trained globally (one model for all entities)
  4. Per-series normalization: RevIN-style normalization before training
  5. Early stopping + validation: 20% temporal holdout for validation

Models added (14 total):
  2025:  TimeFilter (ICML), WPMixer (AAAI), MultiPatchFormer (SciRep)
  2024:  MSGNet (AAAI), PAttn (NeurIPS), MambaSimple (SSM-based)
  2023:  Koopa (NeurIPS), FreTS (NeurIPS), Crossformer (ICLR),
         MICN (ICLR), SegRNN (arXiv)
  2022:  Nonstationary_Transformer (NeurIPS), FiLM (NeurIPS), SCINet (NeurIPS)

Dependencies:
  - TSLib must be cloned to: /mnt/aiongpfs/projects/eint/vendor/Time-Series-Library
  - PyTorch 2.x, einops, scipy (all pre-installed)
  - No mamba_ssm needed (use MambaSimple instead of Mamba)
"""
from __future__ import annotations

import gc
import importlib
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ModelBase, ModelConfig

logger = logging.getLogger(__name__)

# ============================================================================
# TSLib path setup
# ============================================================================
TSLIB_ROOT = Path(os.environ.get(
    "TSLIB_ROOT",
    "/mnt/aiongpfs/projects/eint/vendor/Time-Series-Library"
))


def _ensure_tslib_path():
    """Add TSLib root to sys.path so `from layers.X import Y` works."""
    tslib_str = str(TSLIB_ROOT)
    if tslib_str not in sys.path:
        sys.path.insert(0, tslib_str)


# ============================================================================
# Default hyperparameters per model
# key: TSLib model file name (without .py)
# ============================================================================
TSLIB_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ── 2025 ──
    "TimeFilter": {
        "venue": "ICML 2025",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "activation": "gelu",
        "alpha": 0.1, "top_p": 0.5, "pos": True,
        "factor": 1,
    },
    "WPMixer": {
        "venue": "AAAI 2025",
        "d_model": 128, "d_ff": 256, "n_heads": 4, "e_layers": 3,
        "dropout": 0.1, "level": 3, "factor": 2,
        "activation": "gelu",
    },
    "MultiPatchFormer": {
        "venue": "Scientific Reports 2025",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "activation": "gelu",
        "factor": 1,
    },
    # ── 2024 ──
    "MSGNet": {
        "venue": "AAAI 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "top_k": 5, "activation": "gelu",
        "factor": 1,
        # MSGNet-specific graph convolution params
        "conv_channel": 32, "skip_channel": 32, "gcn_depth": 2,
        "propalpha": 0.05, "node_dim": 40, "individual": 0,
        "subgraph_size": 20,
    },
    "PAttn": {
        "venue": "NeurIPS 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 1,
        "dropout": 0.1, "patch_len": 16, "activation": "gelu",
        "factor": 1,
    },
    "MambaSimple": {
        "venue": "arXiv 2023 (SSM)",
        "d_model": 128, "d_ff": 16, "d_conv": 4, "expand": 2,
        "dropout": 0.1, "e_layers": 2,
        "activation": "gelu", "factor": 1,
    },
    # ── 2023 ──
    "Koopa": {
        "venue": "NeurIPS 2023",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "seg_len": 12, "num_blocks": 3,
        "dynamic_dim": 128, "alpha": 0.2, "activation": "gelu",
        "factor": 1,
    },
    "FreTS": {
        "venue": "NeurIPS 2023",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "activation": "gelu",
        "factor": 1,
        "channel_independence": 1,  # FreTS: 1=channel-independent
    },
    "Crossformer": {
        "venue": "ICLR 2023",
        "d_model": 128, "d_ff": 256, "n_heads": 4, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "seg_len": 12,
        "activation": "gelu", "factor": 1,
    },
    "MICN": {
        "venue": "ICLR 2023",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "decomp_kernel": [33],
        "conv_kernel": [12, 16], "isometric_kernel": [18, 6],
        "activation": "gelu", "factor": 1,
    },
    "SegRNN": {
        "venue": "arXiv 2023",
        "d_model": 128, "d_ff": 256, "dropout": 0.1,
        "seg_len": 12, "activation": "gelu",
        "factor": 1, "rnn_type": "gru",
    },
    # ── 2022 ──
    "Nonstationary_Transformer": {
        "venue": "NeurIPS 2022",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2, "activation": "gelu", "factor": 1,
    },
    "FiLM": {
        "venue": "NeurIPS 2022",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "modes1": 32, "mode_type": 0,
        "activation": "gelu", "factor": 1,
        "ratio": 0.5,  # FiLM: Legendre projection ratio
    },
    "SCINet": {
        "venue": "NeurIPS 2022",
        "d_model": 128, "d_ff": 256, "dropout": 0.1,
        "hidden_size": 4, "num_levels": 3, "num_stacks": 1,
        "num_decoder_layer": 1, "concat_len": 0, "groups": 1,
        "kernel": 5, "dilation": 1, "positionalE": False,
        "single_step_output_One": 0, "RIN": False,
        "activation": "gelu", "factor": 1,
    },
    # ── Phase 9 additions: models available in vendored TSLib ──
    "ETSformer": {
        "venue": "ICML 2023",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "d_layers": 2, "dropout": 0.1, "activation": "gelu",
        "factor": 1, "K": 3, "std": 0.2,
    },
    "LightTS": {
        "venue": "arXiv 2022",
        "d_model": 128, "d_ff": 256, "dropout": 0.1,
        "activation": "gelu", "factor": 1,
        "chunk_size": 12,
    },
    "Pyraformer": {
        "venue": "ICLR 2022",
        "d_model": 128, "d_ff": 256, "n_heads": 4, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "activation": "gelu",
        "factor": 1, "window_size": [4, 4],
        "inner_size": 3,
    },
    "Reformer": {
        "venue": "ICLR 2020",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "d_layers": 1, "dropout": 0.1, "activation": "gelu",
        "factor": 1, "bucket_size": 4, "n_hashes": 4,
    },
    "TiRex": {
        "venue": "ICLR 2025",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "activation": "gelu",
        "factor": 1,
    },
    "Mamba": {
        "venue": "ICLR 2024",
        "d_model": 128, "d_ff": 16, "d_conv": 4, "expand": 2,
        "dropout": 0.1, "e_layers": 4,
        "activation": "gelu", "factor": 1,
    },
    # ── Phase 10 additions ──
    "KANAD": {
        "venue": "arXiv 2024",
        "d_model": 8, "d_ff": 256, "dropout": 0.1,
        "activation": "gelu", "factor": 1,
    },
    "FITS": {
        "venue": "ICLR 2024",
        "cut_freq": 30, "individual": False,
        "dropout": 0.0,
    },
    "SparseTSF": {
        "venue": "ICML 2024",
        "d_model": 128, "period_len": 7, "model_type": "mlp",
        "dropout": 0.0,
    },
    "CATS": {
        "venue": "ICLR 2025",
        "d_model": 128, "d_ff": 256, "n_heads": 16,
        "d_layers": 3, "dropout": 0.1, "patch_len": 24,
        "stride": 24, "padding_patch": "end",
        "QAM_start": 0.1, "QAM_end": 0.5,
        "query_independence": True, "store_attn": False,
    },
    "Fredformer": {
        "venue": "KDD 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "stride": 8,
        "padding_patch": "end", "revin": 1, "affine": 0,
        "subtract_last": 0, "individual": 0,
        "fc_dropout": 0.1, "head_dropout": 0.0,
        "cf_dim": 48, "cf_depth": 1, "cf_heads": 4,
        "cf_head_dim": 32, "cf_drop": 0.0, "cf_mlp": 256,
        "mlp_hidden": 256, "mlp_drop": 0.1, "use_nys": 0,
        "ablation": 0, "output": 0,
    },
    "CycleNet": {
        "venue": "NeurIPS 2024",
        "d_model": 512, "cycle": 7, "model_type": "linear",
        "use_revin": True,
    },
    "xPatch": {
        "venue": "arXiv 2024",
        "patch_len": 16, "stride": 8,
        "padding_patch": "end", "revin": 1,
        "alpha": 0.0, "beta": 0.0, "ma_type": "ema",
    },
    "FilterTS": {
        "venue": "arXiv 2024",
        "d_model": 128, "e_layers": 2,
        "dropout": 0.1, "filter_type": "all",
        "bandwidth": 0.1, "quantile": 0.5,
        "top_K_static_freqs": 5, "use_norm": True,
        "embedding": "fourier_interpolate",
    },
    # ── Phase 11: Section C SOTA models ──
    "CFPT": {
        "venue": "ICLR 2025",
        "d_model": 64, "e_layers": 2, "dropout": 0.1,
        "beta": 0.5, "rda": 2, "rdb": 2, "ksize": 3, "period": 7,
        "time_feature_types": ["dayofweek", "dayofmonth", "dayofyear"],
    },
    "DeformableTST": {
        "venue": "ICLR 2025",
        "dims": [128, 256], "depths": [2, 2], "heads": [4, 8],
        "n_groups": [2, 4], "stage_spec": [["D", "D"], ["D", "D"]],
        "window_size": [7, 7], "dropout": 0.1, "drop_path_rate": 0.1,
        "head_dropout": 0.0, "revin": True, "revin_affine": True,
        "revin_subtract_last": False, "ksize": [5, 5], "stride": [1, 1],
        "nat_ksize": [7, 7], "stem_ratio": 1, "down_ratio": 1,
        "fmap_size": 7, "head_type": "flatten",
        "use_pe": [True, True], "dwc_pe": [True, True],
        "fixed_pe": [False, False],
        "no_off": [False, False], "offset_range_factor": [2, 2],
        "use_dwc_mlp": [True, True], "use_lpu": [True, True],
        "local_kernel_size": [5, 5],
        "expansion": 4, "attn_drop": 0.0, "drop": 0.0,
        "proj_drop": 0.0, "log_cpb": [False, False],
        "use_head_norm": True,
        "layer_scale_value": [-1, -1],
    },
    "ModernTCN": {
        "venue": "ICLR 2024",
        "dims": [64, 64, 64, 64], "dw_dims": [256, 256, 256, 256],
        "large_size": [51, 51, 51, 51], "small_size": [5, 5, 5, 5],
        "kernel_size": 51, "num_blocks": [1, 1, 1, 1], "dropout": 0.1,
        "patch_size": 8, "patch_stride": 4, "stem_ratio": 1.0,
        "downsample_ratio": 2, "ffn_ratio": 2, "freq": "h",
        "small_kernel_merged": False, "use_multi_scale": False,
        "revin": True, "affine": True, "subtract_last": False,
        "individual": False, "head_dropout": 0.0, "decomposition": False,
    },
    "PathFormer": {
        "venue": "ICLR 2024",
        "d_model": 128, "d_ff": 256, "dropout": 0.1,
        "num_experts_list": [4, 4],
        "patch_size_list": [[16, 12, 8, 4], [16, 12, 8, 4]], "layer_nums": 2,
        "k": 2, "residual_connection": True, "revin": True,
        "batch_norm": True, "gpu": 0, "num_nodes": 77,
    },
    "SEMPO": {
        "venue": "ICML 2024",
        "d_model": 64, "e_layers": 2, "d_layers": 1,
        "dropout": 0.1, "patch_len": 16, "stride": 8,
        "c_in": 1, "head_type": "prediction",
        "domain_len": 128, "horizon_lengths": [7, 14, 30, 60],
        "setting": "long_term_forecast", "data": "custom",
    },
    "TimePerceiver": {
        "venue": "arXiv 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8,
        "dropout": 0.1, "patch_len": 16,
        "latent_dim": 64, "latent_d_ff": 128,
        "num_latents": 32, "num_latent_blocks": 2,
        "use_latent": True, "query_share": True,
    },
    "TimeBridge": {
        "venue": "NeurIPS 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8,
        "dropout": 0.1, "attn_dropout": 0.1,
        "ca_layers": 2, "ia_layers": 1, "pd_layers": 2,
        "num_p": 4, "period": 7, "stable_len": 30,
        "revin": True, "activation": "gelu",
    },
    "TQNet": {
        "venue": "ICML 2024",
        "d_model": 128, "dropout": 0.1,
        "cycle": 7, "model_type": "linear", "use_revin": True,
    },
    "PIR": {
        "venue": "NeurIPS 2024",
        "d_model": 128, "n_heads": 8, "dropout": 0.1,
        "backbone": "PatchTST", "retrieval_num": 5,
        "retrieval_stride": 1, "factor": 1,
        "refine_layers": 1, "refine_d_model": 128,
        "refine_d_ff": 256, "use_norm": True,
        "including_time_features": True, "activation": "gelu",
    },
    "CARD": {
        "venue": "ICLR 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "stride": 8,
        "dp_rank": 32, "momentum": 0.1, "merge_size": 2,
        "alpha": 0.0, "use_statistic": True,
        "total_token_number": 0,
    },
    "PDF": {
        "venue": "ICML 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "fc_dropout": 0.1, "head_dropout": 0.0,
        "patch_len": [16], "stride": [8], "padding_patch": "end",
        "kernel_list": [3, 5, 7], "period": [7],
        "serial_conv": True, "wo_conv": False, "add": True,
        "revin": True, "affine": True, "subtract_last": False,
        "individual": False,
    },
    "TimeRecipe": {
        "venue": "NeurIPS 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "patch_len": 16, "activation": "gelu",
        "emb_type": "patch", "ff_type": "mlp",
        "fusion": "temporal", "use_decomp": False, "use_norm": True,
    },
    "DUET": {
        "venue": "NeurIPS 2024",
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "activation": "gelu", "factor": 1,
        "CI": True, "fc_dropout": 0.1,
        "noisy_gating": True, "num_experts": 4, "k": 2,
        "hidden_size": 128,
    },
    "SRSNet": {
        "venue": "arXiv 2024",
        "d_model": 128, "dropout": 0.1,
        "patch_len": 16, "stride": 8,
        "hidden_size": 256, "alpha": 0.5, "pos": True,
        "head_mode": "linear", "affine": True, "subtract_last": False,
    },
}

# Map model names to their TSLib file names (most are identical)
_MODEL_FILE_MAP = {
    "NonstationaryTransformer": "Nonstationary_Transformer",
}


class TSLibModelWrapper(ModelBase):
    """Wrapper for thuml/Time-Series-Library models.

    Converts our panel DataFrames into TSLib's expected tensor format,
    runs a standard PyTorch training loop with early stopping, and
    translates predictions back to our format.

    TSLib models expect:
      - configs: argparse-like namespace with seq_len, pred_len, d_model, etc.
      - forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → predictions

    Our wrapper:
      1. Converts entity-panel DataFrame to windowed (B, seq_len, C) tensors
      2. Trains with MSE loss, Adam optimizer, cosine LR schedule
      3. Early stopping on temporal validation set (last 20% of timeline)
      4. RevIN-style per-window normalization
    """

    def __init__(
        self,
        config: ModelConfig,
        tslib_model_name: str,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 15,
        **kwargs,
    ):
        super().__init__(config)
        self._tslib_model_name = tslib_model_name
        self._max_epochs = max_epochs
        self._lr = learning_rate
        self._batch_size = batch_size
        self._patience = patience
        self._extra_kwargs = kwargs
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._train_feat_mean: Optional[np.ndarray] = None
        self._train_feat_std: Optional[np.ndarray] = None

    # Time feature dimension per TSLib frequency code (from layers/Embed.py)
    _FREQ_FEAT_DIM = {
        "h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3,
    }

    def _build_configs(self, seq_len: int, pred_len: int, enc_in: int) -> SimpleNamespace:
        """Build a TSLib-compatible configs namespace."""
        model_key = _MODEL_FILE_MAP.get(self._tslib_model_name, self._tslib_model_name)
        defaults = TSLIB_CONFIGS.get(model_key, {})

        # For models with seg_len, align pred_len to be >= seg_len and a multiple
        seg_len = defaults.get("seg_len", None)
        if seg_len is not None and pred_len < seg_len:
            pred_len = seg_len
        if seg_len is not None and pred_len % seg_len != 0:
            pred_len = ((pred_len // seg_len) + 1) * seg_len

        # For TimeFilter: patch_len must divide enc_in * seq_len
        raw_patch_len = defaults.get("patch_len", 16)
        if model_key == "TimeFilter":
            total = enc_in * seq_len
            # Find largest divisor of total that's <= raw_patch_len
            patch_len = raw_patch_len
            while patch_len > 1 and total % patch_len != 0:
                patch_len -= 1
            logger.info(
                f"[TSLib-TimeFilter] patch_len adjusted: {raw_patch_len} → {patch_len} "
                f"(enc_in={enc_in}, seq_len={seq_len}, total={total})"
            )
        else:
            patch_len = raw_patch_len

        # FITS: cut_freq defaults to half the sequence length
        if model_key == "FITS":
            if "cut_freq" not in defaults or defaults["cut_freq"] > seq_len // 2:
                defaults = dict(defaults)
                defaults["cut_freq"] = max(1, seq_len // 2)

        cfg = SimpleNamespace(
            # Core temporal dimensions
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=seq_len // 2,  # decoder input overlap
            # Channel dimensions
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=1,  # univariate target prediction
            # Architecture
            d_model=defaults.get("d_model", 128),
            d_ff=defaults.get("d_ff", 256),
            n_heads=defaults.get("n_heads", 8),
            e_layers=defaults.get("e_layers", 2),
            d_layers=defaults.get("d_layers", 1),
            dropout=defaults.get("dropout", 0.1),
            activation=defaults.get("activation", "gelu"),
            factor=defaults.get("factor", 1),
            # Task
            task_name="long_term_forecast",
            output_attention=False,
            # Embedding (freq from model config, default 'd' for daily → 3 time features)
            embed="timeF",
            freq=defaults.get("freq", "d"),
            # Common extras
            top_k=defaults.get("top_k", 5),
            num_kernels=defaults.get("num_kernels", 6),
            moving_avg=25,
            distil=True,
            # Universal attributes needed by many models
            num_class=0,  # not used in forecasting but many models reference it
            n_vars=enc_in,  # DeformableTST RevIN needs n_vars=enc_in
            num_nodes=enc_in,  # PathFormer RevIN needs num_features=num_nodes=enc_in
            batch_size=self._batch_size,  # WPMixer needs this
            device=self._device,  # WPMixer needs torch.device (DWT_Decomposition calls .type)
            use_amp=False,  # WPMixer needs this
            patch_len=patch_len,  # TimeFilter, PAttn, WPMixer (adjusted above)
        )

        # Copy all model-specific params from defaults
        for k, v in defaults.items():
            if k not in ("venue",) and not hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

    # Models whose top-level Model.forward() only accepts encoder input (x_enc)
    # and would raise TypeError if called with the standard 4-arg TSLib interface.
    # Determined by inspecting vendor source: each model's Model.forward() signature.
    _ENCODER_ONLY_MODELS = frozenset({
        "DeformableTST",  # forward(x)
        "Fredformer",     # forward(x)
        "ModernTCN",      # forward(x, te=None)
        "PDF",            # forward(x)
        "PathFormer",     # forward(x)
        "SparseTSF",      # forward(x)
        "TimeRecipe",     # forward(x_enc)
        "xPatch",         # forward(x)
    })

    def _load_tslib_model(self, configs: SimpleNamespace) -> nn.Module:
        """Dynamically load TSLib model class and instantiate it."""
        _ensure_tslib_path()

        model_file = _MODEL_FILE_MAP.get(self._tslib_model_name, self._tslib_model_name)

        try:
            module = importlib.import_module(f"models.{model_file}")
            # All TSLib models export a class named "Model"
            model_cls = getattr(module, "Model")
            model = model_cls(configs)
            return model.to(self._device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TSLib model '{model_file}': {e}\n"
                f"Ensure TSLib is at {TSLIB_ROOT}"
            ) from e

    def _is_encoder_only(self) -> bool:
        """Check if loaded model uses encoder-only forward(x) interface."""
        return self._tslib_model_name in self._ENCODER_ONLY_MODELS

    def _forward_model(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        dec_inp: torch.Tensor,
        dec_mark: torch.Tensor,
    ) -> torch.Tensor:
        """Unified forward call respecting model-specific interface.

        Standard TSLib models: model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        Encoder-only models:   model(x_enc)
        """
        if self._is_encoder_only():
            return self._model(x_enc)
        return self._model(x_enc, x_mark_enc, dec_inp, dec_mark)

    def _create_windows(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seq_len: int,
        pred_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sliding windows from panel DataFrame.

        Returns:
            x_enc: (N_windows, seq_len, n_features)
            y_target: (N_windows, pred_len)
        """
        # Convert target to numpy
        y_arr = y.values.astype(np.float32)

        # Get numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_arr = X[numeric_cols].values.astype(np.float32)
        n_features = X_arr.shape[1]

        # Simple global normalization for target
        self._target_mean = float(np.nanmean(y_arr))
        self._target_std = float(np.nanstd(y_arr))
        if self._target_std < 1e-8:
            self._target_std = 1.0
        y_norm = (y_arr - self._target_mean) / self._target_std

        # Normalize features
        feat_mean = np.nanmean(X_arr, axis=0, keepdims=True)
        feat_std = np.nanstd(X_arr, axis=0, keepdims=True)
        feat_std[feat_std < 1e-8] = 1.0
        X_norm = (X_arr - feat_mean) / feat_std

        # Handle NaN
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        y_norm = np.nan_to_num(y_norm, nan=0.0)

        # Create sliding windows per entity if entity_id exists
        total_len = seq_len + pred_len
        windows_x = []
        windows_y = []

        if "entity_id" in X.columns:
            for eid, grp in X.groupby("entity_id"):
                idx = grp.index
                n = len(idx)
                if n < total_len:
                    continue
                positions = [X.index.get_loc(i) for i in idx]
                for start in range(0, n - total_len + 1, max(1, pred_len // 2)):
                    end_x = start + seq_len
                    end_y = end_x + pred_len
                    x_window = X_norm[positions[start]:positions[start] + seq_len]
                    y_window = y_norm[positions[start] + seq_len:positions[start] + total_len]
                    if x_window.shape[0] == seq_len and y_window.shape[0] == pred_len:
                        windows_x.append(x_window)
                        windows_y.append(y_window)
        else:
            # Flat array: create sequential windows
            n = len(y_arr)
            for start in range(0, n - total_len + 1, max(1, pred_len // 2)):
                windows_x.append(X_norm[start:start + seq_len])
                windows_y.append(y_norm[start + seq_len:start + total_len])

        if not windows_x:
            # Fallback: pad a single window
            padded_x = np.zeros((seq_len, n_features), dtype=np.float32)
            padded_y = np.zeros(pred_len, dtype=np.float32)
            fill = min(len(X_norm), seq_len)
            padded_x[-fill:] = X_norm[:fill]
            windows_x.append(padded_x)
            windows_y.append(padded_y)

        x_enc = torch.tensor(np.array(windows_x), dtype=torch.float32)
        y_target = torch.tensor(np.array(windows_y), dtype=torch.float32)

        logger.info(
            f"[TSLib-{self._tslib_model_name}] Created {len(windows_x)} windows "
            f"(seq={seq_len}, pred={pred_len}, feats={n_features})"
        )
        return x_enc, y_target

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TSLibModelWrapper":
        """Train TSLib model with early stopping."""
        horizon = int(kwargs.get("horizon", 7))
        seq_len = 60  # Fixed context length matching our benchmark
        pred_len = max(horizon, 7)  # Minimum 7 for stable training

        # Count numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        enc_in = len(numeric_cols)

        # Save training normalization statistics for predict-time reuse
        # (prevents test-time data leakage)
        X_arr = X[numeric_cols].values.astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        self._train_feat_mean = np.mean(X_arr, axis=0, keepdims=True)
        self._train_feat_std = np.std(X_arr, axis=0, keepdims=True)
        self._train_feat_std[self._train_feat_std < 1e-8] = 1.0

        # Build TSLib configs (may adjust pred_len for seg_len alignment)
        configs = self._build_configs(seq_len, pred_len, enc_in)
        self._configs = configs
        self._seq_len = seq_len
        # Use configs.pred_len which may have been aligned to seg_len
        pred_len = configs.pred_len
        self._pred_len = pred_len
        self._n_features = enc_in

        # Load model
        self._model = self._load_tslib_model(configs)
        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(
            f"[TSLib-{self._tslib_model_name}] Model loaded: "
            f"{n_params:,} parameters, device={self._device}"
        )

        # Create training windows
        x_enc, y_target = self._create_windows(X, y, seq_len, pred_len)

        if x_enc.shape[0] < 4:
            logger.warning(
                f"[TSLib-{self._tslib_model_name}] Only {x_enc.shape[0]} windows, "
                f"skipping training"
            )
            self._fitted = True
            return self

        # Train/val split (temporal: last 20%)
        n = x_enc.shape[0]
        val_size = max(4, int(n * 0.2))
        x_train, x_val = x_enc[:-val_size], x_enc[-val_size:]
        y_train, y_val = y_target[:-val_size], y_target[-val_size:]

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        train_loader = DataLoader(
            train_ds, batch_size=self._batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self._batch_size, shuffle=False
        )

        # Optimizer + scheduler
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._max_epochs
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        t0 = time.monotonic()

        # Time feature dimension must match TSLib's freq map (freq='d' → 3)
        time_feat_dim = self._FREQ_FEAT_DIM.get(configs.freq, 4)

        for epoch in range(self._max_epochs):
            # Train
            self._model.train()
            train_loss = 0.0
            n_train = 0
            for bx, by in train_loader:
                bx = bx.to(self._device)
                by = by.to(self._device)

                # TSLib forward: (x_enc, x_mark_enc, x_dec, x_mark_dec)
                # Time marks dimension must match freq (daily=3)
                x_mark = torch.zeros(bx.shape[0], bx.shape[1], time_feat_dim, device=self._device)
                # Decoder input: last label_len of encoder + zeros for prediction
                dec_inp = torch.zeros(
                    bx.shape[0], configs.label_len + pred_len, bx.shape[2],
                    device=self._device,
                )
                dec_inp[:, :configs.label_len, :] = bx[:, -configs.label_len:, :]
                dec_mark = torch.zeros(
                    bx.shape[0], configs.label_len + pred_len, time_feat_dim,
                    device=self._device,
                )

                try:
                    out = self._forward_model(bx, x_mark, dec_inp, dec_mark)
                    # Output shape varies by model, but we want (B, pred_len) for target
                    if isinstance(out, tuple):
                        # Most models: (prediction, attn) → take [0]
                        # SEMPO: ([pretrain_heads], prediction) → take [-1]
                        out = out[-1] if isinstance(out[0], list) else out[0]
                    # Take the last pred_len time steps, first feature (target)
                    if out.dim() == 3:
                        out = out[:, -pred_len:, 0]
                    elif out.dim() == 2:
                        out = out[:, -pred_len:]

                    loss = criterion(out, by)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item() * bx.shape[0]
                    n_train += bx.shape[0]
                except Exception as e:
                    logger.warning(f"[TSLib-{self._tslib_model_name}] Train batch error: {e}")
                    continue

            scheduler.step()

            # Validation
            self._model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self._device)
                    by = by.to(self._device)

                    x_mark = torch.zeros(bx.shape[0], bx.shape[1], time_feat_dim, device=self._device)
                    dec_inp = torch.zeros(
                        bx.shape[0], configs.label_len + pred_len, bx.shape[2],
                        device=self._device,
                    )
                    dec_inp[:, :configs.label_len, :] = bx[:, -configs.label_len:, :]
                    dec_mark = torch.zeros(
                        bx.shape[0], configs.label_len + pred_len, time_feat_dim,
                        device=self._device,
                    )

                    try:
                        out = self._forward_model(bx, x_mark, dec_inp, dec_mark)
                        if isinstance(out, tuple):
                            out = out[-1] if isinstance(out[0], list) else out[0]
                        if out.dim() == 3:
                            out = out[:, -pred_len:, 0]
                        elif out.dim() == 2:
                            out = out[:, -pred_len:]

                        loss = criterion(out, by)
                        val_loss += loss.item() * bx.shape[0]
                        n_val += bx.shape[0]
                    except Exception:
                        continue

            avg_train = train_loss / max(n_train, 1)
            avg_val = val_loss / max(n_val, 1)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                logger.info(
                    f"[TSLib-{self._tslib_model_name}] Epoch {epoch}/{self._max_epochs} "
                    f"train_loss={avg_train:.6f} val_loss={avg_val:.6f} "
                    f"best={best_val_loss:.6f} patience={patience_counter}/{self._patience}"
                )

            if patience_counter >= self._patience:
                logger.info(
                    f"[TSLib-{self._tslib_model_name}] Early stopping at epoch {epoch}"
                )
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        elapsed = time.monotonic() - t0
        logger.info(
            f"[TSLib-{self._tslib_model_name}] Training complete in {elapsed:.1f}s, "
            f"best_val_loss={best_val_loss:.6f}, "
            f"epochs={min(epoch + 1, self._max_epochs)}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Save per-entity context windows for per-entity predict() ----
        self._entity_contexts: dict = {}
        if "entity_id" in X.columns:
            X_arr_ectx = X[numeric_cols].values.astype(np.float32)
            X_arr_ectx = np.nan_to_num(X_arr_ectx, nan=0.0)
            X_ectx_norm = (X_arr_ectx - self._train_feat_mean) / self._train_feat_std

            for eid, grp in X.groupby("entity_id"):
                positions = [X.index.get_loc(i) for i in grp.index]
                entity_feats = X_ectx_norm[positions]
                if len(entity_feats) >= seq_len:
                    ctx = entity_feats[-seq_len:]
                else:
                    ctx = np.zeros((seq_len, enc_in), dtype=np.float32)
                    ctx[-len(entity_feats):] = entity_feats
                self._entity_contexts[eid] = torch.tensor(ctx, dtype=torch.float32)

            logger.info(
                f"[TSLib-{self._tslib_model_name}] Saved {len(self._entity_contexts)} "
                f"entity contexts for per-entity inference"
            )

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate per-entity predictions using trained TSLib model.

        When entity contexts are available (saved during fit), runs batched
        forward passes per-entity to produce unique predictions for each
        entity — eliminating the single-window broadcast bug.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError(f"TSLib model {self._tslib_model_name} not fitted")

        horizon = int(kwargs.get("horizon", self._pred_len))
        h = len(X)
        test_raw = kwargs.get("test_raw")

        # ---- PER-ENTITY PREDICTION (correct panel inference) ----
        if (test_raw is not None
                and "entity_id" in test_raw.columns
                and getattr(self, "_entity_contexts", None)):
            target = kwargs.get("target")
            if target and target in test_raw.columns:
                valid_mask = test_raw[target].notna()
                test_entities = test_raw.loc[valid_mask, "entity_id"].values
            else:
                test_entities = test_raw["entity_id"].values

            if len(test_entities) == h:
                return self._predict_per_entity(test_entities, h, horizon)

        # ---- FALLBACK: single-window prediction (legacy) ----
        return self._predict_single_window(X, h, horizon)

    def _predict_per_entity(
        self, test_entities: np.ndarray, h: int, horizon: int
    ) -> np.ndarray:
        """Batched per-entity forward pass through TSLib model."""
        unique_eids = list(dict.fromkeys(test_entities))  # preserve order, dedupe
        entity_preds: dict = {}

        # Collect entities with saved contexts
        batch_inputs = []
        batch_eids = []
        for eid in unique_eids:
            if eid in self._entity_contexts:
                batch_inputs.append(self._entity_contexts[eid])
                batch_eids.append(eid)

        time_feat_dim = self._FREQ_FEAT_DIM.get(self._configs.freq, 4)
        sub_batch = min(256, max(1, len(batch_inputs)))

        if batch_inputs:
            self._model.eval()
            with torch.no_grad():
                for bs in range(0, len(batch_inputs), sub_batch):
                    be = min(bs + sub_batch, len(batch_inputs))
                    x_batch = torch.stack(batch_inputs[bs:be]).to(self._device)
                    B = x_batch.shape[0]

                    x_mark = torch.zeros(
                        B, self._seq_len, time_feat_dim, device=self._device
                    )
                    dec_inp = torch.zeros(
                        B, self._configs.label_len + self._pred_len,
                        x_batch.shape[2], device=self._device,
                    )
                    dec_inp[:, :self._configs.label_len, :] = \
                        x_batch[:, -self._configs.label_len:, :]
                    dec_mark = torch.zeros(
                        B, self._configs.label_len + self._pred_len,
                        time_feat_dim, device=self._device,
                    )

                    try:
                        out = self._forward_model(x_batch, x_mark, dec_inp, dec_mark)
                        if isinstance(out, tuple):
                            out = out[-1] if isinstance(out[0], list) else out[0]
                        if out.dim() == 3:
                            preds = out[:, -self._pred_len:, 0].cpu().numpy()
                        elif out.dim() == 2:
                            preds = out[:, -self._pred_len:].cpu().numpy()
                        else:
                            preds = out.cpu().numpy().reshape(B, -1)

                        # Denormalize
                        preds = preds * self._target_std + self._target_mean

                        # Use horizon-step-ahead prediction (or mean if horizon > pred_len)
                        for i, eid in enumerate(batch_eids[bs:be]):
                            pred_vec = preds[i]
                            if horizon <= len(pred_vec):
                                entity_preds[eid] = float(pred_vec[horizon - 1])
                            else:
                                entity_preds[eid] = float(np.mean(pred_vec))

                    except Exception as e:
                        logger.warning(
                            f"[TSLib-{self._tslib_model_name}] Batch predict error: {e}"
                        )
                        for eid in batch_eids[bs:be]:
                            entity_preds[eid] = self._target_mean

        # Map predictions to test rows
        fallback = self._target_mean
        y_pred = np.empty(h, dtype=np.float64)
        n_matched = 0
        for i, eid in enumerate(test_entities):
            if eid in entity_preds:
                y_pred[i] = entity_preds[eid]
                n_matched += 1
            else:
                y_pred[i] = fallback

        logger.info(
            f"[TSLib-{self._tslib_model_name}] Per-entity predict: "
            f"{len(entity_preds)}/{len(set(test_entities))} entities matched, "
            f"{n_matched}/{h} rows covered, "
            f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
        )
        return y_pred

    def _predict_single_window(
        self, X: pd.DataFrame, h: int, horizon: int
    ) -> np.ndarray:
        """Fallback: single-window prediction (used when no entity contexts)."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_arr = X[numeric_cols].values.astype(np.float32)

        # Use TRAINING normalization statistics (saved during fit) to avoid
        # test-time data leakage. Fall back to test stats if unavailable.
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        if self._train_feat_mean is not None and self._train_feat_std is not None:
            feat_mean = self._train_feat_mean
            feat_std = self._train_feat_std
        else:
            feat_mean = np.mean(X_arr, axis=0, keepdims=True)
            feat_std = np.std(X_arr, axis=0, keepdims=True)
            feat_std[feat_std < 1e-8] = 1.0
        X_norm = (X_arr - feat_mean) / feat_std

        # Pad or truncate to seq_len
        seq_len = self._seq_len
        if len(X_norm) >= seq_len:
            x_input = X_norm[-seq_len:]
        else:
            x_input = np.zeros((seq_len, X_norm.shape[1]), dtype=np.float32)
            x_input[-len(X_norm):] = X_norm

        # Convert to tensor
        x_enc = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Time feature dimension must match freq (daily=3)
        time_feat_dim = self._FREQ_FEAT_DIM.get(self._configs.freq, 4)
        x_mark = torch.zeros(1, seq_len, time_feat_dim, device=self._device)

        configs = self._configs
        dec_inp = torch.zeros(
            1, configs.label_len + self._pred_len, x_enc.shape[2],
            device=self._device,
        )
        dec_inp[:, :configs.label_len, :] = x_enc[:, -configs.label_len:, :]
        dec_mark = torch.zeros(
            1, configs.label_len + self._pred_len, time_feat_dim,
            device=self._device,
        )

        self._model.eval()
        with torch.no_grad():
            try:
                out = self._forward_model(x_enc, x_mark, dec_inp, dec_mark)
                if isinstance(out, tuple):
                    out = out[-1] if isinstance(out[0], list) else out[0]
                if out.dim() == 3:
                    preds = out[0, -self._pred_len:, 0].cpu().numpy()
                elif out.dim() == 2:
                    preds = out[0, -self._pred_len:].cpu().numpy()
                else:
                    preds = out.cpu().numpy().flatten()[:self._pred_len]
            except Exception as e:
                logger.warning(
                    f"[TSLib-{self._tslib_model_name}] Predict failed: {e}, "
                    f"returning mean"
                )
                preds = np.full(self._pred_len, self._target_mean)
                return preds[:h] if h < len(preds) else np.pad(
                    preds, (0, h - len(preds)), constant_values=self._target_mean
                )

        # Denormalize
        preds = preds * self._target_std + self._target_mean

        # Match output length to request
        if h <= len(preds):
            # Take last `h` predictions (e.g., horizon=1, pred_len=7 → take [-1])
            return preds[-h:].astype(np.float64)
        else:
            # Pad with last value
            result = np.full(h, preds[-1] if len(preds) > 0 else self._target_mean)
            result[:len(preds)] = preds
            return result.astype(np.float64)


# ============================================================================
# Factory functions — one per model
# ============================================================================

def _tslib_factory(tslib_name: str, display_name: Optional[str] = None):
    """Create a factory function for a TSLib model."""
    name = display_name or tslib_name

    def create(**kwargs):
        config = ModelConfig(
            name=name,
            model_type="forecasting",
            params={"tslib_model": tslib_name, **kwargs},
            optional_dependency="tslib",
        )
        return TSLibModelWrapper(config, tslib_name, **kwargs)

    create.__doc__ = f"TSLib {name} model ({TSLIB_CONFIGS.get(tslib_name, {}).get('venue', 'unknown')})"
    return create


# 2025
create_timefilter = _tslib_factory("TimeFilter")
create_wpmixer = _tslib_factory("WPMixer")
create_multipatchformer = _tslib_factory("MultiPatchFormer")

# 2024
create_msgnet = _tslib_factory("MSGNet")
create_pattn = _tslib_factory("PAttn")
create_mambasimple = _tslib_factory("MambaSimple")

# 2023
create_koopa = _tslib_factory("Koopa")
create_frets = _tslib_factory("FreTS")
create_crossformer = _tslib_factory("Crossformer")
create_micn = _tslib_factory("MICN")
create_segrnn = _tslib_factory("SegRNN")

# 2022
create_nonstationary_transformer = _tslib_factory(
    "Nonstationary_Transformer", "NonstationaryTransformer"
)
create_film = _tslib_factory("FiLM")
create_scinet = _tslib_factory("SCINet")

# Phase 9 additions
create_etsformer = _tslib_factory("ETSformer")
create_lightts = _tslib_factory("LightTS")
create_pyraformer = _tslib_factory("Pyraformer")
create_reformer = _tslib_factory("Reformer")
create_tirex = _tslib_factory("TiRex")
create_mamba = _tslib_factory("Mamba")

# Phase 10
create_kanad = _tslib_factory("KANAD")
create_fits = _tslib_factory("FITS")
create_sparsetsf = _tslib_factory("SparseTSF")
create_cats = _tslib_factory("CATS")
create_fredformer = _tslib_factory("Fredformer")
create_cyclenet = _tslib_factory("CycleNet")
create_xpatch = _tslib_factory("xPatch")
create_filterts = _tslib_factory("FilterTS")

# Phase 11: Section C SOTA models
create_cfpt = _tslib_factory("CFPT")
create_deformabletst = _tslib_factory("DeformableTST")
create_moderntcn = _tslib_factory("ModernTCN")
create_pathformer = _tslib_factory("PathFormer")
create_sempo = _tslib_factory("SEMPO")
create_timeperceiver = _tslib_factory("TimePerceiver")
create_timebridge = _tslib_factory("TimeBridge")
create_tqnet = _tslib_factory("TQNet")
create_pir = _tslib_factory("PIR")
create_card = _tslib_factory("CARD")
create_pdf = _tslib_factory("PDF")
create_timerecipe = _tslib_factory("TimeRecipe")
create_duet = _tslib_factory("DUET")
create_srsnet = _tslib_factory("SRSNet")


# ============================================================================
# Registry (imported by registry.py)
# ============================================================================

TSLIB_MODELS = {
    # 2025
    "TimeFilter": create_timefilter,
    "WPMixer": create_wpmixer,
    "MultiPatchFormer": create_multipatchformer,
    "TiRex": create_tirex,
    # 2024
    "MSGNet": create_msgnet,
    "PAttn": create_pattn,
    "MambaSimple": create_mambasimple,
    "Mamba": create_mamba,
    # 2023
    "Koopa": create_koopa,
    "FreTS": create_frets,
    "Crossformer": create_crossformer,
    "MICN": create_micn,
    "SegRNN": create_segrnn,
    "ETSformer": create_etsformer,
    # 2022
    "NonstationaryTransformer": create_nonstationary_transformer,
    "FiLM": create_film,
    "SCINet": create_scinet,
    "LightTS": create_lightts,
    "Pyraformer": create_pyraformer,
    "Reformer": create_reformer,
    # Phase 10
    "KANAD": create_kanad,
    "FITS": create_fits,
    "SparseTSF": create_sparsetsf,
    "CATS": create_cats,
    "Fredformer": create_fredformer,
    "CycleNet": create_cyclenet,
    "xPatch": create_xpatch,
    "FilterTS": create_filterts,
    # Phase 11: Section C SOTA models
    "CFPT": create_cfpt,
    "DeformableTST": create_deformabletst,
    "ModernTCN": create_moderntcn,
    "PathFormer": create_pathformer,
    "SEMPO": create_sempo,
    "TimePerceiver": create_timeperceiver,
    "TimeBridge": create_timebridge,
    "TQNet": create_tqnet,
    "PIR": create_pir,
    "CARD": create_card,
    "PDF": create_pdf,
    "TimeRecipe": create_timerecipe,
    "DUET": create_duet,
    "SRSNet": create_srsnet,
}


def list_tslib_models() -> List[str]:
    """Return list of available TSLib model names."""
    return sorted(TSLIB_MODELS.keys())
