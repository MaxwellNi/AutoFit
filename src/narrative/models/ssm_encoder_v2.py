from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, causal: bool):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.conv = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            groups=self.channels,
            bias=True,
        )

    def _full_conv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        xt = x.transpose(1, 2)
        if self.causal:
            pad_left = self.kernel_size - 1
            xt = F.pad(xt, (pad_left, 0))
        else:
            if self.kernel_size % 2 == 0:
                raise ValueError("Non-causal conv requires odd kernel_size")
            pad = self.kernel_size // 2
            xt = F.pad(xt, (pad, pad))
        yt = self.conv(xt)
        return yt.transpose(1, 2)

    def _chunked_conv(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B, T, C = x.shape
        xt = x.transpose(1, 2)
        k = self.kernel_size
        outputs = []

        if self.causal:
            for start in range(0, T, chunk_size):
                end = min(T, start + chunk_size)
                ctx_start = max(0, start - (k - 1))
                xt_chunk = xt[:, :, ctx_start:end]
                pad_left = (k - 1) - (start - ctx_start)
                if pad_left > 0:
                    xt_chunk = F.pad(xt_chunk, (pad_left, 0))
                y = self.conv(xt_chunk)
                outputs.append(y)
        else:
            if k % 2 == 0:
                raise ValueError("Non-causal conv requires odd kernel_size")
            pad = k // 2
            for start in range(0, T, chunk_size):
                end = min(T, start + chunk_size)
                ctx_start = max(0, start - pad)
                ctx_end = min(T, end + pad)
                xt_chunk = xt[:, :, ctx_start:ctx_end]
                left_ctx = start - ctx_start
                right_ctx = ctx_end - end
                pad_left = pad - left_ctx
                pad_right = pad - right_ctx
                if pad_left > 0 or pad_right > 0:
                    xt_chunk = F.pad(xt_chunk, (pad_left, pad_right))
                y = self.conv(xt_chunk)
                outputs.append(y)

        yt = torch.cat(outputs, dim=2)
        return yt.transpose(1, 2)

    def forward(self, x: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        if chunk_size is None or chunk_size <= 0 or chunk_size >= x.shape[1]:
            return self._full_conv(x)
        return self._chunked_conv(x, int(chunk_size))


class SSMEncoderV2(nn.Module):
    """
    SSMEncoderV2: chunked depthwise conv + gated MLP blocks.
    Supports causal / non-causal modes and chunked processing.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        chunk_size: Optional[int] = None,
        causal: bool = True,
        return_sequence: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        self.chunk_size = chunk_size
        self.causal = bool(causal)
        self.return_sequence = bool(return_sequence)

        self.in_proj = nn.Linear(self.input_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [_DepthwiseConv1d(self.d_model, self.kernel_size, causal=self.causal) for _ in range(self.n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
        self.drop = nn.Dropout(float(dropout))
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GLU(dim=-1),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )
        self.out_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be [B,T,C]")
        h = self.in_proj(x)
        for conv, ln in zip(self.blocks, self.norms):
            y = conv(h, chunk_size=self.chunk_size)
            h = ln(h + self.drop(y))
            y2 = self.ff(h)
            h = ln(h + self.drop(y2))
        h = self.out_norm(h)

        if self.return_sequence:
            return h

        if attn_mask is None:
            return h.mean(dim=1)

        m = attn_mask.to(h.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (h * m).sum(dim=1) / denom


__all__ = ["SSMEncoderV2"]
