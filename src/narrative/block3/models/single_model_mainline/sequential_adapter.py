#!/usr/bin/env python3
"""Sequential ED-SSM + MoE Trunk Adapter for panel data.

Converts flat panel data (entity × date × features) into entity-level
sliding-window sequences, trains the EventDrivenSSM → SparseMoETrunk
pipeline end-to-end with auxiliary task supervision, then returns
per-row learned embeddings.

Drop-in replacement for ``LearnableTrunkAdapter`` with temporal awareness.

Usage::

    adapter = SequentialTrunkAdapter(window_size=30, n_epochs=20, device="cuda")
    embeddings = adapter.fit_transform(
        X, y, target_name="is_funded",
        entity_ids=entity_ids, dates=dates,
    )
    # embeddings: (N, expert_dim) numpy array
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .causal_decoders import build_causal_decoder
from .event_driven_ssm import SequentialMoETrunk

_LOG = logging.getLogger("sequential_adapter")

_TARGET_TASK_MAP = {
    "is_funded": 0,
    "funding_raised_usd": 1,
    "investors_count": 2,
}
_BINARY_TARGETS = {"is_funded"}


# ─────────────────────────────────────────────────────────────────────
#  Dataset: entity-level sliding windows
# ─────────────────────────────────────────────────────────────────────


class EntitySequenceDataset(Dataset):
    """Converts panel data into entity-level sliding window sequences.

    For each row *i* belonging to entity *e*, the sequence is the
    lookback window of size ``window_size`` ending at row *i* (within
    entity *e*'s sorted timeline).  Shorter sequences are left-padded.

    Parameters
    ----------
    features : np.ndarray
        (N, D) feature matrix — rows aligned with entity_ids / dates.
    targets : np.ndarray
        (N,) target vector.
    entity_ids : np.ndarray or None
        (N,) entity identifiers.  If None, each row is an independent
        single-step sequence.
    dates : np.ndarray or None
        (N,) dates for sorting within entity.  If None, row order is
        preserved as the temporal axis.
    window_size : int
        Maximum lookback length.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        entity_ids: np.ndarray | None,
        dates: np.ndarray | None,
        window_size: int = 30,
    ):
        N, D = features.shape
        self.window_size = window_size
        self.n_features = D

        # Sort by (entity, date) to build contiguous entity blocks
        if entity_ids is not None:
            sort_keys = np.lexsort(
                (dates if dates is not None else np.arange(N), entity_ids)
            )
        else:
            sort_keys = np.arange(N)

        self._features = np.ascontiguousarray(features[sort_keys].astype(np.float32))
        self._targets = targets[sort_keys].astype(np.float32)

        # Map sorted indices back to original row order
        self._sorted_to_orig = sort_keys.copy()

        # Build entity boundary array: entity_start[i] = first index
        # of the entity that row i belongs to (in sorted order).
        if entity_ids is not None:
            sorted_eids = entity_ids[sort_keys]
            self._entity_start = np.empty(N, dtype=np.int64)
            prev_eid = sorted_eids[0]
            block_start = 0
            for i in range(N):
                if sorted_eids[i] != prev_eid:
                    block_start = i
                    prev_eid = sorted_eids[i]
                self._entity_start[i] = block_start
        else:
            # Each row is its own "entity" → window = 1
            self._entity_start = np.arange(N, dtype=np.int64)

        self._N = N

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int):
        es = int(self._entity_start[idx])
        win_start = max(es, idx - self.window_size + 1)
        actual_len = idx - win_start + 1

        window = self._features[win_start : idx + 1]  # (actual_len, D)

        if actual_len < self.window_size:
            pad = np.zeros(
                (self.window_size - actual_len, self.n_features),
                dtype=np.float32,
            )
            window = np.concatenate([pad, window], axis=0)

        return (
            torch.from_numpy(window.copy()),           # (L, D)
            torch.tensor(actual_len, dtype=torch.long),  # scalar
            torch.tensor(self._targets[idx]),            # scalar
        )

    @property
    def orig_order(self) -> np.ndarray:
        """Mapping from sorted position → original row index."""
        return self._sorted_to_orig


# ─────────────────────────────────────────────────────────────────────
#  Adapter: fit / transform / describe
# ─────────────────────────────────────────────────────────────────────


class SequentialTrunkAdapter:
    """Sequential ED-SSM + MoE adapter for panel time-series data.

    Trains end-to-end with auxiliary task supervision:
      loss = task_loss + α · load_balance + β · diversity

    Parameters
    ----------
    window_size : int
        Lookback window length per entity.
    d_model, d_state, n_ssm_layers, d_jump
        EventDrivenSSM hyperparameters.
    compact_dim, n_experts, expert_dim, top_k
        SparseMoETrunk hyperparameters.
    lr, weight_decay : float
        AdamW optimiser settings.
    n_epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    max_train_sequences : int
        Cap on training sequences (subsample if exceeded).
    random_state : int
        Seed for reproducibility.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"cuda:0"`` etc.
    """

    def __init__(
        self,
        window_size: int = 30,
        d_model: int = 64,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        d_event: int = 0,
        d_jump: int = 64,
        compact_dim: int = 64,
        n_experts: int = 6,
        expert_dim: int = 32,
        top_k: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 512,
        max_train_sequences: int = 100_000,
        random_state: int = 42,
        device: str = "auto",
        decoder_branch: str = "legacy",
        freeze_unified_ssm: bool = False,
    ):
        self.window_size = window_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_ssm_layers = n_ssm_layers
        self.d_event = d_event
        self.d_jump = d_jump
        self.compact_dim = compact_dim
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_train_sequences = max_train_sequences
        self.random_state = random_state
        self.decoder_branch = str(decoder_branch)
        self.freeze_unified_ssm = bool(freeze_unified_ssm)

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._trunk: SequentialMoETrunk | None = None
        self._aux_head: nn.Linear | None = None
        self._causal_decoder: nn.Module | None = None
        self._fitted = False
        self._d_features: int = 0
        self._is_binary: bool = False
        self._train_loss_history: list[float] = []

    @staticmethod
    def _grad_sum(module: nn.Module, attr_name: str) -> float:
        layer = getattr(module, attr_name, None)
        if layer is None:
            return 0.0
        gsum = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                gsum += float(torch.sum(torch.abs(p.grad)).detach().cpu().item())
        return gsum

    def _assert_decoder_grad_heartbeat(self) -> None:
        """Fail fast if decoder head gradients are dead on the first batch."""
        if self._causal_decoder is None:
            return

        branch = self.decoder_branch.strip().lower()
        if branch in {"alpha", "icm_lognormal", "lognormal"}:
            gsum = self._grad_sum(self._causal_decoder, "head_f")
            if gsum <= 0.0:
                raise RuntimeError(
                    "FATAL: Funding head gradient is zero on first batch (alpha/head_f)."
                )
        elif branch in {"beta", "icm_iqn", "iqn"}:
            gsum = self._grad_sum(self._causal_decoder, "q_head")
            if gsum <= 0.0:
                raise RuntimeError(
                    "FATAL: Funding head gradient is zero on first batch (beta/q_head)."
                )
        elif branch in {"gamma", "icm_cfm", "cfm"}:
            gsum = self._grad_sum(self._causal_decoder, "vfield")
            if gsum <= 0.0:
                raise RuntimeError(
                    "FATAL: Funding head gradient is zero on first batch (gamma/vfield)."
                )

    # ── public API ───────────────────────────────────────────────

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str,
        entity_ids: np.ndarray | None = None,
        dates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Train ED-SSM+MoE on entity sequences, return embeddings.

        Args:
            X: (N, d_features) feature matrix.
            y: (N,) target vector.
            target_name: target column name (for task routing).
            entity_ids: (N,) entity identifiers for sequence grouping.
            dates: (N,) date values for temporal sorting.

        Returns:
            (N, expert_dim) learned embeddings aligned to input rows.
        """
        self._d_features = X.shape[1]
        self._is_binary = target_name in _BINARY_TARGETS
        task_idx = _TARGET_TASK_MAP.get(target_name, 0)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build dataset
        ds = EntitySequenceDataset(
            X, y, entity_ids, dates, window_size=self.window_size,
        )

        # Subsample if too large
        if len(ds) > self.max_train_sequences:
            rng = np.random.default_rng(self.random_state)
            subset_idx = rng.choice(len(ds), self.max_train_sequences, replace=False)
            ds_train = torch.utils.data.Subset(ds, subset_idx.tolist())
        else:
            ds_train = ds

        loader = DataLoader(
            ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=(self._device.type == "cuda"),
        )

        # Build model
        self._trunk = SequentialMoETrunk(
            d_cont=self._d_features,
            d_event=self.d_event,
            d_model=self.d_model,
            d_state=self.d_state,
            n_ssm_layers=self.n_ssm_layers,
            d_jump=self.d_jump,
            compact_dim=self.compact_dim,
            n_experts=self.n_experts,
            expert_dim=self.expert_dim,
            top_k=self.top_k,
        ).to(self._device)

        self._aux_head = nn.Linear(self.expert_dim, 1).to(self._device)
        nn.init.kaiming_normal_(self._aux_head.weight, nonlinearity="relu")
        nn.init.zeros_(self._aux_head.bias)

        # Optional causal probabilistic decoder branches for heavy-tail targets.
        if not self._is_binary and self.decoder_branch.lower() != "legacy":
            self._causal_decoder = build_causal_decoder(self.decoder_branch, self.expert_dim).to(self._device)
        else:
            self._causal_decoder = None

        # Freeze the underlying UnifiedJumpDiffusionSSM surface to isolate decoder training.
        if self.freeze_unified_ssm:
            for p in self._trunk.ssm.parameters():
                p.requires_grad = False

        # Optimiser
        params = [p for p in self._trunk.parameters() if p.requires_grad]
        params += list(self._aux_head.parameters())
        if self._causal_decoder is not None:
            params += list(self._causal_decoder.parameters())
        optimiser = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay,
        )

        # Task id tensor (constant for all samples)
        task_onehot = torch.zeros(1, 3, device=self._device)
        task_onehot[0, task_idx] = 1.0

        # ── Training loop ────────────────────────────────────────
        self._trunk.train()
        self._train_loss_history.clear()

        t0 = time.time()
        grad_checked = False
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                x_seq, lengths, targets = batch
                B = x_seq.size(0)
                x_seq = x_seq.to(self._device)
                lengths = lengths.to(self._device)
                targets = targets.to(self._device)
                tid = task_onehot.expand(B, -1)

                out = self._trunk(x_seq, tid, lengths=lengths)
                emb = out["output"]  # (B, expert_dim)

                # Auxiliary prediction
                logits = self._aux_head(emb).squeeze(-1)  # (B,)
                if self._is_binary:
                    task_loss = F.binary_cross_entropy_with_logits(logits, targets)
                elif self._causal_decoder is not None:
                    dec_out = self._causal_decoder.compute_loss(emb, targets)
                    task_loss = dec_out.total_loss
                else:
                    # Log1p-transform targets for scale-invariant MSE: prevents mean
                    # collapse on heavy-tailed distributions (funding_raised_usd,
                    # investors_count whose raw variance dwarfs the signal).
                    targets_log = torch.log1p(targets.clamp(min=0.0))
                    task_loss = F.mse_loss(logits, targets_log)

                # MoE auxiliary losses
                lb_loss = out.get("load_balance_loss", torch.tensor(0.0, device=self._device))
                div_loss = self._trunk.moe.expert_diversity_loss()

                loss = task_loss + lb_loss + 0.1 * div_loss

                optimiser.zero_grad()
                loss.backward()
                if (not grad_checked) and (self._causal_decoder is not None) and (not self._is_binary):
                    self._assert_decoder_grad_heartbeat()
                    grad_checked = True
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._train_loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                _LOG.info(
                    f"  Epoch {epoch+1}/{self.n_epochs}: "
                    f"loss={avg_loss:.4f} ({time.time()-t0:.1f}s)"
                )

        train_time = time.time() - t0
        _LOG.info(f"  Training done in {train_time:.1f}s")
        self._fitted = True

        # ── Extract embeddings for ALL rows ──────────────────────
        return self._extract_embeddings(ds, task_idx)

    def transform(
        self,
        X: np.ndarray,
        target_name: str,
        entity_ids: np.ndarray | None = None,
        dates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform new data using the trained model.

        Args & Returns: same as fit_transform but no training.
        """
        if not self._fitted or self._trunk is None:
            raise RuntimeError("SequentialTrunkAdapter is not fitted")

        task_idx = _TARGET_TASK_MAP.get(target_name, 0)
        y_dummy = np.zeros(len(X), dtype=np.float32)

        ds = EntitySequenceDataset(
            X, y_dummy, entity_ids, dates, window_size=self.window_size,
        )
        return self._extract_embeddings(ds, task_idx)

    def describe(self) -> Dict[str, Any]:
        """Return a description of the adapter state."""
        n_params = 0
        if self._trunk is not None:
            n_params = sum(p.numel() for p in self._trunk.parameters())
        if self._aux_head is not None:
            n_params += sum(p.numel() for p in self._aux_head.parameters())

        return {
            "type": "SequentialTrunkAdapter",
            "fitted": self._fitted,
            "window_size": self.window_size,
            "d_features": self._d_features,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "n_ssm_layers": self.n_ssm_layers,
            "n_experts": self.n_experts,
            "expert_dim": self.expert_dim,
            "top_k": self.top_k,
            "n_params": n_params,
            "device": str(self._device),
            "n_epochs": self.n_epochs,
            "decoder_branch": self.decoder_branch,
            "freeze_unified_ssm": self.freeze_unified_ssm,
            "final_loss": self._train_loss_history[-1] if self._train_loss_history else None,
        }

    # ── internals ────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_embeddings(
        self, ds: EntitySequenceDataset, task_idx: int,
    ) -> np.ndarray:
        """Forward-pass all sequences and return (N, expert_dim) in original row order."""
        self._trunk.eval()

        task_onehot = torch.zeros(1, 3, device=self._device)
        task_onehot[0, task_idx] = 1.0

        loader = DataLoader(
            ds, batch_size=self.batch_size * 2,
            shuffle=False, num_workers=0, drop_last=False,
        )

        all_embs = []
        for batch in loader:
            x_seq, lengths, _ = batch
            B = x_seq.size(0)
            x_seq = x_seq.to(self._device)
            lengths = lengths.to(self._device)
            tid = task_onehot.expand(B, -1)

            out = self._trunk(x_seq, tid, lengths=lengths)
            all_embs.append(out["output"].cpu().numpy())

        # Concatenate in sorted order
        embs_sorted = np.concatenate(all_embs, axis=0)  # (N, expert_dim)

        # Restore original row order
        orig_order = ds.orig_order  # sorted_idx → orig_idx
        embs_orig = np.empty_like(embs_sorted)
        embs_orig[orig_order] = embs_sorted

        self._trunk.train()
        return embs_orig
