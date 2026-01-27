from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from narrative.models.local_registry import build_local_model


@dataclass
class SearchResult:
    candidate: Dict
    val_loss: float
    epochs: int
    checkpoint_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 32):
    def _to_tensor(arr):
        try:
            return torch.from_numpy(arr).float()
        except RuntimeError as exc:  # fallback for torch builds w/o numpy
            if "Numpy is not available" in str(exc):
                return torch.tensor(arr.tolist(), dtype=torch.float32)
            raise

    X_tensor = _to_tensor(X)
    y_tensor = _to_tensor(y)

    dataset = list(range(len(X)))
    n_total = len(dataset)
    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = max(1, n_total - n_train - n_val)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    def collate(batch):
        xs = torch.stack([X_tensor[i] for i in batch])
        ys = torch.stack([y_tensor[i] for i in batch])
        return xs, ys

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


def _train_candidate(
    candidate: Dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seq_len: int,
    epochs: int,
    device: str,
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
) -> float:
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
) -> float:
    _set_seed(seed)
    model = build_local_model(
        name=candidate["backbone"],
        seq_len=seq_len,
        pred_len=1,
        enc_in=enc_in,
        module_flags=candidate.get("module_flags"),
        fusion_type=candidate.get("fusion_type", "none"),
        edgar_dim=int(candidate.get("edgar_dim", enc_in)),
        **(candidate.get("model_cfg") or {}),
    ).to(device)
        **(candidate.get("model_cfg") or {}),
    ).to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(candidate.get("lr", 1e-3)))
    best_val = float("inf")
    no_improve = 0
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.dim() > 1:
                pred = pred.mean(dim=tuple(range(1, pred.dim())))
            loss = criterion(pred, yb.float())
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                if pred.dim() > 1:
                    pred = pred.mean(dim=tuple(range(1, pred.dim())))
                loss = criterion(pred, yb.float())
                val_loss += float(loss.item())
                n_val += 1
        current_val = val_loss / max(1, n_val)
        if current_val < best_val:
            best_val = current_val
            no_improve = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if early_stopping_patience is not None and no_improve >= early_stopping_patience:
                break

    return best_val

def successive_halving(
    candidates: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    *,
    seq_len: int,
    enc_in: int,
    device: str = "cpu",
    budgets: Tuple[int, ...] = (2, 5),
    reduction_factor: int = 2,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    early_stopping_patience: Optional[int] = None,
) -> Tuple[List[SearchResult], Dict]:
    output_dir: Optional[Path] = None,
    early_stopping_patience: Optional[int] = None,
) -> Tuple[List[SearchResult], Dict]:
    active = list(candidates)
    all_results: List[SearchResult] = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        scored: List[SearchResult] = []
        for idx, cand in enumerate(active):
            ckpt = None
            if output_dir is not None:
                ckpt = output_dir / "checkpoints" / f"cand_{idx}_budget{budget}.pt"
                meta_path = output_dir / "checkpoints" / f"cand_{idx}_budget{budget}.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        scored: List[SearchResult] = []
        for idx, cand in enumerate(active):
            ckpt = None
            if output_dir is not None:
                ckpt = output_dir / "checkpoints" / f"cand_{idx}_budget{budget}.pt"
                meta_path = output_dir / "checkpoints" / f"cand_{idx}_budget{budget}.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(cand, indent=2), encoding="utf-8")

            val_loss = _train_candidate(
                cand,
                X,
                y,
                seq_len=seq_len,
                enc_in=enc_in,
                epochs=int(budget),
                device=device,
                seed=seed,
                early_stopping_patience=early_stopping_patience,
                checkpoint_path=ckpt,
            )
            scored.append(SearchResult(candidate=cand, val_loss=val_loss, epochs=int(budget), checkpoint_path=str(ckpt) if ckpt else None))
        scored.sort(key=lambda r: r.val_loss)
        all_results.extend(scored)
        keep = max(1, len(scored) // reduction_factor)
        active = [r.candidate for r in scored[:keep]]

    best = min(all_results, key=lambda r: r.val_loss)
    if output_dir is not None:
        best_path = output_dir / "best_config.yaml"
        best_path.write_text(yaml.safe_dump(best.candidate, sort_keys=False), encoding="utf-8")
    return all_results, best.candidate


def train_candidate(
    candidate: Dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seq_len: int,
    enc_in: int,
    epochs: int,
    device: str,
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
) -> float:
    return _train_candidate(
        candidate,
        X,
        y,
        seq_len=seq_len,
        enc_in=enc_in,
        epochs=epochs,
        device=device,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
    )
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
    )
