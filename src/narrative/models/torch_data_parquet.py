from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _as_numpy_emb(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, list) or isinstance(x, tuple):
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _infer_emb_dim(series: pd.Series) -> int:
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        arr = _as_numpy_emb(v)
        if arr.ndim == 1:
            return int(arr.shape[0])
    return 0


def _normalize_numeric(df: pd.DataFrame) -> np.ndarray:
    arr = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return arr


@dataclass
class ParquetOfferItem:
    static_x: np.ndarray
    seq_x: np.ndarray
    seq_time_delta: np.ndarray
    edgar_x: np.ndarray
    edgar_mask: np.ndarray
    text_emb: Optional[np.ndarray] = None
    nbi: Optional[np.ndarray] = None
    nci: Optional[np.ndarray] = None


class ParquetOffersDataset(Dataset):  # type: ignore[misc]
    """
    Multi-stream dataset:
      - static_x: per-offer static features
      - seq_x: per-offer snapshots sequence features
      - seq_time_delta: irregular deltas (days) between snapshots
      - edgar_x: EDGAR controls aligned to snapshots (forward-filled)
      - edgar_mask: missingness mask for edgar_x
      - optional text_emb / nbi / nci
    """

    def __init__(
        self,
        offers_static: pd.DataFrame,
        snapshots: pd.DataFrame,
        edgar_features: Optional[pd.DataFrame] = None,
        *,
        keys: Tuple[str, str] = ("platform_name", "offer_id"),
        time_col: str = "crawled_date",
        seq_cols: Optional[Sequence[str]] = None,
        static_cols: Optional[Sequence[str]] = None,
        edgar_mask_cols: Optional[Sequence[str]] = None,
        text_emb_col: Optional[str] = None,
        nbi_cols: Optional[Sequence[str]] = None,
        nci_cols: Optional[Sequence[str]] = None,
        max_snapshots: Optional[int] = None,
        use_first_k: bool = True,
    ):
        max_snapshots: Optional[int] = None,
        use_first_k: bool = True,
    ):
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for ParquetOffersDataset")

        self.keys = keys
        self.time_col = time_col
        self.text_emb_col = text_emb_col

        if not all(k in snapshots.columns for k in keys):
            raise KeyError(f"snapshots missing keys {keys}")
        if time_col not in snapshots.columns:
            raise KeyError(f"snapshots missing time_col {time_col}")

        snap = snapshots.copy()
        for k in keys:
            snap[k] = snap[k].astype(str)
        snap[time_col] = pd.to_datetime(snap[time_col], errors="coerce", utc=True)
        snap = snap.dropna(subset=[time_col])
        snap = snap.sort_values(list(keys) + [time_col], kind="mergesort")

        static_df = offers_static.copy()
        for k in keys:
            if k not in static_df.columns:
                static_df[k] = ""
            static_df[k] = static_df[k].astype(str)

        if static_cols is None:
            num = static_df.select_dtypes(include=["number", "bool"]).copy()
            drop = [c for c in keys if c in num.columns]
            static_cols = [c for c in num.columns if c not in drop]
        self.static_cols = list(static_cols or [])

        if edgar_features is not None:
            edgar_features = edgar_features.copy()
            for k in keys:
                if k in edgar_features.columns:
                    edgar_features[k] = edgar_features[k].astype(str)
            if time_col in edgar_features.columns:
                edgar_features[time_col] = pd.to_datetime(edgar_features[time_col], errors="coerce", utc=True)

            merge_cols = [c for c in list(keys) + [time_col] if c in edgar_features.columns]
            if merge_cols and all(c in snap.columns for c in merge_cols):
                snap = snap.merge(edgar_features, on=merge_cols, how="left", suffixes=("", "_edgar"))
            elif "cik" in edgar_features.columns and "cik" in snap.columns and time_col in edgar_features.columns:
                snap = snap.merge(edgar_features, on=["cik", time_col], how="left", suffixes=("", "_edgar"))
            else:
                raise KeyError("edgar_features must align on keys + time_col or cik + time_col")

        if seq_cols is None:
            num = snap.select_dtypes(include=["number", "bool"]).copy()
            drop = list(keys) + [time_col]
            if edgar_features is not None:
                drop += [c for c in num.columns if c.endswith("_is_missing")]
            if text_emb_col and text_emb_col in num.columns:
                drop.append(text_emb_col)
            seq_cols = [c for c in num.columns if c not in drop]
        self.seq_cols = list(seq_cols or [])

        if edgar_cols is None and edgar_features is not None:
            edgar_cols = [
                c
                for c in edgar_features.columns
                if c not in list(keys) + [time_col] and not c.endswith("_is_missing")
            ]
        self.edgar_cols = list(edgar_cols or [])

        if edgar_mask_cols is None and edgar_features is not None:
            edgar_mask_cols = [c for c in edgar_features.columns if c.endswith("_is_missing")]
        self.edgar_mask_cols = list(edgar_mask_cols or [])

        self.text_emb_dim = 0
        if text_emb_col and text_emb_col in snap.columns:
            self.text_emb_dim = _infer_emb_dim(snap[text_emb_col])

        self.nbi_cols = list(nbi_cols or [])
        self.nci_cols = list(nci_cols or [])

        # build static map
        static_map: Dict[Tuple[str, str], np.ndarray] = {}
        if self.static_cols:
            st = static_df[list(keys) + self.static_cols].copy()
            st[self.static_cols] = st[self.static_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for _, row in st.iterrows():
                key = (str(row[keys[0]]), str(row[keys[1]]))
                static_map[key] = row[self.static_cols].to_numpy(dtype=np.float32)

        nbi_map: Dict[Tuple[str, str], np.ndarray] = {}
        if self.nbi_cols:
            nb = static_df[list(keys) + self.nbi_cols].copy()
            nb[self.nbi_cols] = nb[self.nbi_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for _, row in nb.iterrows():
                key = (str(row[keys[0]]), str(row[keys[1]]))
                nbi_map[key] = row[self.nbi_cols].to_numpy(dtype=np.float32)

        nci_map: Dict[Tuple[str, str], np.ndarray] = {}
        if self.nci_cols:
            nc = static_df[list(keys) + self.nci_cols].copy()
            nc[self.nci_cols] = nc[self.nci_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for _, row in nc.iterrows():
                key = (str(row[keys[0]]), str(row[keys[1]]))
                nci_map[key] = row[self.nci_cols].to_numpy(dtype=np.float32)

        self._items: List[ParquetOfferItem] = []
        self._offer_keys: List[Tuple[str, str]] = []

        for key_vals, group in snap.groupby(list(keys), sort=False):
            key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            key = (str(key[0]), str(key[1]))
            if max_snapshots is not None:
                group = group.head(int(max_snapshots)) if use_first_k else group.tail(int(max_snapshots))
            if max_snapshots is not None:
                group = group.head(int(max_snapshots)) if use_first_k else group.tail(int(max_snapshots))

            seq_x = _normalize_numeric(group[self.seq_cols]) if self.seq_cols else np.zeros((len(group), 0), dtype=np.float32)

            times = pd.to_datetime(group[time_col], errors="coerce", utc=True)
            delta = times.diff().dt.total_seconds().fillna(0).to_numpy(dtype=np.float32) / 86400.0

            edgar_x = np.zeros((len(group), len(self.edgar_cols)), dtype=np.float32)
            if self.edgar_cols:
                edgar_x = _normalize_numeric(group[self.edgar_cols])

            if self.edgar_mask_cols:
                edgar_mask = group[self.edgar_mask_cols].fillna(True).to_numpy(dtype=bool)
            else:
                edgar_mask = np.isnan(edgar_x)

            text_emb = None
            if text_emb_col and text_emb_col in group.columns and self.text_emb_dim > 0:
                emb_list = []
                for v in group[text_emb_col].tolist():
                    arr = _as_numpy_emb(v)
                    if arr.shape[0] != self.text_emb_dim:
                        pad = np.zeros((self.text_emb_dim,), dtype=np.float32)
                        if arr.shape[0] > 0:
                            pad[: min(self.text_emb_dim, arr.shape[0])] = arr[: self.text_emb_dim]
                        arr = pad
                    emb_list.append(arr)
                text_emb = np.stack(emb_list, axis=0).astype(np.float32)

            static_x = static_map.get(key, np.zeros((len(self.static_cols),), dtype=np.float32))
            nbi = nbi_map.get(key) if self.nbi_cols else None
            nci = nci_map.get(key) if self.nci_cols else None

            self._items.append(
                ParquetOfferItem(
                    static_x=static_x,
                    seq_x=seq_x,
                    seq_time_delta=delta,
                    edgar_x=edgar_x,
                    edgar_mask=edgar_mask,
                    text_emb=text_emb,
                    nbi=nbi,
                    nci=nci,
                )
            )
            self._offer_keys.append(key)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        it = self._items[i]
        out: Dict[str, np.ndarray] = {
            "static_x": it.static_x,
            "seq_x": it.seq_x,
            "time_delta": it.seq_time_delta,
            "edgar_x": it.edgar_x,
            "edgar_mask": it.edgar_mask,
        }
        if it.text_emb is not None:
            out["text_emb"] = it.text_emb
        if it.nbi is not None:
            out["nbi"] = it.nbi
        if it.nci is not None:
            out["nci"] = it.nci
        return out


def collate_parquet_offers(batch: List[Dict[str, np.ndarray]], fixed_len: int) -> Dict[str, torch.Tensor]:
    """
    Pad/crop to fixed_len with masks and irregular time deltas.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    B = len(batch)
    seq_dim = int(batch[0]["seq_x"].shape[1]) if batch[0]["seq_x"].ndim == 2 else 0
    edgar_dim = int(batch[0]["edgar_x"].shape[1]) if batch[0]["edgar_x"].ndim == 2 else 0
    static_dim = int(batch[0]["static_x"].shape[0]) if batch[0]["static_x"].ndim == 1 else 0

    seq_x = torch.zeros((B, fixed_len, seq_dim), dtype=torch.float32)
    seq_mask = torch.zeros((B, fixed_len), dtype=torch.bool)
    time_delta = torch.zeros((B, fixed_len), dtype=torch.float32)
    edgar_x = torch.zeros((B, fixed_len, edgar_dim), dtype=torch.float32)
    edgar_mask = torch.ones((B, fixed_len, edgar_dim), dtype=torch.bool)
    static_x = torch.zeros((B, static_dim), dtype=torch.float32)

    text_emb = None
    nbi = None
    nci = None

    def _to_tensor(arr, dtype):
        try:
            return torch.from_numpy(np.asarray(arr)).to(dtype)
        except RuntimeError as exc:  # pragma: no cover - fallback for torch builds w/o numpy
            if "Numpy is not available" in str(exc):
                return torch.tensor(np.asarray(arr).tolist(), dtype=dtype)
            raise

    def _to_tensor(arr, dtype):
        try:
            return torch.from_numpy(np.asarray(arr)).to(dtype)
        except RuntimeError as exc:  # pragma: no cover - fallback for torch builds w/o numpy
            if "Numpy is not available" in str(exc):
                return torch.tensor(np.asarray(arr).tolist(), dtype=dtype)
            raise

    for i, item in enumerate(batch):
        L = int(item["seq_x"].shape[0])
        if L > fixed_len:
            s = L - fixed_len
            seq_np = item["seq_x"][s:]
            dt_np = item["time_delta"][s:]
            edgar_np = item["edgar_x"][s:]
            edgar_mask_np = item["edgar_mask"][s:]
            L2 = fixed_len
        else:
            seq_np = item["seq_x"]
            dt_np = item["time_delta"]
            edgar_np = item["edgar_x"]
            edgar_mask_np = item["edgar_mask"]
            L2 = L

        seq_x[i, :L2] = _to_tensor(seq_np, torch.float32)
        time_delta[i, :L2] = _to_tensor(dt_np, torch.float32)
        if edgar_dim > 0:
            edgar_x[i, :L2] = _to_tensor(edgar_np, torch.float32)
            edgar_mask[i, :L2] = _to_tensor(edgar_mask_np, torch.bool)
        seq_mask[i, :L2] = True
        static_x[i] = _to_tensor(item["static_x"], torch.float32)

        if "text_emb" in item:
            if text_emb is None:
                text_dim = int(item["text_emb"].shape[1])
                text_emb = torch.zeros((B, fixed_len, text_dim), dtype=torch.float32)
            text_np = item["text_emb"][s:] if L > fixed_len else item["text_emb"]
            text_emb[i, :L2] = _to_tensor(text_np, torch.float32)

        if "nbi" in item:
            if nbi is None:
                nbi_dim = int(item["nbi"].shape[0])
                nbi = torch.zeros((B, nbi_dim), dtype=torch.float32)
            nbi[i] = _to_tensor(item["nbi"], torch.float32)

        if "nci" in item:
            if nci is None:
                nci_dim = int(item["nci"].shape[0])
                nci = torch.zeros((B, nci_dim), dtype=torch.float32)
            nci[i] = _to_tensor(item["nci"], torch.float32)

    out = {
        "static_x": static_x,
        "seq_x": seq_x,
        "seq_mask": seq_mask,
        "time_delta": time_delta,
        "edgar_x": edgar_x,
        "edgar_mask": edgar_mask,
    }
    if text_emb is not None:
        out["text_emb"] = text_emb
    if nbi is not None:
        out["nbi"] = nbi
    if nci is not None:
        out["nci"] = nci
    return out


__all__ = [
    "ParquetOffersDataset",
    "collate_parquet_offers",
]
