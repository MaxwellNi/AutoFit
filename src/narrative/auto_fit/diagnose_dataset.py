from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Iterable
import json

import numpy as np
import pandas as pd


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def diagnose_dataset(
    df: pd.DataFrame,
    *,
    key_col: str = "offer_id",
    time_col: str = "crawled_date",
    value_cols: Optional[Sequence[str]] = None,
    target_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Diagnose dataset for missingness, periodicity, long-memory, nonstationarity, exog strength.
    """
    if value_cols is None:
        value_cols = list(df.select_dtypes(include=["number", "bool"]).columns)
        value_cols = [c for c in value_cols if c not in {time_col}]

    d = df.copy()
    if time_col in d.columns:
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.sort_values([key_col, time_col], kind="mergesort")

    missing_rate = float(d[value_cols].isna().mean().mean()) if value_cols else 0.0

    periodicity_scores = []
    long_memory_scores = []
    multiscale_scores = []
    nonstat_scores = []
    irregularity_scores = []
    time_delta_stats = []

    for _, g in d.groupby(key_col, sort=False):
        if time_col in g.columns:
            t = pd.to_datetime(g[time_col], errors="coerce", utc=True).dropna()
            if len(t) >= 2:
                deltas = t.sort_values().diff().dt.total_seconds().div(86400.0).dropna()
                if len(deltas) > 0:
                    mean_delta = float(deltas.mean())
                    std_delta = float(deltas.std())
                    if mean_delta > 0:
                        irregularity_scores.append(std_delta / mean_delta)
                    time_delta_stats.append((mean_delta, std_delta))
        for col in value_cols:
            series = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
            if len(series) < 10:
                continue
            p7 = abs(_autocorr(series, lag=7))
            p30 = abs(_autocorr(series, lag=30))
            p90 = abs(_autocorr(series, lag=90))
            if np.isfinite(p7):
                periodicity_scores.append(p7)
            if np.isfinite(p30):
                long_memory_scores.append(p30)
            vals = np.array([p7, p30, p90], dtype=float)
            if np.isfinite(vals).any():
                multiscale_scores.append(np.nanmean(vals))

            first = series[: max(1, len(series) // 3)].mean()
            last = series[-max(1, len(series) // 3) :].mean()
            std = series.std() + 1e-8
            nonstat_scores.append(abs(last - first) / std)

    periodicity_score = float(np.nanmean(periodicity_scores)) if periodicity_scores else 0.0
    long_memory_score = float(np.nanmean(long_memory_scores)) if long_memory_scores else 0.0
    multiscale_score = float(np.nanmean(multiscale_scores)) if multiscale_scores else 0.0
    nonstationarity_score = float(np.nanmean(nonstat_scores)) if nonstat_scores else 0.0
    irregularity_score = float(np.nanmean(irregularity_scores)) if irregularity_scores else 0.0

    if time_delta_stats:
        mean_deltas = [m for m, _ in time_delta_stats]
        std_deltas = [s for _, s in time_delta_stats]
        mean_time_delta = float(np.nanmean(mean_deltas))
        std_time_delta = float(np.nanmean(std_deltas))
    else:
        mean_time_delta = 0.0
        std_time_delta = 0.0

    exog_strength = 0.0
    if target_col and target_col in d.columns:
        exog_cols = [c for c in d.columns if c.startswith("edgar_") or c.startswith("text_")]
        if exog_cols:
            exog_corrs = []
            y = pd.to_numeric(d[target_col], errors="coerce")
            for col in exog_cols:
                x = pd.to_numeric(d[col], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                exog_corrs.append(abs(np.corrcoef(x[valid], y[valid])[0, 1]))
            if exog_corrs:
                exog_strength = float(np.nanmean(exog_corrs))

    diagnostics = {
        "missing_rate": missing_rate,
        "periodicity_score": periodicity_score,
        "long_memory_score": long_memory_score,
        "multiscale_score": multiscale_score,
        "nonstationarity_score": nonstationarity_score,
        "irregularity_score": irregularity_score,
        "exog_strength": exog_strength,
        "n_rows": float(len(d)),
        "n_series": float(d[key_col].nunique() if key_col in d.columns else 0),
        "mean_time_delta_days": mean_time_delta,
        "std_time_delta_days": std_time_delta,
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([diagnostics]).to_parquet(out_dir / "diagnostics_table.parquet", index=False)

    return diagnostics
