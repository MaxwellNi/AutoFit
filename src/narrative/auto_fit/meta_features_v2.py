#!/usr/bin/env python3
"""
AutoFit v2 — Meta-Features (Section A / D1).

Computes per-entity and global meta-features for the AutoFit v2 model-selection
pipeline.  The output drives gating, expert selection, and auditing.

Design contract:
    1. Input:  raw panel DataFrame (entity_id × crawled_date_day × features).
    2. Output: Dict[str, float] meta-features + optional JSON/MD report.
    3. Zero information from labels (target) enters meta-features.
       The only use of `target_col` is to separate it from X features.
    4. All computations are deterministic given the same input.

Meta-feature groups:
    A1. Missingness     – per-column / per-entity missing rates
    A2. Irregularity    – sampling interval CV, gap frequency
    A3. Multi-scale     – autocorrelation at 7/30/90 day lags
    A4. Heavy-tail      – kurtosis, tail-index proxy
    A5. Non-stationarity– ADF statistic, rolling-mean drift
    A6. Exog strength   – mutual-info proxy between exog cols and target
    A7. Leakage suspects– columns highly correlated with future label
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetaFeaturesV2:
    """Container for v2 meta-features with provenance."""

    # A1: Missingness
    missing_rate_global: float = 0.0
    missing_rate_per_entity_mean: float = 0.0
    missing_rate_per_entity_max: float = 0.0
    n_cols_above_50pct_missing: int = 0
    pct_cols_above_50pct_missing: float = 0.0

    # A2: Irregularity
    sampling_interval_cv: float = 0.0
    pct_gaps_gt_7d: float = 0.0
    pct_gaps_gt_30d: float = 0.0
    median_obs_per_entity: float = 0.0

    # A3: Multi-scale
    acf_lag7_mean: float = 0.0
    acf_lag30_mean: float = 0.0
    acf_lag90_mean: float = 0.0
    multiscale_score: float = 0.0

    # A4: Heavy-tail / outlier
    kurtosis_mean: float = 0.0
    kurtosis_max: float = 0.0
    tail_index_proxy: float = 0.0       # Hill estimator on top 5%
    pct_outliers_3sigma: float = 0.0

    # A5: Non-stationarity
    nonstationarity_score: float = 0.0   # fraction of series failing ADF
    rolling_mean_drift: float = 0.0      # normalised max-min of 30-day rolling mean

    # A6: Exogenous strength
    exog_strength: float = 0.0
    edgar_strength: float = 0.0
    text_strength: float = 0.0

    # A7: Leakage suspects
    leakage_suspects: List[str] = field(default_factory=list)
    leakage_max_corr: float = 0.0

    # Panel summary
    n_entities: int = 0
    n_rows: int = 0
    n_feature_cols: int = 0
    date_range_days: int = 0

    # Provenance
    computed_at: str = ""
    git_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    # ---- Backward-compatible mapping to v1 names ----
    def to_v1_compat(self) -> Dict[str, float]:
        """Return dict keyed by v1 meta-feature names for existing composer."""
        return {
            "missing_rate": self.missing_rate_global,
            "periodicity_score": self.acf_lag7_mean,
            "long_memory_score": self.acf_lag30_mean,
            "multiscale_score": self.multiscale_score,
            "nonstationarity_score": self.nonstationarity_score,
            "irregular_score": self.sampling_interval_cv,
            "heavy_tail_score": self.kurtosis_mean / 10.0,   # normalise
            "exog_strength": self.exog_strength,
            "edgar_strength": self.edgar_strength,
            "text_strength": self.text_strength,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_acf(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at a specific lag. Returns NaN if underdetermined."""
    if len(x) <= lag + 1:
        return np.nan
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _hill_estimator(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill tail-index estimator on the top `tail_frac` fraction.

    Returns ξ > 0.  Large ξ ⇒ heavier tail.
    Returns 0.0 if insufficient data or degenerate.
    """
    x = np.abs(x)
    x = x[x > 0]
    if len(x) < 20:
        return 0.0
    k = max(2, int(len(x) * tail_frac))
    x_sorted = np.sort(x)[::-1][:k]
    threshold = x_sorted[-1]
    if threshold <= 0:
        return 0.0
    log_ratios = np.log(x_sorted[:-1] / threshold)
    return float(np.mean(log_ratios)) if len(log_ratios) > 0 else 0.0


def _adf_reject(x: np.ndarray, significance: float = 0.05) -> bool:
    """Return True if series is *stationary* (reject unit root)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(x, maxlag=min(12, len(x) // 4), autolag="AIC")
        return result[1] < significance
    except Exception:
        return False  # Cannot determine → assume non-stationary


def _mutual_info_score(x: np.ndarray, y: np.ndarray) -> float:
    """Discretised mutual-information estimate between two continuous arrays."""
    try:
        from sklearn.metrics import mutual_info_score as mi
        n_bins = max(5, min(50, int(np.sqrt(len(x)))))
        x_d = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
        y_d = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        return float(mi(x_d, y_d))
    except Exception:
        return 0.0


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_meta_features(
    df: pd.DataFrame,
    *,
    entity_col: str = "entity_id",
    date_col: str = "crawled_date_day",
    target_col: str = "funding_raised_usd",
    edgar_prefix: str = "edgar_",
    text_prefix: str = "text_",
    max_entities_sample: int = 500,
    seed: int = 42,
) -> MetaFeaturesV2:
    """
    Compute v2 meta-features from a raw panel DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with entity_col, date_col, and feature columns.
    entity_col : str
        Column identifying entities.
    date_col : str
        Column with dates (will be coerced to datetime).
    target_col : str
        The prediction target.  Used ONLY for exog-strength and
        leakage-suspect detection (never enters meta-features themselves).
    edgar_prefix / text_prefix : str
        Prefixes for EDGAR / text feature columns.
    max_entities_sample : int
        If more entities exist, sample this many for speed.
    seed : int
        Reproducibility seed.

    Returns
    -------
    MetaFeaturesV2
    """
    mf = MetaFeaturesV2()
    mf.computed_at = datetime.now(timezone.utc).isoformat()
    mf.git_hash = _git_hash()

    if df.empty:
        logger.warning("Empty DataFrame passed to compute_meta_features")
        return mf

    # --- Setup ---
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.sort_values([entity_col, date_col])

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in {entity_col, date_col}
    ]
    feature_cols = [c for c in numeric_cols if c != target_col]

    mf.n_rows = len(df)
    mf.n_entities = int(df[entity_col].nunique())
    mf.n_feature_cols = len(feature_cols)

    dates = df[date_col].dropna()
    if len(dates) >= 2:
        mf.date_range_days = int((dates.max() - dates.min()).days)

    # Subsample entities for O(N) bound
    rng = np.random.RandomState(seed)
    all_entities = df[entity_col].unique()
    if len(all_entities) > max_entities_sample:
        sampled = rng.choice(all_entities, size=max_entities_sample, replace=False)
        df_sampled = df[df[entity_col].isin(sampled)]
    else:
        sampled = all_entities
        df_sampled = df

    # -----------------------------------------------------------------
    # A1: Missingness
    # -----------------------------------------------------------------
    if feature_cols:
        col_missing = df_sampled[feature_cols].isna().mean()
        mf.missing_rate_global = float(col_missing.mean())
        entity_miss = df_sampled.groupby(entity_col)[feature_cols].apply(
            lambda g: g.isna().mean().mean()
        )
        mf.missing_rate_per_entity_mean = float(entity_miss.mean())
        mf.missing_rate_per_entity_max = float(entity_miss.max())
        high_miss = (col_missing > 0.5).sum()
        mf.n_cols_above_50pct_missing = int(high_miss)
        mf.pct_cols_above_50pct_missing = float(high_miss / len(feature_cols))

    # -----------------------------------------------------------------
    # A2: Irregularity
    # -----------------------------------------------------------------
    interval_cvs: List[float] = []
    gaps_7d: List[float] = []
    gaps_30d: List[float] = []
    obs_counts: List[int] = []

    for eid, grp in df_sampled.groupby(entity_col, sort=False):
        t = grp[date_col].dropna().sort_values()
        obs_counts.append(len(t))
        if len(t) < 2:
            continue
        deltas = t.diff().dt.total_seconds().div(86400.0).dropna().values
        if len(deltas) == 0:
            continue
        mu = deltas.mean()
        if mu > 0:
            interval_cvs.append(float(deltas.std() / mu))
        gaps_7d.append(float((deltas > 7).mean()))
        gaps_30d.append(float((deltas > 30).mean()))

    mf.sampling_interval_cv = float(np.nanmean(interval_cvs)) if interval_cvs else 0.0
    mf.pct_gaps_gt_7d = float(np.nanmean(gaps_7d)) if gaps_7d else 0.0
    mf.pct_gaps_gt_30d = float(np.nanmean(gaps_30d)) if gaps_30d else 0.0
    mf.median_obs_per_entity = float(np.median(obs_counts)) if obs_counts else 0.0

    # -----------------------------------------------------------------
    # A3: Multi-scale autocorrelation
    # -----------------------------------------------------------------
    acf7s: List[float] = []
    acf30s: List[float] = []
    acf90s: List[float] = []

    target_present = target_col in df_sampled.columns
    for eid, grp in df_sampled.groupby(entity_col, sort=False):
        if not target_present:
            break
        series = pd.to_numeric(grp[target_col], errors="coerce").dropna().values
        if len(series) < 10:
            continue
        a7 = _safe_acf(series, 7)
        a30 = _safe_acf(series, 30)
        a90 = _safe_acf(series, 90)
        if np.isfinite(a7):
            acf7s.append(abs(a7))
        if np.isfinite(a30):
            acf30s.append(abs(a30))
        if np.isfinite(a90):
            acf90s.append(abs(a90))

    mf.acf_lag7_mean = float(np.nanmean(acf7s)) if acf7s else 0.0
    mf.acf_lag30_mean = float(np.nanmean(acf30s)) if acf30s else 0.0
    mf.acf_lag90_mean = float(np.nanmean(acf90s)) if acf90s else 0.0
    vals = [mf.acf_lag7_mean, mf.acf_lag30_mean, mf.acf_lag90_mean]
    mf.multiscale_score = float(np.mean([v for v in vals if v > 0])) if any(v > 0 for v in vals) else 0.0

    # -----------------------------------------------------------------
    # A4: Heavy-tail / outlier
    # -----------------------------------------------------------------
    kurtoses: List[float] = []
    outlier_fracs: List[float] = []
    hill_vals: List[float] = []

    for col in feature_cols[:50]:  # cap at 50 cols for speed
        arr = pd.to_numeric(df_sampled[col], errors="coerce").dropna().values
        if len(arr) < 20:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = sp_stats.kurtosis(arr, nan_policy="omit")
        if np.isfinite(k):
            kurtoses.append(float(k))
        mu, sigma = arr.mean(), arr.std()
        if sigma > 1e-12:
            z = np.abs((arr - mu) / sigma)
            outlier_fracs.append(float((z > 3).mean()))
        h = _hill_estimator(arr)
        if h > 0:
            hill_vals.append(h)

    mf.kurtosis_mean = float(np.nanmean(kurtoses)) if kurtoses else 0.0
    mf.kurtosis_max = float(np.nanmax(kurtoses)) if kurtoses else 0.0
    mf.pct_outliers_3sigma = float(np.nanmean(outlier_fracs)) if outlier_fracs else 0.0
    mf.tail_index_proxy = float(np.nanmean(hill_vals)) if hill_vals else 0.0

    # -----------------------------------------------------------------
    # A5: Non-stationarity
    # -----------------------------------------------------------------
    n_nonstationary = 0
    n_tested = 0
    drift_vals: List[float] = []

    for eid, grp in df_sampled.groupby(entity_col, sort=False):
        if not target_present:
            break
        series = pd.to_numeric(grp[target_col], errors="coerce").dropna().values
        if len(series) < 30:
            continue
        n_tested += 1
        if not _adf_reject(series):
            n_nonstationary += 1
        # Rolling-mean drift
        win = min(30, len(series) // 3)
        if win >= 2:
            rm = pd.Series(series).rolling(win).mean().dropna().values
            if len(rm) >= 2:
                rng_rm = rm.max() - rm.min()
                scale = np.abs(series).mean()
                if scale > 1e-12:
                    drift_vals.append(float(rng_rm / scale))

    mf.nonstationarity_score = float(n_nonstationary / n_tested) if n_tested > 0 else 0.0
    mf.rolling_mean_drift = float(np.nanmean(drift_vals)) if drift_vals else 0.0

    # -----------------------------------------------------------------
    # A6: Exogenous strength (mutual-info proxy)
    # -----------------------------------------------------------------
    if target_present and target_col in df_sampled.columns:
        y_raw = pd.to_numeric(df_sampled[target_col], errors="coerce")
        valid_y = y_raw.notna()

        edgar_cols = [c for c in feature_cols if c.startswith(edgar_prefix)]
        text_cols = [c for c in feature_cols if c.startswith(text_prefix)]
        other_cols = [c for c in feature_cols if c not in edgar_cols and c not in text_cols]

        def _avg_mi(cols: List[str]) -> float:
            mis = []
            for col in cols[:30]:  # cap
                x = pd.to_numeric(df_sampled[col], errors="coerce")
                mask = valid_y & x.notna()
                if mask.sum() < 30:
                    continue
                mi_val = _mutual_info_score(x[mask].values, y_raw[mask].values)
                mis.append(mi_val)
            return float(np.mean(mis)) if mis else 0.0

        mf.edgar_strength = _avg_mi(edgar_cols)
        mf.text_strength = _avg_mi(text_cols)
        exog_all = _avg_mi(edgar_cols + text_cols + other_cols)
        mf.exog_strength = exog_all

    # -----------------------------------------------------------------
    # A7: Leakage suspects — columns that are *too* correlated with target
    # -----------------------------------------------------------------
    if target_present and target_col in df_sampled.columns:
        y_vals = pd.to_numeric(df_sampled[target_col], errors="coerce")
        suspects = []
        max_corr = 0.0
        for col in feature_cols:
            x_vals = pd.to_numeric(df_sampled[col], errors="coerce")
            mask = y_vals.notna() & x_vals.notna()
            if mask.sum() < 30:
                continue
            corr = abs(np.corrcoef(x_vals[mask].values, y_vals[mask].values)[0, 1])
            if np.isfinite(corr) and corr > 0.95:
                suspects.append(col)
                max_corr = max(max_corr, corr)
        mf.leakage_suspects = suspects
        mf.leakage_max_corr = max_corr

    return mf


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_meta_features_report(
    mf: MetaFeaturesV2,
    output_dir: Path,
    *,
    stamp: str = "",
) -> Tuple[Path, Path]:
    """
    Save meta-features as JSON and human-readable Markdown.

    Returns (json_path, md_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "meta_features_v2.json"
    md_path = output_dir / "meta_features_v2_report.md"

    json_path.write_text(
        json.dumps(mf.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# AutoFit v2 Meta-Features Report",
        f"",
        f"**Stamp**: {stamp}",
        f"**Computed at**: {mf.computed_at}",
        f"**Git hash**: {mf.git_hash}",
        f"",
        f"## Panel Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Entities | {mf.n_entities:,} |",
        f"| Rows | {mf.n_rows:,} |",
        f"| Feature cols | {mf.n_feature_cols} |",
        f"| Date range (days) | {mf.date_range_days} |",
        f"",
        f"## A1: Missingness",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Global missing rate | {mf.missing_rate_global:.4f} |",
        f"| Per-entity mean | {mf.missing_rate_per_entity_mean:.4f} |",
        f"| Per-entity max | {mf.missing_rate_per_entity_max:.4f} |",
        f"| Cols >50% missing | {mf.n_cols_above_50pct_missing} ({mf.pct_cols_above_50pct_missing:.1%}) |",
        f"",
        f"## A2: Irregularity",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sampling interval CV | {mf.sampling_interval_cv:.4f} |",
        f"| Gaps >7d | {mf.pct_gaps_gt_7d:.4f} |",
        f"| Gaps >30d | {mf.pct_gaps_gt_30d:.4f} |",
        f"| Median obs/entity | {mf.median_obs_per_entity:.0f} |",
        f"",
        f"## A3: Multi-scale",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| ACF lag 7 | {mf.acf_lag7_mean:.4f} |",
        f"| ACF lag 30 | {mf.acf_lag30_mean:.4f} |",
        f"| ACF lag 90 | {mf.acf_lag90_mean:.4f} |",
        f"| Multi-scale score | {mf.multiscale_score:.4f} |",
        f"",
        f"## A4: Heavy-tail / Outlier",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Kurtosis (mean) | {mf.kurtosis_mean:.2f} |",
        f"| Kurtosis (max) | {mf.kurtosis_max:.2f} |",
        f"| Tail index (Hill) | {mf.tail_index_proxy:.4f} |",
        f"| % outliers (3σ) | {mf.pct_outliers_3sigma:.4f} |",
        f"",
        f"## A5: Non-stationarity",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Non-stationary fraction | {mf.nonstationarity_score:.4f} |",
        f"| Rolling-mean drift | {mf.rolling_mean_drift:.4f} |",
        f"",
        f"## A6: Exogenous Strength",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall exog MI | {mf.exog_strength:.4f} |",
        f"| EDGAR MI | {mf.edgar_strength:.4f} |",
        f"| Text MI | {mf.text_strength:.4f} |",
        f"",
        f"## A7: Leakage Suspects",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Max leakage corr | {mf.leakage_max_corr:.4f} |",
        f"| Suspect cols | {len(mf.leakage_suspects)} |",
    ]

    if mf.leakage_suspects:
        lines.append(f"")
        lines.append(f"**Suspect columns** (|corr| > 0.95):")
        for s in mf.leakage_suspects:
            lines.append(f"- `{s}`")

    lines.append(f"")
    lines.append(f"## v1-Compatible Mapping")
    lines.append(f"```json")
    lines.append(json.dumps(mf.to_v1_compat(), indent=2))
    lines.append(f"```")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, md_path


# ---------------------------------------------------------------------------
# Convenience: compute from FreezePointer
# ---------------------------------------------------------------------------

def compute_from_pointer(
    pointer_path: str = "docs/audits/FULL_SCALE_POINTER.yaml",
    target_col: str = "funding_raised_usd",
    max_entities_sample: int = 500,
    output_dir: Optional[str] = None,
) -> MetaFeaturesV2:
    """
    Load data via FreezePointer and compute meta-features.

    Loads offers_core_daily only (main panel).  Edgar / text columns
    are detected by prefix.
    """
    import yaml

    pointer_data = yaml.safe_load(Path(pointer_path).read_text(encoding="utf-8"))
    stamp = pointer_data["stamp"]
    core_dir = pointer_data["paths"]["offers_core_daily"]["dir"]

    logger.info(f"Loading offers_core_daily from {core_dir} ...")
    import pyarrow.parquet as pq
    table = pq.read_table(core_dir)
    df = table.to_pandas()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} cols")

    mf = compute_meta_features(
        df,
        target_col=target_col,
        max_entities_sample=max_entities_sample,
    )

    if output_dir is not None:
        save_meta_features_report(mf, Path(output_dir), stamp=stamp)
        logger.info(f"Report saved to {output_dir}")

    return mf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute AutoFit v2 meta-features")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--max-entities", type=int, default=500)
    parser.add_argument("--output-dir", default=None,
                        help="If not set, prints JSON to stdout")
    args = parser.parse_args()

    mf = compute_from_pointer(
        pointer_path=args.pointer,
        target_col=args.target,
        max_entities_sample=args.max_entities,
        output_dir=args.output_dir,
    )

    if args.output_dir is None:
        print(json.dumps(mf.to_dict(), indent=2))
    else:
        print(f"Done. Report at {args.output_dir}/meta_features_v2_report.md")
