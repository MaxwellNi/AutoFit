#!/usr/bin/env python3
"""
Block 3 Data Profile Generator.

Generates comprehensive data profiles for AutoFit meta-feature extraction.

Profiles include:
- Sequence length distributions (per entity)
- Missing patterns (by entity, by feature group)
- Time interval/gap distributions
- Heavy-tail/outlier detection
- Non-stationarity/structural break indicators
- Seasonality/multi-scale indicators
- Exogenous feature strength (EDGAR, text)

Outputs:
- profile.json
- profile.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer


def compute_sequence_stats(df: pd.DataFrame, entity_col: str = "entity_id", time_col: str = "crawled_date_day") -> Dict[str, Any]:
    """Compute sequence length statistics per entity."""
    seq_lengths = df.groupby(entity_col).size()
    
    return {
        "n_entities": int(len(seq_lengths)),
        "seq_length_mean": float(seq_lengths.mean()),
        "seq_length_std": float(seq_lengths.std()),
        "seq_length_min": int(seq_lengths.min()),
        "seq_length_max": int(seq_lengths.max()),
        "seq_length_median": float(seq_lengths.median()),
        "seq_length_p10": float(seq_lengths.quantile(0.1)),
        "seq_length_p90": float(seq_lengths.quantile(0.9)),
    }


def compute_missing_patterns(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute missing patterns by feature group."""
    results = {}
    
    # Overall missing rate
    overall_missing = df.isnull().mean()
    results["overall_missing_rate"] = float(overall_missing.mean())
    
    # By feature group
    for group_name, cols in feature_groups.items():
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            group_missing = df[existing_cols].isnull().mean()
            results[f"{group_name}_missing_rate"] = float(group_missing.mean())
            results[f"{group_name}_cols_available"] = len(existing_cols)
        else:
            results[f"{group_name}_missing_rate"] = 1.0
            results[f"{group_name}_cols_available"] = 0
    
    # By entity
    entity_missing = df.groupby("entity_id").apply(lambda x: x.isnull().mean().mean())
    results["entity_missing_rate_mean"] = float(entity_missing.mean())
    results["entity_missing_rate_std"] = float(entity_missing.std())
    
    return results


def compute_time_gaps(df: pd.DataFrame, entity_col: str = "entity_id", time_col: str = "crawled_date_day") -> Dict[str, Any]:
    """Compute time gap distributions."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    gaps = []
    for entity_id, group in df.groupby(entity_col):
        group = group.sort_values(time_col)
        if len(group) > 1:
            entity_gaps = group[time_col].diff().dt.days.dropna()
            gaps.extend(entity_gaps.tolist())
    
    if not gaps:
        return {"time_gap_mean": 1.0, "time_gap_std": 0.0, "irregular_score": 0.0}
    
    gaps = np.array(gaps)
    
    # Irregularity score: coefficient of variation of gaps
    irregular_score = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
    
    return {
        "time_gap_mean": float(np.mean(gaps)),
        "time_gap_std": float(np.std(gaps)),
        "time_gap_median": float(np.median(gaps)),
        "time_gap_p95": float(np.percentile(gaps, 95)),
        "irregular_score": irregular_score,
        "pct_gap_gt_7d": float(np.mean(gaps > 7)),
        "pct_gap_gt_30d": float(np.mean(gaps > 30)),
    }


def compute_heavy_tail_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Compute heavy-tail and outlier statistics."""
    results = {}
    
    existing_cols = [c for c in numeric_cols if c in df.columns]
    
    kurtosis_values = []
    outlier_rates = []
    
    for col in existing_cols[:20]:  # Limit to avoid too long computation
        vals = df[col].dropna()
        if len(vals) > 100:
            # Kurtosis (excess kurtosis, normal = 0)
            k = float(stats.kurtosis(vals, nan_policy='omit'))
            kurtosis_values.append(k)
            
            # Outlier rate (beyond 3 std)
            mean, std = vals.mean(), vals.std()
            if std > 0:
                outlier_rate = float(((vals < mean - 3*std) | (vals > mean + 3*std)).mean())
                outlier_rates.append(outlier_rate)
    
    if kurtosis_values:
        results["kurtosis_mean"] = float(np.mean(kurtosis_values))
        results["kurtosis_max"] = float(np.max(kurtosis_values))
        results["heavy_tail_score"] = float(np.mean([k > 3 for k in kurtosis_values]))
    
    if outlier_rates:
        results["outlier_rate_mean"] = float(np.mean(outlier_rates))
        results["outlier_rate_max"] = float(np.max(outlier_rates))
    
    return results


def compute_nonstationarity_score(df: pd.DataFrame, target_col: str = "funding_raised_usd", entity_col: str = "entity_id") -> Dict[str, Any]:
    """Compute non-stationarity indicators."""
    results = {}
    
    if target_col not in df.columns:
        return {"nonstationarity_score": 0.5, "note": f"{target_col} not found"}
    
    # Check variance ratio (early vs late)
    df_sorted = df.sort_values(["crawled_date_day"])
    n = len(df_sorted)
    
    if n > 100:
        early = df_sorted[target_col].iloc[:n//3].dropna()
        late = df_sorted[target_col].iloc[-n//3:].dropna()
        
        if len(early) > 10 and len(late) > 10:
            var_early = early.var()
            var_late = late.var()
            var_ratio = var_late / max(var_early, 1e-9)
            
            # Mean shift
            mean_early = early.mean()
            mean_late = late.mean()
            mean_shift = abs(mean_late - mean_early) / max(abs(mean_early), 1e-9)
            
            # Non-stationarity score: combine variance ratio and mean shift
            nonstationarity_score = min(1.0, 0.5 * (abs(np.log(var_ratio + 1e-9)) / 2 + mean_shift))
            
            results["var_ratio"] = float(var_ratio)
            results["mean_shift"] = float(mean_shift)
            results["nonstationarity_score"] = float(nonstationarity_score)
        else:
            results["nonstationarity_score"] = 0.5
    else:
        results["nonstationarity_score"] = 0.5
    
    return results


def compute_multiscale_score(df: pd.DataFrame, target_col: str = "funding_raised_usd") -> Dict[str, Any]:
    """Compute multi-scale/seasonality indicators."""
    results = {}
    
    if target_col not in df.columns:
        return {"multiscale_score": 0.0, "periodicity_score": 0.0}
    
    # Aggregate by date
    daily = df.groupby("crawled_date_day")[target_col].mean().dropna()
    
    if len(daily) < 30:
        return {"multiscale_score": 0.0, "periodicity_score": 0.0}
    
    # Simple FFT-based periodicity detection
    try:
        vals = daily.values - np.mean(daily.values)
        fft = np.fft.fft(vals)
        power = np.abs(fft[:len(fft)//2]) ** 2
        
        # Find dominant frequencies
        freq_idx = np.argsort(power)[-5:]  # Top 5 frequencies
        total_power = power.sum()
        top_power = power[freq_idx].sum()
        
        periodicity_score = float(top_power / max(total_power, 1e-9))
        
        # Multi-scale: check power at different frequency bands
        n = len(power)
        low_freq_power = power[:n//10].sum()  # Long-term trends
        mid_freq_power = power[n//10:n//3].sum()  # Medium cycles
        high_freq_power = power[n//3:].sum()  # Short-term noise
        
        multiscale_score = float(1 - (high_freq_power / max(total_power, 1e-9)))
        
        results["periodicity_score"] = periodicity_score
        results["multiscale_score"] = multiscale_score
        results["low_freq_ratio"] = float(low_freq_power / max(total_power, 1e-9))
        results["mid_freq_ratio"] = float(mid_freq_power / max(total_power, 1e-9))
    except Exception:
        results["periodicity_score"] = 0.0
        results["multiscale_score"] = 0.0
    
    return results


def compute_exog_strength(
    core_df: pd.DataFrame,
    edgar_df: Optional[pd.DataFrame],
    text_df: Optional[pd.DataFrame],
    target_col: str = "funding_raised_usd",
) -> Dict[str, Any]:
    """
    Compute exogenous feature strength via simple correlation analysis.
    
    This is a proxy for information gain. Full ablation done in benchmark.
    """
    results = {}
    
    if target_col not in core_df.columns:
        return {"edgar_strength": 0.0, "text_strength": 0.0, "exog_strength": 0.0}
    
    target = core_df[target_col].dropna()
    if len(target) < 100:
        return {"edgar_strength": 0.0, "text_strength": 0.0, "exog_strength": 0.0}
    
    # EDGAR strength: correlation with EDGAR features
    edgar_corrs = []
    if edgar_df is not None:
        # Join on cik + crawled_date_day
        merged = core_df.merge(edgar_df, on=["cik", "crawled_date_day"], how="inner", suffixes=("", "_edgar"))
        if len(merged) > 100:
            edgar_cols = [c for c in edgar_df.columns if c not in ["cik", "crawled_date_day"] and merged[c].dtype in [np.float64, np.int64]]
            for col in edgar_cols[:20]:
                try:
                    corr = merged[[target_col, col]].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        edgar_corrs.append(abs(corr))
                except Exception:
                    pass
    
    edgar_strength = float(np.mean(edgar_corrs)) if edgar_corrs else 0.0
    results["edgar_strength"] = edgar_strength
    results["edgar_max_corr"] = float(np.max(edgar_corrs)) if edgar_corrs else 0.0
    
    # Text strength: correlation with text length features (proxy for text richness)
    text_corrs = []
    if text_df is not None:
        # Join on entity_id + crawled_date_day
        text_df_copy = text_df.copy()
        if "crawled_date_day" not in text_df_copy.columns and "snapshot_ts" in text_df_copy.columns:
            text_df_copy["crawled_date_day"] = pd.to_datetime(text_df_copy["snapshot_ts"]).dt.date.astype(str)
        
        merged = core_df.merge(text_df_copy, on=["entity_id", "crawled_date_day"], how="inner", suffixes=("", "_text"))
        if len(merged) > 100:
            # Look for text length columns
            text_len_cols = [c for c in merged.columns if "len" in c.lower() or "token" in c.lower()]
            for col in text_len_cols[:10]:
                try:
                    corr = merged[[target_col, col]].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        text_corrs.append(abs(corr))
                except Exception:
                    pass
    
    text_strength = float(np.mean(text_corrs)) if text_corrs else 0.0
    results["text_strength"] = text_strength
    results["text_max_corr"] = float(np.max(text_corrs)) if text_corrs else 0.0
    
    # Combined exog strength
    results["exog_strength"] = max(edgar_strength, text_strength)
    
    return results


def generate_profile(ds: Block3Dataset, output_dir: Path, sample_frac: float = 0.1) -> Dict[str, Any]:
    """Generate comprehensive data profile."""
    print("Loading data...")
    
    # Load core (sample for efficiency)
    core_df = ds.get_offers_core_daily()
    if sample_frac < 1.0:
        core_df = core_df.sample(frac=sample_frac, random_state=42)
    
    print(f"Core rows (sampled): {len(core_df):,}")
    
    # Define feature groups
    feature_groups = {
        "target": ["funding_raised_usd", "funding_goal_usd", "investors_count", "is_funded"],
        "financial": ["price_per_share_usd", "valuation_pre_money_usd", "valuation_cap_usd"],
        "temporal": ["datetime_open_offering", "datetime_close_offering", "crawled_date_day"],
        "categorical": ["platform_name", "regulation", "status", "company_type"],
    }
    
    numeric_cols = [
        "funding_raised_usd", "funding_goal_usd", "investors_count",
        "price_per_share_usd", "valuation_pre_money_usd", "valuation_cap_usd",
    ]
    
    profile = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stamp": ds.pointer.stamp,
        "variant": ds.pointer.variant,
        "sample_frac": sample_frac,
        "core_rows_sampled": len(core_df),
    }
    
    # Sequence stats
    print("Computing sequence stats...")
    profile["sequence"] = compute_sequence_stats(core_df)
    
    # Missing patterns
    print("Computing missing patterns...")
    profile["missing"] = compute_missing_patterns(core_df, feature_groups)
    
    # Time gaps
    print("Computing time gaps...")
    profile["time_gaps"] = compute_time_gaps(core_df)
    
    # Heavy tail
    print("Computing heavy tail stats...")
    profile["heavy_tail"] = compute_heavy_tail_stats(core_df, numeric_cols)
    
    # Non-stationarity
    print("Computing non-stationarity...")
    profile["nonstationarity"] = compute_nonstationarity_score(core_df)
    
    # Multi-scale
    print("Computing multi-scale score...")
    profile["multiscale"] = compute_multiscale_score(core_df)
    
    # Exogenous strength
    print("Computing exogenous strength...")
    try:
        edgar_df = ds.get_edgar_store()
        text_df = ds.get_offers_text()
        profile["exogenous"] = compute_exog_strength(core_df, edgar_df, text_df)
    except Exception as e:
        print(f"Warning: Could not compute exog strength: {e}")
        profile["exogenous"] = {"edgar_strength": 0.0, "text_strength": 0.0, "exog_strength": 0.0}
    
    # Meta-features for AutoFit
    profile["meta_features"] = {
        "nonstationarity_score": profile["nonstationarity"].get("nonstationarity_score", 0.5),
        "periodicity_score": profile["multiscale"].get("periodicity_score", 0.0),
        "multiscale_score": profile["multiscale"].get("multiscale_score", 0.0),
        "long_memory_score": min(1.0, profile["sequence"].get("seq_length_mean", 10) / 100),
        "irregular_score": profile["time_gaps"].get("irregular_score", 0.0),
        "heavy_tail_score": profile["heavy_tail"].get("heavy_tail_score", 0.0),
        "exog_strength": profile["exogenous"].get("exog_strength", 0.0),
        "edgar_strength": profile["exogenous"].get("edgar_strength", 0.0),
        "text_strength": profile["exogenous"].get("text_strength", 0.0),
        "missing_rate": profile["missing"].get("overall_missing_rate", 0.5),
    }
    
    return profile


def write_profile_md(profile: Dict[str, Any], md_path: Path):
    """Write profile as markdown."""
    lines = [
        "# Block 3 Data Profile",
        "",
        f"**Generated:** {profile['generated_at']}",
        f"**Stamp:** `{profile['stamp']}`",
        f"**Variant:** `{profile['variant']}`",
        f"**Sample fraction:** {profile['sample_frac']}",
        f"**Core rows sampled:** {profile['core_rows_sampled']:,}",
        "",
        "## Sequence Statistics",
        "",
        f"- Entities: {profile['sequence']['n_entities']:,}",
        f"- Mean sequence length: {profile['sequence']['seq_length_mean']:.1f}",
        f"- Median sequence length: {profile['sequence']['seq_length_median']:.1f}",
        f"- Min/Max: {profile['sequence']['seq_length_min']} / {profile['sequence']['seq_length_max']}",
        "",
        "## Missing Patterns",
        "",
        f"- Overall missing rate: {profile['missing']['overall_missing_rate']:.2%}",
        f"- Target missing rate: {profile['missing'].get('target_missing_rate', 'N/A')}",
        f"- Financial missing rate: {profile['missing'].get('financial_missing_rate', 'N/A')}",
        "",
        "## Time Gap Distribution",
        "",
        f"- Mean gap (days): {profile['time_gaps']['time_gap_mean']:.1f}",
        f"- Median gap (days): {profile['time_gaps'].get('time_gap_median', 'N/A')}",
        f"- Irregular score: {profile['time_gaps']['irregular_score']:.3f}",
        f"- % gaps > 7 days: {profile['time_gaps'].get('pct_gap_gt_7d', 0):.1%}",
        "",
        "## Heavy Tail / Outliers",
        "",
        f"- Mean kurtosis: {profile['heavy_tail'].get('kurtosis_mean', 'N/A')}",
        f"- Heavy tail score: {profile['heavy_tail'].get('heavy_tail_score', 'N/A')}",
        f"- Mean outlier rate: {profile['heavy_tail'].get('outlier_rate_mean', 'N/A')}",
        "",
        "## Non-stationarity",
        "",
        f"- Non-stationarity score: {profile['nonstationarity'].get('nonstationarity_score', 'N/A')}",
        f"- Variance ratio (late/early): {profile['nonstationarity'].get('var_ratio', 'N/A')}",
        f"- Mean shift: {profile['nonstationarity'].get('mean_shift', 'N/A')}",
        "",
        "## Multi-scale / Seasonality",
        "",
        f"- Periodicity score: {profile['multiscale'].get('periodicity_score', 'N/A')}",
        f"- Multi-scale score: {profile['multiscale'].get('multiscale_score', 'N/A')}",
        "",
        "## Exogenous Feature Strength",
        "",
        f"- EDGAR strength: {profile['exogenous'].get('edgar_strength', 'N/A')}",
        f"- Text strength: {profile['exogenous'].get('text_strength', 'N/A')}",
        f"- Combined exog strength: {profile['exogenous'].get('exog_strength', 'N/A')}",
        "",
        "## Meta-Features (for AutoFit)",
        "",
        "| Feature | Value |",
        "|---------|-------|",
    ]
    
    for k, v in profile["meta_features"].items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate Block 3 data profile")
    parser.add_argument("--pointer", type=Path, default=Path("docs/audits/FULL_SCALE_POINTER.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-frac", type=float, default=0.1, help="Fraction of data to sample")
    args = parser.parse_args()
    
    # Load dataset
    ds = Block3Dataset.from_pointer(args.pointer)
    
    # Determine output directory (pointer-resolved)
    if args.output_dir is None:
        # Use pointer-resolved base path
        from src.narrative.auto_fit.rule_based_composer import get_profile_path_from_pointer
        profile_dir = get_profile_path_from_pointer(args.pointer)
        output_dir = profile_dir.parent
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate profile
    profile = generate_profile(ds, output_dir, sample_frac=args.sample_frac)
    
    # Write outputs
    json_path = output_dir / "profile.json"
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    
    md_path = output_dir / "profile.md"
    write_profile_md(profile, md_path)
    print(f"Wrote {md_path}")
    
    print("\nMeta-features for AutoFit:")
    for k, v in profile["meta_features"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
