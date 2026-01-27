from __future__ import annotations

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


_METRIC_PATTERNS = {
    "rmse": re.compile(r"(?i)\bRMSE\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
    "mae": re.compile(r"(?i)\bMAE\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
    "mse": re.compile(r"(?i)\bMSE\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
    "r2": re.compile(r"(?i)\b(R2|R\^2|R²)\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)"),
}

_TOP_KEYWORDS = re.compile(r"(?i)(top[- ]?1\b|\bbest\b|最优|最佳|冠军)")


def _load_main_results(paper_tables_dir: Path | None, bench_dirs: Sequence[str]) -> pd.DataFrame:
    if paper_tables_dir:
        main_path = paper_tables_dir / "main_results.parquet"
        if main_path.exists():
            return pd.read_parquet(main_path)

    dfs: List[pd.DataFrame] = []
    for pattern in bench_dirs:
        for path in sorted(glob.glob(pattern)):
            metrics_path = Path(path) / "metrics.parquet"
            if metrics_path.exists():
                try:
                    dfs.append(pd.read_parquet(metrics_path))
                except Exception:
                    continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError("No main_results.parquet or metrics.parquet found for verification")


def _collect_metric_values(df: pd.DataFrame) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {"rmse": [], "mae": [], "mse": [], "r2": []}
    cols = {c.lower(): c for c in df.columns}
    if {"metric_name", "metric_value"}.issubset(cols):
        name_col = cols["metric_name"]
        value_col = cols["metric_value"]
        for _, row in df[[name_col, value_col]].iterrows():
            metric = str(row[name_col]).lower()
            if metric in values and pd.notna(row[value_col]):
                values[metric].append(float(row[value_col]))
        return values

    for metric in values:
        metric_cols = [c for c in df.columns if metric in c.lower()]
        for col in metric_cols:
            col_vals = df[col].dropna().astype(float).tolist()
            values[metric].extend(col_vals)
    return values


def _collect_model_names(df: pd.DataFrame) -> List[str]:
    for col in ["backbone", "model", "model_name", "model_id"]:
        if col in df.columns:
            names = sorted({str(v) for v in df[col].dropna().unique() if str(v).strip()})
            return names
    return []


def _best_models(df: pd.DataFrame) -> Dict[str, set]:
    best: Dict[str, set] = {"rmse": set(), "mae": set(), "mse": set(), "r2": set()}
    name_col = None
    for col in ["backbone", "model", "model_name", "model_id"]:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        return best

    if {"metric_name", "metric_value"}.issubset(df.columns):
        for metric in ["rmse", "mae", "mse", "r2"]:
            subset = df[df["metric_name"].str.lower() == metric]
            subset = subset[pd.notna(subset["metric_value"])]
            if subset.empty:
                continue
            if metric == "r2":
                best_val = subset["metric_value"].max()
            else:
                best_val = subset["metric_value"].min()
            best[metric] = set(subset.loc[subset["metric_value"] == best_val, name_col].astype(str))
        return best

    for metric in ["rmse", "mae", "mse", "r2"]:
        metric_cols = [c for c in df.columns if metric in c.lower()]
        if not metric_cols:
            continue
        col = sorted(metric_cols)[0]
        subset = df[pd.notna(df[col])]
        if subset.empty:
            continue
        if metric == "r2":
            best_val = subset[col].max()
        else:
            best_val = subset[col].min()
        best[metric] = set(subset.loc[subset[col] == best_val, name_col].astype(str))
    return best


def _matches_table_value(table_values: List[float], doc_value: float, decimals: int) -> bool:
    if not table_values:
        return False
    if decimals <= 0:
        return any(int(round(v)) == int(round(doc_value)) for v in table_values)
    rounded = round(doc_value, decimals)
    for v in table_values:
        if round(v, decimals) == rounded:
            return True
    return False


def _parse_metric_mentions(text: str) -> List[Tuple[str, float, int, str]]:
    mentions: List[Tuple[str, float, int, str]] = []
    for metric, pattern in _METRIC_PATTERNS.items():
        for match in pattern.finditer(text):
            if metric == "r2":
                value_str = match.group(2)
            else:
                value_str = match.group(1)
            try:
                value = float(value_str)
            except ValueError:
                continue
            decimals = len(value_str.split(".")[1]) if "." in value_str else 0
            line_start = text.rfind("\n", 0, match.start())
            line_end = text.find("\n", match.end())
            if line_start == -1:
                line_start = 0
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            mentions.append((metric, value, decimals, line))
    return mentions


def _find_top_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        if _TOP_KEYWORDS.search(line):
            lines.append(line.strip())
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify docs against run outputs")
    parser.add_argument("--paper_tables_dir", type=Path, default=None)
    parser.add_argument("--bench_dirs", nargs="*", default=[])
    parser.add_argument("--doc_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "docs_mismatch_report.txt"

    df = _load_main_results(args.paper_tables_dir, args.bench_dirs)
    metric_values = _collect_metric_values(df)
    model_names = _collect_model_names(df)
    best_models = _best_models(df)

    mismatches: List[str] = []

    for doc_path in args.doc_paths:
        path = Path(doc_path)
        if not path.exists():
            mismatches.append(f"[missing_doc] {doc_path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")

        for metric, value, decimals, line in _parse_metric_mentions(text):
            table_vals = metric_values.get(metric, [])
            if not _matches_table_value(table_vals, value, decimals):
                mismatches.append(
                    f"[metric_mismatch] {doc_path} metric={metric} value={value} line='{line}'"
                )

        if model_names:
            model_pattern = re.compile(
                r"|".join([re.escape(name) for name in sorted(model_names, key=len, reverse=True)]),
                flags=re.IGNORECASE,
            )
            for line in _find_top_lines(text):
                found = {m.group(0) for m in model_pattern.finditer(line)} if model_pattern.pattern else set()
                if not found:
                    mismatches.append(f"[top_model_missing] {doc_path} line='{line}'")
                    continue
                best_union = set().union(*best_models.values()) if best_models else set()
                for name in found:
                    if best_union and name.lower() not in {b.lower() for b in best_union}:
                        mismatches.append(
                            f"[top_model_mismatch] {doc_path} model='{name}' line='{line}'"
                        )

    report_lines = [
        f"mismatch_count={len(mismatches)}",
        f"paper_tables_dir={args.paper_tables_dir}",
        f"bench_dirs={args.bench_dirs}",
    ]
    if mismatches:
        report_lines.append("details:")
        report_lines.extend(mismatches)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
