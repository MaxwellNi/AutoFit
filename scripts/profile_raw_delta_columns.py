#!/usr/bin/env python
"""
Profile raw Delta tables (offers, edgar) for column stats.
Uses Delta _delta_log stats when available; fallback to parquet metadata or scan sample.
Output: raw_offers_profile.json/.md, raw_edgar_profile.json/.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent


def _delta_stats_from_log(delta_path: Path, mode: str) -> Dict[str, Any]:
    """Parse Delta _delta_log for add actions with stats. Aggregate min/max/nullCount."""
    log_dir = delta_path / "_delta_log"
    if not log_dir.exists():
        return {}
    stats_by_col: Dict[str, Dict[str, Any]] = {}
    total_records = 0
    n_files_with_stats = 0
    for f in sorted(log_dir.glob("*.json")):
        try:
            content = f.read_text(encoding="utf-8")
            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                add = obj.get("add", {})
                if not add:
                    continue
                st = add.get("stats")
                if not st:
                    continue
                if isinstance(st, str):
                    st = json.loads(st)
                n_rec = st.get("numRecords", 0)
                total_records += n_rec
                n_files_with_stats += 1
                minv = st.get("minValues", {})
                maxv = st.get("maxValues", {})
                nullc = st.get("nullCount", {})
                for col in set(list(minv.keys()) + list(maxv.keys()) + list(nullc.keys())):
                    if col not in stats_by_col:
                        stats_by_col[col] = {
                            "min": None, "max": None,
                            "null_count_sum": 0, "record_count_sum": 0,
                            "n_files": 0,
                        }
                    stats_by_col[col]["record_count_sum"] += n_rec
                    stats_by_col[col]["n_files"] += 1
                    if col in nullc:
                        stats_by_col[col]["null_count_sum"] += nullc[col]
                    if col in minv and (stats_by_col[col]["min"] is None or minv[col] < stats_by_col[col]["min"]):
                        stats_by_col[col]["min"] = minv[col]
                    if col in maxv and (stats_by_col[col]["max"] is None or maxv[col] > stats_by_col[col]["max"]):
                        stats_by_col[col]["max"] = maxv[col]
        except Exception:
            continue
    if not stats_by_col:
        return {}
    cols = {}
    for col, v in stats_by_col.items():
        rc = v["record_count_sum"]
        nc = v["null_count_sum"]
        cols[col] = {
            "dtype": "string",
            "source": "delta_log",
            "approx_non_null_rate": 1 - (nc / rc) if rc else None,
            "approx_null_count": nc,
            "record_count": rc,
            "min": v["min"],
            "max": v["max"],
        }
    return {
        "mode": mode,
        "source": "delta_log",
        "total_records_est": total_records,
        "n_files_with_stats": n_files_with_stats,
        "columns": cols,
    }


def _schema_from_deltalake(delta_path: Path) -> Dict[str, str]:
    """Get schema from DeltaTable."""
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(delta_path))
        return {f.name: str(f.type) for f in dt.schema().fields}
    except Exception:
        return {}


def _welford_update(n: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
    n1 = n + 1
    delta = x - mean
    mean1 = mean + delta / n1
    m2_1 = m2 + delta * (x - mean1)
    return n1, mean1, m2_1


def _profile_from_scan_streaming(
    delta_path: Path, mode: str, scan_rows: int, seed: int, batch_size: int = 500_000
) -> Dict[str, Any]:
    """Streaming scan: Welford mean/std, min/max, reservoir quantiles, top_k for categorical."""
    try:
        from deltalake import DeltaTable
        import pandas as pd
        import numpy as np
        dt = DeltaTable(str(delta_path))
        dset = dt.to_pyarrow_dataset()
        all_cols = [f.name for f in dt.schema().fields]
        read_cols = all_cols[:120]
        acc: Dict[str, Dict[str, Any]] = {c: {"welford": (0, 0.0, 0.0), "min": None, "max": None, "reservoir": [], "counter": {}, "null_count": 0} for c in read_cols}
        reservoir_size = min(100_000, scan_rows // 10)
        total_seen = 0
        rng = np.random.default_rng(seed)
        for batch in dset.scanner(columns=read_cols, batch_size=batch_size).to_batches():
            df = batch.to_pandas()
            total_seen += len(df)
            for c in read_cols:
                if c not in df.columns:
                    continue
                s = df[c].dropna()
                acc[c]["null_count"] += int(df[c].isna().sum())
                if s.empty:
                    continue
                if pd.api.types.is_numeric_dtype(s):
                    arr = pd.to_numeric(s, errors="coerce").dropna().values
                    n, mean, m2 = acc[c]["welford"]
                    for v in arr:
                        vf = float(v)
                        if np.isfinite(vf):
                            n, mean, m2 = _welford_update(n, mean, m2, vf)
                            if acc[c]["min"] is None or vf < acc[c]["min"]:
                                acc[c]["min"] = vf
                            if acc[c]["max"] is None or vf > acc[c]["max"]:
                                acc[c]["max"] = vf
                            if len(acc[c]["reservoir"]) < reservoir_size:
                                acc[c]["reservoir"].append(vf)
                            elif rng.random() < reservoir_size / (n + 1):
                                acc[c]["reservoir"][rng.integers(0, reservoir_size)] = vf
                    acc[c]["welford"] = (n, mean, m2)
                else:
                    for v in s.astype(str):
                        acc[c]["counter"][v] = acc[c]["counter"].get(v, 0) + 1
                        if len(acc[c]["counter"]) > 100_000:
                            acc[c]["counter"] = dict(sorted(acc[c]["counter"].items(), key=lambda x: -x[1])[:50_000])
            if total_seen >= scan_rows:
                break
        cols_out = {}
        for c in read_cols:
            a = acc[c]
            n, mean, m2 = a["welford"]
            rc = total_seen
            nc = a["null_count"]
            nnr = 1 - (nc / rc) if rc else None
            dtype = "float64" if n > 0 else "object"
            col: Dict[str, Any] = {
                "column": c,
                "dtype": dtype,
                "source": "scan_sample",
                "non_null_rate_sample": nnr,
                "record_count": rc,
                "approx_null_count": nc,
            }
            if n > 0:
                col["mean_sample"] = mean
                col["std_sample"] = float((m2 / n) ** 0.5) if n else 0.0
                col["min_sample"] = a["min"]
                col["max_sample"] = a["max"]
                rv = a["reservoir"]
                if len(rv) >= 10:
                    arr = np.array(rv)
                    col["p01_sample"] = float(np.percentile(arr, 1))
                    col["p50_sample"] = float(np.percentile(arr, 50))
                    col["p99_sample"] = float(np.percentile(arr, 99))
            if a["counter"]:
                top = sorted(a["counter"].items(), key=lambda x: -x[1])[:50]
                col["distinct_sample"] = len(a["counter"])
                col["top_k"] = [x[0] for x in top]
            col["approx_non_null_rate"] = nnr
            cols_out[c] = col
        return {
            "mode": mode,
            "source": "scan_sample",
            "total_records_est": total_seen,
            "columns": cols_out,
        }
    except Exception as e:
        return {"mode": mode, "source": "error", "error": str(e), "columns": {}}


def _profile_from_scan(delta_path: Path, mode: str, scan_rows: int, seed: int) -> Dict[str, Any]:
    """Fallback: scan sample rows for column stats (small sample, in-memory)."""
    try:
        from deltalake import DeltaTable
        import pandas as pd
        dt = DeltaTable(str(delta_path))
        dset = dt.to_pyarrow_dataset()
        cols = [f.name for f in dt.schema().fields][:80]
        df = dset.scanner(columns=cols, batch_size=min(scan_rows, 200_000)).to_pandas()
        if scan_rows and len(df) > scan_rows:
            df = df.sample(n=scan_rows, random_state=seed, replace=False)
        if df.empty:
            return {"mode": mode, "source": "scan_sample", "columns": {}, "total_records_est": 0}
        cols_out = {}
        for c in df.columns:
            s = df[c]
            nc = s.isna().sum()
            rc = len(s)
            cols_out[c] = {
                "column": c,
                "dtype": str(s.dtype),
                "source": "scan_sample",
                "non_null_rate_sample": 1 - (nc / rc) if rc else None,
                "approx_null_count": int(nc),
                "record_count": rc,
                "min": str(s.min()) if s.dtype == "object" or "str" in str(s.dtype) else (float(s.min()) if s.dtype.kind in "fc" else s.min()),
                "max": str(s.max()) if s.dtype == "object" or "str" in str(s.dtype) else (float(s.max()) if s.dtype.kind in "fc" else s.max()),
                "distinct_sample": int(s.nunique()) if rc else None,
            }
        return {
            "mode": mode,
            "source": "scan_sample",
            "total_records_est": len(df),
            "columns": cols_out,
        }
    except Exception as e:
        return {"mode": mode, "source": "error", "error": str(e), "columns": {}}


def _to_md(profile: Dict[str, Any], title: str) -> str:
    """Convert profile to markdown grouped by modeling value."""
    lines = [f"# {title}", ""]
    lines.append(f"- Source: {profile.get('source', 'unknown')}")
    lines.append(f"- Total records (est): {profile.get('total_records_est', 'N/A')}")
    lines.append("")
    cols = profile.get("columns", {})
    keys_time = ["platform_name", "offer_id", "entity_id", "crawled_date", "crawled_date_day", "filed_date"]
    outcome = ["funding_goal_usd", "funding_raised_usd", "investors_count", "is_funded"]
    pricing = ["funding_goal", "funding_raised", "minimum_investment_accepted"]
    status = ["status", "web_fundraising_status", "regulation"]
    company = ["company_name", "company_legal_name", "cik", "platform_country"]
    text = ["headline", "title", "description_text", "company_description", "financial_condition"]
    edgar = ["total_offering_amount", "total_amount_sold", "submission_offering_data"]
    groups = [
        ("Keys/Time", keys_time),
        ("Outcome/Trajectory", outcome),
        ("Pricing/Terms", pricing),
        ("Status/Regime", status),
        ("Company traits", company),
        ("Arrays/Text", text),
        ("EDGAR fundamentals", edgar),
    ]
    seen = set()
    for gname, gcols in groups:
        found = [c for c in gcols if c in cols and c not in seen]
        if not found:
            continue
        lines.append(f"## {gname}")
        for c in found:
            v = cols[c]
            nnr = v.get("approx_non_null_rate")
            nnr_s = f"{nnr:.2%}" if nnr is not None else "N/A"
            std_s = f" std={v.get('std_sample'):.4f}" if v.get('std_sample') is not None else ""
            dist_s = f" distinct={v.get('distinct_sample')}" if v.get('distinct_sample') is not None else ""
            lines.append(f"- **{c}**: non_null≈{nnr_s}{std_s}{dist_s}")
            seen.add(c)
        lines.append("")
    other = [c for c in cols if c not in seen][:30]
    if other:
        lines.append("## Other columns")
        for c in other:
            v = cols[c]
            nnr = v.get("approx_non_null_rate")
            nnr_s = f"{nnr:.2%}" if nnr is not None else "N/A"
            lines.append(f"- {c}: non_null≈{nnr_s}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile raw Delta columns for Block3 data freeze.")
    parser.add_argument("--raw_delta", type=Path, default=None, help="Delta table path (overrides raw_offers/raw_edgar when mode is set)")
    parser.add_argument("--raw_offers", type=Path, default=Path("data/raw/offers"))
    parser.add_argument("--raw_edgar", type=Path, default=Path("data/raw/edgar/accessions"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--mode", choices=["offers", "edgar", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_files_frac", type=float, default=1.0)
    parser.add_argument("--scan_sample_rows", type=int, default=0, help="0=no scan; >0 stream sample for stats (use streaming when >= 500k)")
    parser.add_argument("--inventory_out", type=Path, default=None, help="Output parquet for column inventory (used when mode is single)")
    parser.add_argument("--inventory_out_offers", type=Path, default=None)
    parser.add_argument("--inventory_out_edgar", type=Path, default=None)
    args = parser.parse_args()

    delta_offers = args.raw_delta if args.raw_delta and args.mode == "offers" else args.raw_offers
    delta_edgar = args.raw_delta if args.raw_delta and args.mode == "edgar" else args.raw_edgar
    out_dir = args.output_dir or Path("runs/orchestrator/20260129_073037/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run_mode(mode: str, delta: Path) -> None:
        prof = _delta_stats_from_log(delta.resolve(), mode)
        if args.scan_sample_rows >= 500_000:
            prof = _profile_from_scan_streaming(delta.resolve(), mode, args.scan_sample_rows, args.seed)
        elif not prof.get("columns") and args.scan_sample_rows:
            prof = _profile_from_scan(delta.resolve(), mode, args.scan_sample_rows, args.seed)
        elif not prof.get("columns"):
            prof = _profile_from_scan(delta.resolve(), mode, 50_000, args.seed)
        schema = _schema_from_deltalake(delta.resolve())
        for col, dtype in schema.items():
            if col not in prof.get("columns", {}):
                prof.setdefault("columns", {})[col] = {"column": col, "dtype": dtype, "source": "schema_only", "approx_non_null_rate": None}
        out_json = args.output_json if args.output_json else (out_dir / f"raw_{mode}_profile.json")
        out_md = args.output_md if args.output_md else (out_dir / f"raw_{mode}_profile.md")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(prof, indent=2), encoding="utf-8")
        out_md.write_text(_to_md(prof, f"Raw {mode.title()} Profile"), encoding="utf-8")
        print(f"Wrote {out_json}, {out_md}", flush=True)
        inv_path = None
        if args.inventory_out and args.mode == mode:
            inv_path = args.inventory_out
        elif args.mode == "both":
            inv_path = (args.inventory_out_offers if mode == "offers" else args.inventory_out_edgar) or (out_dir / f"raw_{mode}_column_inventory.parquet")
        if inv_path:
            inv_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pandas as pd
                rows = []
                for col_name, v in prof.get("columns", {}).items():
                    r = {"column": col_name, **{k: val for k, val in v.items() if k != "top_k" and not isinstance(val, (list, dict))}}
                    rows.append(r)
                inv_df = pd.DataFrame(rows)
                inv_df.to_parquet(inv_path, index=False)
                print(f"Wrote inventory {inv_path}", flush=True)
            except Exception as e:
                print(f"WARN: inventory write failed: {e}", flush=True)

    if args.mode in ("offers", "both"):
        _run_mode("offers", delta_offers)
    if args.mode in ("edgar", "both"):
        _run_mode("edgar", delta_edgar)


if __name__ == "__main__":
    main()
