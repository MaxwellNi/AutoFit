#!/usr/bin/env python
"""
Profile raw Delta tables (offers, edgar) for column stats.
Streaming scan with checkpointing by active_files.
Outputs raw_{mode}_profile.json/.md and inventory parquet (one row per column).
"""
from __future__ import annotations

import argparse
import json
import pickle
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


def _init_acc(schema: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    acc = {}
    for col in schema:
        acc[col] = {
            "welford": (0, 0.0, 0.0),
            "min": None,
            "max": None,
            "reservoir": [],
            "counter": {},
            "null_count": 0,
            "str_len_sum": 0,
            "str_len_count": 0,
            "distinct_overflow": False,
        }
    return acc


def _dtype_kind(type_str: str) -> str:
    tl = str(type_str).lower()
    if "int" in tl or "float" in tl or "double" in tl or "decimal" in tl:
        return "numeric"
    if "bool" in tl:
        return "bool"
    if "timestamp" in tl or "date" in tl:
        return "datetime"
    if "string" in tl or "binary" in tl:
        return "string"
    if "list" in tl or "map" in tl or "struct" in tl or "array" in tl:
        return "nested"
    return "other"


def _profile_from_scan_streaming(
    delta_path: Path,
    mode: str,
    scan_rows: int,
    seed: int,
    state_path: Path,
    max_distinct: int = 200_000,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Streaming scan with checkpointing by active_files."""
    try:
        from deltalake import DeltaTable
        import numpy as np
        import pyarrow.parquet as pq
    except Exception as e:
        return {"mode": mode, "source": "error", "error": str(e), "columns": {}}

    dt = DeltaTable(str(delta_path))
    schema = {f.name: str(f.type) for f in dt.schema().fields}
    files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
    files_list = [str(f) for f in files_list]
    files_list = sorted(files_list)

    if state_path.exists():
        try:
            state = pickle.loads(state_path.read_bytes())
            acc = state.get("acc") or _init_acc(schema)
            processed = set(state.get("processed_files", []))
            total_seen = int(state.get("total_seen", 0))
        except Exception:
            acc = _init_acc(schema)
            processed = set()
            total_seen = 0
    else:
        acc = _init_acc(schema)
        processed = set()
        total_seen = 0

    rng = np.random.default_rng(seed)
    reservoir_size = min(100_000, max(1_000, scan_rows // 10)) if scan_rows else 100_000

    scan_cols = [c for c, t in schema.items() if _dtype_kind(t) != "nested"]
    if not scan_cols:
        return {"mode": mode, "source": "schema_only", "total_records_est": 0, "columns": {}}

    for fpath in files_list:
        if fpath in processed:
            continue
        try:
            pf = pq.ParquetFile(fpath)
        except Exception:
            processed.add(fpath)
            continue
        for rg in range(pf.num_row_groups):
            try:
                table = pf.read_row_group(rg, columns=scan_cols)
            except Exception:
                continue
            total_seen += table.num_rows
            for col, t in schema.items():
                if col not in table.column_names:
                    continue
                arr = table[col]
                acc[col]["null_count"] += arr.null_count
                kind = _dtype_kind(t)
                if kind == "nested":
                    continue
                if kind == "numeric":
                    try:
                        vals = arr.combine_chunks().to_numpy(zero_copy_only=False)
                    except Exception:
                        vals = None
                    if vals is not None:
                        n, mean, m2 = acc[col]["welford"]
                        for v in vals:
                            try:
                                vf = float(v)
                            except Exception:
                                continue
                            n, mean, m2 = _welford_update(n, mean, m2, vf)
                            if acc[col]["min"] is None or vf < acc[col]["min"]:
                                acc[col]["min"] = vf
                            if acc[col]["max"] is None or vf > acc[col]["max"]:
                                acc[col]["max"] = vf
                            if len(acc[col]["reservoir"]) < reservoir_size:
                                acc[col]["reservoir"].append(vf)
                            elif rng.random() < reservoir_size / max(1, n):
                                acc[col]["reservoir"][rng.integers(0, reservoir_size)] = vf
                        acc[col]["welford"] = (n, mean, m2)
                elif kind in ("string", "bool"):
                    try:
                        vals = arr.combine_chunks().to_pylist()
                    except Exception:
                        vals = []
                    for v in vals:
                        if v is None:
                            continue
                        s = str(v)
                        if not acc[col]["distinct_overflow"]:
                            acc[col]["counter"][s] = acc[col]["counter"].get(s, 0) + 1
                            if len(acc[col]["counter"]) > max_distinct:
                                acc[col]["distinct_overflow"] = True
                                acc[col]["counter"] = dict(sorted(acc[col]["counter"].items(), key=lambda x: -x[1])[:max_distinct])
                        acc[col]["str_len_sum"] += len(s)
                        acc[col]["str_len_count"] += 1
            if scan_rows and total_seen >= scan_rows:
                break
        processed.add(fpath)
        state = {
            "processed_files": list(processed),
            "total_seen": total_seen,
            "acc": acc,
        }
        state_path.write_bytes(pickle.dumps(state))
        if scan_rows and total_seen >= scan_rows:
            break

    cols_out: Dict[str, Any] = {}
    for col, t in schema.items():
        a = acc.get(col, {})
        n, mean, m2 = a.get("welford", (0, 0.0, 0.0))
        rc = total_seen
        nc = a.get("null_count", 0)
        nnr = 1 - (nc / rc) if rc else None
        col_rec: Dict[str, Any] = {
            "column": col,
            "dtype": str(t),
            "source": "scan_sample",
            "non_null_rate_sample": nnr,
            "record_count": rc,
            "approx_null_count": nc,
        }
        if n > 0:
            col_rec["mean_sample"] = mean
            col_rec["std_sample"] = float((m2 / n) ** 0.5) if n else 0.0
            col_rec["min_sample"] = a.get("min")
            col_rec["max_sample"] = a.get("max")
            rv = a.get("reservoir") or []
            if len(rv) >= 10:
                import numpy as np
                arr = np.array(rv)
                col_rec["p01_sample"] = float(np.percentile(arr, 1))
                col_rec["p50_sample"] = float(np.percentile(arr, 50))
                col_rec["p99_sample"] = float(np.percentile(arr, 99))
        if a.get("counter"):
            top = sorted(a["counter"].items(), key=lambda x: -x[1])[:top_k]
            col_rec["distinct_sample"] = len(a["counter"])
            col_rec["top_k"] = [x[0] for x in top]
        if a.get("str_len_count", 0) > 0:
            col_rec["avg_str_len"] = a["str_len_sum"] / a["str_len_count"]
        col_rec["approx_non_null_rate"] = nnr
        cols_out[col] = col_rec

    return {
        "mode": mode,
        "source": "scan_sample",
        "total_records_est": total_seen,
        "columns": cols_out,
        "state_path": str(state_path),
    }


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
            std_s = f" std={v.get('std_sample'):.4f}" if v.get("std_sample") is not None else ""
            dist_s = f" distinct={v.get('distinct_sample')}" if v.get("distinct_sample") is not None else ""
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
    parser.add_argument("--scan_sample_rows", type=int, default=0, help="0=no scan; >0 stream sample for stats")
    parser.add_argument("--inventory_out", type=Path, default=None, help="Output parquet for column inventory (used when mode is single)")
    parser.add_argument("--inventory_out_offers", type=Path, default=None)
    parser.add_argument("--inventory_out_edgar", type=Path, default=None)
    parser.add_argument("--docs_audits_dir", type=Path, default=None)
    parser.add_argument("--audit_stamp", type=str, default=None)
    args = parser.parse_args()

    delta_offers = args.raw_delta if args.raw_delta and args.mode == "offers" else args.raw_offers
    delta_edgar = args.raw_delta if args.raw_delta and args.mode == "edgar" else args.raw_edgar
    out_dir = args.output_dir or Path("runs/orchestrator/20260129_073037/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run_mode(mode: str, delta: Path) -> None:
        prof = _delta_stats_from_log(delta.resolve(), mode)
        if args.scan_sample_rows > 0:
            state_path = out_dir / f".raw_profile_state_{mode}.pkl"
            prof = _profile_from_scan_streaming(delta.resolve(), mode, args.scan_sample_rows, args.seed, state_path)
        elif not prof.get("columns"):
            prof = _delta_stats_from_log(delta.resolve(), mode)

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
                    r = {
                        "column": col_name,
                        "dtype": v.get("dtype"),
                        "source": v.get("source"),
                        "non_null_rate": v.get("non_null_rate_sample") or v.get("approx_non_null_rate"),
                        "approx_distinct": v.get("distinct_sample"),
                        "mean": v.get("mean_sample"),
                        "std": v.get("std_sample"),
                        "min": v.get("min_sample") or v.get("min"),
                        "max": v.get("max_sample") or v.get("max"),
                        "p01": v.get("p01_sample"),
                        "p50": v.get("p50_sample"),
                        "p99": v.get("p99_sample"),
                        "avg_str_len": v.get("avg_str_len"),
                        "top_k": json.dumps(v.get("top_k")) if v.get("top_k") is not None else None,
                    }
                    rows.append(r)
                inv_df = pd.DataFrame(rows)
                inv_df.to_parquet(inv_path, index=False)
                print(f"Wrote inventory {inv_path}", flush=True)
            except Exception as e:
                print(f"WARN: inventory write failed: {e}", flush=True)

        if args.docs_audits_dir:
            stamp = args.audit_stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            args.docs_audits_dir.mkdir(parents=True, exist_ok=True)
            anchor_path = args.docs_audits_dir / f"raw_{mode}_profile_{stamp}.md"
            anchor_path.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
            manifest_path = args.docs_audits_dir / "MANIFEST.json"
            if manifest_path.exists():
                import hashlib
                h = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
                rel_path = f"docs/audits/raw_{mode}_profile_{stamp}.md"
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                entries = [e for e in m.get("entries", []) if e.get("path") != rel_path]
                entries.append({"path": rel_path, "sha256": h})
                m["entries"] = entries
                manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    if args.mode in ("offers", "both"):
        _run_mode("offers", delta_offers)
    if args.mode in ("edgar", "both"):
        _run_mode("edgar", delta_edgar)


if __name__ == "__main__":
    main()
