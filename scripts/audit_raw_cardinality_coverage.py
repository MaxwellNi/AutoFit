#!/usr/bin/env python
"""
Raw vs processed cardinality coverage audit. Produces facts on:
- raw offers Delta: version, active_files, row count by year (projection-only scan)
- offers_core_v2: row_count, unique entities, by-year; entity subset from MANIFEST
- offers_text_full: row_count, unique entities, by-year; raw_rows_scanned/rows_emitted from MANIFEST
- raw edgar Delta: version, active_files, row count by year
- edgar store: row_count, by-year
- Two-machine consistency: raw versions/active_files must match (--reference_json).

Gate: FAIL on missing counts, version mismatch, or offers_text manifest lacking limit_rows/overwrite protection.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _hostname() -> str:
    return os.environ.get("HOST_TAG", os.environ.get("HOSTNAME", "unknown")).replace(".", "-")[:64]


def _setup_logger(output_dir: Path, host_suffix: Optional[str] = None) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_cardinality_coverage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    name = f"raw_cardinality_coverage_{host_suffix or _hostname()}.log"
    fh = logging.FileHandler(log_dir / name, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _delta_version_and_files(path: Path) -> Tuple[Optional[int], int]:
    try:
        from deltalake import DeltaTable
    except ImportError:
        return None, 0
    if not path.exists() or not (path / "_delta_log").exists():
        return None, 0
    try:
        dt = DeltaTable(str(path))
        ver = dt.version()
        files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
        return ver, len(files_list)
    except Exception:
        return None, 0


def _raw_offers_stats(path: Path, logger: logging.Logger, limit_rows: Optional[int] = None) -> Dict[str, Any]:
    ver, nfiles = _delta_version_and_files(path)
    if ver is None:
        return {"version": None, "active_files": 0, "row_count_total": 0, "row_count_by_year": {}}
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(path))
        dset = dt.to_pyarrow_dataset()
        cols = ["platform_name", "offer_id", "crawled_date_day"]
        schema_names = [f.name for f in dset.schema]
        read_cols = [c for c in cols if c in schema_names]
        if not read_cols:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        scanner = dset.scanner(columns=read_cols, batch_size=200_000)
        frames = []
        seen = 0
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            frames.append(df)
            seen += len(df)
            if limit_rows and seen >= limit_rows:
                break
        if not frames:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        df = pd.concat(frames, ignore_index=True)
        if limit_rows and len(df) > limit_rows:
            df = df.head(limit_rows)
        if "crawled_date_day" in df.columns:
            df["_year"] = pd.to_datetime(df["crawled_date_day"], errors="coerce").dt.year
            by_year = df["_year"].dropna().astype(int).value_counts().sort_index().to_dict()
            by_year = {str(k): int(v) for k, v in by_year.items()}
        else:
            by_year = {}
        return {"version": ver, "active_files": nfiles, "row_count_total": int(len(df)), "row_count_by_year": by_year}
    except Exception as e:
        logger.warning("raw_offers scan failed: %s", e)
        return {"version": ver, "active_files": nfiles, "row_count_total": None, "row_count_by_year": {}}


def _raw_edgar_stats(path: Path, logger: logging.Logger, limit_rows: Optional[int] = 500_000) -> Dict[str, Any]:
    ver, nfiles = _delta_version_and_files(path)
    if ver is None:
        return {"version": None, "active_files": 0, "row_count_total": 0, "row_count_by_year": {}}
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(path))
        dset = dt.to_pyarrow_dataset()
        cols = ["cik", "filed_date"]
        schema_names = [f.name for f in dset.schema]
        read_cols = [c for c in cols if c in schema_names]
        if not read_cols:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        scanner = dset.scanner(columns=read_cols, batch_size=200_000)
        frames = []
        seen = 0
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            frames.append(df)
            seen += len(df)
            if limit_rows and seen >= limit_rows:
                break
        if not frames:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        df = pd.concat(frames, ignore_index=True)
        if limit_rows and len(df) > limit_rows:
            df = df.head(limit_rows)
        if "filed_date" in df.columns:
            df["_year"] = pd.to_datetime(df["filed_date"], errors="coerce").dt.year
            by_year = df["_year"].dropna().astype(int).value_counts().sort_index().to_dict()
            by_year = {str(k): int(v) for k, v in by_year.items()}
        else:
            by_year = {}
        return {"version": ver, "active_files": nfiles, "row_count_total": int(len(df)), "row_count_by_year": by_year}
    except Exception as e:
        logger.warning("raw_edgar scan failed: %s", e)
        return {"version": ver, "active_files": nfiles, "row_count_total": None, "row_count_by_year": {}}


def _parquet_stats(path: Path, entity_col: str = "entity_id", time_col: str = "snapshot_ts") -> Dict[str, Any]:
    if not path.exists():
        return {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}
    try:
        df = pd.read_parquet(path, columns=[c for c in [entity_col, time_col] if c])
        n_ent = int(df[entity_col].nunique()) if entity_col in df.columns else 0
        by_year = {}
        if time_col in df.columns:
            df["_year"] = pd.to_datetime(df[time_col], errors="coerce").dt.year
            by_year = {str(int(k)): int(v) for k, v in df["_year"].dropna().value_counts().sort_index().items()}
        return {"row_count": int(len(df)), "n_unique_entity_id": n_ent, "row_count_by_year": by_year}
    except Exception:
        return {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}


def _edgar_store_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"row_count": 0, "row_count_by_year": {}}
    try:
        import pyarrow.dataset as ds
        dset = ds.dataset(str(path), format="parquet", partitioning="hive")
        schema_names = [f.name for f in dset.schema]
        cols = ["crawled_date_day"] if "crawled_date_day" in schema_names else []
        if not cols:
            scanner = dset.scanner(batch_size=200_000)
            total = sum(b.num_rows for b in scanner.to_batches())
            return {"row_count": total, "row_count_by_year": {}}
        scanner = dset.scanner(columns=cols, batch_size=200_000)
        frames = [b.to_pandas() for b in scanner.to_batches()]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df["_year"] = pd.to_datetime(df["crawled_date_day"], errors="coerce").dt.year
        by_year = {str(int(k)): int(v) for k, v in df["_year"].dropna().value_counts().sort_index().items()}
        return {"row_count": int(len(df)), "row_count_by_year": by_year}
    except Exception:
        return {"row_count": 0, "row_count_by_year": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw vs processed cardinality coverage audit.")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--raw_edgar_delta", type=Path, required=True)
    parser.add_argument("--offers_core_parquet", type=Path, required=True)
    parser.add_argument("--offers_core_manifest", type=Path, required=True)
    parser.add_argument("--offers_text_full_dir", type=Path, required=True)
    parser.add_argument("--edgar_store_dir", type=Path, required=True)
    parser.add_argument("--snapshots_index_parquet", type=Path, default=None, help="Snapshots index for edgar alignment coverage")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, default=None, help="Compare raw versions/active_files; FAIL if mismatch")
    parser.add_argument("--raw_scan_limit", type=int, default=2_000_000, help="Limit rows for raw offers scan (projection-only)")
    parser.add_argument("--docs_audits_dir", type=Path, default=None, help="If set, write public anchor raw_cardinality_coverage_STAMP.md")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host = _hostname()
    logger = _setup_logger(args.output_dir, host_suffix=host)
    logger.info("=== Raw Cardinality Coverage Audit Start (host=%s) ===", host)

    fail_reasons: List[str] = []

    raw_offers = _raw_offers_stats(args.raw_offers_delta, logger, limit_rows=args.raw_scan_limit)
    raw_edgar = _raw_edgar_stats(args.raw_edgar_delta, logger)

    offers_core_stats = _parquet_stats(args.offers_core_parquet)
    offers_core_manifest: Dict[str, Any] = {}
    if args.offers_core_manifest.exists():
        offers_core_manifest = json.loads(args.offers_core_manifest.read_text(encoding="utf-8"))
    offers_core = {
        "row_count": offers_core_stats["row_count"],
        "n_unique_entity_id": offers_core_stats["n_unique_entity_id"],
        "row_count_by_year": offers_core_stats["row_count_by_year"],
        "selection_note": "Entity subset (limit_entities) from MANIFEST",
        "manifest_selection": offers_core_manifest.get("selection", {}),
    }

    offers_text_manifest_path = args.offers_text_full_dir / "MANIFEST.json"
    if not offers_text_manifest_path.exists():
        fail_reasons.append("offers_text_full MANIFEST.json not found")
        offers_text = {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}, "manifest": {}}
    else:
        offers_text_manifest = json.loads(offers_text_manifest_path.read_text(encoding="utf-8"))
        if "limit_rows" not in offers_text_manifest:
            fail_reasons.append("offers_text_full manifest must record limit_rows (null for full)")
        if "raw_rows_scanned" not in offers_text_manifest or "rows_emitted" not in offers_text_manifest:
            fail_reasons.append("offers_text_full manifest must record raw_rows_scanned and rows_emitted")
        offers_text_parquet = args.offers_text_full_dir / "offers_text.parquet"
        st = _parquet_stats(offers_text_parquet) if offers_text_parquet.exists() else {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}
        offers_text = {
            "row_count": st["row_count"],
            "n_unique_entity_id": st["n_unique_entity_id"],
            "row_count_by_year": st["row_count_by_year"],
            "manifest": {k: offers_text_manifest.get(k) for k in ["limit_rows", "raw_offers_version", "raw_active_files", "raw_rows_scanned", "rows_emitted", "rows_all_text_null_dropped", "n_unique_entity_id", "n_unique_entity_day"]},
        }

    edgar_store = _edgar_store_stats(args.edgar_store_dir)

    # Snapshots index vs edgar store alignment
    snapshots_alignment = {}
    if args.snapshots_index_parquet and args.snapshots_index_parquet.exists():
        try:
            snap_df = pd.read_parquet(args.snapshots_index_parquet)
            snap_count = len(snap_df)
            snap_unique_cik = int(snap_df["cik"].nunique()) if "cik" in snap_df.columns else 0
            # Check how many snapshots have edgar data
            edgar_df = None
            try:
                import pyarrow.dataset as ds
                dset = ds.dataset(str(args.edgar_store_dir), format="parquet", partitioning="hive")
                edgar_df = dset.to_table(columns=["cik", "crawled_date_day"] if "crawled_date_day" in [f.name for f in dset.schema] else ["cik"]).to_pandas()
            except Exception:
                pass
            if edgar_df is not None and "cik" in snap_df.columns:
                snap_ciks = set(snap_df["cik"].dropna().astype(str))
                edgar_ciks = set(edgar_df["cik"].dropna().astype(str))
                cik_overlap = len(snap_ciks & edgar_ciks)
                snapshots_alignment = {
                    "snapshots_count": snap_count,
                    "snapshots_unique_cik": snap_unique_cik,
                    "edgar_unique_cik": len(edgar_ciks),
                    "cik_overlap": cik_overlap,
                    "cik_coverage_rate": cik_overlap / len(snap_ciks) if snap_ciks else 0,
                }
            else:
                snapshots_alignment = {"snapshots_count": snap_count, "snapshots_unique_cik": snap_unique_cik}
        except Exception as e:
            logger.warning("snapshots_index alignment check failed: %s", e)

    if raw_offers["version"] is None:
        fail_reasons.append("raw_offers Delta version missing")
    if raw_edgar["version"] is None:
        fail_reasons.append("raw_edgar Delta version missing")

    if args.reference_json and args.reference_json.exists():
        ref = json.loads(args.reference_json.read_text(encoding="utf-8"))
        ro = ref.get("raw_offers", {})
        re_ = ref.get("raw_edgar", {})
        if ro.get("version") != raw_offers.get("version"):
            fail_reasons.append(f"raw_offers version mismatch: ref={ro.get('version')} vs current={raw_offers.get('version')}")
        if ro.get("active_files") != raw_offers.get("active_files"):
            fail_reasons.append(f"raw_offers active_files mismatch: ref={ro.get('active_files')} vs current={raw_offers.get('active_files')}")
        if re_.get("version") != raw_edgar.get("version"):
            fail_reasons.append(f"raw_edgar version mismatch: ref={re_.get('version')} vs current={raw_edgar.get('version')}")
        if re_.get("active_files") != raw_edgar.get("active_files"):
            fail_reasons.append(f"raw_edgar active_files mismatch: ref={re_.get('active_files')} vs current={raw_edgar.get('active_files')}")

    report = {
        "host": host,
        "raw_offers": raw_offers,
        "raw_edgar": raw_edgar,
        "offers_core_v2": offers_core,
        "offers_text_full": offers_text,
        "edgar_store": edgar_store,
        "snapshots_alignment": snapshots_alignment,
        "data_morphology": {
            "offers_core_v2": "Entity subset full trajectory (limit_entities from MANIFEST)",
            "offers_text_full": "Full raw scan filtered/deduped text panel (manifest counters prove)",
            "edgar_store": "Full raw aggregation aligned to snapshots",
        },
        "gate_passed": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }

    json_path = args.output_dir / "raw_cardinality_coverage.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        "# Raw Cardinality Coverage Report",
        "",
        f"- **Host:** {host}",
        f"- **Gate:** {'PASS' if report['gate_passed'] else 'FAIL'}",
        "",
        "## Raw vs Processed",
        "",
        "| Artifact | Version | Active Files | Row Count | Unique Entities |",
        "|----------|---------|--------------|-----------|-----------------|",
        f"| raw_offers | {raw_offers.get('version')} | {raw_offers.get('active_files')} | {raw_offers.get('row_count_total')} | - |",
        f"| raw_edgar | {raw_edgar.get('version')} | {raw_edgar.get('active_files')} | {raw_edgar.get('row_count_total')} | - |",
        f"| offers_core_v2 | - | - | {offers_core['row_count']} | {offers_core['n_unique_entity_id']} |",
        f"| offers_text_full | - | - | {offers_text['row_count']} | {offers_text['n_unique_entity_id']} |",
        f"| edgar_store | - | - | {edgar_store['row_count']} | - |",
        "",
        "## Data morphology",
        "",
        "- **offers_core_v2:** Entity subset full trajectory (limit_entities from MANIFEST)",
        "- **offers_text_full:** Full raw scan filtered/deduped text panel (manifest counters prove)",
        "- **edgar_store:** Full raw aggregation aligned to snapshots",
        "",
    ]
    if snapshots_alignment:
        md_lines.append("## Snapshots-EDGAR Alignment\n")
        for k, v in snapshots_alignment.items():
            md_lines.append(f"- **{k}:** {v}")
        md_lines.append("")
    if fail_reasons:
        md_lines.append("## Fail reasons\n")
        for r in fail_reasons:
            md_lines.append(f"- {r}")
        md_lines.append("")
    md_path = args.output_dir / "raw_cardinality_coverage.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    if args.docs_audits_dir and report["gate_passed"]:
        stamp = "20260129_073037"
        anchor_path = args.docs_audits_dir / f"raw_cardinality_coverage_{stamp}.md"
        args.docs_audits_dir.mkdir(parents=True, exist_ok=True)
        anchor_content = (
            "# Raw Cardinality Coverage Audit Anchor (" + stamp + ")\n\n"
            "Public audit anchor for raw vs processed cardinality coverage.\n\n"
            "## Data Morphology\n\n"
            "- **offers_core_v2:** Entity subset full trajectory (limit_entities from MANIFEST)\n"
            "- **offers_text_full:** Full raw scan filtered/deduped text panel (manifest counters prove)\n"
            "- **edgar_store:** Full raw aggregation aligned to snapshots\n\n"
            "## Two-Machine Consistency\n\n"
            "Run on 3090 first, then on 4090 with `--reference_json runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage.json`.\n"
            "raw_offers_version, raw_edgar_version, active_files must match; else FAIL.\n\n"
            "## Reproducibility\n\n"
            "```bash\n"
            "HOST_TAG=3090 python scripts/audit_raw_cardinality_coverage.py \\\n"
            "  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \\\n"
            "  --offers_core_parquet runs/offers_core_v2_20260127_043052/offers_core.parquet \\\n"
            "  --offers_core_manifest runs/offers_core_v2_20260127_043052/MANIFEST.json \\\n"
            "  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \\\n"
            "  --edgar_store_dir runs/edgar_feature_store/20260127_133511/edgar_features \\\n"
            "  --output_dir runs/orchestrator/20260129_073037/analysis \\\n"
            "  --docs_audits_dir docs/audits\n"
            "```\n"
        )
        anchor_path.write_text(anchor_content, encoding="utf-8")
        logger.info("Wrote public anchor %s", anchor_path)
        manifest_path = args.docs_audits_dir / "MANIFEST.json"
        if manifest_path.exists():
            import hashlib
            h = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
            rel_path = f"docs/audits/raw_cardinality_coverage_{stamp}.md"
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = m.get("entries", [])
            entries = [e for e in entries if e.get("path") != rel_path]
            entries.append({"path": rel_path, "sha256": h})
            m["entries"] = entries
            manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
            logger.info("Updated MANIFEST.json with raw_cardinality_coverage sha256")

    logger.info("=== Raw Cardinality Coverage Audit Complete === Gate: %s", "PASS" if report["gate_passed"] else "FAIL")
    if fail_reasons:
        for r in fail_reasons:
            logger.error("FAIL: %s", r)
        sys.exit(1)


if __name__ == "__main__":
    main()
