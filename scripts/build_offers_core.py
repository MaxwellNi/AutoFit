from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Add repo root for narrative imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.offers_core_builder import (
    STATIC_COLS_DEFAULT,
    infer_snapshot_time_col,
    build_offers_core_from_snapshots,
)
from narrative.data_preprocessing.parquet_catalog import scan_snapshots


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_offers_core")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _scan_random_fragments(
    base_dir: Path,
    *,
    columns: Sequence[str],
    limit_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    dataset = ds.dataset(
        str(base_dir),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )
    cols = [c for c in columns if c in dataset.schema.names]
    fragments = list(dataset.get_fragments())
    if not fragments:
        return pd.DataFrame(columns=cols)
    rng = np.random.default_rng(seed)
    rng.shuffle(fragments)
    frames = []
    seen = 0
    for frag in fragments:
        frag_cols = [c for c in cols if c in frag.physical_schema.names]
        if not frag_cols:
            continue
        table = frag.to_table(columns=frag_cols)
        df = table.to_pandas()
        if df.empty:
            continue
        if limit_rows is not None:
            remaining = int(limit_rows) - seen
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.sample(n=remaining, random_state=seed)
        frames.append(df)
        seen += len(df)
        if limit_rows is not None and seen >= int(limit_rows):
            break
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def _build_entity_keys(df: pd.DataFrame, id_cols: Sequence[str]) -> pd.Series:
    cols = [c for c in id_cols if c in df.columns]
    if not cols:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[cols].astype(str).agg("|".join, axis=1)


def _fingerprint_dir(path: Path) -> Dict[str, Any]:
    files = sorted([p for p in path.rglob("*.parquet*") if p.is_file() and "_delta_log" not in str(p)])
    total_size = 0
    lines: List[str] = []
    for p in files:
        stat = p.stat()
        total_size += stat.st_size
        lines.append(f"{p}|{stat.st_size}|{stat.st_mtime}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() if lines else None
    return {"file_count": len(files), "size_bytes_total": total_size, "fingerprint_sha256": digest}


def _load_snapshots(
    offers_path: Path,
    *,
    columns: Sequence[str],
    limit_rows: int | None,
    sample_strategy: str,
    sample_seed: int,
    batch_size: int,
    limit_entities: int | None,
    entity_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    selection: Dict[str, Any] = {
        "limit_entities": limit_entities,
        "entity_seed": entity_seed,
        "entity_key_cols": None,
        "selected_entity_hash": None,
        "selected_entity_count": None,
        "selected_offer_id_count": None,
        "filter_col": None,
        "selection_note": None,
    }

    if offers_path.is_file():
        if offers_path.suffix.lower() == ".csv":
            df = pd.read_csv(offers_path)
        else:
            df = pd.read_parquet(offers_path)
        if limit_entities:
            id_cols = ["platform_name", "offer_id"] if all(c in df.columns for c in ["platform_name", "offer_id"]) else ["offer_id"]
            selection["entity_key_cols"] = id_cols
            keys = _build_entity_keys(df, id_cols)
            unique_keys = keys.dropna().unique().tolist()
            rng = np.random.RandomState(entity_seed)
            rng.shuffle(unique_keys)
            selected = unique_keys[: int(limit_entities)]
            selection["selected_entity_hash"] = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
            selection["selected_entity_count"] = len(selected)
            df = df[keys.isin(selected)].copy()
            if "offer_id" in df.columns:
                selection["filter_col"] = "offer_id"
                selection["selected_offer_id_count"] = int(df["offer_id"].nunique())
            selection["selection_note"] = "limit_entities applied on full file"
        elif limit_rows is not None and limit_rows > 0:
            df = df.head(int(limit_rows)).copy()
        return df, selection

    dataset = ds.dataset(
        str(offers_path),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )
    schema_cols = set(dataset.schema.names)
    id_cols = ["platform_name", "offer_id"] if {"platform_name", "offer_id"}.issubset(schema_cols) else ["offer_id"]
    if "offer_id" not in id_cols:
        raise KeyError("offer_id not found in dataset schema")
    selection["entity_key_cols"] = id_cols

    if limit_entities:
        candidate_rows = limit_rows if limit_rows is not None else 200000
        if sample_strategy == "random_fragments":
            sample_df = _scan_random_fragments(
                offers_path,
                columns=id_cols,
                limit_rows=candidate_rows,
                seed=int(sample_seed),
            )
        else:
            sample_df = scan_snapshots(
                [],
                base_dir=offers_path,
                columns=id_cols,
                allow_all=True,
                limit_rows=candidate_rows,
                batch_size=batch_size,
            )
        if sample_df.empty:
            raise ValueError("No rows available for entity sampling")
        keys = _build_entity_keys(sample_df, id_cols)
        unique_keys = keys.dropna().unique().tolist()
        rng = np.random.RandomState(entity_seed)
        rng.shuffle(unique_keys)
        selected = unique_keys[: int(limit_entities)]
        selection["selected_entity_hash"] = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
        selection["selected_entity_count"] = len(selected)
        selection["selection_note"] = f"limit_entities applied; candidates sampled rows={candidate_rows}"

        offer_ids = sample_df[sample_df[id_cols].astype(str).agg("|".join, axis=1).isin(selected)]["offer_id"].dropna().unique().tolist()
        selection["filter_col"] = "offer_id"
        selection["selected_offer_id_count"] = len(offer_ids)
        expr = ds.field("offer_id").isin(offer_ids)
        cols = [c for c in columns if c in dataset.schema.names]
        table = dataset.to_table(columns=cols, filter=expr)
        return table.to_pandas(), selection

    if sample_strategy == "random_fragments":
        df = _scan_random_fragments(
            offers_path,
            columns=columns,
            limit_rows=limit_rows,
            seed=int(sample_seed),
        )
    else:
        df = scan_snapshots(
            [],
            base_dir=offers_path,
            columns=columns,
            allow_all=True,
            limit_rows=limit_rows,
            batch_size=batch_size,
        )
    return df, selection


def _entity_id_checks(df: pd.DataFrame, id_cols: Sequence[str]) -> Dict[str, int]:
    key_pairs = df[list(id_cols) + ["entity_id"]].drop_duplicates()
    n_entities = key_pairs["entity_id"].nunique()
    n_keys = key_pairs[list(id_cols)].drop_duplicates().shape[0]
    n_key_to_entity = (
        key_pairs.groupby(list(id_cols))["entity_id"].nunique().gt(1).sum()
    )
    n_entity_to_key = key_pairs.groupby("entity_id").size().gt(1).sum()
    return {
        "n_entities": int(n_entities),
        "n_keys": int(n_keys),
        "n_key_to_entity_violations": int(n_key_to_entity),
        "n_entity_to_key_violations": int(n_entity_to_key),
    }


def _monotonic_checks(df: pd.DataFrame, id_cols: Sequence[str]) -> Dict[str, int]:
    diffs = (
        df.groupby(list(id_cols), sort=False)["snapshot_ts"]
        .diff()
        .dt.total_seconds()
    )
    n_negative = int((diffs < 0).sum())
    return {"n_negative_deltas": n_negative}


def _t_index_checks(df: pd.DataFrame, id_cols: Sequence[str]) -> Dict[str, int]:
    stats = df.groupby(list(id_cols), sort=False)["t_index"].agg(["min", "max", "count"])
    bad = (stats["min"] != 0) | (stats["max"] != stats["count"] - 1)
    return {"n_non_contiguous": int(bad.sum())}


def _cutoff_checks(df: pd.DataFrame) -> Dict[str, int]:
    if "cutoff_ts" not in df.columns:
        return {
            "n_cutoff_missing": len(df),
            "n_cutoff_gt_snapshot": 0,
            "n_snapshot_gt_cutoff": 0,
        }
    cutoff = pd.to_datetime(df["cutoff_ts"], errors="coerce", utc=True)
    snap = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    has_cutoff = cutoff.notna()
    n_missing = int((~has_cutoff).sum())
    n_cutoff_gt_snap = int((cutoff[has_cutoff] > snap[has_cutoff]).sum())
    n_snap_gt_cutoff = int((snap[has_cutoff] > cutoff[has_cutoff]).sum())
    return {
        "n_cutoff_missing": n_missing,
        "n_cutoff_gt_snapshot": n_cutoff_gt_snap,
        "n_snapshot_gt_cutoff": n_snap_gt_cutoff,
    }


def _join_coverage(
    snapshots_df: pd.DataFrame,
    static_df: pd.DataFrame,
    id_cols: Sequence[str],
) -> Dict[str, float]:
    snap_keys = snapshots_df[list(id_cols)].drop_duplicates()
    static_keys = static_df[list(id_cols)].drop_duplicates()
    if len(snap_keys) == 0:
        return {"snap_keys": 0.0, "static_keys": 0.0, "coverage": 0.0}
    joined = snap_keys.merge(static_keys, on=list(id_cols), how="inner")
    coverage = len(joined) / len(snap_keys)
    return {
        "snap_keys": float(len(snap_keys)),
        "static_keys": float(len(static_keys)),
        "coverage": float(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offers_core parquet from snapshots")
    parser.add_argument("--offers_path", type=str, default="data/raw/offers")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--cutoff_time", type=str, default=None)
    parser.add_argument("--cutoff_mode", choices=["start", "end"], default="start")
    parser.add_argument("--batch_size", type=int, default=200_000)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--limit_entities", type=int, default=None)
    parser.add_argument("--entity_seed", type=int, default=42)
    parser.add_argument("--sample_strategy", choices=["head", "random_fragments"], default="head")
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("runs") / "offers_core" / ts
    output_dir = _resolve_path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "build_offers_core.log"
    logger = _setup_logger(log_path)

    logger.info("build_offers_core start")
    logger.info("offers_path=%s", args.offers_path)
    logger.info("output_dir=%s", output_dir)
    logger.info("limit_rows=%s", args.limit_rows)
    logger.info("limit_entities=%s entity_seed=%s", args.limit_entities, args.entity_seed)
    logger.info("sample_strategy=%s sample_seed=%s", args.sample_strategy, args.sample_seed)
    logger.info("cutoff_mode=%s", args.cutoff_mode)
    logger.info("cutoff_time=%s", args.cutoff_time)

    columns = list(dict.fromkeys(STATIC_COLS_DEFAULT + (
        "funding_goal_usd",
        "funding_raised_usd",
        "investors_count",
        "is_funded",
        "crawled_date",
        "snapshot_date",
        "crawled_date_day",
    )))

    offers_path = _resolve_path(args.offers_path)
    snapshots, selection = _load_snapshots(
        offers_path,
        columns=columns,
        limit_rows=args.limit_rows,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        limit_entities=args.limit_entities,
        entity_seed=args.entity_seed,
    )

    logger.info("loaded snapshots rows=%d cols=%d", len(snapshots), len(snapshots.columns))
    if snapshots.empty:
        raise ValueError("No snapshot rows loaded; check offers_path or filters")

    id_cols = ["platform_name", "offer_id"]
    if not all(c in snapshots.columns for c in id_cols):
        raise KeyError(f"Required id columns missing: {id_cols}")

    time_col = args.time_col or infer_snapshot_time_col(snapshots.columns)
    if time_col is None:
        raise KeyError("No snapshot time column found in data")
    logger.info("snapshot_time_col=%s", time_col)

    core, static_df, info = build_offers_core_from_snapshots(
        snapshots,
        id_cols=id_cols,
        snapshot_time_col=time_col,
        static_cols=STATIC_COLS_DEFAULT,
        cutoff_time=args.cutoff_time,
        cutoff_mode=args.cutoff_mode,
    )

    rename_map: Dict[str, str] = {}
    if "funding_goal_usd" not in core.columns and "funding_goal" in core.columns:
        rename_map["funding_goal"] = "funding_goal_usd"
    if "funding_raised_usd" not in core.columns and "funding_raised" in core.columns:
        rename_map["funding_raised"] = "funding_raised_usd"
    if rename_map:
        core = core.rename(columns=rename_map)

    logger.info("core rows=%d cols=%d", len(core), len(core.columns))
    logger.info("static rows=%d cols=%d", len(static_df), len(static_df.columns))
    logger.info("cutoff_source=%s", info.get("cutoff_source"))

    entity_stats = _entity_id_checks(core, id_cols)
    mono_stats = _monotonic_checks(core, id_cols)
    t_index_stats = _t_index_checks(core, id_cols)
    cutoff_stats = _cutoff_checks(core)
    join_stats = _join_coverage(core, static_df, id_cols)

    logger.info(
        "entity_id deterministic: entities=%d keys=%d key->entity_viol=%d entity->key_viol=%d",
        entity_stats["n_entities"],
        entity_stats["n_keys"],
        entity_stats["n_key_to_entity_violations"],
        entity_stats["n_entity_to_key_violations"],
    )
    logger.info("snapshot_ts monotonic per entity: negative_deltas=%d", mono_stats["n_negative_deltas"])
    logger.info("t_index continuous: non_contiguous=%d", t_index_stats["n_non_contiguous"])
    logger.info(
        "cutoff_ts<=snapshot_ts violations=%d (raw_missing=%d adjusted=%d raw_violations=%d)",
        cutoff_stats["n_cutoff_gt_snapshot"],
        info.get("cutoff_missing_raw"),
        info.get("cutoff_adjusted"),
        info.get("cutoff_raw_violations"),
    )
    logger.info(
        "snapshot_ts<=cutoff_ts violations=%d",
        cutoff_stats["n_snapshot_gt_cutoff"],
    )
    logger.info(
        "static↔snapshots join coverage: snap_keys=%d static_keys=%d coverage=%.4f",
        int(join_stats["snap_keys"]),
        int(join_stats["static_keys"]),
        join_stats["coverage"],
    )

    out_core = output_dir / "offers_core.parquet"
    out_static = output_dir / "offers_static.parquet"
    core.to_parquet(out_core, index=False)
    static_df.to_parquet(out_static, index=False)

    metrics = {
        "rows": int(len(core)),
        "columns": int(len(core.columns)),
        "static_rows": int(len(static_df)),
        "static_columns": int(len(static_df.columns)),
        "snapshot_time_col": info.get("snapshot_time_col"),
        "cutoff_source": info.get("cutoff_source"),
        "cutoff_mode": info.get("cutoff_mode"),
        "cutoff_missing_raw": info.get("cutoff_missing_raw"),
        "cutoff_adjusted": info.get("cutoff_adjusted"),
        "cutoff_raw_violations": info.get("cutoff_raw_violations"),
        "entity_id_checks": entity_stats,
        "monotonic_checks": mono_stats,
        "t_index_checks": t_index_stats,
        "cutoff_checks": cutoff_stats,
        "join_coverage": join_stats,
    }
    (output_dir / "offers_core_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    input_sha = None
    input_size = None
    input_fingerprint = None
    if offers_path.is_file():
        input_sha = _sha256(offers_path)
        input_size = offers_path.stat().st_size
    else:
        input_fingerprint = _fingerprint_dir(offers_path)

    manifest = {
        "offers_path": str(offers_path),
        "input_sha256": input_sha,
        "input_size_bytes": input_size,
        "input_fingerprint": input_fingerprint,
        "output_dir": str(output_dir),
        "rows": int(len(core)),
        "columns": list(core.columns),
        "static_rows": int(len(static_df)),
        "static_columns": list(static_df.columns),
        "rename_map": rename_map,
        "selection": selection,
        "output_sha256": {
            "offers_core.parquet": _sha256(out_core),
            "offers_static.parquet": _sha256(out_static),
        },
        "command": " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("saved offers_core=%s", out_core)
    logger.info("saved offers_static=%s", out_static)
    logger.info("saved metrics=%s", output_dir / "offers_core_metrics.json")
    logger.info("build_offers_core done")


if __name__ == "__main__":
    main()
