#!/usr/bin/env python3
"""
Run a narrow real-data smoke slice for V740-alpha without touching the live
benchmark harness outputs.

This script intentionally reuses the current Block 3 data semantics:
  - canonical WIDE2 freeze
  - same ablation join logic
  - same strict temporal split
  - same leakage-safe feature preparation

But it does NOT register V740-alpha into the active benchmark line and does NOT
write under runs/benchmarks/block3_phase9_fair/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.metrics import (
    mae as metric_mae,
    mape as metric_mape,
    rmse as metric_rmse,
    smape as metric_smape,
)
from src.narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper
from src.narrative.block3.unified_protocol import (
    TemporalSplitConfig,
    apply_temporal_split,
    load_block3_tasks_config,
)
from src.narrative.block3.models.v740_multisource_features import KNOWN_EDGAR_COLUMNS


_TARGET_LEAK_GROUPS: Dict[str, set] = {
    "funding_raised_usd": {
        "funding_raised_usd", "funding_raised",
        "is_funded",
        "investors_count", "non_national_investors",
    },
    "investors_count": {
        "investors_count", "non_national_investors",
        "funding_raised_usd", "funding_raised",
        "is_funded",
    },
    "is_funded": {
        "is_funded",
        "funding_raised_usd", "funding_raised",
        "investors_count", "non_national_investors",
    },
    "funding_goal_usd": {
        "funding_goal_usd", "funding_goal",
        "funding_goal_maximum", "funding_goal_maximum_usd",
    },
}
_ALWAYS_DROP = {
    "entity_id", "crawled_date_day", "cik", "date",
    "offer_id", "snapshot_ts", "crawled_date", "processed_datetime",
}
def _candidate_repo_roots() -> list[Path]:
    roots: list[Path] = [REPO_ROOT]
    env_root = os.environ.get("BLOCK3_CANONICAL_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path.home() / "repo_root")
    roots.append(Path("/work/projects/eint/repo_root"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _resolve_repo_relative(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for root in _candidate_repo_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


_TEXT_EMB_PATH = _resolve_repo_relative("runs/text_embeddings/text_embeddings.parquet")


def _load_pointer() -> Dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "docs" / "audits" / "FULL_SCALE_POINTER.yaml").read_text())


def _iter_entities_with_target_coverage(
    core_path: Path,
    target: str,
    limit: int,
    temporal_config: TemporalSplitConfig,
    require_cik: bool = False,
) -> list[str]:
    pf = pq.ParquetFile(str(core_path))
    stats: Dict[str, Dict[str, Any]] = {}
    train_end = pd.Timestamp(temporal_config.train_end)
    test_start = pd.Timestamp(temporal_config.val_end) + pd.Timedelta(days=temporal_config.embargo_days)
    test_end = pd.Timestamp(temporal_config.test_end)
    columns = ["entity_id", "crawled_date_day", target]
    if require_cik:
        columns.append("cik")

    for batch in pf.iter_batches(columns=columns, batch_size=65536):
        batch_df = batch.to_pandas()
        batch_df["crawled_date_day"] = pd.to_datetime(batch_df["crawled_date_day"], utc=True, errors="coerce")
        batch_df["crawled_date_day"] = batch_df["crawled_date_day"].dt.tz_convert(None)
        for row in batch_df.itertuples(index=False):
            eid = str(row.entity_id)
            st = stats.setdefault(
                eid,
                {
                    "train": False,
                    "test": False,
                    "cik": False,
                    "train_count": 0,
                    "test_count": 0,
                    "train_pos": 0,
                    "train_neg": 0,
                    "test_pos": 0,
                    "test_neg": 0,
                },
            )
            ts = getattr(row, "crawled_date_day")
            value = getattr(row, target)
            if pd.notna(value) and pd.notna(ts):
                is_pos = bool(float(value) > 0.5) if target == "is_funded" else False
                if ts <= train_end:
                    st["train"] = True
                    st["train_count"] += 1
                    if target == "is_funded":
                        st["train_pos" if is_pos else "train_neg"] += 1
                elif test_start < ts <= test_end:
                    st["test"] = True
                    st["test_count"] += 1
                    if target == "is_funded":
                        st["test_pos" if is_pos else "test_neg"] += 1
            if require_cik and pd.notna(getattr(row, "cik", None)):
                st["cik"] = True

    qualified = [
        (eid, st)
        for eid, st in stats.items()
        if st["train"] and st["test"] and (not require_cik or st["cik"])
    ]
    if target == "is_funded":
        def _binary_rank(item: tuple[str, Dict[str, Any]]) -> tuple[Any, ...]:
            eid, st = item
            total = st["train_count"] + st["test_count"]
            total_pos = st["train_pos"] + st["test_pos"]
            pos_rate = (total_pos / total) if total else 0.5
            both_train_classes = st["train_pos"] > 0 and st["train_neg"] > 0
            both_test_classes = st["test_pos"] > 0 and st["test_neg"] > 0
            any_test_pos = st["test_pos"] > 0
            any_test_neg = st["test_neg"] > 0
            return (
                0 if both_train_classes else 1,
                0 if both_test_classes else 1,
                0 if any_test_pos and any_test_neg else 1,
                abs(pos_rate - 0.5),
                -total,
                eid,
            )

        qualified.sort(key=_binary_rank)
    else:
        qualified.sort(key=lambda item: (-(item[1]["train_count"] + item[1]["test_count"]), item[0]))
    return [eid for eid, _ in qualified[:limit]]


def _numeric_schema_columns(parquet_path: Path) -> list[str]:
    schema = pq.ParquetFile(str(parquet_path)).schema_arrow
    out: list[str] = []
    for field in schema:
        t = field.type
        if (
            pa.types.is_integer(t)
            or pa.types.is_floating(t)
            or pa.types.is_decimal(t)
            or pa.types.is_boolean(t)
        ):
            out.append(field.name)
    return out


def _read_parquet_filtered(path: Path, columns: list[str], filters: list[tuple[str, str, list[str]]] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=filters)


def _load_core_slice(pointer: Dict[str, Any], entity_ids: list[str]) -> pd.DataFrame:
    core_path = _resolve_repo_relative(pointer["offers_core_daily"]["dir"]) / "offers_core_daily.parquet"
    numeric_cols = _numeric_schema_columns(core_path)
    keep = ["entity_id", "crawled_date_day", "cik", "snapshot_ts"]
    keep += [c for c in ["funding_raised_usd", "investors_count", "is_funded"] if c not in keep]
    keep += [c for c in numeric_cols if c not in keep]
    return _read_parquet_filtered(core_path, keep, filters=[("entity_id", "in", entity_ids)])


def _load_text_slice(entity_ids: list[str]) -> pd.DataFrame:
    cols = ["entity_id", "crawled_date_day"] + [f"text_emb_{i}" for i in range(64)]
    return _read_parquet_filtered(_TEXT_EMB_PATH, cols, filters=[("entity_id", "in", entity_ids)])


def _load_edgar_slice(pointer: Dict[str, Any], cik_values: list[str]) -> pd.DataFrame:
    base_dir = _resolve_repo_relative(pointer["edgar_store_full_daily"]["dir"])
    ed_dir = base_dir / "edgar_features"
    ed_path = ed_dir if ed_dir.exists() else base_dir
    cols = ["cik", "crawled_date_day", "cutoff_ts", "edgar_filed_date"] + [c for c in KNOWN_EDGAR_COLUMNS if c not in {"snapshot_year"}]
    cols += ["snapshot_year"]
    cols = list(dict.fromkeys(cols))
    return _read_parquet_filtered(ed_path, cols, filters=[("cik", "in", cik_values)])


def _join_text_embeddings(core_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    df = core_df.copy()
    text = text_df.copy()
    df["_cdd_str"] = pd.to_datetime(df["crawled_date_day"]).dt.strftime("%Y-%m-%d")
    text["_cdd_str"] = pd.to_datetime(text["crawled_date_day"]).dt.strftime("%Y-%m-%d")
    emb_cols = [c for c in text.columns if c.startswith("text_emb_")]
    merge_df = text[["entity_id", "_cdd_str"] + emb_cols]
    merged = df.merge(merge_df, on=["entity_id", "_cdd_str"], how="left")
    merged.drop(columns=["_cdd_str"], inplace=True)
    return merged


def _join_edgar_asof(core_df: pd.DataFrame, edgar_df: pd.DataFrame) -> pd.DataFrame:
    df = core_df.copy()
    ed = edgar_df.copy()
    df["crawled_date_day"] = pd.to_datetime(df["crawled_date_day"], utc=True).dt.tz_convert(None).astype("datetime64[ns]")
    ed["crawled_date_day"] = pd.to_datetime(ed["crawled_date_day"], utc=True).dt.tz_convert(None).astype("datetime64[ns]")

    df_cik = df[df["cik"].notna()].sort_values("crawled_date_day").copy()
    df_no_cik = df[df["cik"].isna()].copy()
    ed_sorted = ed.sort_values("crawled_date_day")
    ed_feature_cols = [c for c in ed.columns if c not in ("cik", "crawled_date_day")]

    merged = pd.merge_asof(
        df_cik,
        ed_sorted,
        by="cik",
        on="crawled_date_day",
        direction="backward",
        tolerance=pd.Timedelta("90D"),
        suffixes=("", "_edgar"),
    )
    for ec in ed_feature_cols:
        if ec not in df_no_cik.columns:
            df_no_cik[ec] = np.nan
    return pd.concat([merged, df_no_cik], ignore_index=True)


def _load_smoke_frame(args: argparse.Namespace, temporal_config: TemporalSplitConfig) -> pd.DataFrame:
    pointer = _load_pointer()
    core_path = _resolve_repo_relative(pointer["offers_core_daily"]["dir"]) / "offers_core_daily.parquet"
    entity_ids = _iter_entities_with_target_coverage(
        core_path,
        args.target,
        args.max_entities,
        temporal_config,
        require_cik=args.ablation in {"core_edgar", "core_edgar_seed2", "full"},
    )
    if not entity_ids:
        raise RuntimeError(
            f"No entities found with train/test coverage for target={args.target} "
            f"ablation={args.ablation}"
        )
    df = _load_core_slice(pointer, entity_ids)

    if args.ablation in {"core_text", "full"}:
        df = _join_text_embeddings(df, _load_text_slice(entity_ids))

    if args.ablation in {"core_edgar", "core_edgar_seed2", "full"}:
        ciks = [str(x) for x in df["cik"].dropna().astype(str).unique().tolist()]
        if ciks:
            edgar_df = _load_edgar_slice(pointer, ciks)
            edgar_df["cik"] = edgar_df["cik"].astype(str)
            df["cik"] = df["cik"].astype(str)
            df = _join_edgar_asof(df, edgar_df)

    return df


def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    valid_mask = df[target].notna()
    df_clean = df[valid_mask].copy()
    if len(df_clean) == 0:
        return pd.DataFrame(), pd.Series(dtype=np.float64)

    leak_group = _TARGET_LEAK_GROUPS.get(target, {target})
    drop_cols = _ALWAYS_DROP | leak_group
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    X = df_clean[feature_cols].fillna(0)
    y = df_clean[target]
    return X, y


def _downsample_preserve_time(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    ordered = df.sort_values(["crawled_date_day", "entity_id"], kind="mergesort").reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, num=max_rows, dtype=int)
    idx = np.unique(idx)
    sampled = ordered.iloc[idx].copy()
    return sampled.sort_values(["entity_id", "crawled_date_day"], kind="mergesort").reset_index(drop=True)


def _downsample_binary_preserve_time(df: pd.DataFrame, target: str, max_rows: int) -> pd.DataFrame:
    if target != "is_funded" or max_rows <= 0 or len(df) <= max_rows:
        return _downsample_preserve_time(df, max_rows)
    ordered = df.sort_values(["crawled_date_day", "entity_id"], kind="mergesort").reset_index(drop=True)
    labels = ordered[target].to_numpy(dtype=np.float64)
    pos_idx = np.flatnonzero(labels > 0.5)
    neg_idx = np.flatnonzero(labels <= 0.5)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return _downsample_preserve_time(df, max_rows)

    raw_pos_rate = float(len(pos_idx) / len(labels))
    desired_pos_rate = min(0.75, max(0.25, raw_pos_rate))
    pos_keep = min(len(pos_idx), max(1, int(round(max_rows * desired_pos_rate))))
    neg_keep = min(len(neg_idx), max(1, max_rows - pos_keep))

    chosen = set()
    chosen.update(pos_idx[np.linspace(0, len(pos_idx) - 1, num=pos_keep, dtype=int)].tolist())
    chosen.update(neg_idx[np.linspace(0, len(neg_idx) - 1, num=neg_keep, dtype=int)].tolist())
    if len(chosen) < max_rows:
        remaining = [i for i in range(len(ordered)) if i not in chosen]
        if remaining:
            extra = np.linspace(0, len(remaining) - 1, num=min(max_rows - len(chosen), len(remaining)), dtype=int)
            chosen.update(remaining[i] for i in extra.tolist())

    sampled = ordered.iloc[sorted(chosen)].copy()
    return sampled.sort_values(["entity_id", "crawled_date_day"], kind="mergesort").reset_index(drop=True)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return metrics
    metrics["mae"] = float(metric_mae(y_true, y_pred))
    metrics["rmse"] = float(metric_rmse(y_true, y_pred))
    metrics["smape"] = float(metric_smape(y_true, y_pred))
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        metrics["mape"] = float(metric_mape(y_true[nonzero], y_pred[nonzero]))
    return metrics


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["task1_outcome", "task2_forecast", "task3_risk_adjust"])
    ap.add_argument("--ablation", required=True, choices=[
        "core_only", "core_only_seed2", "core_text", "core_edgar", "core_edgar_seed2", "full",
    ])
    ap.add_argument("--target", required=True, choices=["funding_raised_usd", "investors_count", "is_funded"])
    ap.add_argument("--horizon", type=_positive_int, required=True)
    ap.add_argument("--max-entities", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--input-size", type=int, default=60)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-covariates", type=int, default=15)
    ap.add_argument("--max-windows", type=int, default=50000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-teacher-distill", action="store_true")
    ap.add_argument("--disable-event-head", action="store_true")
    ap.add_argument("--disable-task-modulation", action="store_true")
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-preds", type=Path, default=None)
    ap.add_argument(
        "--skip-if-output-exists",
        action="store_true",
        help="Exit successfully without re-running if output-json already exists.",
    )
    return ap.parse_args()


def _summarize_result(
    args: argparse.Namespace,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preds: np.ndarray,
    metrics: Dict[str, Any],
    model: V740AlphaPrototypeWrapper,
) -> Dict[str, Any]:
    preds = np.asarray(preds, dtype=np.float64)
    finite = np.isfinite(preds)
    coverage = float(finite.mean()) if len(preds) else 0.0
    std = float(np.nanstd(preds)) if len(preds) else 0.0
    constant_prediction = bool(len(preds) > 1 and std < 1e-8)
    return {
        "task": args.task,
        "ablation": args.ablation,
        "target": args.target,
        "horizon": args.horizon,
        "input_size": args.input_size,
        "horizon_to_context_ratio": float(args.horizon / max(1, args.input_size)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]) if len(X_train) else 0,
        "selected_entities": int(
            pd.concat([train["entity_id"], val["entity_id"], test["entity_id"]], ignore_index=True).nunique()
        ),
        "edgar_cols_used": int(len(model._edgar_cols)),
        "text_cols_used": int(len(model._text_cols)),
        "coverage_ratio": coverage,
        "constant_prediction": constant_prediction,
        "prediction_std": std,
        "binary_train_rate": float(getattr(model, "_binary_train_rate", 0.0)) if getattr(model, "_binary_target", False) else None,
        "binary_event_rate": float(getattr(model, "_binary_event_rate", 0.0)) if getattr(model, "_binary_target", False) else None,
        "binary_transition_rate": float(getattr(model, "_binary_transition_rate", 0.0)) if getattr(model, "_binary_target", False) else None,
        "binary_pos_weight": float(getattr(model, "_binary_pos_weight", 1.0)) if getattr(model, "_binary_target", False) else None,
        "binary_temperature": float(getattr(model, "_binary_temperature", 1.0)) if getattr(model, "_binary_target", False) else None,
        "teacher_distill_enabled": bool(getattr(model, "enable_teacher_distill", False)),
        "event_head_enabled": bool(getattr(model, "enable_event_head", False)),
        "task_mod_enabled": bool(getattr(model, "_effective_task_modulation", getattr(model, "enable_task_modulation", False))),
        "binary_teacher_weight": float(getattr(model, "_binary_teacher_weight", 0.0)) if getattr(model, "_binary_target", False) else None,
        "binary_event_weight": float(getattr(model, "_binary_event_weight", 0.0)) if getattr(model, "_binary_target", False) else None,
        "teacher_logistic_mix": float(getattr(model, "_teacher_logistic_mix", 0.0)) if getattr(model, "_binary_target", False) else None,
        "teacher_tree_mix": float(getattr(model, "_teacher_tree_mix", 0.0)) if getattr(model, "_binary_target", False) else None,
        "edgar_source_density": float(getattr(model, "_edgar_source_density", 0.0)),
        "text_source_density": float(getattr(model, "_text_source_density", 0.0)),
        "metrics": metrics,
    }


def main() -> int:
    args = _parse_args()
    if args.skip_if_output_exists and args.output_json is not None and args.output_json.exists():
        print(
            f"[v740-smoke] skip existing artifact: {args.output_json}",
            flush=True,
        )
        return 0
    t0 = time.time()
    print(f"[v740-smoke] start task={args.task} ablation={args.ablation} target={args.target} h={args.horizon}", flush=True)

    tasks_config = load_block3_tasks_config()
    print(f"[v740-smoke] config ready after {time.time() - t0:.2f}s", flush=True)

    t_load = time.time()
    split_cfg = tasks_config.get("split", {})
    temporal_config = TemporalSplitConfig(
        train_end=split_cfg.get("train_end", "2025-06-30"),
        val_end=split_cfg.get("val_end", "2025-09-30"),
        test_end=split_cfg.get("test_end", "2025-12-31"),
        embargo_days=split_cfg.get("embargo_days", 7),
    )
    df = _load_smoke_frame(args, temporal_config)
    train, val, test, _ = apply_temporal_split(df, temporal_config)
    if args.max_rows and len(train) > args.max_rows:
        train = _downsample_binary_preserve_time(train, args.target, args.max_rows)
    print(
        f"[v740-smoke] data loaded in {time.time() - t_load:.2f}s "
        f"(train={len(train):,} val={len(val):,} test={len(test):,})",
        flush=True,
    )

    t_prep = time.time()
    X_train, y_train = _prepare_features(train, args.target)
    X_test, y_test = _prepare_features(test, args.target)
    print(
        f"[v740-smoke] features prepared in {time.time() - t_prep:.2f}s "
        f"(train={len(X_train):,} test={len(X_test):,} feat={X_train.shape[1] if len(X_train) else 0})",
        flush=True,
    )
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows after preparation: train={len(X_train)}, test={len(X_test)}"
        )

    model = V740AlphaPrototypeWrapper(
        input_size=args.input_size,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_covariates=args.max_covariates,
        max_entities=args.max_entities,
        max_windows=args.max_windows,
        patience=args.patience,
        enable_teacher_distill=not args.disable_teacher_distill,
        enable_event_head=not args.disable_event_head,
        enable_task_modulation=not args.disable_task_modulation,
        seed=args.seed,
    )
    t_fit = time.time()
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=args.target,
        task=args.task,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    print(
        f"[v740-smoke] fit complete in {time.time() - t_fit:.2f}s "
        f"(edgar_cols={len(model._edgar_cols)} text_cols={len(model._text_cols)})",
        flush=True,
    )
    t_pred = time.time()
    preds = model.predict(
        X_test,
        test_raw=test,
        target=args.target,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    print(f"[v740-smoke] predict complete in {time.time() - t_pred:.2f}s", flush=True)
    metrics = _compute_metrics(y_test.values, preds)
    summary = _summarize_result(args, train, val, test, X_train, X_test, y_test, preds, metrics, model)
    summary["wall_time_seconds"] = round(time.time() - t0, 3)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_preds is not None:
        args.output_preds.parent.mkdir(parents=True, exist_ok=True)
        out = test.loc[y_test.index, ["entity_id", "crawled_date_day"]].copy()
        out["y_true"] = y_test.values
        out["y_pred"] = preds
        out.to_parquet(args.output_preds, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
